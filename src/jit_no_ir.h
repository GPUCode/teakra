#pragma once
#include "shared_memory.h"
#include <utility>
#include <atomic>
#include <tuple>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <xbyak/xbyak.h>
#include "bit.h"
#include <set>
#include <stack>
#include "core_timing.h"
#include "memory_interface.h"
#include "interpreter.h"
#include "operand.h"
#include "hash.h"
#include "mmio.h"
#include "jit_regs.h"
#include "register.h"
#include "xbyak_abi.h"

#ifdef WIN32
static constexpr bool IsWindows() {
    return true;
}
#else
static constexpr bool IsWindows() {
    return false;
}
#endif

namespace Teakra::Disassembler {

struct ArArpSettings {
    std::array<std::uint16_t, 2> ar;
    std::array<std::uint16_t, 4> arp;
};

std::string Do(std::uint16_t opcode, std::uint16_t expansion = 0,
               std::optional<ArArpSettings> ar_arp = std::nullopt);
}

namespace Teakra {

constexpr size_t MAX_CODE_SIZE = 256 * 1024 * 1024;

#define NOT_IMPLEMENTED() UNREACHABLE()

struct alignas(16) StackLayout {
    s64 cycles_remaining;
    s64 cycles_to_run;
};

class EmitX64 {
    // Technically there's prpage which would make this 22 bits wide, but it's always zero so whatever.
    static constexpr size_t BlockCacheSize = 1ULL << 18;
public:
    EmitX64(CoreTiming& core_timing, JitRegisters& regs, MemoryInterface& mem)
        : core_timing(core_timing), regs(regs), mem(mem), c(MAX_CODE_SIZE) {
        block_cache = std::make_unique<BlockList[]>(BlockCacheSize);
        auto& miu = mem.memory_interface_unit;
        miu.SetOffsets(&regs.x_offset, &regs.y_offset, &regs.z_offset);
        miu.SetPageMode(&regs.page_mode);
        miu.SetMmioBase(&regs.mmio_base);
        EmitDispatcher();
    }

    using RunCodeFuncType = void (*)(EmitX64*);
    RunCodeFuncType run_code{};

    using BlockFunc = void(*)(JitRegisters*);

    struct BlockSwapState {
        Mod1 mod1{};
        Mod0 mod0{};
        Mod2 mod2{};
        std::array<ArpU, 4> arp{};
        std::array<ArU, 2> ar{};
        u16 pad{};
    };

    struct BlockKey {
        BlockSwapState curr{};
        BlockSwapState shadow{};
        Cfg cfgi, cfgj;
        u16 stepi0, stepj0;
        Cfg cfgib, cfgjb;
        u16 stepi0b, stepj0b;
        u32 pc;
        u32 pad;

        FORCE_INLINE bool operator==(const BlockKey& other) const {
            return std::memcmp(this, &other, sizeof(BlockKey)) == 0;
        }
    };

    static_assert(sizeof(BlockKey) == 64);

    struct Block {
        BlockFunc func;
        s32 cycles;
    };

    enum JitStatus {
        Compiling = 0,
        EndStaticJump = 1,
        EndDynJump = 2,
    };

    CoreTiming& core_timing;
    JitRegisters& regs;
    MemoryInterface& mem;
    Xbyak::CodeGenerator c;
    s32 cycles_remaining;
    Xbyak::Label block_exit;
    const std::vector<Matcher<EmitX64>> decoders = GetDecoderTable<EmitX64>();
    std::set<u32> bkrep_end_locations;
    std::set<u32> rep_end_locations;
    bool compiling = false;
    JitStatus status = JitStatus::Compiling;
    using BlockList = std::vector<std::pair<BlockKey, Block>>;
    std::unique_ptr<BlockList[]> block_cache;
    std::stack<u32> call_stack;
    Block* current_blk{};
    BlockKey blk_key{};
    bool unimplemented = false;

    // Emit a call to a class member function, passing "this_object" (+ an adjustment if necessary)
    // As the function's "this" pointer. Only works with classes with single, non-virtual
    // inheritance, hence the static asserts. Those are all we need though, thankfully.
    template <typename T>
    void CallMemberFunction(T func, void* this_object) {
        void* function_ptr;
        uintptr_t this_ptr = reinterpret_cast<uintptr_t>(this_object);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
        static_assert(sizeof(T) == 8, "[x64 JIT] Invalid size for member function pointer");
        std::memcpy(&function_ptr, &func, sizeof(T));
#else
        static_assert(sizeof(T) == 16, "[x64 JIT] Invalid size for member function pointer");
        uint64_t arr[2];
        std::memcpy(arr, &func, sizeof(T));
        // First 8 bytes correspond to the actual pointer to the function
        function_ptr = reinterpret_cast<void*>(arr[0]);
        // Next 8 bytes correspond to the "this" pointer adjustment
        this_ptr += arr[1];
#endif

        // Load the "this" pointer to arg1 and emit a call to the function
        c.mov(ABI_PARAM1, this_ptr);
        CallFarFunction(c, function_ptr);
    }

    void Reset() {
        // Reset registers
        regs.Reset();

        // Clear any program data from previous runs
        block_cache = std::make_unique<BlockList[]>(BlockCacheSize);
        bkrep_end_locations.clear();
        rep_end_locations.clear();

        // Reset code generator and emit the dispatcher again
        c.reset();
        EmitDispatcher();
    }

    u32 Run(s64 cycles) {
        cycles_remaining = cycles;
        current_blk = nullptr;
        regs.idle = false;
        run_code(this);
        return std::abs(cycles_remaining);
    }

    void EmitDispatcher() {
        run_code = c.getCurr<RunCodeFuncType>();
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLEE_SAVED, 8, 16);

        Xbyak::Label dispatcher_start, dispatcher_end;
        c.L(dispatcher_start);

        // Call LookupBlock. This will compile a new block and increment timers for us
        CallMemberFunction(&EmitX64::LookupNewBlock, this);
        c.test(ABI_RETURN, ABI_RETURN);
        c.jz(dispatcher_end);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&regs));
        c.jmp(ABI_RETURN);
        c.L(block_exit);
        CallMemberFunction(&EmitX64::DoInterruptsAndRunDebug, this);
        c.jmp(dispatcher_start);
        c.L(dispatcher_end);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLEE_SAVED, 8, 16);
        c.ret();
        c.ready();
    }

    BlockFunc LookupNewBlock() {
        if (cycles_remaining <= 0) {
            return nullptr;
        }

        // State for bank exchange.
        blk_key.pc = regs.pc;
        std::memcpy(&blk_key.cfgi, &regs.cfgi, sizeof(u16) * 8);
        // Current state
        std::memcpy(&blk_key.curr.mod1, &regs.mod1, sizeof(u16) * 3);
        std::memcpy(&blk_key.curr.arp, &regs.arp, sizeof(regs.arp) + sizeof(regs.ar));
        // Shadow state.
        std::memcpy(&blk_key.shadow.mod1, &regs.mod1b, sizeof(u16) * 3);
        std::memcpy(&blk_key.shadow.arp, &regs.arpb, sizeof(regs.arpb) + sizeof(regs.arb));

        LookupBlock();

        // Check if we are idle, and skip ahead
        if (regs.idle) {
            u64 skipped = core_timing.Skip(cycles_remaining - 1);
            cycles_remaining -= skipped;
            // Skip additional tick so to let components fire interrupts
            if (cycles_remaining > 1) {
                cycles_remaining--;
                core_timing.Tick();
            }
        }

        // Check for interrupts.
        for (std::size_t i = 0; i < 3; ++i) {
            if (interrupt_pending[i]) {
                regs.ip[i] = 1;
                interrupt_pending[i] = false;
            }
        }

        if (vinterrupt_pending) {
            regs.ipv = 1;
            vinterrupt_pending = false;
        }

        // Return the block function to execute.
        return current_blk->func;
    }

    FORCE_INLINE void LookupBlock() {
        auto& vec = block_cache[regs.pc];
        for (auto& [key, block] : vec) {
            if (key == blk_key) {
                current_blk = &block;
                return;
            }
        }

        auto& [key, blk] = vec.emplace_back();
        key = blk_key;
        current_blk = &blk;
        CompileBlock(blk);
        //printf("Compiling block at 0x%x with size = %d\n", blk_key.pc, blk.cycles);
    }

    void DoInterruptsAndRunDebug() {
        if (regs.ie && !regs.rep) {
            bool interrupt_handled = false;
            for (u32 i = 0; i < regs.im.size(); ++i) {
                if (regs.im[i] && regs.ip[i]) {
                    regs.ip[i] = 0;
                    regs.ie = 0;
                    PushPC();
                    regs.pc = 0x0006 + i * 8;
                    regs.idle = false;
                    interrupt_handled = true;
                    if (regs.ic[i]) {
                        ContextStore();
                    }
                    break;
                }
            }
            if (!interrupt_handled && regs.imv && regs.ipv) {
                regs.ipv = 0;
                regs.ie = 0;
                PushPC();
                regs.pc = vinterrupt_address;
                regs.idle = false;
                if (vinterrupt_context_switch) {
                    ContextStore();
                }
            }
        }

        // Count the cycles of the previous executed block.
        core_timing.Tick(current_blk->cycles);
        cycles_remaining -= current_blk->cycles;
    }

    void CompileBlock(Block& blk) {
        // Load block state
        blk.func = c.getCurr<BlockFunc>();
        c.mov(REGS, ABI_PARAM1);
        c.mov(R0_1_2_3, qword[REGS + offsetof(JitRegisters, r)]);
        c.mov(R4_5_6_7, qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4]);
        c.mov(FACTORS, qword[REGS + offsetof(JitRegisters, y)]);
        c.mov(A[0], qword[REGS + offsetof(JitRegisters, a)]);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)]);
        c.mov(B[0], qword[REGS + offsetof(JitRegisters, b)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)]);
        c.mov(FLAGS, word[REGS + offsetof(JitRegisters, flags)]);

        call_stack = {};

        //Disassembler::ArArpSettings settings;
        //std::memcpy(&settings.ar, &blk_key.curr.ar, sizeof(settings.ar));
        //std::memcpy(&settings.arp, &blk_key.curr.arp, sizeof(settings.arp));

        compiling = true;
        while (compiling) {
            const u32 current_pc = regs.pc;
            u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            auto& decoder = decoders[opcode];
            u16 expand_value = 0;
            if (decoder.NeedExpansion()) {
                expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            }

            decoder.call(*this, opcode, expand_value);
            blk.cycles++;

            if (rep_end_locations.contains(current_pc)) {
                Xbyak::Label end_label, jump_back_label;
                c.cmp(word[REGS + offsetof(JitRegisters, repc)], 0);
                c.jnz(jump_back_label);
                c.mov(word[REGS + offsetof(JitRegisters, rep)], false);
                c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc); // Loop done, move to next
                c.jmp(end_label);
                c.L(jump_back_label);
                c.sub(word[REGS + offsetof(JitRegisters, repc)], 1); // Loop not done, go back to current
                c.mov(dword[REGS + offsetof(JitRegisters, pc)], current_pc);
                c.L(end_label);
                compiling = false;
            }

            if (bkrep_end_locations.contains(regs.pc - 1)) {
                EmitBkrepReturn(regs.pc);
            }

            //const auto name = Disassembler::Do(opcode, expand_value, settings);
            //printf("%s\n", name.c_str());
        }

        // Flush block state
        EmitBlockExit();
    }

    void EmitBlockExit() {
        c.mov(qword[REGS + offsetof(JitRegisters, r)], R0_1_2_3);
        c.mov(qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4], R4_5_6_7);
        c.mov(qword[REGS + offsetof(JitRegisters, y)], FACTORS);
        c.mov(qword[REGS + offsetof(JitRegisters, a)], A[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b)], B[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)], B[1]);
        c.mov(word[REGS + offsetof(JitRegisters, flags)], FLAGS.cvt16());
        c.jmp(block_exit);
    }

    void EmitBkrepReturn(u32 next_pc) {
        using Frame = JitRegisters::BlockRepeatFrame;
        // if (regs.lp && regs.bkrep_stack[regs.bcn - 1].end + 1 == regs.pc) {
        //      if (regs.bkrep_stack[regs.bcn - 1].lc == 0) {
        //         --regs.bcn;
        //         regs.lp = regs.bcn != 0;
        //      } else {
        //         --regs.bkrep_stack[regs.bcn - 1].lc;
        //         regs.pc = regs.bkrep_stack[regs.bcn - 1].start;
        //      }
        // }
        Xbyak::Label end_label, jump_to_target;
        const Reg64 bcn = rax;
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.jz(end_label);
        c.movzx(bcn, word[REGS + offsetof(JitRegisters, bcn)]);
        c.sub(bcn, 1);
        c.lea(rbx, ptr[bcn + bcn * 2]);
        c.lea(rbx, ptr[REGS + offsetof(JitRegisters, bkrep_stack) + rbx * 4]);
        c.cmp(dword[rbx + offsetof(Frame, end)], regs.pc - 1);
        c.jne(end_label);
        c.cmp(word[rbx + offsetof(Frame, lc)], 0);
        c.jne(jump_to_target);
        c.mov(word[REGS + offsetof(JitRegisters, bcn)], bcn.cvt16());
        c.test(bcn, bcn);
        c.setnz(byte[REGS + offsetof(JitRegisters, lp)]);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], next_pc);
        c.jmp(end_label);
        c.L(jump_to_target);
        c.sub(word[rbx + offsetof(Frame, lc)], 1);
        c.mov(rbx.cvt32(), dword[rbx + offsetof(Frame, start)]);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], rbx.cvt32());
        c.L(end_label);
        compiling = false;
    }

    void EmitPushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        const Reg16 sp = bx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        if (regs.cpc == 1) {
            NOT_IMPLEMENTED();
        } else {
            StoreToMemory(sp, l);
            c.sub(sp, 1);
            StoreToMemory(sp, h);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void PushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        if (regs.cpc == 1) {
            mem.DataWrite(--regs.sp, h);
            mem.DataWrite(--regs.sp, l);
        } else {
            mem.DataWrite(--regs.sp, l);
            mem.DataWrite(--regs.sp, h);
        }
    }

    void EmitPopPC() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 pc = rcx;
        c.xor_(pc, pc);
        if (regs.cpc == 1) {
            NOT_IMPLEMENTED();
        } else {
            EmitLoadFromMemory<true>(pc, sp);
            c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
            c.add(sp, 1);
            c.shl(pc, 16);
            EmitLoadFromMemory<true>(pc, sp);
            c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
            c.add(sp, 2);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc.cvt32());
    }

    void PopPC() {
        u16 h, l;
        if (regs.cpc == 1) {
            l = mem.DataRead(regs.sp++);
            h = mem.DataRead(regs.sp++);
        } else {
            h = mem.DataRead(regs.sp++);
            l = mem.DataRead(regs.sp++);
        }
        SetPC(l | ((u32)h << 16));
    }

    void SetPC(u32 new_pc) {
        ASSERT(new_pc < 0x40000);
        regs.pc = new_pc;
        compiling = false;
    }

    void SignalInterrupt(u32 i) {
        interrupt_pending[i] = true;
    }
    void SignalVectoredInterrupt(u32 address, bool context_switch) {
        vinterrupt_address = address;
        vinterrupt_pending = true;
        vinterrupt_context_switch = context_switch;
    }

    using instruction_return_type = void;

    std::array<bool, 3> interrupt_pending{};
    bool vinterrupt_pending{false};
    bool vinterrupt_context_switch;
    u32 vinterrupt_address;

    void nop() {
        // literally nothing
    }

    void undefined(u16 opcode) {
        std::printf("Undefined opcode: 0x%x\n", opcode);
        NOT_IMPLEMENTED();
    }

    void ContextStore() {
        NOT_IMPLEMENTED();
    }

    void ContextRestore() {
        NOT_IMPLEMENTED();
    }

    void norm(Ax a, Rn b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void swap(SwapType swap) {
        RegName s0, d0, s1, d1;
        const Reg64 u = rax, v = rbx;
        switch (swap.GetName()) {
        case SwapTypeValue::a0b0:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a1b0:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a1b1:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1b1:
            GetAcc(u, RegName::a1);
            GetAcc(v, RegName::b1);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b1, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1a1b0:
            GetAcc(u, RegName::a1);
            GetAcc(v, RegName::b0);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b0, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b0;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a0b1a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b1;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a1b0a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b0;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::a1b1a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b1;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::b0a0b1:
            s0 = d1 = RegName::a0;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b0a1b1:
            s0 = d1 = RegName::a1;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b1a0b0:
            s0 = d1 = RegName::a0;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        case SwapTypeValue::b1a1b0:
            s0 = d1 = RegName::a1;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        default:
            UNREACHABLE();
        }
        GetAcc(u, s0);
        GetAcc(v, s1);
        SatAndSetAccAndFlag(d0, u);
        SatAndSetAccAndFlag(d1, v); // only this one affects flags (except for fl)
    }
    void trap() {
        NOT_IMPLEMENTED();
    }

    void EmitLoadFunctionCall(Reg64 out, Reg64 address) {
        // TODO: Non MMIO reads can be performed inside the JIT.
        // Push all registers because our JIT assumes everything is non volatile
        c.push(rbp);
        c.push(rbx);
        c.push(rcx);
        c.push(rdx);
        c.push(rsi);
        c.push(rdi);
        c.push(r8);
        c.push(r9);
        c.push(r10);
        c.push(r11);
        c.push(r12);
        c.push(r13);
        c.push(r14);
        c.push(r15);

        c.mov(rbp, rsp);
        // Reserve a bunch of stack space for Windows shadow stack et al, then force align rsp to 16 bytes to respect the ABI
        c.sub(rsp, 64);
        c.and_(rsp, ~0xF);

        c.movzx(ABI_PARAM2, address.cvt16());
        c.xor_(ABI_PARAM3.cvt32(), ABI_PARAM3.cvt32()); // bypass_mmio = false
        CallMemberFunction(&MemoryInterface::DataRead, &mem);

        // Undo anything we did
        c.mov(rsp, rbp);
        c.pop(r15);
        c.pop(r14);
        c.pop(r13);
        c.pop(r12);
        c.pop(r11);
        c.pop(r10);
        c.pop(r9);
        c.pop(r8);
        c.pop(rdi);
        c.pop(rsi);
        c.pop(rdx);
        c.pop(rcx);
        c.pop(rbx);
        c.pop(rbp);
        c.mov(out.cvt16(), ABI_RETURN.cvt16());
    }

    void EmitConvertAddress(Reg64 addr, Reg64 scratch) {
        // NOTE: This assumes x_size[0] is always 0x1E!
        c.mov(scratch.cvt32(), dword[REGS + offsetof(JitRegisters, x_offset)]);
        c.cmp(addr, 0x1E * MemoryInterfaceUnit::XYSizeResolution);
        c.cmovg(scratch.cvt32(), dword[REGS + offsetof(JitRegisters, y_offset)]);
        c.cmp(word[REGS + offsetof(JitRegisters, page_mode)], 0);
        c.cmove(scratch.cvt32(), dword[REGS + offsetof(JitRegisters, z_offset)]);
        c.add(addr, scratch);
    }

    template <bool bypass_mmio = false>
    void EmitLoadFromMemory(Reg64 out, Reg64 address) {
        Xbyak::Label end_label, read_label;
        const Reg64 scratch = rsi;
        if constexpr (!bypass_mmio) {
            c.movzx(scratch, word[REGS + offsetof(JitRegisters, mmio_base)]);
            c.cmp(address, scratch);
            c.jl(read_label);
            c.add(scratch, MemoryInterfaceUnit::MMIOSize);
            c.cmp(address, scratch);
            c.jge(read_label);
            EmitLoadFunctionCall(out, address);
            c.jmp(end_label);
            c.L(read_label);
        }

        EmitConvertAddress(address, scratch);
        c.mov(scratch, reinterpret_cast<uintptr_t>(mem.shared_memory.raw));
        c.mov(out.cvt16(), word[scratch + address * 2]);

        if constexpr (!bypass_mmio) {
            c.L(end_label);
        }
    }

    void LoadFromMemory(Reg64 out, MemImm8 addr) {
        LoadFromMemory(out, addr.Unsigned16() + (blk_key.curr.mod1.page << 8));
    }

    template <typename T>
    void LoadFromMemory(Reg64 out, T addr) {
        // TODO: Non MMIO reads can be performed inside the JIT.
        // Push all registers because our JIT assumes everything is non volatile
        c.push(rbp);
        c.push(rbx);
        c.push(rcx);
        c.push(rdx);
        c.push(rsi);
        c.push(rdi);
        c.push(r8);
        c.push(r9);
        c.push(r10);
        c.push(r11);
        c.push(r12);
        c.push(r13);
        c.push(r14);
        c.push(r15);

        c.mov(rbp, rsp);
        // Reserve a bunch of stack space for Windows shadow stack et al, then force align rsp to 16 bytes to respect the ABI
        c.sub(rsp, 64);
        c.and_(rsp, ~0xF);

        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            c.movzx(ABI_PARAM2, addr.cvt16());
        } else {
            c.mov(ABI_PARAM2, addr & 0xFFFF);
        }
        c.xor_(ABI_PARAM3.cvt32(), ABI_PARAM3.cvt32()); // bypass_mmio = false
        CallMemberFunction(&MemoryInterface::DataRead, &mem);

        // Undo anything we did
        c.mov(rsp, rbp);
        c.pop(r15);
        c.pop(r14);
        c.pop(r13);
        c.pop(r12);
        c.pop(r11);
        c.pop(r10);
        c.pop(r9);
        c.pop(r8);
        c.pop(rdi);
        c.pop(rsi);
        c.pop(rdx);
        c.pop(rcx);
        c.pop(rbx);
        c.pop(rbp);
        c.mov(out.cvt16(), ABI_RETURN.cvt16());
    }

    template <typename T>
    void LoadFromMemory(Xbyak::Address out, T addr) {
        // TODO: Non MMIO reads can be performed inside the JIT.
        // Push all registers because our JIT assumes everything is non volatile
        c.push(rbp);
        c.push(rbx);
        c.push(rcx);
        c.push(rdx);
        c.push(rsi);
        c.push(rdi);
        c.push(r8);
        c.push(r9);
        c.push(r10);
        c.push(r11);
        c.push(r12);
        c.push(r13);
        c.push(r14);
        c.push(r15);

        c.mov(rbp, rsp);
        // Reserve a bunch of stack space for Windows shadow stack et al, then force align rsp to 16 bytes to respect the ABI
        c.sub(rsp, 64);
        c.and_(rsp, ~0xF);

        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            c.movzx(ABI_PARAM2, addr.cvt16());
        } else {
            c.mov(ABI_PARAM2, addr & 0xFFFF);
        }
        c.xor_(ABI_PARAM3.cvt32(), ABI_PARAM3.cvt32()); // bypass_mmio = false
        CallMemberFunction(&MemoryInterface::DataRead, &mem);

        // Undo anything we did
        c.mov(rsp, rbp);
        c.pop(r15);
        c.pop(r14);
        c.pop(r13);
        c.pop(r12);
        c.pop(r11);
        c.pop(r10);
        c.pop(r9);
        c.pop(r8);
        c.pop(rdi);
        c.pop(rsi);
        c.pop(rdx);
        c.pop(rcx);
        c.pop(rbx);
        c.pop(rbp);
        c.mov(out, ABI_RETURN.cvt16());
    }

    void DoMultiplication(u32 unit, Reg32 x, Reg32 y, bool x_sign, bool y_sign) {
        c.mov(x.cvt64(), FACTORS);
        if (unit == 0) {
            c.movzx(y, FACTORS.cvt16());
            c.shr(x.cvt64(), 32);
            c.and_(x, 0xFFFF);
        } else {
            c.shr(x.cvt64(), 16);
            c.movzx(y, x.cvt16());
            c.shr(x.cvt64(), 32);
        }
        if (blk_key.curr.mod0.hwm == 1 || (blk_key.curr.mod0.hwm == 3 && unit == 0)) {
            c.shr(y, 8);
        } else if (blk_key.curr.mod0.hwm == 2 || (blk_key.curr.mod0.hwm == 3 && unit == 1)) {
            c.and_(y, 0xFF);
        }
        if (x_sign) {
            SignExtend(x, 16);
        }
        if (y_sign) {
            SignExtend(y, 16);
        }
        c.imul(x, y);
        c.mov(dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit], x);
        if (x_sign || y_sign) {
            c.shr(x, 31);
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], x.cvt16());
        } else {
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], 0);
        }
    }

    void AlmGeneric(AlmOp op, Reg64 a, Ax b) {
        const Reg64 value = rax;
        switch (op) {
        case AlmOp::Or: {
            GetAcc(value, b.GetName());
            c.or_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::And: {
            GetAcc(value, b.GetName());
            c.and_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Xor: {
            GetAcc(value, b.GetName());
            c.xor_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Tst0: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.test(value, a);
            c.setz(a.cvt8());
            c.shl(a, 15);
            c.shr(a, 15 - decltype(Flags::fz)::position);
            c.or_(FLAGS.cvt16(), a.cvt16()); // regs.fz = (value & a) == 0;
            break;
        }
        case AlmOp::Tst1: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.not_(a);
            c.test(value, a);
            c.setz(a.cvt8());
            c.shl(a, 15);
            c.shr(a, 15 - decltype(Flags::fz)::position);
            c.or_(FLAGS, a); // regs.fz = (value & ~a) == 0;
            break;
        }
        case AlmOp::Cmp:
        case AlmOp::Cmpu:
        case AlmOp::Sub:
        case AlmOp::Subl:
        case AlmOp::Subh:
        case AlmOp::Add:
        case AlmOp::Addl:
        case AlmOp::Addh: {
            GetAcc(value, b.GetName());
            const bool sub = !(op == AlmOp::Add || op == AlmOp::Addl || op == AlmOp::Addh);
            const Reg64 result = rcx;
            AddSub(value, a, result, sub);
            if (op == AlmOp::Cmp || op == AlmOp::Cmpu) {
                SetAccFlag(result);
            } else {
                SatAndSetAccAndFlag(b.GetName(), result);
            }
            break;
        }
        case AlmOp::Msu: {
            NOT_IMPLEMENTED();
            GetAcc(value, b.GetName());
            const Reg64 product = rcx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rdx;
            AddSub(value, product, result, true);
            SatAndSetAccAndFlag(b.GetName(), result);

            c.shl(a, 48);
            c.shr(a, 16);
            c.mov(result, ~decltype(Factors::x0)::mask);
            c.and_(FACTORS, result);
            c.or_(FACTORS, a); // regs.x[0] = a & 0xFFFF;
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }
        case AlmOp::Sqra: {
            GetAcc(value, b.GetName());
            const Reg64 product = rdx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rcx;
            AddSub(value, product, result, false);
            SatAndSetAccAndFlag(b.GetName(), result);
        }
            [[fallthrough]];
        case AlmOp::Sqr: {
            c.and_(a, 0xFFFF);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.ror(FACTORS, 32);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.rol(FACTORS, 32);
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }

        default:
            NOT_IMPLEMENTED();
        }
    }

    void ExtendOperandForAlm(AlmOp op, Reg64 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            SignExtend(a, 16);
            break;
        case AlmOp::Addh:
        case AlmOp::Subh:
            c.shl(a, 16);
            SignExtend(a, 32);
            break;
        default:
            c.movzx(a.cvt64(), a.cvt16());
            break;
        }
    }

    u64 ExtendOperandForAlm(AlmOp op, u16 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            return ::SignExtend<16, u64>(a);
        case AlmOp::Addh:
        case AlmOp::Subh:
            return ::SignExtend<32, u64>((u64)a << 16);
        default:
            return a;
        }
    }

    void alm(Alm op, MemImm8 a, Ax b) {
        const Reg64 value = rbx;
        const Reg64 address = rcx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alm(Alm op, Rn a, StepZIDS as, Ax b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alm(Alm op, Register a, Ax b) {
        const Reg64 value = rbx;
        auto CheckBus40OperandAllowed = [&] {
            static const std::unordered_set<AlmOp> allowed_instruction{
                AlmOp::Or, AlmOp::And, AlmOp::Xor, AlmOp::Add, AlmOp::Cmp, AlmOp::Sub,
            };
            if (allowed_instruction.count(op.GetName()) == 0)
                NOT_IMPLEMENTED(); // weird effect. probably undefined
        };
        switch (a.GetName()) {
        // need more test
        case RegName::p:
            CheckBus40OperandAllowed();
            ProductToBus40(value, Px{0});
            break;
        case RegName::a0:
        case RegName::a1:
            CheckBus40OperandAllowed();
            GetAcc(value, a.GetName());
            break;
        default:
            RegToBus16(a.GetName(), value);
            ExtendOperandForAlm(op.GetName(), value);
            break;
        }
        AlmGeneric(op.GetName(), value, b);
    }
    void alm_r6(Alm op, Ax b) {
        NOT_IMPLEMENTED();
    }

    void alu(Alu op, MemImm16 a, Ax b) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16());
        EmitLoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alu(Alu op, MemR7Imm16 a, Ax b) {
        const Reg64 address = rax;
        const Reg64 value = rbx;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address, a.Unsigned16());
        EmitLoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alu(Alu op, Imm16 a, Ax b) {
        u16 value = a.Unsigned16();
        c.mov(rbx, ExtendOperandForAlm(op.GetName(), value));
        AlmGeneric(op.GetName(), rbx, b);
    }
    void alu(Alu op, Imm8 a, Ax b) {
        u16 value = a.Unsigned16();
        const Reg64 and_backup = rsi;
        c.xor_(and_backup.cvt32(), and_backup.cvt32());
        if (op.GetName() == AlmOp::And) {
            // AND instruction has a special treatment:
            // bit 8~15 are unaffected in the accumulator, but the flags are set as if they are
            // affected
            GetAcc(and_backup, b.GetName());
            c.and_(and_backup, 0xFF00);
        }
        value = ExtendOperandForAlm(op.GetName(), value);
        c.mov(rbx, value);
        AlmGeneric(op.GetName(), rbx, b);
        if (op.GetName() == AlmOp::And) {
            const Reg64 and_new = rax;
            GetAcc(and_new, b.GetName());
            c.and_(and_new.cvt16(), 0x00FF);
            c.or_(and_new, and_backup);
            SetAcc(b.GetName(), and_new);
        }
    }
    void alu(Alu op, MemR7Imm7s a, Ax b) {
        const Reg64 address = rax;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address.cvt32(), a.Signed16());
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }

    void or_(Ab a, Ax b, Ax c) {
        GetAcc(rax, a.GetName());
        GetAcc(rbx, b.GetName());
        this->c.or_(rax, rbx);
        SetAccAndFlag(c.GetName(), rax);
    }
    void or_(Ax a, Bx b, Ax c) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        this->c.or_(value, GetAccDirect(b.GetName()));
        SetAccAndFlag(c.GetName(), value);
    }
    void or_(Bx a, Bx b, Ax c) {
        NOT_IMPLEMENTED();
    }

    u16 GenericAlbConst(Alb op, u16 a, u16 b) {
        u16 result;
        switch (op.GetName()) {
        case AlbOp::Set: {
            result = a | b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Rst: {
            result = ~a & b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Chng: {
            result = a ^ b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Addv: {
            u32 r = a + b;
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            result = r & 0xFFFF;
            break;
        }
        case AlbOp::Tst0: {
            result = (a & b) != 0;
            break;
        }
        case AlbOp::Tst1: {
            result = (a & ~b) != 0;
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            u32 r = b - a;
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            result = r & 0xFFFF;
            break;
        }
        default:
            UNREACHABLE();
        }
        if (result == 0) {
            c.bts(FLAGS, decltype(Flags::fz)::position);
        } else {
            c.btr(FLAGS, decltype(Flags::fz)::position);
        }
        return result;
    }

    void GenericAlb(Alb op, u16 a, Reg16 b, Reg16 result) {
        switch (op.GetName()) {
        case AlbOp::Set: {
            c.or_(b.cvt32(), a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Rst: {
            [[maybe_unused]] const u16 a_not = ~a;
            c.and_(b.cvt32(), a_not);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Chng: {
            c.xor_(b.cvt32(), a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS, b);
            break;
        }
        case AlbOp::Addv: {
            c.movsx(result.cvt32(), b);
            c.add(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS, result);

            c.movzx(result.cvt32(), b);
            c.add(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        case AlbOp::Tst0: {
            c.xor_(result, result);
            c.and_(b.cvt32(), a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Tst1: {
            c.xor_(result, result);
            c.not_(b);
            c.and_(b.cvt32(), a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            c.movsx(result.cvt32(), b);
            c.sub(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS, result);

            c.movzx(result.cvt32(), b);
            c.sub(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        default:
            UNREACHABLE();
        }
        c.xor_(b, b);
        c.test(result.cvt16(), result.cvt16());
        c.setz(b.cvt8());
        c.shl(b, decltype(Flags::fz)::position);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.or_(FLAGS, b);
    }

    static bool IsAlbModifying(Alb op) {
        switch (op.GetName()) {
        case AlbOp::Set:
        case AlbOp::Rst:
        case AlbOp::Chng:
        case AlbOp::Addv:
        case AlbOp::Subv:
            return true;
        case AlbOp::Tst0:
        case AlbOp::Tst1:
        case AlbOp::Cmpv:
            return false;
        default:
            UNREACHABLE();
        }
    }

    static bool IsAlbConst(SttMod b) {
        switch (b.GetName()) {
        case RegName::mod0:
        case RegName::mod1:
        case RegName::mod2:
            return true;
        default:
            return false;
        }
    }

    void alb(Alb op, Imm16 a, MemImm8 b) {
        const Reg64 bv = rax;
        const Reg64 address = rbx;
        c.mov(address, b.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(bv, address);
        const Reg64 result = rbx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            StoreToMemory(b.Unsigned16() + (blk_key.curr.mod1.page << 8), result);
        }
    }
    void alb(Alb op, Imm16 a, Rn b, StepZIDS bs) {
        const Reg64 address = rbx;
        RnAddressAndModify(b.Index(), bs.GetName(), address);
        const Reg64 bv = rax;
        EmitLoadFromMemory(bv, address);
        const Reg64 result = rcx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            StoreToMemory(address, result);
        }
    }
    void alb(Alb op, Imm16 a, Register b) {
        const Reg64 bv = rax;
        if (b.GetName() == RegName::p) {
            ProductToBus40(bv, Px{0});
            c.shr(bv, 16);
        } else if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
            NOT_IMPLEMENTED(); // weird effect;
        } else {
            RegToBus16(b.GetName(), bv);
        }
        const Reg64 result = rbx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            switch (b.GetName()) {
            case RegName::a0:
            case RegName::a1:
                NOT_IMPLEMENTED();
            // operation on accumulators doesn't go through regular bus with flag and saturation
            case RegName::a0l:
                c.mov(A[0].cvt16(), result.cvt16());
                break;
            case RegName::a1l:
                c.mov(A[1].cvt16(), result.cvt16());
                break;
            case RegName::b0l:
                c.mov(B[0].cvt16(), result.cvt16());
                break;
            case RegName::b1l:
                c.mov(B[1].cvt16(), result.cvt16());
                break;
            case RegName::a0h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(A[0], rsi);
                c.or_(A[0], result);
                break;
            case RegName::a1h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(A[1], rsi);
                c.or_(A[1], result);
                break;
            case RegName::b0h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(B[0], rsi);
                c.or_(B[0], result);
                break;
            case RegName::b1h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(B[1], rsi);
                c.or_(B[1], result);
                break;
            default:
                RegFromBus16(b.GetName(), result.cvt64()); // including RegName:p (p0h)
            }
        }
    }
    void alb_r6(Alb op, Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void alb(Alb op, Imm16 a, SttMod b) {
        if (IsAlbConst(b)) {
            u16 bv;
            switch (b.GetName()) {
            case RegName::mod0:
                bv = blk_key.curr.mod0.raw;
                break;
            case RegName::mod1:
                bv = blk_key.curr.mod1.raw;
                break;
            case RegName::mod2:
                bv = blk_key.curr.mod2.raw;
                break;
            default:
                UNREACHABLE();
            }
            u16 result = GenericAlbConst(op, a.Unsigned16(), bv);
            if (IsAlbModifying(op)) {
                RegFromBus16(b.GetName(), result);
            }
            return;
        }
        const Reg64 bv = rax;
        RegToBus16(b.GetName(), bv);
        const Reg16 result = bx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result);
        if (IsAlbModifying(op)) {
            RegFromBus16(b.GetName(), result.cvt64());
        }
    }

    void add(Ab a, Bx b) {
        const Reg64 value_a = rax;
        const Reg64 value_b = rbx;
        GetAcc(value_a, a.GetName());
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Bx a, Ax b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add_p1(Ax b) {
        const Reg64 value_a = rax;
        ProductToBus40(value_a, Px{1});
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Px a, Bx b) {
        const Reg64 value_a = rax;
        ProductToBus40(value_a, a);
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }

    void sub(Ab a, Bx b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub(Bx a, Ax b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub_p1(Ax b) {
        NOT_IMPLEMENTED();
    }
    void sub(Px a, Bx b) {
        NOT_IMPLEMENTED();
    }

    void app(Ab c, SumBase base, bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        ProductSum(base, c.GetName(), sub_p0, p0_align, sub_p1, p1_align);
    }

    void add_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_sv(ArRn1 a, ArStep1 as, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_sv(ArRn1 a, ArStep1 as, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_i_mov_j_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_j_mov_i_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_i_mov_j(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_j_mov_i(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }

    template <typename T>
    constexpr size_t BitSize() {
        if constexpr (std::is_same_v<T, Reg64>) {
            return 64;
        } else if constexpr (std::is_same_v<T, Reg32>) {
            return 32;
        } else if constexpr (std::is_same_v<T, Reg16>) {
            return 16;
        } else {
            return sizeof(std::make_unsigned_t<T>) * CHAR_BIT;
        }
    }

    template <typename T>
    void SignExtend(T value, u32 bit_count) {
        if (bit_count == 16) {
            c.movsx(value, value.cvt16());
            return;
        }
        if (bit_count == 32) {
            if constexpr (std::is_same_v<T, Xbyak::Reg64>) {
                c.movsxd(value, value.cvt32());
                return;
            }
            NOT_IMPLEMENTED();
        }
        c.shl(value, BitSize<T>() - bit_count);
        c.sar(value, BitSize<T>() - bit_count);
    }

    void ShiftBus40(Reg64 value, u16 sv, RegName dest) {
        c.mov(rbx, 0xFF'FFFF'FFFF);
        c.xor_(ecx, ecx); // rcx = 0
        c.and_(value, rbx);
        const Reg64 original_sign = rbx;
        c.mov(original_sign, value);
        c.shr(original_sign, 39);
        if ((sv >> 15) == 0) {
            // left shift
            if (sv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    // regs.fv = value != 0;
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.test(value, value);
                    c.setne(cl); // u32 mask = (value != 0) ? 1 : 0
                    c.shl(cx, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS, cx); // flags |= mask
                    // if (regs.fv) {
                    //    regs.fvl = 1;
                    // }
                    static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
                    // mask is either 0 or 1. If it was 0, then the or_ wont have any effect.
                    // If it was 1, it will set fvl, which is what we want.
                    c.shr(cx, 3); // mask >>= 3;
                    c.or_(FLAGS, cx);
                }
                c.xor_(value.cvt32(), value.cvt32()); // value = 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
            } else {
                if (blk_key.curr.mod0.s == 0) {
                    // TODO: This can probably be done better
                    c.mov(rdx, value);
                    c.mov(rcx, value);
                    SignExtend(rcx, 40);
                    SignExtend(rdx, 40 - sv);

                           // regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.cmp(rcx, rdx);
                    c.setne(cl); // u32 mask = (SignExtend<40>(value) != SignExtend(value, 40 - sv) ? 1 : 0
                    c.shl(cl, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS.cvt8(), cl); // flags |= mask
                    // if (regs.fv) {
                    //     regs.fvl = 1;
                    // }
                    c.shr(cl, 3); // mask >>= 3;
                    c.or_(FLAGS.cvt8(), cl);
                }
                c.shl(value, sv); // value <<= sv;
                // regs.fc0 = (value & ((u64)1 << 40)) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, 40);
                c.setc(cl); // u32 mask = (value & ((u64)1 << 40));
                c.shl(cl, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS.cvt8(), cl);
            }
        } else {
            // right shift
            u16 nsv = ~sv + 1;
            if (nsv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    NOT_IMPLEMENTED();
                    // regs.fc0 = (value >> 39) & 1;
                    // value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
                } else {
                    c.xor_(value.cvt32(), value.cvt32()); // value = 0;
                    c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
                }
            } else {
                // regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, nsv - 1);
                c.setc(cl); // u32 mask = (value & ((u64)1 << (nsv - 1)));
                c.shl(cx, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS, cx);
                c.shr(value, nsv); // value >>= nsv;
                if (blk_key.curr.mod0.s == 0) {
                    SignExtend(value, 40 - nsv);
                }
            }

            if (blk_key.curr.mod0.s == 0) {
                c.and_(FLAGS, ~decltype(Flags::fv)::mask); // regs.fv = 0;
            }
        }

        SignExtend(value, 40);
        SetAccFlag(value);
        if (blk_key.curr.mod0.s == 0 && blk_key.curr.mod0.sata == 0) {

            Xbyak::Label end_label, saturate_label;
            c.bt(FLAGS, decltype(Flags::fv)::position);
            c.jc(saturate_label);
            c.movsxd(rcx, value.cvt32());
            c.cmp(rcx, value);
            c.je(end_label);
            c.L(saturate_label);
            c.or_(FLAGS, decltype(Flags::flm)::mask);
            c.mov(value, 0x7FFF'FFFF);
            c.mov(rcx, 0xFFFF'FFFF'8000'0000);
            c.cmp(original_sign, 1);
            c.cmove(value, rcx);
            c.L(end_label);
        }
        SetAcc(dest, value);
    }

    void ShiftBus40(Reg64 value, Reg16 sv, RegName dest) {
        ASSERT(sv.getIdx() == Xbyak::Operand::CL);
        c.xor_(esi, esi);
        c.mov(rbx, 0xFF'FFFF'FFFF);
        c.and_(value, rbx);
        const Reg64 original_sign = rbx;
        c.mov(original_sign, value);
        c.shr(original_sign, 39);
        Xbyak::Label end_label, right_shift;
        c.bt(sv, 15);
        c.jc(right_shift);
        // left shift
        Xbyak::Label normal_shift, end_left_shift;
        c.cmp(sv, 40);
        c.jl(normal_shift);
        if (blk_key.curr.mod0.s == 0) {
            // regs.fv = value != 0;
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
            c.test(value, value);
            c.setne(sil);                             // u32 mask = (value != 0) ? 1 : 0
            c.shl(si, decltype(Flags::fv)::position); // mask <<= fv_pos
            c.or_(FLAGS, si);                         // flags |= mask
            // if (regs.fv) {
            //    regs.fvl = 1;
            // }
            static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
            // mask is either 0 or 1. If it was 0, then the or_ wont have any effect.
            // If it was 1, it will set fvl, which is what we want.
            c.shr(si, 3); // mask >>= 3;
            c.or_(FLAGS, si);
        }
        c.xor_(value, value);                       // value = 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
        c.jmp(end_left_shift);
        c.L(normal_shift);
        if (blk_key.curr.mod0.s == 0) {
            // TODO: This can probably be done better
            // rdx = SignExtend(value, 40 - sv);
            c.mov(rdx, value);
            c.add(sv, 24);
            c.shl(rdx, sv.cvt8());
            c.sar(rdx, sv.cvt8());
            c.sub(sv, 24);

            // rcx = SignExtend(value, 40);
            c.mov(rsi, value);
            SignExtend(rsi, 40);

            // regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
            c.cmp(rsi, rdx);
            c.setne(sil); // u32 mask = (SignExtend<40>(value) != SignExtend(value, 40 - sv) ? 1 : 0
            c.and_(si, 1);
            c.shl(si, decltype(Flags::fv)::position); // mask <<= fv_pos
            c.or_(FLAGS.cvt8(), si);                  // flags |= mask
            // if (regs.fv) {
            //     regs.fvl = 1;
            // }
            c.shr(si, 3); // mask >>= 3;
            c.or_(FLAGS.cvt8(), si);
        }
        c.shl(value, sv.cvt8()); // value <<= sv;
        c.xor_(esi, esi);
        // regs.fc0 = (value & ((u64)1 << 40)) != 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
        c.bt(value, 40);
        c.setc(sil);                               // u32 mask = (value & ((u64)1 << 40));
        c.shl(si, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
        c.or_(FLAGS.cvt8(), si);
        c.L(end_left_shift);
        c.jmp(end_label);
        c.L(right_shift);
        // right shift
        const Reg16 nsv = sv;
        c.not_(nsv);
        c.add(nsv, 1);
        Xbyak::Label normal_right_shift, end_right_shift;
        c.cmp(nsv, 40);
        c.jl(normal_right_shift);
        if (blk_key.curr.mod0.s == 0) {
            c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
            c.xor_(esi, esi);
            c.bt(value, 39);
            c.setc(sil);
            c.shl(sil, decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), sil);
            // regs.fc0 = (value >> 39) & 1;
            c.shl(value, 64 - 40);
            c.sar(value, 63);
            // value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
        } else {
            c.xor_(value, value);                       // value = 0;
            c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
        }
        c.jmp(end_right_shift);
        c.L(normal_right_shift);
        // regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
        c.xor_(esi, esi);
        c.sub(nsv, 1);
        c.bt(value, nsv.cvt64());
        c.setc(sil);                               // u32 mask = (value & ((u64)1 << (nsv - 1)));
        c.shl(si, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
        c.or_(FLAGS, si);
        c.add(nsv, 1);
        c.shr(value, nsv.cvt8()); // value >>= nsv;
        if (blk_key.curr.mod0.s == 0) {
            c.add(nsv, 24);
            c.shl(value, nsv.cvt8());
            c.sar(value, nsv.cvt8());
            c.sub(nsv, 24);
        }
        c.L(end_right_shift);

        if (blk_key.curr.mod0.s == 0) {
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // regs.fv = 0;
        }
        c.L(end_label);

        SignExtend(value, 40);
        SetAccFlag(value);
        if (blk_key.curr.mod0.s == 0 && blk_key.curr.mod0.sata == 0) {
            Xbyak::Label end_label, saturate_label;
            c.bt(FLAGS, decltype(Flags::fv)::position);
            c.jc(saturate_label);
            c.movsxd(rcx, value.cvt32());
            c.cmp(rcx, value);
            c.je(end_label);
            c.L(saturate_label);
            c.or_(FLAGS, decltype(Flags::flm)::mask);
            c.mov(value, 0x7FFF'FFFF);
            c.mov(rcx, 0xFFFF'FFFF'8000'0000);
            c.cmp(original_sign, 1);
            c.cmove(value, rcx);
            c.L(end_label);
        }
        SetAcc(dest, value);
    }

    static void PrintValue(u64 value, const char* fmt) {
        printf(fmt, value);
        std::fflush(stdout);
    }

    template <typename T>
    void EmitPrint(T value, const char* fmt) {
        c.push(rax);
        c.push(rbx);
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, value);
        c.mov(ABI_PARAM2, reinterpret_cast<uintptr_t>(fmt));
        CallFarFunction(c, PrintValue);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.pop(rbx);
        c.pop(rax);
    }

    void AddSub(Reg64 a, Reg64 b, Reg64 result, bool sub) {
        c.mov(rsi, 0xFF'FFFF'FFFF);
        c.and_(a, rsi);
        c.and_(b, rsi);
        c.mov(result, a);
        if (sub) {
            c.sub(result, b);
        } else {
            c.add(result, b);
        }
        c.and_(FLAGS, ~(decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask)); // clear fc0, fv
        c.xor_(esi, esi);
        c.bt(result, 40);
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fc0)::position);
        c.or_(FLAGS, rsi.cvt16()); // regs.fc0 = (result >> 40) & 1;
        if (sub) {
            c.not_(b);
        }
        c.xor_(b, a);
        c.not_(b);
        c.xor_(a, result);
        c.and_(a, b);
        c.bt(a, 39); // ((~(a ^ b) & (a ^ result)) >> 39) & 1;
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fv)::position);
        c.or_(FLAGS, rsi.cvt16());
        // if (regs.fv) {
        //     regs.fvl = 1;
        // }
        static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
        c.shr(rsi, 3);
        c.or_(FLAGS, rsi.cvt16());
        SignExtend(result, 40);
    }

    void Moda(ModaOp op, RegName a, Cond cond) {
        ConditionPass(cond, [&] {
            Xbyak::Reg64 acc = rax;
            switch (op) {
            case ModaOp::Shr: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFF, a);
                break;
            }
            case ModaOp::Shr4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFC, a);
                break;
            }
            case ModaOp::Shl: {
                GetAcc(acc, a);
                ShiftBus40(acc, 1, a);
                break;
            }
            case ModaOp::Shl4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 4, a);
                break;
            }
            case ModaOp::Ror: {
                GetAcc(acc, a);
                c.mov(rbx, 0xFF'FFFF'FFFF); // Maybe use BZHI?
                c.and_(acc, rbx);
                c.xor_(rbx, rbx);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(bl); // u16 old_fc = regs.fc0;
                c.shl(rbx, 40);
                c.or_(acc, rbx); // value |= (u64)old_fc << 40;
                c.bt(acc, 1); // u32 mask = value & 1;
                c.setc(bl); // mask <<= fc0_pos;
                c.shl(bl, decltype(Flags::fc0)::position);
                c.or_(FLAGS.cvt8(), bx); // flags |= mask;
                c.shr(acc, 1); // value >>= 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Rol: {
                GetAcc(acc, a);
                c.xor_(rbx, rbx);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(bl); // u16 old_fc = regs.fc0;
                c.shl(acc, 1); // value <<= 1;
                c.or_(acc, rbx); // value |= old_fc;
                c.bt(acc, 40);
                c.setc(bl);
                c.shl(bl, decltype(Flags::fc0)::position);
                c.or_(FLAGS, bx); // regs.fc0 = (value >> 40) & 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Clr: {
                SatAndSetAccAndFlag(a, 0ULL);
                break;
            }
            case ModaOp::Not: {
                GetAcc(acc, a);
                c.not_(acc);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Neg: {
                GetAcc(acc, a);
                c.and_(FLAGS, ~(decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask)); // clear fc0, fv
                const Reg64 scratch = rbx;
                c.test(acc, acc);
                c.setnz(scratch.cvt8());
                c.shl(scratch, decltype(Flags::fc0)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                c.mov(scratch, 0xFFFF'FF80'0000'0000);
                c.cmp(acc, scratch);
                c.sete(scratch.cvt8());
                c.shl(scratch, decltype(Flags::fv)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                // if (regs.fv)
                //    regs.fvl = 1;
                c.shr(scratch, decltype(Flags::fv)::position - decltype(Flags::fvl)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                c.not_(acc);
                c.add(acc, 1);
                SignExtend(acc, 40);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Rnd: {
                GetAcc(acc, a);
                c.mov(rbx, 0x8000);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Pacr: {
                /*u64 value = ProductToBus40(Px{0});
                u64 result = AddSub(value, 0x8000, false);
                SatAndSetAccAndFlag(a, result);
                break;*/
                NOT_IMPLEMENTED();
            }
            case ModaOp::Clrr: {
                SatAndSetAccAndFlag(a, 0x8000ULL);
                break;
            }
            case ModaOp::Inc: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Dec: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, true);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Copy: {
                // note: bX doesn't support
                GetAcc(acc, a == RegName::a0 ? RegName::a1 : RegName::a0);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            default:
                UNREACHABLE();
            }
        });
    }

    void moda4(Moda4 op, Ax a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void moda3(Moda3 op, Bx a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void pacr1(Ax a) {
        NOT_IMPLEMENTED();
    }

    void clr(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }
    void clrr(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void BlockRepeat(Reg64 lc, u32 address) {
        using Frame = JitRegisters::BlockRepeatFrame;
        static_assert(sizeof(Frame) == 12);
        c.movzx(rbx, word[REGS + offsetof(JitRegisters, bcn)]);
        c.lea(rcx, ptr[rbx + rbx * 2]);
        c.lea(rcx, ptr[REGS + offsetof(JitRegisters, bkrep_stack) + rcx * 4]);
        c.mov(dword[rcx + offsetof(Frame, start)], regs.pc);
        c.mov(dword[rcx + offsetof(Frame, end)], address);
        c.mov(word[rcx + offsetof(Frame, lc)], lc.cvt16());
        c.add(rbx, 1);
        c.or_(rbx, 1 << 16);
        c.mov(dword[REGS + offsetof(JitRegisters, bcn)], rbx.cvt32());
        static_assert(offsetof(JitRegisters, lp) - offsetof(JitRegisters, bcn) == sizeof(u16));
        // TODO: Block linking so small loops dont go the dispatcher
        bkrep_end_locations.insert(address);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        compiling = false;
    }

    void bkrep(Imm8 a, Address16 addr) {
        // TODO: Can probably inline a loop for this with some checks
        const Reg64 lc = rax;
        c.mov(lc, a.Unsigned16());
        u32 address = addr.Address32() | (regs.pc & 0x30000);
        BlockRepeat(lc, address);
    }
    void bkrep(Register a, Address18_16 addr_low, Address18_2 addr_high) {
        const Reg64 lc = rax;
        RegToBus16(a.GetName(), lc);
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }
    void bkrep_r6(Address18_16 addr_low, Address18_2 addr_high) {
        const Reg64 lc = rax;
        RegToBus16(RegName::r6, lc);
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }

    static void DoBkrepStackCopyThunk(JitRegisters* regs) {
        std::copy(regs->bkrep_stack.begin() + 1, regs->bkrep_stack.begin() + regs->bcn,
                  regs->bkrep_stack.begin());
        --regs->bcn;
        if (regs->bcn == 0) {
            regs->lp = 0;
        }
    }

    static void DoBkrepStackCopyThunkRestore(JitRegisters* regs) {
        ASSERT(regs->bcn <= 3);
        std::copy_backward(regs->bkrep_stack.begin(), regs->bkrep_stack.begin() + regs->bcn,
                           regs->bkrep_stack.begin() + regs->bcn + 1);
        ++regs->bcn;
    }

    void RestoreBlockRepeat(Reg64 address_reg) {
        using Frame = JitRegisters::BlockRepeatFrame;
        const Reg64 flag = rcx;
        LoadFromMemory(flag, address_reg);
        c.add(address_reg, 1);

        Xbyak::Label end_label, not_in_loop;
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.jz(not_in_loop);
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, REGS);
        CallFarFunction(c, DoBkrepStackCopyThunkRestore);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.jmp(end_label);
        c.L(not_in_loop);
        c.bt(flag, 15);
        c.setc(byte[REGS + offsetof(JitRegisters, lp)]);
        c.setc(byte[REGS + offsetof(JitRegisters, bcn)]);
        c.L(end_label);

        const Reg64 start_end = IsWindows() ? rsi : rdx;
        c.rorx(start_end, flag, 8);
        c.and_(start_end, 0x3);
        c.shl(start_end, 16);
        LoadFromMemory(start_end, address_reg);
        c.add(address_reg, 1);
        c.shl(start_end, 16);
        c.and_(flag, 0x3);
        c.mov(start_end.cvt16(), flag.cvt16());
        c.shl(start_end, 16);
        LoadFromMemory(start_end, address_reg);
        c.add(address_reg, 1);
        c.mov(qword[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(Frame, start)], start_end);
        LoadFromMemory(word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(Frame, lc)], address_reg);
        c.add(address_reg, 1);
    }

    void StoreBlockRepeat(Reg64 address_reg) {
        using Frame = JitRegisters::BlockRepeatFrame;
        const Reg64 flag = rcx;
        c.movzx(flag, word[REGS + offsetof(JitRegisters, lp)]);
        c.shl(flag, 15);
        const Reg64 start_end = IsWindows() ? rsi : rdx;
        c.mov(start_end, qword[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(Frame, start)]);
        c.sub(address_reg, 1);
        StoreToMemory(address_reg, word[REGS + offsetof(JitRegisters, bkrep_stack) + offsetof(Frame, lc)]);
        c.sub(address_reg, 1);
        StoreToMemory(address_reg, start_end);
        c.shr(start_end, 16);
        c.or_(flag.cvt8(), start_end.cvt8());
        c.shr(start_end, 16);
        c.sub(address_reg, 1);
        StoreToMemory(address_reg, start_end);
        c.and_(start_end, 0xFFFF0000);
        c.shr(start_end, 16);
        c.shl(start_end, 8);
        c.or_(flag, start_end);
        c.sub(address_reg, 1);
        StoreToMemory(address_reg, flag);

        Xbyak::Label end_label;
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.jz(end_label);
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.mov(ABI_PARAM1, REGS);
        CallFarFunction(c, DoBkrepStackCopyThunk);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLER_SAVED_GPR, 8);
        c.L(end_label);
    }

    void bkreprst(ArRn2 a) {
        NOT_IMPLEMENTED();
    }
    void bkreprst_memsp() {
        const Reg64 sp = rbx;
        c.movzx(sp.cvt32(), word[REGS + offsetof(JitRegisters, sp)]);
        RestoreBlockRepeat(sp);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void bkrepsto(ArRn2 a) {
        NOT_IMPLEMENTED();
    }
    void bkrepsto_memsp() {
        const Reg64 sp = rbx;
        c.movzx(sp.cvt32(), word[REGS + offsetof(JitRegisters, sp)]);
        StoreBlockRepeat(sp);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void banke(BankFlags flags) {
        if (flags.Cfgi()) {
            c.mov(word[REGS + offsetof(JitRegisters, cfgi)], blk_key.cfgib.raw);
            c.mov(word[REGS + offsetof(JitRegisters, cfgib)], blk_key.cfgi.raw);
            std::swap(blk_key.cfgi, blk_key.cfgib);
            if (blk_key.curr.mod1.stp16) {
                c.mov(word[REGS + offsetof(JitRegisters, stepi0)], blk_key.stepi0b);
                c.mov(word[REGS + offsetof(JitRegisters, stepi0b)], blk_key.stepi0);
                std::swap(blk_key.stepi0, blk_key.stepi0b);
            }
        }
        if (flags.R4()) {
            c.xchg(word[REGS + offsetof(JitRegisters, r4b)], R4_5_6_7.cvt16());
        }
        if (flags.R1()) {
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.xchg(word[REGS + offsetof(JitRegisters, r1b)], R0_1_2_3.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
        }
        if (flags.R0()) {
            c.xchg(word[REGS + offsetof(JitRegisters, r0b)], R0_1_2_3.cvt16());
        }
        if (flags.R7()) {
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.xchg(word[REGS + offsetof(JitRegisters, r7b)], R4_5_6_7.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
        }
        if (flags.Cfgj()) {
            c.mov(word[REGS + offsetof(JitRegisters, cfgj)], blk_key.cfgjb.raw);
            c.mov(word[REGS + offsetof(JitRegisters, cfgjb)], blk_key.cfgj.raw);
            std::swap(blk_key.cfgj, blk_key.cfgjb);
            if (blk_key.curr.mod1.stp16) {
                c.mov(word[REGS + offsetof(JitRegisters, stepj0)], blk_key.stepj0b);
                c.mov(word[REGS + offsetof(JitRegisters, stepj0b)], blk_key.stepj0);
                std::swap(blk_key.stepj0, blk_key.stepj0b);
            }
        }
    }
    void bankr() {
        NOT_IMPLEMENTED();
    }
    void bankr(Ar a) {
        NOT_IMPLEMENTED();
    }
    void bankr(Ar a, Arp b) {
        NOT_IMPLEMENTED();
    }
    void bankr(Arp a) {
        NOT_IMPLEMENTED();
    }

    void bitrev(Rn a) {
        NOT_IMPLEMENTED();
    }
    void bitrev_dbrv(Rn a) {
        NOT_IMPLEMENTED();
    }
    void bitrev_ebrv(Rn a) {
        NOT_IMPLEMENTED();
    }

    void br(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        ConditionPass(cond, [&] {
            regs.pc = Address32(addr_low, addr_high);
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
    }

    void brr(RelAddr7 addr, Cond cond) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        ConditionPass(cond, [&] {
            // note: pc is the address of the NEXT instruction
            regs.pc += addr.Relative32();
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            if (addr.Relative32() == 0xFFFFFFFF) {
                c.mov(dword[REGS + offsetof(JitRegisters, idle)], true);
                compiling = false; // Always end compilation for idle loops.
            } else {
                // For static jump we can continue compiling.
                compiling = cond.GetName() == CondValue::True;
            }
        });
    }

    void break_() {
        NOT_IMPLEMENTED();
    }

    void call(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        const u32 ret_pc = regs.pc;
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], ret_pc);
        ConditionPass(cond, [&] {
            EmitPushPC();
            regs.pc = Address32(addr_low, addr_high);
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
        if (compiling) {
            call_stack.push(ret_pc);
        }
    }
    void calla(Axl a) {
        NOT_IMPLEMENTED();
    }
    void calla(Ax a) {
        EmitPushPC();
        const Reg64 pc = rax;
        GetAcc(pc, a.GetName());
        c.and_(pc, 0x3FFFF);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc.cvt32());
        compiling = false;
    }
    void callr(RelAddr7 addr, Cond cond) {
        const u32 ret_pc = regs.pc;
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], ret_pc);
        ConditionPass(cond, [&] {
            EmitPushPC();
            regs.pc += addr.Relative32();
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
        if (compiling) {
            call_stack.push(ret_pc);
        }
    }

    void cntx_s() {
        regs.ShadowStore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);
        // if (!regs.crep) {
        //     regs.repcs = regs.repc;
        // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.mov(word[REGS + offsetof(JitRegisters, repcs)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(qword[REGS + offsetof(JitRegisters, a1s)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b1s)], B[1]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.mov(rax, B[1]);
        c.mov(B[1], A[1]);
        SetAccAndFlag(RegName::a1, rax); // Flag set on b1->a1
        c.L(end_label);
    }
    void cntx_r() {
        regs.ShadowRestore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);

               // if (!regs.crep) {
               //     regs.repc = regs.repcs;
               // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a1s)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b1s)]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.xchg(A[1], B[1]);
        c.L(end_label);
    }

    void ret(Cond cond) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        ConditionPass(cond, [&] {
            EmitPopPC();
        });
        // If the last call instruction had a static target and this instruction
        // always returns, we don't have to stop compiling.
        if (cond.GetName() == CondValue::True && !call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }
    void retd() {
        NOT_IMPLEMENTED();
    }
    void reti(Cond cond) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        ConditionPass(cond, [&]{
            EmitPopPC();
            c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
        });
        if (cond.GetName() == CondValue::True && !call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }
    void retic(Cond cond) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        ConditionPass(cond, [&] {
            EmitPopPC();
            c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
            cntx_r();
        });
        if (cond.GetName() == CondValue::True && !call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }
    void retid() {
        NOT_IMPLEMENTED();
    }
    void retidc() {
        NOT_IMPLEMENTED();
    }
    void rets(Imm8 a) {
        EmitPopPC();
        c.add(word[REGS + offsetof(JitRegisters, sp)], a.Unsigned16());
        if (!call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }

    void load_ps(Imm2 a) {
        const u16 mask = (a.Unsigned16() & 3) << decltype(Mod0::ps0)::position;
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~decltype(Mod0::ps0)::mask);
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(a.Unsigned16());
    }
    void load_stepi(Imm7s a) {
        // Although this is signed, we still only store the lower 7 bits
        const u8 stepi = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepi);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.cfgi.step.Assign(stepi);
    }
    void load_stepj(Imm7s a) {
        const u8 stepj = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepj);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.cfgj.step.Assign(stepj);
    }
    void load_page(Imm8 a) {
        const u8 page = static_cast<u8>(a.Unsigned16());
        c.mov(byte[REGS + offsetof(JitRegisters, mod1)], page);
        blk_key.curr.mod1.page.Assign(page);
    }
    void load_modi(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(rax, ~decltype(Cfg::mod)::mask);
        c.or_(rax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.cfgi.mod.Assign(a.Unsigned16());
    }
    void load_modj(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(rax, ~decltype(Cfg::mod)::mask);
        c.or_(rax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.cfgj.mod.Assign(a.Unsigned16());
    }
    void load_movpd(Imm2 a) {
        c.mov(word[REGS + offsetof(JitRegisters, pcmhi)], a.Unsigned16());
    }
    void load_ps01(Imm4 a) {
        const u16 ps0 = a.Unsigned16() & 3;
        const u16 ps1 = a.Unsigned16() >> 2;
        const u16 mask = (ps0 << decltype(Mod0::ps0)::position) | (ps1 << decltype(Mod0::ps1)::position);
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~(decltype(Mod0::ps0)::mask | decltype(Mod0::ps1)::mask));
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(ps0);
        blk_key.curr.mod0.ps1.Assign(ps1);
    }

    void push(Imm16 a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, a.Unsigned16());
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push(Register a) {
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value, true);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push(Abe a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        c.shr(value, 32);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    std::optional<u16> IsArpArpSttModConst(ArArpSttMod a) {
        switch (a.GetName()) {
        case RegName::ar0:
            return blk_key.curr.ar[0].raw;
        case RegName::ar1:
            return blk_key.curr.ar[1].raw;
        case RegName::arp0:
            return blk_key.curr.arp[0].raw;
        case RegName::arp1:
            return blk_key.curr.arp[1].raw;
        case RegName::arp2:
            return blk_key.curr.arp[2].raw;
        case RegName::arp3:
            return blk_key.curr.arp[3].raw;
        case RegName::mod0:
            return blk_key.curr.mod0.raw;
        case RegName::mod1:
            return blk_key.curr.mod1.raw;
        case RegName::mod2:
            return blk_key.curr.mod2.raw;
        default:
            return std::nullopt;
        }
    }

    void push(ArArpSttMod a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        if (auto value = IsArpArpSttModConst(a); value.has_value()) {
            StoreToMemory(sp, value.value());
            c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
            return;
        }
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_prpage() {
        NOT_IMPLEMENTED();
    }
    void push(Px a) {
        const Reg64 value = rbx;
        ProductToBus40(value, a);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_r6() {
        const Reg64 value = rax;
        RegToBus16(RegName::r6, value);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_repc() {
        const Reg64 value = rax;
        c.mov(value, word[REGS + offsetof(JitRegisters, repc)]);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_x0() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 32);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_x1() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 48);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_y1() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 16);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void pusha(Ax a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void pusha(Bx a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void pop(Register a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Abe a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        const Reg32 tmp = eax; // Register used just for truncating the accumulator register to 32 bits
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.movsx(value.cvt32(), value.cvt8());
        c.shl(value, 32);
        // Zero-extend bottom 32 bits of accumulator to tmp.cvt64() and merge it into value
        c.mov(tmp, GetAccDirect(a.GetName()).cvt32());
        c.or_(value, tmp.cvt64());
        SetAccAndFlag(a.GetName(), value);
    }
    void pop(ArArpSttMod a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Bx a) {
        NOT_IMPLEMENTED();
    }
    void pop_prpage() {
        NOT_IMPLEMENTED();
    }
    void pop(Px a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.shl(value, 16);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
        ProductFromBus32(a, value.cvt32());
    }
    void pop_r6() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(RegName::r6, value);
    }
    void pop_repc() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], value.cvt16());
    }
    void pop_x0() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
    }
    void pop_x1() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 48);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 16);
    }
    void pop_y1() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        EmitLoadFromMemory<true>(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 16);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 48);
    }
    void popa(Ab a) {
        const Reg64 value = rbx;
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.shl(value, 16);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        SignExtend(value, 32);
        SetAccAndFlag(a.GetName(), value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void rep(Imm8 a) {
        u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        auto& decoder = decoders[opcode];
        u16 expand_value = 0;
        if (decoder.NeedExpansion()) {
            expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        }

        for (int i = 0; i <= a.Unsigned16(); i++) {
            decoder.call(*this, opcode, expand_value);
            current_blk->cycles++;
            ASSERT(compiling); // Ensure the instruction doesn't break the block
        }
    }
    void rep(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        c.mov(word[REGS + offsetof(JitRegisters, rep)], true);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], value.cvt16());
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        compiling = false;
        rep_end_locations.insert(regs.pc);
    }
    void rep_r6() {
        NOT_IMPLEMENTED();
    }

    void shfc(Ab a, Ab b, Cond cond) {
        ConditionPass(cond, [&] {
            const Reg64 value = rax;
            GetAcc(value, a.GetName());
            const Reg16 sv = cx;
            c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
            ShiftBus40(value, sv, b.GetName());
        });
    }
    void shfi(Ab a, Ab b, Imm6s s) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void tst4b(ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }
    void tst4b(ArRn2 b, ArStep2 bs, Ax c) {
        NOT_IMPLEMENTED();
    }
    void tstb(MemImm8 a, Imm4 b) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        c.xor_(eax, eax);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.bt(value, b.Unsigned16());
        c.setc(ah);
        c.or_(FLAGS, eax);
    }
    void tstb(Rn a, StepZIDS as, Imm4 b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        c.xor_(eax, eax);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.bt(value, b.Unsigned16());
        c.setc(ah);
        c.or_(FLAGS, eax);
    }
    void tstb(Register a, Imm4 b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        const Reg64 mask = rbx;
        c.xor_(mask.cvt32(), mask.cvt32());
        c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
        c.bt(value, b.Unsigned16());
        c.setc(mask.cvt8());
        c.shl(mask, decltype(Flags::fz)::position);
        c.or_(FLAGS, mask);
    }
    void tstb_r6(Imm4 b) {
        NOT_IMPLEMENTED();
    }
    void tstb(SttMod a, Imm16 b) {
        NOT_IMPLEMENTED();
    }

    void and_(Ab a, Ab b, Ax c) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        this->c.and_(value, value_b);
        SetAccAndFlag(c.GetName(), value);
    }

    void dint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 0);
    }
    void eint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
    }

    void MulGeneric(MulOp op, Ax a) {
        if (op != MulOp::Mpy && op != MulOp::Mpysu) {
            const Reg64 value = rax;
            GetAcc(value, a.GetName());
            const Reg64 product = rbx;
            ProductToBus40(product, Px{0});
            if (op == MulOp::Maa || op == MulOp::Maasu) {
                c.shr(product, 16);
                SignExtend(product, 24);
            }
            const Reg64 result = rcx;
            AddSub(value, product, result, false);
            SatAndSetAccAndFlag(a.GetName(), result);
        }

        switch (op) {
        case MulOp::Mpy:
        case MulOp::Mac:
        case MulOp::Maa:
            DoMultiplication(0, eax, ebx, true, true);
            break;
        case MulOp::Mpysu:
        case MulOp::Macsu:
        case MulOp::Maasu:
            // Note: the naming conventin of "mpysu" is "multiply signed *y* by unsigned *x*"
            DoMultiplication(0, eax, ebx, false, true);
            break;
        case MulOp::Macus:
            DoMultiplication(0, eax, ebx, true, false);
            break;
        case MulOp::Macuu:
            DoMultiplication(0, eax, ebx, false, false);
            break;
        }
    }

    void mul(Mul3 op, Rn y, StepZIDS ys, Imm16 x, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mul_y0(Mul3 op, Rn x, StepZIDS xs, Ax a) {
        const Reg64 address = rax;
        RnAddressAndModify(x.Index(), xs.GetName(), address);
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0(Mul3 op, Register x, Ax a) {
        const Reg64 x0 = rax;
        RegToBus16(x.GetName(), x0);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), x0.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul(Mul3 op, R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        const Reg64 address_y = rax;
        const Reg64 address_x = rbx;
        RnAddressAndModify(y.Index(), ys.GetName(), address_y);
        RnAddressAndModify(x.Index(), xs.GetName(), address_x);
        EmitLoadFromMemory(FACTORS, address_y);
        c.rorx(FACTORS, FACTORS, 32);
        EmitLoadFromMemory(FACTORS, address_x);
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0_r6(Mul3 op, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mul_y0(Mul2 op, MemImm8 x, Ax a) {
        const Reg64 x0 = rax;
        const Reg64 address = rbx;
        c.mov(address, x.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(x0, address);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), x0.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }

    void mpyi(Imm8s x) {
        c.ror(FACTORS, 32);
        c.mov(FACTORS.cvt16(), x.Signed16());
        c.rol(FACTORS, 32);
        DoMultiplication(0, eax, ebx, true, true);
    }

    void msu(R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        NOT_IMPLEMENTED();
    }
    void msu(Rn y, StepZIDS ys, Imm16 x, Ax a) {
        NOT_IMPLEMENTED();
    }
    void msusu(ArRn2 x, ArStep2 xs, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mac_x1to0(Ax a) {
        NOT_IMPLEMENTED();
    }
    void mac1(ArpRn1 xy, ArpStep1 xis, ArpStep1 yjs, Ax a) {
        NOT_IMPLEMENTED();
    }

    void modr(Rn a, StepZIDS as) {
        const u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, as.GetName(), reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_dmod(Rn a, StepZIDS as) {
        NOT_IMPLEMENTED();
    }
    void modr_i2(Rn a) {
        u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, StepValue::Increase2Mode1, reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_i2_dmod(Rn a) {
        NOT_IMPLEMENTED();
    }
    void modr_d2(Rn a) {
        u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, StepValue::Decrease2Mode1, reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_d2_dmod(Rn a) {
        NOT_IMPLEMENTED();
    }
    void modr_eemod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModifyNoPreserve(uniti, stepi, rax);
        RnAndModifyNoPreserve(unitj, stepj, rax);
    }
    void modr_edmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }
    void modr_demod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }
    void modr_ddmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }

    void ProgramRead(Reg64 out, Reg64 address) {
        c.movzx(address, address.cvt16());
        c.mov(out, reinterpret_cast<uintptr_t>(mem.GetMemory().raw));
        c.movzx(out, word[out + address * 2]);
    }

    void movd(R0123 a, StepZIDS as, R45 b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void movp(Axl a, Register b) {
        NOT_IMPLEMENTED();
    }
    void movp(Ax a, Register b) {
        const Reg64 address = rax;
        GetAcc(address, a.GetName());
        c.and_(address, 0x3FFFF);
        const Reg64 value = rbx;
        ProgramRead(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void movp(Rn a, StepZIDS as, R0123 b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void movpdw(Ax a) {
        const Reg64 address = rbx;
        GetAcc(address, a.GetName());
        c.and_(address, 0x3FFFF);
        const Reg64 pc = rax;
        // the endianess doesn't seem to be affected by regs.cpc
        ProgramRead(pc, address);
        c.add(address, 1);
        c.shl(pc, 16);
        ProgramRead(pc, address);
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc);
        compiling = false;
    }

    void mov(Ab a, Ab b) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        SatAndSetAccAndFlag(b.GetName(), value);
    }
    void mov_dvm(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_x0(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_x1(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_y1(Abl a) {
        const Reg64 value16 = rax;
        RegToBus16(a.GetName(), value16, true);
        c.rorx(FACTORS, FACTORS, 16);
        c.mov(FACTORS.cvt16(), value16.cvt16());
        c.rorx(FACTORS, FACTORS, 48);
    }

    void StoreToMemory(MemImm8 addr, Reg64 value) {
        StoreToMemory(addr.Unsigned16() + (blk_key.curr.mod1.page << 8), value);
    }

    template <typename T1, typename T2>
    void StoreToMemory(T1 addr, T2 value) {
        // TODO: Non MMIO writes can be performed inside the JIT.
        // Push all registers because our JIT assumes everything is non volatile
        c.push(rbp);
        c.push(rbx);
        c.push(rcx);
        c.push(rdx);
        c.push(rsi);
        c.push(rdi);
        c.push(r8);
        c.push(r9);
        c.push(r10);
        c.push(r11);
        c.push(r12);
        c.push(r13);
        c.push(r14);
        c.push(r15);

        c.mov(rbp, rsp);
        // Reserve a bunch of stack space for Windows shadow stack et al, then force align rsp to 16 bytes to respect the ABI
        c.sub(rsp, 64);
        c.and_(rsp, ~0xF);

        if constexpr (std::is_base_of_v<Xbyak::Reg, T1>) {
            c.movzx(ABI_PARAM2, addr.cvt16());
        } else {
            c.mov(ABI_PARAM2, addr);
        }
        c.mov(ABI_PARAM3, value);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&mem));
        c.xor_(ABI_PARAM4.cvt32(), ABI_PARAM4.cvt32()); // bypass_mmio = false
        CallMemberFunction(&MemoryInterface::DataWrite, &mem);

        // Undo anything we did
        c.mov(rsp, rbp);
        c.pop(r15);
        c.pop(r14);
        c.pop(r13);
        c.pop(r12);
        c.pop(r11);
        c.pop(r10);
        c.pop(r9);
        c.pop(r8);
        c.pop(rdi);
        c.pop(rsi);
        c.pop(rdx);
        c.pop(rcx);
        c.pop(rbx);
        c.pop(rbp);
    }

    void mov(Ablh a, MemImm8 b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemImm16 b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        StoreToMemory(b.Unsigned16(), value16);
    }
    void mov(Axl a, MemR7Imm16 b) {
        const Reg64 value16 = rax;
        RegToBus16(a.GetName(), value16, true);
        const Reg64 address = rbx;
        RegToBus16(RegName::r7, address);
        c.add(address, b.Unsigned16());
        StoreToMemory(address, value16);
    }
    void mov(Axl a, MemR7Imm7s b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        const Reg64 address = rcx;
        RegToBus16(RegName::r7, address);
        c.add(address, b.Signed16());
        StoreToMemory(address, value16);
    }

    void mov(MemImm16 a, Ax b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16());
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ab b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ablh b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_eu(MemImm8 a, Axh b) {
        NOT_IMPLEMENTED();
    }
    void mov(MemImm8 a, RnOld b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(MemImm8 a) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        c.mov(word[REGS + offsetof(JitRegisters, sv)], value.cvt16());
    }
    void mov_dvm_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_icr_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm16 a, Bx b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm16 a, Register b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_icr(Imm5 a) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm8s a, Axh b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm8s a, RnOld b) {
        u16 value = a.Signed16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(Imm8s a) {
        const u16 value = a.Signed16();
        c.mov(word[REGS + offsetof(JitRegisters, sv)], value);
    }
    void mov(Imm8 a, Axl b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm16 a, Ax b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address, a.Unsigned16());
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm7s a, Ax b) {
        const Reg64 address = rax;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address.cvt32(), ::SignExtend<16, u32>(a.Signed16()));
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Bx b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Register b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        EmitLoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_memsp_to(Register b) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp_to(Register b) {
        const Reg64 value = rax;
        c.mov(value, word[REGS + offsetof(JitRegisters, mixp)]);
        RegFromBus16(b.GetName(), value);
    }
    void mov(RnOld a, MemImm8 b) {
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value);
        StoreToMemory(b.Unsigned16() + (blk_key.curr.mod1.page << 8), value);
    }
    void mov_icr(Register a) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], value.cvt16());
    }
    void mov(Register a, Rn b, StepZIDS bs) {
        // a = a0 or a1 is overrided
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value, true);
        const Reg64 address = rcx;
        RnAddressAndModify(b.Index(), bs.GetName(), address);
        StoreToMemory(address, value);
    }
    void mov(Register a, Bx b) {
        const Reg64 value = rax;
        if (a.GetName() == RegName::p) {
            ProductToBus40(value, Px{0});
            SatAndSetAccAndFlag(b.GetName(), value);
        } else if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            // Is there any difference from the mov(Ab, Ab) instruction?
            GetAcc(value, a.GetName());
            SatAndSetAccAndFlag(b.GetName(), value);
        } else {
            RegToBus16(a.GetName(), value, true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov(Register a, Register b) {
        // a = a0 or a1 is overrided
        if (a.GetName() == RegName::p) {
            // b loses its typical meaning in this case
            const RegName b_name = b.GetNameForMovFromP();
            const Reg64 value = rax;
            ProductToBus40(value, Px{0});
            SatAndSetAccAndFlag(b_name, value);
        } else if (a.GetName() == RegName::pc) {
            if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
                SatAndSetAccAndFlag(b.GetName(), static_cast<u64>(regs.pc));
            } else {
                RegFromBus16(b.GetName(), regs.pc & 0xFFFF);
            }
        } else {
            const Reg64 value = rax;
            RegToBus16(a.GetName(), value, true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov_repc_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_sv_to(MemImm8 b) {
        StoreToMemory(b.Unsigned16() + (blk_key.curr.mod1.page << 8), word[REGS + offsetof(JitRegisters, sv)]);
    }
    void mov_x0_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_x1_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_y1_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm16 a, ArArp b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Imm16 a) {
        u16 value = a.Unsigned16();
        RegFromBus16(RegName::r6, value);
    }

    static void ThrowException() {
        throw UnimplementedException();
    }

    void mov_repc(Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void mov_stepi0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepi0)], value);
        blk_key.stepi0 = value;
    }
    void mov_stepj0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepj0)], value);
        blk_key.stepj0 = value;
    }
    void mov(Imm16 a, SttMod b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_prpage(Imm4 a) {
        NOT_IMPLEMENTED();
    }

    void mov_a0h_stepi0() {
        const Reg64 value = rax;
        RegToBus16(RegName::a0h, value, true);
        c.mov(word[REGS + offsetof(JitRegisters, stepi0)], value.cvt16());
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        compiling = false;
    }
    void mov_a0h_stepj0() {
        const Reg64 value = rax;
        RegToBus16(RegName::a0h, value, true);
        c.mov(word[REGS + offsetof(JitRegisters, stepj0)], value.cvt16());
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
        compiling = false;
    }
    void mov_stepi0_a0h() {
        RegFromBus16(RegName::a0h, blk_key.stepi0);
    }
    void mov_stepj0_a0h() {
        RegFromBus16(RegName::a0h, blk_key.stepj0);
    }

    void mov_prpage(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_repc(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov(Abl a, ArArp b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Abl a, SttMod b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(b.GetName(), value);
    }

    void mov_prpage_to(Abl b) {
        NOT_IMPLEMENTED();
    }
    void mov_repc_to(Abl b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArp a, Abl b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        RegFromBus16(b.GetName(), value);
    }
    void mov(SttMod a, Abl b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        RegFromBus16(b.GetName(), value);
    }

    void mov_repc_to(ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArp a, ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }
    void mov(SttMod a, ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }

    void mov_repc(ArRn1 a, ArStep1 as) {
        NOT_IMPLEMENTED();
    }
    void mov(ArRn1 a, ArStep1 as, ArArp b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArRn1 a, ArStep1 as, SttMod b) {
        NOT_IMPLEMENTED();
    }

    void mov_repc_to(MemR7Imm16 b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArpSttMod a, MemR7Imm16 b) {
        NOT_IMPLEMENTED();
    }

    void mov_repc(MemR7Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void mov(MemR7Imm16 a, ArArpSttMod b) {
        NOT_IMPLEMENTED();
    }

    void mov_pc(Ax a) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], GetAccDirect(a.GetName()).cvt32());
        compiling = false;
    }
    void mov_pc(Bx a) {
        NOT_IMPLEMENTED();
    }

    void mov_mixp_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_to(Register b) {
        NOT_IMPLEMENTED();
    }
    void mov_p0(Ab a) {
        const Reg64 value = rax;
        GetAndSatAcc(value, a.GetName());
        ProductFromBus32(Px{0}, value.cvt32());
    }
    void mov_p1_to(Ab b) {
        const Reg64 value = rax;
        ProductToBus40(value, Px{1});
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov2(Px a, ArRn2 b, ArStep2 bs) {
        const Reg64 value = rbx;
        c.mov(value, dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * a.Index()]);
        u16 unit = GetArRnUnit(b);
        const Reg64 address = rcx;
        RnAddressAndModify(unit, GetArStep(bs), address);
        const Reg64 address2 = rax;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(bs));
        // NOTE: keep the write order exactly like this.
        StoreToMemory(address2, value);
        c.shr(value, 16);
        StoreToMemory(address, value);
    }
    void mov2s(Px a, ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }
    void mov2(ArRn2 a, ArStep2 as, Px b) {
        u16 unit = GetArRnUnit(a);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(as), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(as));
        const Reg64 value = rcx;
        EmitLoadFromMemory(value, address);
        c.shl(value, 16);
        EmitLoadFromMemory(value, address2);
        ProductFromBus32(b, value.cvt32());
    }
    void mova(Ab a, ArRn2 b, ArStep2 bs) {
        u16 unit = GetArRnUnit(b);
        const Reg64 value = rcx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 address = rbx;
        const Reg64 address2 = rax;
        RnAddressAndModify(unit, GetArStep(bs), address);
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(bs));
        // NOTE: keep the write order exactly like this. The second one overrides the first one if
        // the offset is zero.
        StoreToMemory(address2, value);
        c.shr(value, 16);
        StoreToMemory(address, value);
    }
    void mova(ArRn2 a, ArStep2 as, Ab b) {
        u16 unit = GetArRnUnit(a);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(as), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(as));
        const Reg64 value = rcx;
        EmitLoadFromMemory(value, address);
        c.ror(value.cvt32(), 16);
        EmitLoadFromMemory(value, address2);
        c.movsxd(value, value.cvt32());
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov_r6_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_r6_mixp() {
        NOT_IMPLEMENTED();
    }
    void mov_r6_to(Register b) {
        const Reg64 value = rax;
        RegToBus16(RegName::r6, value);
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(RegName::r6, value);
    }
    void mov_memsp_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_r6_to(Rn b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void mov_r6(Rn a, StepZIDS as) {
        NOT_IMPLEMENTED();
    }

    void mov2_axh_m_y0_m(Axh a, ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }

    void mov2_ax_mij(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        NOT_IMPLEMENTED();
    }
    void mov2_ax_mji(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        NOT_IMPLEMENTED();
    }
    void mov2_mij_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov2_mji_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov2_abh_m(Abh ax, Abh ay, ArRn1 b, ArStep1 bs) {
        const Reg64 u = rbx;
        const Reg64 v = rax;
        GetAndSatAccNoFlag(u, ax.GetName());
        c.shr(u, 16);
        GetAndSatAccNoFlag(v, ay.GetName());
        c.shr(v, 16);
        u16 unit = GetArRnUnit(b);
        const Reg64 ua = rcx;
        RnAddressAndModify(unit, GetArStep(bs), ua);
        const Reg64 va = rdx;
        c.mov(va, ua);
        OffsetAddress(unit, va.cvt16(), GetArOffset(bs));
        // keep the order
        StoreToMemory(va, v);
        StoreToMemory(ua, u);
    }
    void exchange_iaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }
    void exchange_riaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }
    void exchange_jai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        const Reg64 i = rax;
        const Reg64 j = rbx;
        RnAddressAndModify(ui, si, i);
        RnAddressAndModify(uj, sj, j);
        const Reg64 value = rcx;
        GetAndSatAccNoFlag(value, a.GetName());
        c.shr(value, 16);
        StoreToMemory(i, value);
        EmitLoadFromMemory(value, j);
        c.shl(value, 16);
        SignExtend(value, 32);
        SetAcc(a.GetName(), value);
    }
    void exchange_rjai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }

    void movs(MemImm8 a, Ab b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        EmitLoadFromMemory(value, address);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Rn a, StepZIDS as, Ab b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rax;
        EmitLoadFromMemory(value, address);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Register a, Ab b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs_r6_to(Ax b) {
        NOT_IMPLEMENTED();
    }
    void movsi(RnOld a, Ab b, Imm5s s) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        SignExtend(value, 16);
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void movr(ArRn2 a, ArStep2 as, Abh b) {
        NOT_IMPLEMENTED();
    }
    void movr(Rn a, StepZIDS as, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr(Register a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr(Bx a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr_r6_to(Ax b) {
        NOT_IMPLEMENTED();
    }

    template <typename T>
    void Exp(Reg64 value, T count) {
        c.mov(rsi, value);
        c.not_(rsi);
        c.bt(value, 39);
        c.cmovc(value, rsi);
        c.shl(value, 64 - 39);
        c.or_(value, 0x1FFFFFF);
        c.lzcnt(count, value);
        c.sub(count, 8);
    }

    void exp(Bx a) {
        NOT_IMPLEMENTED();
    }
    void exp(Bx a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void exp(Rn a, StepZIDS as) {
        NOT_IMPLEMENTED();
    }
    void exp(Rn a, StepZIDS as, Ax b) {
        NOT_IMPLEMENTED();
    }
    void exp(Register a) {
        const Reg64 value = rax;
        if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            GetAcc(value, a.GetName());
        } else {
            // RegName::p follows the usual rule
            RegToBus16(a.GetName(), value);
            c.shl(value, 16);
            SignExtend(value, 32);
        }
        const Reg64 count = rbx;
        Exp(value, count);
        c.mov(word[REGS + offsetof(JitRegisters, sv)], count.cvt16());
    }
    void exp(Register a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void exp_r6() {
        NOT_IMPLEMENTED();
    }
    void exp_r6(Ax b) {
        NOT_IMPLEMENTED();
    }

    void lim(Ax a, Ax b) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        SaturateAcc<true>(value);
        SetAccAndFlag(b.GetName(), value);
    }

    void vtrclr0() {
        NOT_IMPLEMENTED();
    }
    void vtrclr1() {
        NOT_IMPLEMENTED();
    }
    void vtrclr() {
        NOT_IMPLEMENTED();
    }
    void vtrmov0(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrmov1(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrmov(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrshr() {
        NOT_IMPLEMENTED();
    }

    void clrp0() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{0}, tmp);
    }
    void clrp1() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{1}, tmp);
    }
    void clrp() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{0}, tmp);
        ProductFromBus32(Px{1}, tmp);
    }

    void max_ge(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void max_gt(Ax a, StepZIDS bs) {
        const Reg64 u = rax;
        GetAcc(u, a.GetName());
        const Reg64 v = rbx;
        GetAcc(v, CounterAcc(a.GetName()));
        const Reg64 d = rcx;
        c.mov(d, v);
        c.sub(d, u);
        const Reg64 r0 = u;
        RnAndModify(0, bs.GetName(), r0);

        Xbyak::Label end_label, zero_fm;
        c.test(d, d);
        c.jle(zero_fm);
        c.bts(FLAGS, decltype(Flags::fm)::position);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], r0.cvt16());
        SetAcc(a.GetName(), v);
        c.jmp(end_label);
        c.L(zero_fm);
        c.btr(FLAGS, decltype(Flags::fm)::position);
        c.L(end_label);
    }
    void min_le(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_lt(Ax a, StepZIDS bs) {
        const Reg64 u = rax;
        const Reg64 v = rbx;
        GetAcc(u, a.GetName());
        GetAcc(v, CounterAcc(a.GetName()));
        const Reg64 d = rcx;
        c.mov(d, v);
        c.sub(d, u);
        const Reg64 r0 = u;
        RnAndModify(0, bs.GetName(), r0);

        Xbyak::Label end_label, false_label;
        c.bt(d, 63);
        c.jnc(false_label);
        c.or_(FLAGS, decltype(Flags::fm)::mask);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], r0.cvt16());
        SetAcc(a.GetName(), v);
        c.jmp(end_label);
        c.L(false_label);
        c.and_(FLAGS, ~decltype(Flags::fm)::mask);
        c.L(end_label);
    }

    void max_ge_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void max_gt_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_le_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_lt_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }

    void divs(MemImm8 a, Ax b) {
        NOT_IMPLEMENTED();
    }

    void sqr_sqr_add3(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void sqr_sqr_add3(ArRn2 a, ArStep2 as, Ab b) {
        NOT_IMPLEMENTED();
    }

    void sqr_mpysu_add3a(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void cmp(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void cmp_b0_b1() {
        NOT_IMPLEMENTED();
    }
    void cmp_b1_b0() {
        NOT_IMPLEMENTED();
    }
    void cmp(Bx a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void cmp_p1_to(Ax b) {
        NOT_IMPLEMENTED();
    }

    void max2_vtr(Ax a) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr(Ax a) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }

    template <typename ArpStepX>
    void mov_sv_app(ArRn1 a, ArpStepX as, Bx b, SumBase base, bool sub_p0, bool p0_align,
                    bool sub_p1, bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void cbs(Axh a, CbsCond c) {
        NOT_IMPLEMENTED();
    }
    void cbs(Axh a, Bxh b, CbsCond c) {
        NOT_IMPLEMENTED();
    }
    void cbs(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, CbsCond c) {
        NOT_IMPLEMENTED();
    }

    void ProductSum(SumBase base, RegName acc, bool sub_p0, bool p0_align, bool sub_p1,
                    bool p1_align) {
        const Reg64 value_a = rax;
        const Reg64 value_b = rbx;
        ProductToBus40(value_a, Px{0});
        ProductToBus40(value_b, Px{1});
        if (p0_align) {
            c.shr(value_a, 16);
            SignExtend(value_a, 24);
        }
        if (p1_align) {
            c.shr(value_b, 16);
            SignExtend(value_b, 24);
        }
        const Reg64 value_c = rcx;
        switch (base) {
        case SumBase::Zero:
            c.xor_(value_c, value_c);
            break;
        case SumBase::Acc:
            GetAcc(value_c, acc);
            break;
        case SumBase::Sv:
            c.movzx(value_c, word[REGS + offsetof(JitRegisters, sv)]);
            c.shl(value_c, 16);
            SignExtend(value_c, 32);
            break;
        case SumBase::SvRnd:
            c.movzx(value_c, word[REGS + offsetof(JitRegisters, sv)]);
            c.shl(value_c, 16);
            SignExtend(value_c, 32);
            c.or_(value_c, 0x8000);
            break;
        default:
            UNREACHABLE();
        }
        const Reg64 result = rdx;
        AddSub(value_c, value_a, result, sub_p0);
        const Reg32 temp = value_a.cvt32();
        c.mov(temp, FLAGS);
        c.and_(temp, decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask); // Keep fc0 and fv
        const Reg64 result2 = value_c;
        AddSub(result, value_b, result2, sub_p1);
        // Is this correct?
        if (sub_p0 == sub_p1) {
            c.or_(FLAGS, temp);
        } else {
            c.xor_(FLAGS, temp);
        }
        SatAndSetAccAndFlag(acc, result2);
    }

    void mma(RegName a, bool x0_sign, bool y0_sign, bool x1_sign, bool y1_sign, SumBase base,
             bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        NOT_IMPLEMENTED();
    }

    template <typename ArpRnX, typename ArpStepX>
    void mma(ArpRnX xy, ArpStepX i, ArpStepX j, bool dmodi, bool dmodj, RegName a, bool x0_sign,
             bool y0_sign, bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align,
             bool sub_p1, bool p1_align) {
        ProductSum(base, a, sub_p0, p0_align, sub_p1, p1_align);
        auto [ui, uj] = GetArpRnUnit(xy);
        auto [si, sj] = GetArpStep(i, j);
        auto [oi, oj] = GetArpOffset(i, j);
        const Reg64 x = rbx;
        RnAddressAndModify(ui, si, x, dmodi);
        const Reg64 x2 = rax;
        c.mov(x2, x);
        OffsetAddress(ui, x2.cvt16(), oi, dmodi);
        const Reg64 factors = rcx;
        EmitLoadFromMemory<true>(factors, x2);
        c.shl(factors, 16);
        EmitLoadFromMemory<true>(factors, x);
        c.shl(factors, 16);
        const Reg64 y = x;
        RnAddressAndModify(uj, sj, y, dmodj);
        const Reg64 y2 = x2;
        c.mov(y2, y);
        OffsetAddress(uj, y2.cvt16(), oj, dmodj);
        EmitLoadFromMemory<true>(factors, y2);
        c.shl(factors, 16);
        EmitLoadFromMemory<true>(factors, y);
        c.mov(FACTORS, factors);
        DoMultiplication(0, eax, ebx, x0_sign, y0_sign);
        DoMultiplication(1, eax, ebx, x1_sign, y1_sign);
    }

    void mma_mx_xy(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_xy_mx(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_my_my(ArRn1 x, ArStep1 xs, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        ProductSum(base, a, sub_p0, p0_align, sub_p1, p1_align);
        u16 unit = GetArRnUnit(x);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(xs), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(xs));
        c.rorx(FACTORS, FACTORS, 32);
        EmitLoadFromMemory<true>(FACTORS, address);
        c.rorx(FACTORS, FACTORS, 16);
        EmitLoadFromMemory<true>(FACTORS, address2);
        c.rorx(FACTORS, FACTORS, 16);
        DoMultiplication(0, eax, ebx, x0_sign, y0_sign);
        DoMultiplication(1, eax, ebx, x1_sign, y1_sign);
    }

    void mma_mov(Axh u, Bxh v, ArRn1 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign,
                 bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_mov(ArRn2 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                 bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void addhp(ArRn2 a, ArStep2 as, Px b, Ax c) {
        NOT_IMPLEMENTED();
    }

    void mov_ext0(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext1(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext2(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext3(Imm8s a) {
        NOT_IMPLEMENTED();
    }

private:
    void ConditionPass(Cond cond, auto&& func) {
        Xbyak::Label end_cond;
        switch (cond.GetName()) {
        case CondValue::True:
            func();
            return;
        case CondValue::Eq:
            // return fz == 1;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Neq:
            // return fz == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Gt:
            // return fz == 0 && fm == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask | decltype(Flags::fm)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Ge:
            // return fm == 0;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Lt:
            // return fm == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Le:
            // return fm == 1 || fz == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask | decltype(Flags::fz)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Nn:
            // return fn == 0;
            c.test(FLAGS, decltype(Flags::fn)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::C:
            // return fc0 == 1;
            c.test(FLAGS, decltype(Flags::fc0)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::V:
            // return fv == 1;
            c.test(FLAGS, decltype(Flags::fv)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::E:
            // return fe == 1;
            c.test(FLAGS, decltype(Flags::fe)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::L:
            // return flm == 1 || fvl == 1;
            c.test(FLAGS, decltype(Flags::flm)::mask | decltype(Flags::fvl)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Nr:
            // return fr == 0;
            c.test(FLAGS, decltype(Flags::fr)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Niu0:
        case CondValue::Iu0:
        case CondValue::Iu1:
        default:
            NOT_IMPLEMENTED();
            return;
        }
        func();
        c.L(end_cond);
    }

    void RegToBus16(RegName reg, Reg64 out, bool enable_sat_for_mov = false) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            GetAcc(out, reg);
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            if (enable_sat_for_mov) {
                GetAndSatAcc(out, reg);
            } else {
                GetAcc(out, reg);
            }
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            if (enable_sat_for_mov) {
                GetAndSatAcc(out, reg);
            } else {
                GetAcc(out, reg);
            }
            c.shr(out, 16);
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            NOT_IMPLEMENTED();

        case RegName::r0:
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case RegName::r1:
            c.rorx(out, R0_1_2_3, 16);
            break;
        case RegName::r2:
            c.rorx(out, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(out, R0_1_2_3, 48);
            break;
        case RegName::r4:
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case RegName::r5:
            c.rorx(out, R4_5_6_7, 16);
            break;
        case RegName::r6:
            c.rorx(out, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(out, R4_5_6_7, 48);
            break;
        case RegName::y0:
            c.movzx(out, FACTORS.cvt16());
            break;

        case RegName::p:
            ProductToBus40(out, Px{0});
            c.shr(out, 16);
            c.and_(out, 0xFFFF);
            break;

        case RegName::sp:
            c.movzx(out, word[REGS + offsetof(JitRegisters, sp)]);
            break;
        case RegName::sv:
            c.mov(out, word[REGS + offsetof(JitRegisters, sv)]);
            break;
        case RegName::lc:
            regs.GetLc(c, out);
            break;
        case RegName::stt0:
            regs.GetStt0(c, out.cvt16());
            break;
        case RegName::stt1:
            regs.GetStt1(c, out.cvt16());
            break;
        case RegName::stt2:
            regs.GetStt2(c, out.cvt16());
            break;
        case RegName::st0:
            regs.GetSt0(c, out.cvt16(), blk_key.curr.mod0);
            break;
        case RegName::st1:
            regs.GetSt1(c, out.cvt16(), blk_key.curr.mod0, blk_key.curr.mod1);
            break;

        case RegName::cfgi:
            c.mov(out, blk_key.cfgi.raw);
            break;
        case RegName::cfgj:
            c.mov(out, blk_key.cfgj.raw);
            break;

        case RegName::mod0:
            c.mov(out, blk_key.curr.mod0.raw & Mod0::Mask());
            break;
        case RegName::mod1:
            c.mov(out, blk_key.curr.mod1.raw & Mod1::Mask());
            break;
        case RegName::mod2:
            c.mov(out, blk_key.curr.mod2.raw);
            break;
        case RegName::mod3:
            regs.GetMod3(c, out);
            break;

        case RegName::ar0:
            c.mov(out, blk_key.curr.ar[0].raw & ArU::Mask());
            break;
        case RegName::ar1:
            c.mov(out, blk_key.curr.ar[1].raw & ArU::Mask());
            break;
        case RegName::arp0:
            c.mov(out, blk_key.curr.arp[0].raw & ArpU::Mask());
            break;
        case RegName::arp1:
            c.mov(out, blk_key.curr.arp[1].raw & ArpU::Mask());
            break;
        case RegName::arp2:
            c.mov(out, blk_key.curr.arp[2].raw & ArpU::Mask());
            break;
        case RegName::arp3:
            c.mov(out, blk_key.curr.arp[3].raw & ArpU::Mask());
            break;

        default:
            UNREACHABLE();
        }
    }

    void GetAcc(Xbyak::Reg64 out, RegName name) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(out, A[0]);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(out, A[1]);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(out, B[0]);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(out, B[1]);
            break;
        default:
            NOT_IMPLEMENTED();
        }
    }

    Reg64 GetAccDirect(RegName name) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            return A[0];
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            return A[1];
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            return B[0];
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            return B[1];
        default:
            UNREACHABLE();
        }
    }

    void RegFromBus16(RegName reg, Reg64 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SignExtend(value, 16);
            SatAndSetAccAndFlag(reg, value);
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            c.movzx(value, value.cvt16());
            SatAndSetAccAndFlag(reg, value);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            c.shl(value, 16);
            SignExtend(value, 32);
            SatAndSetAccAndFlag(reg, value);
            break;

        case RegName::r0:
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            break;
        case RegName::r1:
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            break;
        case RegName::r2:
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            break;
        case RegName::r4:
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            break;
        case RegName::r5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case RegName::r6:
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            break;

        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Modifies static state, end block
            break;
        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value.cvt16());
            break;
        case RegName::y0:
            c.mov(FACTORS.cvt16(), value.cvt16());
            break;
        case RegName::p:
            c.cmp(value.cvt16(), 0x7FFF);
            c.setg(byte[REGS + offsetof(JitRegisters, pe)]);
            c.mov(word[REGS + offsetof(JitRegisters, p) + sizeof(u16)], value.cvt16());
            break;

        case RegName::sv:
            c.mov(word[REGS + offsetof(JitRegisters, sv)], value.cvt16());
            break;

        case RegName::lc:
            regs.SetLc(c, value);
            break;

        case RegName::cfgi:
            c.mov(word[REGS + offsetof(JitRegisters, cfgi)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::cfgj:
            c.mov(word[REGS + offsetof(JitRegisters, cfgj)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;

        case RegName::st0:
            regs.SetSt0(c, value, blk_key.curr.mod0);
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;

        case RegName::stt0:
            regs.SetStt0(c, value);
            break;
        case RegName::stt1:
            regs.SetStt1(c, value);
            break;
        case RegName::stt2:
            regs.SetStt2(c, value);
            break;

        case RegName::mod0:
            c.mov(word[REGS + offsetof(JitRegisters, mod0)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false;
            break;
        case RegName::mod1:
            c.mov(word[REGS + offsetof(JitRegisters, mod1)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false;
            break;
        case RegName::mod2:
            c.mov(word[REGS + offsetof(JitRegisters, mod2)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false;
            break;
        case RegName::mod3:
            regs.SetMod3(c, value);
            break;

        case RegName::ar0:
            c.mov(word[REGS + offsetof(JitRegisters, ar)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::ar1:
            c.mov(word[REGS + offsetof(JitRegisters, ar) + sizeof(u16)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::arp0:
            c.mov(word[REGS + offsetof(JitRegisters, arp)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::arp1:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16)], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::arp2:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 2], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;
        case RegName::arp3:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 3], value.cvt16());
            c.mov(dword[REGS + offsetof(JitRegisters, pc)], regs.pc);
            compiling = false; // Static state changed, end block
            break;

        default:
            UNREACHABLE();
        }
    }

    void RegFromBus16(RegName reg, u16 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SatAndSetAccAndFlag(reg, ::SignExtend<16, u64>(value));
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            SatAndSetAccAndFlag(reg, (u64)value);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(u32(value) << 16));
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            NOT_IMPLEMENTED();

        case RegName::r0:
            c.mov(R0_1_2_3.cvt16(), value);
            break;
        case RegName::r1:
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            break;
        case RegName::r2:
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            break;
        case RegName::r4:
            c.mov(R4_5_6_7.cvt16(), value);
            break;
        case RegName::r5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case RegName::r6:
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            break;

        case RegName::y0:
            c.mov(FACTORS.cvt16(), value);
            break;
        case RegName::p:
            c.mov(word[REGS + offsetof(JitRegisters, pe)], value > 0x7FFF);
            c.mov(word[REGS + offsetof(JitRegisters, p) + sizeof(u16)], value);
            break;

        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value);
            break;
        case RegName::mod0:
            blk_key.curr.mod0.raw = value;
            regs.SetMod0(c, value);
            break;
        case RegName::mod1:
            blk_key.curr.mod1.raw = value;
            regs.SetMod1(c, value);
            break;
        case RegName::mod2:
            blk_key.curr.mod2.raw = value;
            regs.SetMod2(c, value);
            break;
        case RegName::mod3:
            regs.SetMod3(c, value);
            break;

        case RegName::cfgi:
            blk_key.cfgi.raw = value;
            regs.SetCfgi(c, value);
            break;
        case RegName::cfgj:
            blk_key.cfgj.raw = value;
            regs.SetCfgj(c, value);
            break;

        case RegName::sv:
            c.mov(word[REGS + offsetof(JitRegisters, sv)], value);
            break;

        case RegName::st0:
            regs.SetSt0(c, value, blk_key.curr.mod0);
            break;
        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            break;
        case RegName::st2:
            regs.SetSt2(c, value, blk_key.curr.mod0, blk_key.curr.mod2);
            break;

        case RegName::ar0:
            blk_key.curr.ar[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar)], value);
            break;
        case RegName::ar1:
            blk_key.curr.ar[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar) + sizeof(u16)], value);
            break;

        case RegName::arp0:
            blk_key.curr.arp[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp)], value);
            break;
        case RegName::arp1:
            blk_key.curr.arp[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16)], value);
            break;
        case RegName::arp2:
            blk_key.curr.arp[2].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 2], value);
            break;
        case RegName::arp3:
            blk_key.curr.arp[3].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 3], value);
            break;

        default:
            NOT_IMPLEMENTED();
        }
    }

    template <typename T>
    void SetAccFlag(T value) {
        constexpr u16 ACC_MASK = ~(decltype(Flags::fz)::mask | decltype(Flags::fm)::mask |
                                   decltype(Flags::fe)::mask | decltype(Flags::fn)::mask);
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            const Reg64 scratch = rdx;
            c.xor_(scratch.cvt32(), scratch.cvt32());
            c.and_(FLAGS.cvt32(), ACC_MASK); // clear fz, fm, fe, fn
            c.test(value, value);
            c.setz(scratch.cvt8()); // mask = (value == 0) ? 1 : 0;
            c.shl(scratch, decltype(Flags::fz)::position); // mask <<= fz_pos
            // regs.fz = value == 0;
            c.or_(FLAGS, scratch); // flags |= mask
            static_assert(decltype(Flags::fz)::position - decltype(Flags::fn)::position == 2);
            c.shr(scratch, 2); // mask >>= (fz_pos - fn_pos);
            c.or_(FLAGS, scratch);
            // regs.fm = (value >> 39) != 0;
            c.mov(scratch, value);
            c.shr(scratch, 39);
            c.test(scratch, scratch);
            c.setnz(scratch.cvt8());
            c.shl(scratch, decltype(Flags::fm)::position);
            c.or_(FLAGS.cvt8(), scratch.cvt8());
            // regs.fe = value != SignExtend<32>(value);
            c.movsxd(scratch, value.cvt32());
            c.cmp(scratch, value);
            c.setne(scratch.cvt8());
            c.shl(scratch, decltype(Flags::fe)::position);
            c.or_(FLAGS.cvt8(), scratch.cvt8());
            c.shl(scratch, 16 - decltype(Flags::fe)::position);
            c.btc(scratch, 16);
            c.bt(value, 31);  // u64 bit31 = (value >> 31) & 1;
            c.setc(dl);
            c.bt(value, 30); // u64 bit30 = (value >> 30) & 1;
            c.setc(dh);
            c.xor_(dh, dl);
            c.shr(scratch, 8);
            c.and_(scratch, 0x101);
            c.and_(dl, dh);
            static_assert(decltype(Flags::fn)::position < 8);
            c.shl(dl, decltype(Flags::fn)::position);
            c.or_(FLAGS.cvt8(), dl);
            // regs.fn = regs.fz || (!regs.fe && (bit31 ^ bit30) != 0);
        } else {
            const u64 bit31 = (value >> 31) & 1;
            const u64 bit30 = (value >> 30) & 1;
            const u16 fz = value == 0;
            const u16 fm = (value >> 39) != 0;
            const u16 fe = value != ::SignExtend<32>(value);
            const u16 fn = fz || (!fe && (bit31 ^ bit30) != 0);

            u16 mask = 0;
            mask |= fz << decltype(Flags::fz)::position;
            mask |= fm << decltype(Flags::fm)::position;
            mask |= fe << decltype(Flags::fe)::position;
            mask |= fn << decltype(Flags::fn)::position;

            c.and_(FLAGS, ACC_MASK); // clear fz, fm, fe, fn
            c.or_(FLAGS, mask);
        }
    }

    template <typename T>
    void SetAcc(RegName name, T value) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(A[0], value);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(A[1], value);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(B[0], value);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(B[1], value);
            break;
        default:
            UNREACHABLE();
        }
    }

    template <bool flag, typename T>
    void SaturateAcc(T& value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            Xbyak::Label end_saturate;
            c.movsxd(rsi, value.cvt32()); // rbx = SignExtend<32>(value);
            c.cmp(value, rsi);
            c.je(end_saturate);
            if constexpr (flag) {
                c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
            }
            c.shr(value, 39);
            c.mov(rsi, 0x0000'0000'7FFF'FFFF);
            c.test(value, value);
            c.mov(value, 0xFFFF'FFFF'8000'0000);
            c.cmovz(value, rsi);
            // note: flm doesn't change value otherwise
            c.L(end_saturate);
        } else {
            if (value != ::SignExtend<32>(value)) {
                if constexpr (flag) {
                    c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
                }
                if ((value >> 39) != 0)
                    value = 0xFFFF'FFFF'8000'0000;
                else
                    value = 0x0000'0000'7FFF'FFFF;
            }
            // note: flm doesn't change value otherwise
        }
    }

    template <typename T>
    void SatAndSetAccAndFlag(RegName name, T value) {
        SetAccFlag(value);
        if (!blk_key.curr.mod0.sata) {
            SaturateAcc<true>(value);
        }
        SetAcc(name, value);
    }

    void SetAccAndFlag(RegName name, Reg64 value) {
        SetAccFlag(value);
        SetAcc(name, value);
    }

    void GetAndSatAcc(Reg64 out, RegName name) {
        GetAcc(out, name);
        if (!blk_key.curr.mod0.sat) {
            SaturateAcc<true>(out);
        }
    }

    void GetAndSatAccNoFlag(Reg64 out, RegName name) {
        GetAcc(out, name);
        if (!blk_key.curr.mod0.sat) {
            SaturateAcc<false>(out);
        }
    }

    void ProductToBus40(Reg64 value, Px reg) {
        const u16 unit = reg.Index();
        c.mov(value, word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit]);
        c.shl(value, 32);
        c.xor_(rsi, rsi);
        c.mov(esi, dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit]);
        c.or_(value, rsi);
        switch (unit == 0 ? blk_key.curr.mod0.ps0.Value() : blk_key.curr.mod0.ps1.Value()) {
        case 0:
            SignExtend(value, 33);
            break;
        case 1:
            c.shr(value, 1);
            SignExtend(value, 32);
            break;
        case 2:
            c.shl(value, 1);
            SignExtend(value, 34);
            break;
        case 3:
            c.shl(value, 2);
            SignExtend(value, 35);
            break;
        }
    }

    void RnAddress(u32 unit, Reg64 value) {
        if (blk_key.curr.mod2.IsBr(unit) && !blk_key.curr.mod2.IsM(unit)) {
            c.rol(value.cvt16(), 8);
            c.mov(esi, value.cvt32());
            c.and_(esi, 3855);
            c.shl(esi, 4);
            c.shr(value.cvt32(), 4);
            c.and_(value.cvt32(), 3855);
            c.or_(value.cvt32(), esi);
            c.mov(esi, value.cvt32());
            c.and_(esi, 13107);
            c.shr(value.cvt32(), 2);
            c.and_(value.cvt32(), 13107);
            c.lea(esi, ptr[value + 4*rsi]);
            c.mov(edx, esi);
            c.and_(edx, 21845);
            c.shr(esi, 1);
            c.and_(esi, 21845);
            c.lea(esi, ptr[rsi + 2*rdx]);
            c.mov(value.cvt32(), esi);
        }
    }

    enum class OffsetValue : u16 {
        Zero = 0,
        PlusOne = 1,
        MinusOne = 2,
        MinusOneDmod = 3,
    };

    void OffsetAddress(u32 unit, Reg16 address, OffsetValue offset, bool dmod = false) {
        if (offset == OffsetValue::Zero)
            return;
        if (offset == OffsetValue::MinusOneDmod) {
            c.sub(address, 1);
            return;
        }
        const bool emod = blk_key.curr.mod2.IsM(unit) && !blk_key.curr.mod2.IsBr(unit) && !dmod;
        u16 mod = unit < 4 ? blk_key.cfgi.mod : blk_key.cfgj.mod;
        [[maybe_unused]] u16 mask = 1; // mod = 0 still have one bit mask
        for (u32 i = 0; i < 9; ++i) {
            mask |= mod >> i;
        }
        if (offset == OffsetValue::PlusOne) {
            if (!emod) {
                c.add(address, 1);
                return;
            }
            // if ((address & mask) == mod)
            //    return address & ~mask;
            // return address + 1;
            Xbyak::Label addressDoesNotEqualMask;

            // Note: This section has a LOT of register pressure!
            const Reg16 tmp = si;
            c.mov(tmp, address);
            c.add(address, 1); 
            c.and_(tmp, mask);
            c.cmp(tmp, mod);
            c.jne(addressDoesNotEqualMask);
            // Subtract 1 to get the original value
            c.mov(tmp, ~mask);
            c.sub(address, 1);
            // Xbyak seems to have issues with AND r, imm16, so we need to store the value to a register
            c.and_(address, tmp);
            c.L(addressDoesNotEqualMask);
        } else { // OffsetValue::MinusOne
            if (!emod) {
                c.sub(address, 1);
                return;
            }
            NOT_IMPLEMENTED();
            // TODO: sometimes this would return two addresses,
            // neither of which is the original Rn value.
            // This only happens for memory writing, but not for memory reading.
            // Might be some undefined behaviour.
            // if ((address & mask) == 0)
            //    return address | mod;
            // return address - 1;
        }
    }

    void StepAddress(u32 unit, Reg16 address, StepValue step, bool dmod = false) {
        u16 s;
        bool legacy = blk_key.curr.mod1.cmd;
        bool step2_mode1 = false;
        bool step2_mode2 = false;
        switch (step) {
        case StepValue::Zero:
            s = 0;
            break;
        case StepValue::Increase:
            s = 1;
            break;
        case StepValue::Decrease:
            s = 0xFFFF;
            break;
        // TODO: Increase/Decrease2Mode1/2 sometimes have wrong result if Offset=+/-1.
        // This however never happens with modr instruction.
        case StepValue::Increase2Mode1:
            s = 2;
            step2_mode1 = !legacy;
            break;
        case StepValue::Decrease2Mode1:
            s = 0xFFFE;
            step2_mode1 = !legacy;
            break;
        case StepValue::Increase2Mode2:
            s = 2;
            step2_mode2 = !legacy;
            break;
        case StepValue::Decrease2Mode2:
            s = 0xFFFE;
            step2_mode2 = !legacy;
            break;
        case StepValue::PlusStep: {
            if (blk_key.curr.mod2.IsBr(unit) && !blk_key.curr.mod2.IsM(unit)) {
                s = unit < 4 ? blk_key.stepi0 : blk_key.stepj0;
            } else {
                s = unit < 4 ? blk_key.cfgi.step : blk_key.cfgj.step;
                s = ::SignExtend<7>(s);
            }
            if (blk_key.curr.mod1.stp16 == 1 && !legacy) {
                s = unit < 4 ? regs.stepi0 : regs.stepj0;
                if (blk_key.curr.mod2.IsM(unit)) {
                    s = ::SignExtend<9>(s);
                }
            }
            break;
        }
        default:
            UNREACHABLE();
        }

        if (s == 0)
            return;
        if (!dmod && !blk_key.curr.mod2.IsBr(unit) && blk_key.curr.mod2.IsM(unit)) {
            u16 mod = unit < 4 ? blk_key.cfgi.mod : blk_key.cfgj.mod;

            if (mod == 0) {
                return;
            }

            if (mod == 1 && step2_mode2) {
                return;
            }

            u32 iteration = 1;
            if (step2_mode1) {
                iteration = 2;
                s = ::SignExtend<15, u16>(s >> 1);
            }

            for (u32 i = 0; i < iteration; ++i) {
                if (legacy || step2_mode2) {
                    bool negative = false;
                    u16 m = mod;
                    if (s >> 15) {
                        negative = true;
                        m |= ~s;
                    } else {
                        m |= s;
                    }

                    u16 mask = (1 << std20::log2p1(m)) - 1;
                    const Reg16 scratch = rsi.cvt16();
                    const Reg16 next = rdx.cvt16();
                    if (!negative) {
                        if (!((!step2_mode2 || mod != mask))) {
                            c.mov(next, address);
                            c.add(next, s);
                            c.and_(next, mask);
                        } else {
                            c.mov(next, address);
                            c.add(next, s);
                            c.and_(next, mask);
                            c.mov(scratch, address);
                            c.and_(scratch, mask);
                            c.cmp(scratch, mod);
                            c.setne(scratch.cvt8());
                            c.shl(scratch, 15);
                            c.sar(scratch, 15);
                            c.and_(next, scratch);
                        }
                    } else {
                        if (!((!step2_mode2 || mod != mask))) {
                            c.mov(next, address);
                            c.add(next.cvt32(), s);
                            c.and_(next, mask);
                        } else {
                            c.mov(next, address);
                            c.add(next.cvt32(), s);
                            c.and_(next, mask);
                            c.mov(scratch, mod);
                            c.test(address, mask);
                            c.cmovz(next, scratch);
                        }
                    }
                    c.and_(address, ~mask);
                    c.or_(address, next);
                } else {
                    u16 mask = (1 << std20::log2p1(mod)) - 1;
                    const Reg16 scratch = rsi.cvt16();
                    const Reg16 next = rdx.cvt16();
                    if (s < 0x8000) {
                        c.mov(next, address);
                        c.add(next, s);
                        c.and_(next, mask);
                        c.xor_(scratch, scratch);
                        c.cmp(next, (mod + 1) & mask);
                        c.cmove(next, scratch);
                    } else {
                        c.mov(next, address);
                        c.and_(next, mask);
                        c.mov(scratch, mod + 1);
                        c.test(next, next);
                        c.cmovz(next, scratch);
                        c.add(next.cvt32(), s);
                        c.and_(next, mask);
                    }
                    c.and_(address, ~mask);
                    c.or_(address, next);
                }
            }
        } else {
            const s16 s_s = static_cast<s16>(s);
            if (s_s < 0) {
                c.sub(address, -s_s);
            } else {
                c.add(address, s);
            }
        }
    }

    void RnAndModifyNoPreserve(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        if ((unit == 3 && blk_key.curr.mod1.epi) || (unit == 7 && blk_key.curr.mod1.epj)) {
            if (step != StepValue::Increase2Mode1 && step != StepValue::Decrease2Mode1 &&
                step != StepValue::Increase2Mode2 && step != StepValue::Decrease2Mode2) {
                switch (unit) {
                case 0:
                    c.xor_(R0_1_2_3.cvt16(), R0_1_2_3.cvt16());
                    break;
                case 1:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 2:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 3:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 4:
                    c.xor_(R4_5_6_7.cvt16(), R4_5_6_7.cvt16());
                    break;
                case 5:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 6:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 7:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                default:
                    UNREACHABLE();
                }
                c.xor_(out, out);
                return;
            }
        }
        switch (unit) {
        case 0:
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case 1:
            c.ror(R0_1_2_3, 16);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 16);
            break;
        case 2:
            c.ror(R0_1_2_3, 32);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 32);
            break;
        case 3:
            c.ror(R0_1_2_3, 48);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 48);
            break;
        case 4:
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case 5:
            c.ror(R4_5_6_7, 16);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R4_5_6_7.cvt16());
            c.rol(R4_5_6_7, 16);
            break;
        case 6:
            c.ror(R4_5_6_7, 32);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R4_5_6_7, 32);
            break;
        case 7:
            c.ror(R4_5_6_7, 48);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R4_5_6_7, 48);
            break;
        default:
            UNREACHABLE();
        }
    }

    void RnAndModify(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        switch (unit) {
        case 0:
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case 1:
            c.rorx(out, R0_1_2_3, 16);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 2:
            c.rorx(out, R0_1_2_3, 32);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 3:
            c.rorx(out, R0_1_2_3, 48);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 4:
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case 5:
            c.rorx(out, R4_5_6_7, 16);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 6:
            c.rorx(out, R4_5_6_7, 32);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 7:
            c.rorx(out, R4_5_6_7, 48);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        default:
            UNREACHABLE();
        }

        if ((unit == 3 && blk_key.curr.mod1.epi) || (unit == 7 && blk_key.curr.mod1.epj)) {
            if (step != StepValue::Increase2Mode1 && step != StepValue::Decrease2Mode1 &&
                step != StepValue::Increase2Mode2 && step != StepValue::Decrease2Mode2) {
                switch (unit) {
                case 0:
                    c.xor_(R0_1_2_3.cvt16(), R0_1_2_3.cvt16());
                    break;
                case 1:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 2:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 3:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 4:
                    c.xor_(R4_5_6_7.cvt16(), R4_5_6_7.cvt16());
                    break;
                case 5:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 6:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 7:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                default:
                    UNREACHABLE();
                }
                return;
            }
        }
        switch (unit) {
        case 0:
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            break;
        case 1:
            c.ror(R0_1_2_3, 16);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 16);
            break;
        case 2:
            c.ror(R0_1_2_3, 32);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 32);
            break;
        case 3:
            c.ror(R0_1_2_3, 48);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 48);
            break;
        case 4:
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            break;
        case 5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case 6:
            c.ror(R4_5_6_7, 32);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rol(R4_5_6_7, 32);
            break;
        case 7:
            c.ror(R4_5_6_7, 48);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rol(R4_5_6_7, 48);
            break;
        default:
            UNREACHABLE();
        }
    }

    void RnAddressAndModify(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        RnAndModify(unit, step, out, dmod);
        RnAddress(unit, out);
    }

    template <typename ArRnX>
    u16 GetArRnUnit(ArRnX arrn) const {
        static_assert(std::is_same_v<ArRnX, ArRn1> || std::is_same_v<ArRnX, ArRn2>);
        switch (arrn.Index()) {
        case 0:
            return blk_key.curr.ar[0].arrn0.Value();
        case 1:
            return blk_key.curr.ar[0].arrn1.Value();
        case 2:
            return blk_key.curr.ar[1].arrn0.Value();
        case 3:
            return blk_key.curr.ar[1].arrn1.Value();
        default:
            UNREACHABLE();
        }
    }

    static StepValue ConvertArStep(u16 arvalue) {
        switch (arvalue) {
        case 0:
            return StepValue::Zero;
        case 1:
            return StepValue::Increase;
        case 2:
            return StepValue::Decrease;
        case 3:
            return StepValue::PlusStep;
        case 4:
            return StepValue::Increase2Mode1;
        case 5:
            return StepValue::Decrease2Mode1;
        case 6:
            return StepValue::Increase2Mode2;
        case 7:
            return StepValue::Decrease2Mode2;
        default:
            UNREACHABLE();
        }
    }

    template <typename ArStepX>
    StepValue GetArStep(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep1Alt> ||
                      std::is_same_v<ArStepX, ArStep2>);
        const u16 value = [&] {
            switch (arstep.Index()) {
            case 0:
                return blk_key.curr.ar[0].arstep0.Value();
            case 1:
                return blk_key.curr.ar[0].arstep1.Value();
            case 2:
                return blk_key.curr.ar[1].arstep0.Value();
            case 3:
                return blk_key.curr.ar[1].arstep1.Value();
            default:
                UNREACHABLE();
            }
        }();
        return ConvertArStep(value);
    }

    template <typename ArpStepX>
    std::tuple<OffsetValue, OffsetValue> GetArpOffset(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple((OffsetValue)blk_key.curr.arp[arpstepi.Index()].arpoffseti.Value(),
                               (OffsetValue)blk_key.curr.arp[arpstepj.Index()].arpoffsetj.Value());
    }

    template <typename ArpRnX>
    std::tuple<u16, u16> GetArpRnUnit(ArpRnX arprn) const {
        static_assert(std::is_same_v<ArpRnX, ArpRn1> || std::is_same_v<ArpRnX, ArpRn2>);
        return std::make_tuple(blk_key.curr.arp[arprn.Index()].arprni, blk_key.curr.arp[arprn.Index()].arprnj + 4);
    }

    template <typename ArpStepX>
    std::tuple<StepValue, StepValue> GetArpStep(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple(ConvertArStep(blk_key.curr.arp[arpstepi.Index()].arpstepi),
                               ConvertArStep(blk_key.curr.arp[arpstepj.Index()].arpstepj));
    }

    template <typename ArStepX>
    OffsetValue GetArOffset(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep2>);
        const u16 value = [&] {
            switch (arstep.Index()) {
            case 0:
                return blk_key.curr.ar[0].aroffset0.Value();
            case 1:
                return blk_key.curr.ar[0].aroffset1.Value();
            case 2:
                return blk_key.curr.ar[1].aroffset0.Value();
            case 3:
                return blk_key.curr.ar[1].aroffset1.Value();
            default:
                UNREACHABLE();
            }
        }();
        return static_cast<OffsetValue>(value);
    }

    void ProductFromBus32(Px reg, Reg32 value) {
        u16 unit = reg.Index();
        c.mov(dword[REGS + offsetof(JitRegisters, p) + unit * sizeof(u32)], value);
        c.bt(value, 31);
        c.setc(word[REGS + offsetof(JitRegisters, pe) + unit * sizeof(u16)]);
    }

    static RegName CounterAcc(RegName in) {
        static std::unordered_map<RegName, RegName> map{
                                                        {RegName::a0, RegName::a1},   {RegName::a1, RegName::a0},
                                                        {RegName::b0, RegName::b1},   {RegName::b1, RegName::b0},
                                                        {RegName::a0l, RegName::a1l}, {RegName::a1l, RegName::a0l},
                                                        {RegName::b0l, RegName::b1l}, {RegName::b1l, RegName::b0l},
                                                        {RegName::a0h, RegName::a1h}, {RegName::a1h, RegName::a0h},
                                                        {RegName::b0h, RegName::b1h}, {RegName::b1h, RegName::b0h},
                                                        {RegName::a0e, RegName::a1e}, {RegName::a1e, RegName::a0e},
                                                        {RegName::b0e, RegName::b1e}, {RegName::b1e, RegName::b0e},
                                                        };
        return map.at(in);
    }
};

} // namespace Teakra
