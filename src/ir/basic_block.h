#pragma once

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>

#include <mcl/container/intrusive_list.hpp>
#include <mcl/stdint.hpp>

#include "microinstruction.h"

namespace Teakra::Common {
class Pool;
}

enum class CondValue;

namespace Teakra::IR {

class LocationDescriptor {
public:
    explicit LocationDescriptor(u64 value)
        : value(value) {}

    bool operator==(const LocationDescriptor& o) const {
        return value == o.Value();
    }

    bool operator!=(const LocationDescriptor& o) const {
        return !operator==(o);
    }

    u64 Value() const { return value; }

private:
    u64 value;
};

/**
 * XpertTeak registers that are treated as static state in the JIT.
 * Their value should be known at compile time, otherwise an error is triggered.
 */
struct StaticRegs {
    bool sat = false;  // 1-bit, disable saturation when moving from acc
    bool sata = true; // 1-bit, disable saturation when moving to acc
    u16 hwm = 0;     // 2-bit, half word mode, modify y on multiplication
};

enum class Opcode;

/**
 * A basic block. It consists of zero or more instructions followed by exactly one terminal.
 * Note that this is a linear IR and not a pure tree-based IR: i.e.: there is an ordering to
 * the microinstructions. This only matters before chaining is done in order to correctly
 * order memory accesses.
 */
class Block final {
public:
    using InstructionList = mcl::intrusive_list<Inst>;
    using size_type = InstructionList::size_type;
    using iterator = InstructionList::iterator;
    using const_iterator = InstructionList::const_iterator;
    using reverse_iterator = InstructionList::reverse_iterator;
    using const_reverse_iterator = InstructionList::const_reverse_iterator;

    explicit Block(const LocationDescriptor& location);
    ~Block();

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    Block(Block&&);
    Block& operator=(Block&&);

    bool empty() const { return instructions.empty(); }
    size_type size() const { return instructions.size(); }

    Inst& front() { return instructions.front(); }
    const Inst& front() const { return instructions.front(); }

    Inst& back() { return instructions.back(); }
    const Inst& back() const { return instructions.back(); }

    iterator begin() { return instructions.begin(); }
    const_iterator begin() const { return instructions.begin(); }
    iterator end() { return instructions.end(); }
    const_iterator end() const { return instructions.end(); }

    reverse_iterator rbegin() { return instructions.rbegin(); }
    const_reverse_iterator rbegin() const { return instructions.rbegin(); }
    reverse_iterator rend() { return instructions.rend(); }
    const_reverse_iterator rend() const { return instructions.rend(); }

    const_iterator cbegin() const { return instructions.cbegin(); }
    const_iterator cend() const { return instructions.cend(); }

    const_reverse_iterator crbegin() const { return instructions.crbegin(); }
    const_reverse_iterator crend() const { return instructions.crend(); }

    /**
     * Appends a new instruction to the end of this basic block,
     * handling any allocations necessary to do so.
     *
     * @param op   Opcode representing the instruction to add.
     * @param args A sequence of Value instances used as arguments for the instruction.
     */
    void AppendNewInst(Opcode op, std::initializer_list<Value> args);

    /**
     * Prepends a new instruction to this basic block before the insertion point,
     * handling any allocations necessary to do so.
     *
     * @param insertion_point Where to insert the new instruction.
     * @param op              Opcode representing the instruction to add.
     * @param args            A sequence of Value instances used as arguments for the instruction.
     * @returns Iterator to the newly created instruction.
     */
    iterator PrependNewInst(iterator insertion_point, Opcode op, std::initializer_list<Value> args);

    /// Gets the starting location for this basic block.
    LocationDescriptor Location() const;
    /// Gets the end location for this basic block.
    LocationDescriptor EndLocation() const;
    /// Sets the end location for this basic block.
    void SetEndLocation(const LocationDescriptor& descriptor);

    /// Gets the condition required to pass in order to execute this block.
    CondValue GetCondition() const;
    /// Sets the condition required to pass in order to execute this block.
    void SetCondition(CondValue condition);

    /// Gets the location of the block to execute if the predicated condition fails.
    LocationDescriptor ConditionFailedLocation() const;
    /// Sets the location of the block to execute if the predicated condition fails.
    void SetConditionFailedLocation(LocationDescriptor fail_location);
    /// Determines whether or not a predicated condition failure block is present.
    bool HasConditionFailedLocation() const;

    /// Gets a mutable reference to the condition failed cycle count.
    size_t& ConditionFailedCycleCount();
    /// Gets an immutable reference to the condition failed cycle count.
    const size_t& ConditionFailedCycleCount() const;

    /// Gets a mutable reference to the instruction list for this basic block.
    InstructionList& Instructions();
    /// Gets an immutable reference to the instruction list for this basic block.
    const InstructionList& Instructions() const;

    /// Gets the terminal instruction for this basic block.
    //Terminal GetTerminal() const;
    /// Sets the terminal instruction for this basic block.
    //void SetTerminal(Terminal term);
    /// Replaces the terminal instruction for this basic block.
    //void ReplaceTerminal(Terminal term);
    /// Determines whether or not this basic block has a terminal instruction.
    bool HasTerminal() const;

    /// Gets a mutable reference to the cycle count for this basic block.
    size_t& CycleCount();
    /// Gets an immutable reference to the cycle count for this basic block.
    const size_t& CycleCount() const;

private:
    /// Description of the starting location of this block
    LocationDescriptor location;
    /// Description of the end location of this block
    LocationDescriptor end_location;
    /// Conditional to pass in order to execute this block
    CondValue cond;
    /// Block to execute next if `cond` did not pass.
    std::optional<LocationDescriptor> cond_failed = {};
    /// Number of cycles this block takes to execute if the conditional fails.
    size_t cond_failed_cycle_count = 0;

    /// Expected static register state when jumping to this block.
    StaticRegs entry_state;
    /// Static register state after execution of the block is finished.
    StaticRegs exit_state;

    /// List of instructions in this block.
    InstructionList instructions;
    /// Memory pool for instruction list
    std::unique_ptr<Common::Pool> instruction_alloc_pool;
    /// Terminal instruction of this block.
    //Terminal terminal = Term::Invalid{};

    /// Number of cycles this block takes to execute.
    size_t cycle_count = 0;
};

/// Returns a string representation of the contents of block. Intended for debugging.
std::string DumpBlock(const IR::Block& block);

}  // namespace Dynarmic::IR
