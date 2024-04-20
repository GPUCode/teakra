include(CreateDirectoryGroups)

add_library(teakra
    ../include/teakra/disassembler.h
    ../include/teakra/teakra.h
    ahbm.cpp
    ahbm.h
    apbp.cpp
    apbp.h
    bit.h
    bit_field.h
    btdmp.cpp
    btdmp.h
    common_types.h
    core_timing.h
    crash.h
    decoder.h
    disassembler.cpp
    dma.cpp
    dma.h
    timer.cpp
    timer.h
    icu.h
    interpreter.h
    matcher.h
    memory_interface.cpp
    memory_interface.h
    memory_pool.cpp
    memory_pool.h
    mmio.cpp
    mmio.h
    operand.h
    parser.cpp
    processor.cpp
    processor.h
    register.h
    shared_memory.h
    swap.h
    teakra.cpp
    test.h
    test_generator.cpp
    test_generator.h
    xbyak_abi.h
    #ir/basic_block.cpp
    #ir/basic_block.h
    #ir/ir_emitter.cpp
    #ir/ir_emitter.h
    #ir/microinstruction.cpp
    #ir/microinstruction.h
    #ir/value.cpp
    #ir/value.h
    #ir/opcode.h
    jit_no_ir.cpp
    jit_no_ir.h
    jit_regs.h
    cityhash.cpp
    cityhash.h
    hash.h
)

create_target_directory_groups(teakra)

target_link_libraries(teakra PRIVATE Threads::Threads tsl::robin_map xbyak::xbyak)
target_include_directories(teakra
                           PUBLIC ../include
                           PRIVATE .)
target_compile_options(teakra PRIVATE ${TEAKRA_CXX_FLAGS})

add_library(teakra_c
    ../include/teakra/disassembler_c.h
    ../include/teakra/teakra_c.h
    disassembler_c.cpp
    teakra_c.cpp
)
target_link_libraries(teakra_c PRIVATE teakra)

if (TEAKRA_BUILD_TOOLS)
    add_subdirectory(coff_reader)
    add_subdirectory(dsp1_reader)
    add_subdirectory(test_generator)
    add_subdirectory(test_verifier)
    add_subdirectory(mod_test_generator)
    add_subdirectory(step2_test_generator)
    add_subdirectory(makedsp1)
endif()
