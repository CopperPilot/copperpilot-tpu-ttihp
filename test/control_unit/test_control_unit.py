import cocotb
from cocotb.triggers import ClockCycles
from cocotb.clock import Clock


@cocotb.test()
async def test_control_unit_reset(dut):
    """Test control unit reset functionality"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Apply reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)

    # Check reset state
    assert dut.mem_addr.value == 0, f"mem_addr should be 0 after reset, got {dut.mem_addr.value}"
    assert dut.mmu_en.value == 0, f"mmu_en should be 0 after reset, got {dut.mmu_en.value}"
    assert dut.mmu_cycle.value == 0, f"mmu_cycle should be 0 after reset, got {dut.mmu_cycle.value}"

    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    dut._log.info("Reset test passed")


@cocotb.test()
async def test_control_unit_idle_state(dut):
    """Test control unit stays in IDLE when load_en is not asserted"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Stay in idle for several cycles
    for _ in range(5):
        assert dut.mem_addr.value == 0, f"mem_addr should remain 0 in idle, got {dut.mem_addr.value}"
        assert dut.mmu_en.value == 0, f"mmu_en should remain 0 in idle, got {dut.mmu_en.value}"
        assert dut.mmu_cycle.value == 0, f"mmu_cycle should remain 0 in idle, got {dut.mmu_cycle.value}"
        await ClockCycles(dut.clk, 1)

    dut._log.info("Idle state test passed")


@cocotb.test()
async def test_control_unit_load_matrices(dut):
    """Test control unit matrix loading phase"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Start loading - first load_en pulse should trigger transition to LOAD_MATS
    dut.load_en.value = 1
    await ClockCycles(dut.clk, 1)

    # Check memory address increments correctly during loading
    expected_addrs = [0, 1, 2, 3, 4, 5, 6, 7]
    for i, expected_addr in enumerate(expected_addrs):
        # Check current state BEFORE the clock edge
        assert int(dut.mem_addr.value) == expected_addr, (
            f"Cycle {i+1}: mem_addr should be {expected_addr}, got {dut.mem_addr.value}"
        )

        current_loaded = int(dut.mem_addr.value)
        # We check when loaded=6 and not when=5.
        # The value is "set" when=5; sequential regs capture it on the next clock edge.
        if current_loaded >= 6:
            assert dut.mmu_en.value == 1, (
                f"Cycle {i+1}: mmu_en should be 1 when mat_elems_loaded >= 6"
            )
        else:
            assert dut.mmu_en.value == 0, (
                f"Cycle {i+1}: mmu_en should be 0 when mat_elems_loaded < 5"
            )

        # Wait for next clock edge (this is when assignments happen)
        await ClockCycles(dut.clk, 1)

    # At this point, mem_addr should have wrapped to 0 after loading all 8 elements
    assert dut.mem_addr.value == 0, "mem_addr should be 0 after loading all 8 elements"
    assert dut.mmu_en.value == 1, "mmu_en should be 1 after loading all 8 elements"
    dut.load_en.value = 0

    dut._log.info("Matrix loading test passed")


@cocotb.test()
async def test_control_unit_mmu_compute_phase(dut):
    """Test control unit MMU compute and writeback phase"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Load all 8 elements quickly
    dut.load_en.value = 1
    await ClockCycles(dut.clk, 8)
    dut.load_en.value = 0

    # Now in MMU_FEED_COMPUTE_WB state
    for expected_cycle in range(2, 8):
        await ClockCycles(dut.clk, 1)
        assert dut.mmu_en.value == 1, "mmu_en should remain 1 during compute phase"
        assert dut.mmu_cycle.value.integer == expected_cycle, (
            f"mmu_cycle should be {expected_cycle}, got {dut.mmu_cycle.value}"
        )

    # One more step should wrap mmu_cycle back to 0
    await ClockCycles(dut.clk, 1)
    assert dut.mmu_cycle.value.integer == 0, "mmu_cycle should reset to 0 in compute"

    dut._log.info("MMU compute phase test passed")


@cocotb.test()
async def test_control_unit_full_cycle(dut):
    """Test complete control unit operation cycle"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Phase 1: Load matrices (8 cycles)
    dut.load_en.value = 1
    for cycle in range(8):
        await ClockCycles(dut.clk, 1)
        if cycle < 7:  # During loading
            assert dut.mem_addr.value.integer == cycle, f"mem_addr should be {cycle} during loading"

    # Phase 2: MMU compute (cycles 2..7)
    for cycle in range(2, 8):
        await ClockCycles(dut.clk, 1)
        assert dut.mmu_en.value == 1, "mmu_en should be 1 during MMU phase"

    dut._log.info("Full cycle test passed")


@cocotb.test()
async def test_control_unit_shaky_load_en(dut):
    """Test that load_en is ignored during MMU compute phase"""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Load matrices (8 cycles)
    dut.load_en.value = 1
    for cycle in range(8):
        await ClockCycles(dut.clk, 1)
        assert dut.mem_addr.value.integer == cycle

    # Now in MMU compute phase: toggle load_en randomly
    for cycle in range(2, 8):
        dut.load_en.value = 1 if cycle % 2 == 0 else 0
        await ClockCycles(dut.clk, 1)
        assert dut.mmu_cycle.value.integer == cycle, "mmu_cycle should keep incrementing in compute phase"

    dut._log.info("Shaky load enable during compute test passed")

