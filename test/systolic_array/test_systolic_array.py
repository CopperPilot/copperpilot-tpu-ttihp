import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


@cocotb.test()
async def test_systolic_array_basic(dut):
    """Test basic 2x2 matrix multiplication through the systolic array."""
    cocotb.log.info("Starting systolic array test")

    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst.value = 1
    dut.clear.value = 1
    dut.activation.value = 0
    dut.a_data0.value = 0
    dut.a_data1.value = 0
    dut.b_data0.value = 0
    dut.b_data1.value = 0
    await Timer(20, units="ns")

    dut.rst.value = 0
    dut.clear.value = 1
    await RisingEdge(dut.clk)

    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Drive a small matrix multiply: A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    matrix_A = [[1, 2], [3, 4]]
    matrix_B = [[5, 6], [7, 8]]

    weights = [matrix_A[0][0], matrix_A[0][1], matrix_A[1][0], matrix_A[1][1]]  # weight0..3
    inputs = [matrix_B[0][0], matrix_B[0][1], matrix_B[1][0], matrix_B[1][1]]  # input0..3

    # Cycle 0
    dut.a_data0.value = weights[0]
    dut.a_data1.value = 0
    dut.b_data0.value = inputs[0]
    dut.b_data1.value = 0
    await RisingEdge(dut.clk)

    # Cycle 1
    dut.a_data0.value = weights[1]
    dut.a_data1.value = weights[2]
    dut.b_data0.value = inputs[2]
    dut.b_data1.value = inputs[1]
    await RisingEdge(dut.clk)

    # Cycle 2
    dut.a_data0.value = 0
    dut.a_data1.value = weights[3]
    dut.b_data0.value = 0
    dut.b_data1.value = inputs[3]
    await RisingEdge(dut.clk)

    # Clear inputs for subsequent cycles
    dut.a_data0.value = 0
    dut.a_data1.value = 0
    dut.b_data0.value = 0
    dut.b_data1.value = 0

    # Wait for 2 more cycles to let systolic array process
    await ClockCycles(dut.clk, 2)

    c00 = dut.c00.value.signed_integer
    c01 = dut.c01.value.signed_integer
    c10 = dut.c10.value.signed_integer
    c11 = dut.c11.value.signed_integer

    assert c00 == 19, f"C00 expected 19 but got {c00}"
    assert c01 == 22, f"C01 expected 22 but got {c01}"
    assert c10 == 43, f"C10 expected 43 but got {c10}"
    assert c11 == 50, f"C11 expected 50 but got {c11}"

    cocotb.log.info("Systolic array multiplication test passed")

