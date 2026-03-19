import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

import numpy as np


def to_s8(v: int) -> int:
    """Interpret an 8-bit unsigned value as signed."""
    return v if v < 128 else v - 256


def to_u8(v: int) -> int:
    """Convert signed int into two's complement 8-bit pattern."""
    return v & 0xFF


def to_s16_from_u16(word: int) -> int:
    """Interpret 16-bit unsigned word as signed."""
    if word >= 0x8000:
        return word - 0x10000
    return word


async def reset_dut(dut):
    dut.rst.value = 1
    dut.en.value = 0
    dut.mmu_cycle.value = 0
    dut.transpose.value = 0

    dut.input0.value = 0
    dut.input1.value = 0
    dut.input2.value = 0
    dut.input3.value = 0
    dut.weight0.value = 0
    dut.weight1.value = 0
    dut.weight2.value = 0
    dut.weight3.value = 0

    dut.c00.value = 0
    dut.c01.value = 0
    dut.c10.value = 0
    dut.c11.value = 0

    dut.host_outdata.value = 0
    await ClockCycles(dut.clk, 1)

    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)


@cocotb.test()
async def test_mmu_feeder_streaming_bytes(dut):
    """Validate a/b selection, done/clear, and host_outdata byte stream."""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    test_vectors = [
        # (name, transpose, A=[w0,w1,w2,w3], B=[i0,i1,i2,i3])
        ("regular", 0, [1, 2, 3, 4], [5, 6, 7, 8]),
        ("signed_edge", 0, [127, -128, 127, -128], [5, 5, 5, 5]),
        ("transpose", 1, [1, 2, 3, 4], [5, 6, 7, 8]),
    ]

    for name, transpose, W, I in test_vectors:
        dut._log.info(f"Running {name}")

        # Reset between vectors
        await reset_dut(dut)
        dut.en.value = 1
        dut.transpose.value = transpose

        w0, w1, w2, w3 = W
        i0, i1, i2, i3 = I

        # Drive weight/input bytes (two's complement patterns)
        dut.weight0.value = to_u8(w0)
        dut.weight1.value = to_u8(w1)
        dut.weight2.value = to_u8(w2)
        dut.weight3.value = to_u8(w3)
        dut.input0.value = to_u8(i0)
        dut.input1.value = to_u8(i1)
        dut.input2.value = to_u8(i2)
        dut.input3.value = to_u8(i3)

        A = np.array([[to_s8(to_u8(w0)), to_s8(to_u8(w1))],
                      [to_s8(to_u8(w2)), to_s8(to_u8(w3))]], dtype=np.int16)
        B_raw = np.array([[to_s8(to_u8(i0)), to_s8(to_u8(i1))],
                           [to_s8(to_u8(i2)), to_s8(to_u8(i3))]], dtype=np.int16)
        B = B_raw.T if transpose else B_raw
        C = A @ B
        flat = C.flatten().tolist()  # [c00,c01,c10,c11] in the order used by feeder tests

        c00, c01, c10, c11 = [int(x) for x in flat]

        result_bytes = []

        # Cycle 0: clear should assert when mmu_cycle==0
        dut.mmu_cycle.value = 0
        await RisingEdge(dut.clk)
        assert dut.clear.value == 1
        assert dut.done.value == 0

        # Cycle 1: set c00 (first word) and verify a/b selection
        dut.mmu_cycle.value = 1
        dut.c00.value = c00
        await RisingEdge(dut.clk)
        assert dut.clear.value == 0
        assert dut.done.value == 0

        # Cycle 2: host_outdata should stream c00[15:8]
        dut.mmu_cycle.value = 2
        dut.c01.value = c01
        dut.c10.value = c10
        await RisingEdge(dut.clk)
        result_bytes.append(dut.host_outdata.value.integer)
        assert dut.done.value == 1

        # Cycle 3: host_outdata should stream c00[7:0]
        dut.mmu_cycle.value = 3
        dut.c11.value = c11
        await RisingEdge(dut.clk)
        result_bytes.append(dut.host_outdata.value.integer)

        # Remaining bytes for c01/c10/c11 hi/lo
        for mmu_c in range(4, 8):
            dut.mmu_cycle.value = mmu_c
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        # Output stream continues for the "final two" bytes after mmu_cycle wraps
        for mmu_c in range(0, 2):
            dut.mmu_cycle.value = mmu_c
            await RisingEdge(dut.clk)
            result_bytes.append(dut.host_outdata.value.integer)

        assert len(result_bytes) == 8

        words = [
            to_s16_from_u16((result_bytes[0] << 8) | result_bytes[1]),
            to_s16_from_u16((result_bytes[2] << 8) | result_bytes[3]),
            to_s16_from_u16((result_bytes[4] << 8) | result_bytes[5]),
            to_s16_from_u16((result_bytes[6] << 8) | result_bytes[7]),
        ]

        for k, (exp, got) in enumerate(zip(flat, words)):
            assert exp == got, f"{name}: C[{k}] got {got}, expected {exp}"

        dut._log.info(f"{name} passed")

