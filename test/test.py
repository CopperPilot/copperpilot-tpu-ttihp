# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb  # type: ignore
from cocotb.clock import Clock  # type: ignore
from cocotb.triggers import ClockCycles  # type: ignore

import numpy as np  # type: ignore


def to_u8(v: int) -> int:
    return v & 0xFF


def u16_to_s16(word: int) -> int:
    return word - 0x10000 if word >= 0x8000 else word


async def reset_dut(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)


def expected_matmul(A_flat, B_flat, transpose: int, relu: int):
    A = np.array(A_flat, dtype=np.int16).reshape((2, 2))
    B_raw = np.array(B_flat, dtype=np.int16).reshape((2, 2))
    B = B_raw.T if transpose else B_raw
    C = A @ B
    if relu:
        C = np.maximum(C, 0)
    return [int(x) for x in C.flatten()]  # [c00,c01,c10,c11]


async def load_2x2(dut, A_flat, B_flat, transpose: int, relu: int):
    # uio_in[0]=load_en, uio_in[1]=transpose, uio_in[2]=activation(relu)
    ctrl = (transpose << 1) | (relu << 2) | 1

    # Load A -> memory addresses 0..3
    for i in range(4):
        dut.ui_in.value = to_u8(A_flat[i])
        dut.uio_in.value = ctrl
        await ClockCycles(dut.clk, 1)

    # Load B -> memory addresses 4..7
    for i in range(4):
        dut.ui_in.value = to_u8(B_flat[i])
        dut.uio_in.value = ctrl
        await ClockCycles(dut.clk, 1)

    # Stop loading
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 0
    await ClockCycles(dut.clk, 1)


async def capture_output_byte_window(dut, nbytes: int = 16):
    """
    Capture `nbytes` from `uo_out` once `uio_out[7]` (done) asserts.
    We capture a small window and let the caller align/interpret bytes.
    """
    for _ in range(200):
        await ClockCycles(dut.clk, 1)
        done = (int(dut.uio_out.value.integer) >> 7) & 1
        if done == 1:
            out = []
            for _ in range(nbytes):
                out.append(dut.uo_out.value.integer)
                if len(out) < nbytes:
                    await ClockCycles(dut.clk, 1)
            return out

    raise AssertionError("Timed out waiting for done==1")


def decode_words_from_bytes(raw_bytes, offset: int, swap_hi_lo: bool):
    """
    Decode four signed 16-bit words from 8 consecutive bytes.
    If `swap_hi_lo` is True, interpret each word as (lo<<8)|hi.
    """
    b = raw_bytes[offset : offset + 8]
    if len(b) != 8:
        return None

    def bytes_to_word(i: int) -> int:
        hi = b[i * 2 + 0]
        lo = b[i * 2 + 1]
        word_u16 = ((lo << 8) | hi) if swap_hi_lo else ((hi << 8) | lo)
        return u16_to_s16(word_u16)

    return [bytes_to_word(i) for i in range(4)]


def match_expected_words(raw_bytes, exp_words):
    """
    Try multiple alignments and hi/lo interpretations until we match expected words.
    Returns (decoded_words, offset, swap_hi_lo).
    """
    for offset in range(0, 8):  # enough for small off-by-one alignment errors
        for swap_hi_lo in (False, True):
            decoded = decode_words_from_bytes(raw_bytes, offset=offset, swap_hi_lo=swap_hi_lo)
            if decoded is None:
                continue
            if decoded == exp_words:
                return decoded, offset, swap_hi_lo
    return None, None, None


@cocotb.test()
async def test_tpu_matmul_stream_regular(dut):
    """End-to-end test: A*B with transpose=0, relu=0."""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    A = [1, 2, 3, 4]          # row-major A00,A01,A10,A11
    B = [5, 6, 7, 8]          # row-major B00,B01,B10,B11
    transpose = 0
    relu = 0

    await load_2x2(dut, A, B, transpose=transpose, relu=relu)
    raw_bytes = await capture_output_byte_window(dut, nbytes=16)

    exp = expected_matmul(A, B, transpose=transpose, relu=relu)
    words, offset, swap = match_expected_words(raw_bytes, exp_words=exp)
    assert words is not None, f"Output bytes did not align: raw={raw_bytes} exp={exp}"

    dut._log.info("TPU regular matmul passed")


@cocotb.test()
async def test_tpu_matmul_stream_transpose_relu(dut):
    """End-to-end test: A*B^T with relu=1."""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]
    transpose = 1
    relu = 1

    await load_2x2(dut, A, B, transpose=transpose, relu=relu)
    raw_bytes = await capture_output_byte_window(dut, nbytes=16)

    exp = expected_matmul(A, B, transpose=transpose, relu=relu)
    words, offset, swap = match_expected_words(raw_bytes, exp_words=exp)
    assert words is not None, f"Output bytes did not align: raw={raw_bytes} exp={exp}"

    dut._log.info("TPU transpose+ReLU matmul passed")


@cocotb.test()
async def test_tpu_matmul_stream_signed_edges(dut):
    """End-to-end test with signed saturation edges (two's complement inputs)."""
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Includes negative values
    A = [127, -128, 127, -128]
    B = [7, 6, 5, 4]
    transpose = 0
    relu = 0

    await load_2x2(dut, A, B, transpose=transpose, relu=relu)
    raw_bytes = await capture_output_byte_window(dut, nbytes=16)

    exp = expected_matmul(A, B, transpose=transpose, relu=relu)
    words, offset, swap = match_expected_words(raw_bytes, exp_words=exp)
    assert words is not None, f"Output bytes did not align: raw={raw_bytes} exp={exp}"

    dut._log.info("TPU signed-edge matmul passed")
