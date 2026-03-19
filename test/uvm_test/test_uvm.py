import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import numpy as np

from pyuvm import uvm_test, uvm_root


def to_s8(v: int) -> int:
    return v if v < 128 else v - 256


def to_u8(v: int) -> int:
    return v & 0xFF


def u16_to_s16(word: int) -> int:
    return word - 0x10000 if word >= 0x8000 else word


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_2x2(dut, A, B, transpose: int, relu: int):
    # Load A (weights): addresses 0..3
    for i in range(4):
        dut.ui_in.value = to_u8(A[i])
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1
        await RisingEdge(dut.clk)

    # Load B (inputs): addresses 4..7
    for i in range(4):
        dut.ui_in.value = to_u8(B[i])
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1
        await RisingEdge(dut.clk)

    # Stop loading
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 0
    await RisingEdge(dut.clk)


async def read_stream_when_done(dut, n_bytes: int = 8):
    """
    Wait until uio_out[7] (done) becomes 1, then read n_bytes from uo_out,
    sampling one byte per clock.
    """
    seen = False
    out = []
    for _ in range(80):
        await RisingEdge(dut.clk)
        done = (int(dut.uio_out.value.integer) >> 7) & 1
        if not seen:
            if done == 1:
                seen = True
                # First byte is valid on this same edge
                out.append(dut.uo_out.value.integer)
        else:
            if len(out) < n_bytes:
                out.append(dut.uo_out.value.integer)
            if len(out) >= n_bytes:
                break

    assert len(out) == n_bytes, f"Timed out reading stream; got {len(out)} bytes"
    return out


class CopperpilotMatMulUvmTest(uvm_test):
    async def run_phase(self):
        dut = cocotb.top

        # Simple deterministic transaction (enough to validate end-to-end)
        A = [1, 2, 3, 4]  # [A00,A01,A10,A11]
        B = [5, 6, 7, 8]  # [B00,B01,B10,B11]
        transpose = 1
        relu = 1

        await reset_dut(dut)
        await load_2x2(dut, A, B, transpose=transpose, relu=relu)

        raw_bytes = await read_stream_when_done(dut, n_bytes=8)

        words_u16 = [
            (raw_bytes[0] << 8) | raw_bytes[1],
            (raw_bytes[2] << 8) | raw_bytes[3],
            (raw_bytes[4] << 8) | raw_bytes[5],
            (raw_bytes[6] << 8) | raw_bytes[7],
        ]
        words = [u16_to_s16(w) for w in words_u16]

        A_mat = np.array(A, dtype=np.int16).reshape((2, 2))
        B_mat = np.array(B, dtype=np.int16).reshape((2, 2))
        C = A_mat @ (B_mat.T if transpose else B_mat)
        C = np.maximum(C, 0) if relu else C
        C_flat = C.flatten().tolist()

        for k in range(4):
            assert words[k] == int(C_flat[k]), f"Mismatch at C[{k}]: got {words[k]}, expected {C_flat[k]}"

        dut._log.info("UVM-style full TPU matmul test passed")


@cocotb.test()
async def run_uvm(dut):
    # Drive clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await uvm_root().run_test("CopperpilotMatMulUvmTest")

