<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project is a small 2x2 matrix multiply (systolic TPU) implemented in Verilog and submitted through Tiny Tapeout.

At a high level, the chip supports signed 8-bit inputs and produces signed 16-bit accumulation results. The systolic array contains 4 processing elements (PEs) wired as a 2x2 grid; each PE performs a multiply-accumulate of one streamed operand pair.

### Dataflow

The design is organized into these blocks:

- `memory`: stores 4 weight bytes (matrix A) and 4 input bytes (matrix B)
- `control_unit`: sequences loading from `memory` and orchestrates the compute/output cycles
- `mmu_feeder`: selects which weight/input bytes are presented to the systolic array each cycle, and streams output bytes back out
- `systolic_array_2x2`: performs the multiply-accumulate in a 2x2 systolic layout

The 16-bit results from the systolic array are serialized into two bytes per output element, then buffered to align output timing.

```mermaid
flowchart TD
  Pins[ui_in + uio_in] --> Mem[memory]
  Mem --> Ctrl[control_unit]
  Ctrl --> Feeder[mmu_feeder]
  Feeder --> Array[systolic_array_2x2]
  Array --> Acc[PE accumulations (16-bit)]
  Acc --> Buff[buffer delay chain]
  Buff --> Out[uo_out + uio_out]
```

## How to test

You can test the design with the cocotb-based verification included in this repo.

On each push/PR, Tiny Tapeout CI will run:
- RTL simulation / cocotb tests (`.github/workflows/test.yaml`)
- Docs datasheet generation (`.github/workflows/docs.yaml`)
- GDS build + viewer (`.github/workflows/gds.yaml`)

To run tests locally:
```sh
cd test
make -B
```

### Functional pin usage
- `uio_in[0]`: `load_en` (when 1, each cycle writes `ui_in` into internal memory)
- `uio_in[1]`: `transpose` (treats the B-matrix as transposed)
- `uio_in[2]`: `activation` (when 1, applies ReLU to the systolic outputs)
- `ui_in[7:0]`: signed 8-bit matrix elements being loaded
- `uo_out[7:0]`: streamed output bytes (two bytes per 16-bit result element)
- `uio_out[7]`: `done` (output stream valid)
- `uio_out[6:5]`: control state (timing-aligned)

## External hardware

None. This is a pure digital design using only the Tiny Tapeout GPIO interface.
