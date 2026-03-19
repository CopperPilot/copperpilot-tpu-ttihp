# Makefile
# Tiny Tapeout Verilog template test runner (cocotb).
#
# North-star development: `test/test.py` (full-chip regression).

SIM ?= icarus
FST ?= -fst # Use more efficient FST format (Icarus)
TOPLEVEL_LANG ?= verilog
DEFAULT_GOAL_TARGET ?= all
.DEFAULT_GOAL := $(DEFAULT_GOAL_TARGET)

SRC_DIR = $(PWD)/../src
WAVES_ROOT = $(PWD)

TIMESTAMP = $(shell TZ=US/Eastern date +%Y%m%d_%H:%M:%S)
COMPILE_ARGS += -DTIMESTAMP=\"$(TIMESTAMP)\"

# RTL source lists
TPU_RTL_SOURCES = \
  copperpilot_tpu.v \
  memory.v \
  control_unit.v \
  mmu_feeder.v \
  systolic_array_2x2.v \
  PE.v \
  delay_cell.v

ARRAY_RTL_SOURCES = \
  systolic_array_2x2.v \
  PE.v

ifeq ($(GATES),yes)
# Gate-level mode: rely on the hardened netlist copied in by the GDS workflow.
TPU_TEST_VERILOG_SOURCES = \
  $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_io/verilog/sg13g2_io.v \
  $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v \
  $(PWD)/gate_level_netlist.v \
  $(PWD)/tb.v
else
TPU_TEST_VERILOG_SOURCES = \
  $(PWD)/tb.v \
  $(addprefix $(SRC_DIR)/,$(TPU_RTL_SOURCES))
endif

# Build directories
ifeq ($(GATES),yes)
  SIM_BUILD = sim_build/gl
  COMPILE_ARGS += -DGL_TEST -DFUNCTIONAL -DSIM
endif

# Shared include path for design sources
COMPILE_ARGS += -I$(SRC_DIR)

.PHONY: all test-top test-uvm test-memory test-systolic-array test-control-unit test-mmu-feeder clean

ifeq ($(GATES),yes)
# In gate-level mode we only have the hardened top module netlist.
all: test-top
else
# RTL mode runs the full block test suite + the full TPU north-star.
all: test-uvm test-memory test-systolic-array test-control-unit test-mmu-feeder test-top
endif

# Provide safe defaults for cocotb's own `sim` target.
# If cocotb ends up invoking `sim` without our MODULE/TOPLEVEL overrides,
# it will still run the TPU north-star correctly.
TOPLEVEL ?= tb
COCOTB_TEST_MODULES ?= test
PYTHONPATH ?= $(WAVES_ROOT)
VERILOG_SOURCES ?= $(TPU_TEST_VERILOG_SOURCES)

clean::
	rm -rf sim_build* results.xml test/output* $(WAVES_ROOT)/*.fst $(WAVES_ROOT)/*.vcd

test-top:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=tb \
	  MODULE=test \
	  PYTHONPATH=$(WAVES_ROOT) \
	  VERILOG_SOURCES="$(TPU_TEST_VERILOG_SOURCES)" \
	  WAVES_DIR=$(WAVES_ROOT) \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

test-uvm:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=tb \
	  MODULE=test_uvm \
	  PYTHONPATH="$(WAVES_ROOT)/uvm_test" \
	  VERILOG_SOURCES="$(WAVES_ROOT)/uvm_test/tb.v $(addprefix $(SRC_DIR)/,$(TPU_RTL_SOURCES))" \
	  WAVES_DIR="$(WAVES_ROOT)/uvm_test" \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

test-control-unit:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=control_unit_tb \
	  MODULE=test_control_unit \
	  PYTHONPATH="$(WAVES_ROOT)/control_unit" \
	  VERILOG_SOURCES="$(WAVES_ROOT)/control_unit/control_unit_tb.v $(addprefix $(SRC_DIR)/,control_unit.v)" \
	  WAVES_DIR="$(WAVES_ROOT)/control_unit" \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

test-memory:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=memory_tb \
	  MODULE=test_memory \
	  PYTHONPATH="$(WAVES_ROOT)/memory" \
	  VERILOG_SOURCES="$(WAVES_ROOT)/memory/memory_tb.v $(addprefix $(SRC_DIR)/,memory.v)" \
	  WAVES_DIR="$(WAVES_ROOT)/memory" \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

test-mmu-feeder:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=mmu_feeder_tb \
	  MODULE=test_mmu_feeder \
	  PYTHONPATH="$(WAVES_ROOT)/mmu_feeder" \
	  VERILOG_SOURCES="$(WAVES_ROOT)/mmu_feeder/mmu_feeder_tb.v $(addprefix $(SRC_DIR)/,mmu_feeder.v)" \
	  WAVES_DIR="$(WAVES_ROOT)/mmu_feeder" \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

test-systolic-array:
	$(MAKE) clean
	$(MAKE) sim \
	  TOPLEVEL=systolic_array_tb \
	  MODULE=test_systolic_array \
	  PYTHONPATH="$(WAVES_ROOT)/systolic_array" \
	  VERILOG_SOURCES="$(WAVES_ROOT)/systolic_array/systolic_array_tb.v $(addprefix $(SRC_DIR)/,$(ARRAY_RTL_SOURCES))" \
	  WAVES_DIR="$(WAVES_ROOT)/systolic_array" \
	  COMPILE_ARGS='$(COMPILE_ARGS)'
	! grep -q failure results.xml

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim

# Override cocotb's default `all` target so `make` runs our full suite.
# (cocotb's Makefile.sim may define `all` -> `sim` with default/empty vars)
ifeq ($(GATES),yes)
all:
	$(MAKE) test-top
else
all:
	$(MAKE) test-uvm
	$(MAKE) test-memory
	$(MAKE) test-systolic-array
	$(MAKE) test-control-unit
	$(MAKE) test-mmu-feeder
	$(MAKE) test-top
endif
