# Objects depend on this Makefile, so editing it rebuilds them.
TYR_MAKEFILE := $(lastword $(MAKEFILE_LIST))
.DEFAULT_GOAL := all

# Detect Lean
ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

LEAN_INCLUDE := $(LEAN_HOME)/include

# Detect LIBTORCH


# TORCH FLAGS
TORCHDIR=../external/wheels/torch
TORCH_INC_FLAGS=-isystem $(TORCHDIR)/include -isystem $(TORCHDIR)/include/torch/csrc/api/include -isystem $(TORCHDIR)/include/TH -isystem $(TORCHDIR)/include/THC
TORCH_LIBPATH=$(TORCHDIR)/lib
TORCH_CUDA_VENDOR_LIBPATH=../external/wheels/nvidia/cu13/lib
GLOO_DIR ?= ../external/git/gloo
SOXR_DIR=../external/git/soxr
SOXR_BUILD_DIR=build/soxr
SOXR_INC_FLAGS=-I$(SOXR_DIR)/src -I$(SOXR_BUILD_DIR)
SOXR_LIBPATH=$(SOXR_BUILD_DIR)/src
SOXR_LIB=$(SOXR_LIBPATH)/libsoxr.a
SOXR_OBJ_DIR=$(SOXR_BUILD_DIR)/obj
SOXR_CONFIG=$(SOXR_BUILD_DIR)/soxr-config.h
SOXR_SRCS := soxr data-io dbesi0 filter fft4g64 cr cr32 fft4g32 cr64 vr32
SOXR_OBJS := $(addprefix $(SOXR_OBJ_DIR)/,$(addsuffix .o,$(SOXR_SRCS)))
SOXR_CFLAGS := -std=gnu89 -O3 -fPIC -I$(SOXR_DIR)/src -I$(SOXR_BUILD_DIR) -DSOXR_LIB

# ARROW/PARQUET FLAGS
# The Arrow and Parquet libraries of the pyarrow wheel fetched by deps/fetch.sh,
# linked by file name (they carry the version, e.g. libarrow.so.2500).
UNAME_S := $(shell uname -s)
ARROW_DIR := ../external/wheels/pyarrow
ifeq ($(UNAME_S),Darwin)
ARROW_LIBS := $(wildcard $(ARROW_DIR)/libarrow.*.dylib $(ARROW_DIR)/libparquet.*.dylib)
else
ARROW_LIBS := $(wildcard $(ARROW_DIR)/libarrow.so.* $(ARROW_DIR)/libparquet.so.*)
endif
ARROW_INC_FLAGS := -isystem $(ARROW_DIR)/include
ARROW_LINK_FLAGS := $(ARROW_LIBS)



# Config

MKPATH := mkdir -p
RMPATH := rm -rf

AR := ar
CC := gcc
CXX := c++
# CUDA_HOME is passed by lakefile.lean from `lake -R -Kcuda=...`, which also
# checks it against the fetched libtorch. Unset or empty: CPU build.
ifneq ($(strip $(CUDA_HOME)),)
NVCC := $(CUDA_HOME)/bin/nvcc
HAS_NVCC := 1
CUDA_MAJOR_VERSION := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n1)
else
HAS_NVCC := 0
endif
CUDA_INC_FLAGS :=
HAS_CUDA_TOOLKIT := 0
CUDA_LIB_FLAGS :=
ifeq ($(HAS_NVCC),1)
ifneq ($(wildcard $(CUDA_HOME)/include/cuda_runtime.h),)
CUDA_INC_FLAGS += -I$(CUDA_HOME)/include
endif
ifneq ($(wildcard $(CUDA_HOME)/include/cuda_runtime_api.h),)
HAS_CUDA_TOOLKIT := 1
endif
CUBLASLT_LINK_FLAG := -lcublasLt
ifneq ($(wildcard $(TORCH_CUDA_VENDOR_LIBPATH)/libcublasLt.so*),)
CUDA_LIB_FLAGS += -L$(TORCH_CUDA_VENDOR_LIBPATH)
CUBLASLT_LINK_FLAG := -l:libcublasLt.so.13
endif
ifneq ($(wildcard $(CUDA_HOME)/lib64),)
CUDA_LIB_FLAGS += -L$(CUDA_HOME)/lib64
endif
ifneq ($(wildcard $(CUDA_HOME)/targets/aarch64-linux/lib),)
CUDA_LIB_FLAGS += -L$(CUDA_HOME)/targets/aarch64-linux/lib
endif
ifeq ($(HAS_CUDA_TOOLKIT),1)
CUDA_LIB_FLAGS += $(CUBLASLT_LINK_FLAG)
endif
endif
NCCL_ROOT ?= ../external/wheels/nvidia/nccl
NCCL_INC_FLAGS :=
ifneq ($(wildcard $(NCCL_ROOT)/include/nccl.h),)
NCCL_INC_FLAGS += -I$(NCCL_ROOT)/include
endif
NCCL_LIB_FLAGS :=
ifneq ($(wildcard $(NCCL_ROOT)/lib),)
NCCL_LIB_FLAGS += -L$(NCCL_ROOT)/lib
endif
CXX_STD ?= c++20
CXX_STD_FLAG := --std=$(CXX_STD)
EXTRA_CXX_FLAGS := -fPIC -O3 -Wall -Wno-sign-compare \
		 -Wunused-parameter -Wno-missing-field-initializers -Wno-comment
NVCCFLAGS := -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -lineinfo -Xcompiler -fPIC
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__

# Physical GPU target (controls generated SASS).
# Set by lakefile.lean from `-Kgpu`; H100 when building without CUDA.
GPU ?= H100

ifeq ($(GPU),H100)
GPU_ARCH := 90a
else ifeq ($(GPU),A100)
GPU_ARCH := 80a
else ifeq ($(GPU),B200)
GPU_ARCH := 100a
else ifeq ($(GPU),B300)
GPU_ARCH := 103a
else ifeq ($(GPU),GB10)
GPU_ARCH := 121
else
$(error Unsupported GPU: $(GPU). Use -Kgpu=H100, A100, B200, B300 or GB10)
endif
GPU_COMPUTE := compute_$(GPU_ARCH)
GPU_CODE := sm_$(GPU_ARCH)

NVCCFLAGS += -gencode arch=$(GPU_COMPUTE),code=$(GPU_CODE)

# The pinned torch wheels include Gloo's implementation but omit its public
# headers. deps/git.lock supplies the exact matching upstream header revision.
TORCH_INC_FLAGS += -isystem $(GLOO_DIR)

# Optional distributed backend toggles based on available LibTorch artifacts.
ifneq ($(wildcard $(TORCHDIR)/include/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libtorch_cuda.*),)
ifneq ($(wildcard $(TORCH_LIBPATH)/libc10_cuda.*),)
ifneq ($(strip $(NCCL_INC_FLAGS)),)
EXTRA_CXX_FLAGS += -DUSE_C10D_NCCL
TORCH_CUDA_LINK_FLAGS := -ltorch_cuda -lc10_cuda
endif
endif
endif
endif

# Clang-only warning flag (unsupported by GCC).
ifneq ($(findstring clang,$(shell $(CXX) --version 2>/dev/null)),)
EXTRA_CXX_FLAGS += -Winconsistent-missing-override
endif

SRC_DIR := src
HDR_DIR := include
OUT_DIR := build
TOOLS_DIR := tools
BUILD_FLAGS := $(OUT_DIR)/build-flags.txt

SRCS := \
	tyr.cpp\
	tyr_gated_delta.cpp\
	tyr_distributed.cpp\
	tyr_polar.cpp\
	tyr_media_stub.cpp\
	tyr_parquet.cpp
MM_SRCS :=
ifeq ($(UNAME_S),Darwin)
MM_SRCS += apple_mps.mm
MM_SRCS += apple_vl_processor.mm
MM_SRCS += apple_audio_input.mm
MM_SRCS += apple_audio_output.mm
APPLE_FRAMEWORK_FLAGS := \
	-framework Foundation \
	-framework CoreFoundation \
	-framework Metal \
	-framework CoreGraphics \
	-framework ImageIO \
	-framework AVFoundation \
	-framework CoreMedia \
	-framework CoreVideo \
	-framework VideoToolbox \
	-framework Accelerate \
	-framework AudioToolbox
else
APPLE_FRAMEWORK_FLAGS :=
endif
# tyr_laguna_moe.cu (fused NVFP4 MoE expert forward) is arch-portable CUDA and
# compiles for every selected GPU target.
ifeq ($(HAS_NVCC),1)
CU_SRCS := $(SRC_DIR)/tyr_laguna_moe.cu
else
CU_SRCS :=
endif
# Without NVCC, src/tyr_laguna_moe.cu is not compiled. Compile its non-CUDA
# stub branch (the #ifndef __CUDACC__ section of
# the same file) as plain C++ so the `lean_torch_laguna_moe_fp4_forward`
# extern symbol still resolves for Lean executables.
ifneq ($(HAS_NVCC),1)
LAGUNA_MOE_STUB_OBJ := $(OUT_DIR)/tyr_laguna_moe_stub.o
else
LAGUNA_MOE_STUB_OBJ :=
endif

LIB_NAME := TyrC
ifeq ($(UNAME_S),Darwin)
DYLIB := lib${LIB_NAME}.dylib
SHARED_FLAG := -dynamiclib
else
DYLIB := lib${LIB_NAME}.so
SHARED_FLAG := -shared
endif

CPP_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(SRCS:.cpp=.o))
MM_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(MM_SRCS:.mm=.o))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OUT_DIR)/%.o,$(CU_SRCS))
OBJ_FILES := $(CPP_OBJ_FILES) $(MM_OBJ_FILES) $(CU_OBJ_FILES) $(LAGUNA_MOE_STUB_OBJ)
DEP_FILES := $(OBJ_FILES:.o=.d) $(SOXR_OBJS:.o=.d)

# Lean runtime library
LEAN_LIB := $(LEAN_HOME)/lib/lean

# Shared library link flags. libTyrC is linked into Lean executables and
# precompiled Lean modules, which already contain the Lean runtime, so it does
# not link libleanshared: its `lean_*` references resolve from the loading
# process. Linking it here with the system toolchain keeps the system glibc and
# libstdc++ behind this library, so Lake can link Lean code with its bundled
# clang.
ifeq ($(UNAME_S),Darwin)
DYLIB_LEAN_FLAGS := -undefined dynamic_lookup -install_name @rpath/$(DYLIB)
RPATH_ORIGIN := @loader_path
DYLIB_EXPORTS := map/libTyrC.exports
DYLIB_EXPORT_FLAGS := -Wl,-exported_symbols_list,$(DYLIB_EXPORTS)
else
DYLIB_LEAN_FLAGS :=
RPATH_ORIGIN := \$$ORIGIN
DYLIB_EXPORTS := map/libTyrC.map
DYLIB_EXPORT_FLAGS := -Wl,--version-script=$(DYLIB_EXPORTS)
endif
# Vendored libraries, relative to $(OUT_DIR) so the checkout can move.
DYLIB_RPATH_FLAGS := $(foreach dir,torch/lib pyarrow nvidia/cu13/lib,-Wl,-rpath,$(RPATH_ORIGIN)/../../external/wheels/$(dir))
DYLIB_LINK_FLAGS := $(DYLIB_LEAN_FLAGS) \
	-L$(TORCH_LIBPATH) -ltorch -ltorch_cpu -lc10 $(TORCH_CUDA_LINK_FLAGS) \
	$(ARROW_LINK_FLAGS) \
	-L$(SOXR_LIBPATH) -lsoxr \
	$(CUDA_LIB_FLAGS) $(NCCL_LIB_FLAGS) \
	$(DYLIB_RPATH_FLAGS) \
	$(APPLE_FRAMEWORK_FLAGS)

# Build Rules

# Everything that changes compiled output, one `KEY=value` per line. Objects
# depend on this file, so a changed compiler, flag or GPU rebuilds
# them. It is rewritten only when its content changes (FORCE runs the recipe
# every time; an unchanged timestamp leaves dependents alone).
shell_quote = '$(subst ','"'"',$(1))'
BUILD_FLAGS_KEYS := CC CXX NVCC LEAN_HOME TORCHDIR SOXR_DIR GLOO_DIR \
	CXX_STD_FLAG EXTRA_CXX_FLAGS NVCCFLAGS SOXR_CFLAGS \
	TORCH_INC_FLAGS CUDA_INC_FLAGS NCCL_INC_FLAGS ARROW_INC_FLAGS \
	GPU GPU_COMPUTE GPU_CODE HAS_NVCC HAS_CUDA_TOOLKIT \
	CUDA_HOME CUDA_MAJOR_VERSION OBJ_FILES DYLIB_LINK_FLAGS

.PHONY: check-deps FORCE
FORCE:

$(BUILD_FLAGS): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' $(foreach key,$(BUILD_FLAGS_KEYS),$(call shell_quote,$(key)=$($(key)))) > $@.new
	@if cmp -s $@.new $@; then rm $@.new; else mv $@.new $@; fi

-include $(DEP_FILES)

check-deps:
	@for compiler in $(firstword $(CC)) $(firstword $(CXX)); do \
		if ! command -v "$$compiler" >/dev/null 2>&1; then \
			echo "Missing C/C++ compiler $$compiler. On Debian/Ubuntu: sudo apt install g++"; \
			exit 1; \
		fi; \
	done
	@if [ ! -f "$(SOXR_DIR)/CMakeLists.txt" ]; then \
		echo "Missing $(SOXR_DIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	@if [ ! -f "$(TORCHDIR)/include/torch/csrc/api/include/torch/torch.h" ] || [ ! -d "$(TORCH_LIBPATH)" ]; then \
		echo "Missing LibTorch at $(TORCHDIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	@if [ ! -f "$(GLOO_DIR)/gloo/algorithm.h" ]; then \
		echo "Missing Gloo headers at $(GLOO_DIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	@if [ "$(words $(ARROW_LIBS))" != 2 ]; then \
		echo "Missing the Arrow and Parquet libraries in $(ARROW_DIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi

all: dylib

dylib: check-deps soxr $(OUT_DIR)/$(DYLIB)

soxr: check-deps $(SOXR_LIB)

$(SOXR_LIB): $(SOXR_OBJS) | check-deps
	@if [ ! -f "$(SOXR_DIR)/CMakeLists.txt" ]; then \
		echo "Missing $(SOXR_DIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	$(MKPATH) $(SOXR_LIBPATH)
	$(RMPATH) $@
	$(AR) rcs $@ $(SOXR_OBJS)

$(SOXR_CONFIG): | check-deps
	$(MKPATH) $(SOXR_BUILD_DIR)
	@printf '%s\n' \
		'#if !defined soxr_config_included' \
		'#define soxr_config_included' \
		'#define AVCODEC_FOUND 0' \
		'#define AVUTIL_FOUND 0' \
		'#define WITH_PFFFT 0' \
		'#define HAVE_FENV_H 1' \
		'#define HAVE_STDBOOL_H 1' \
		'#define HAVE_STDINT_H 1' \
		'#define HAVE_LRINT 1' \
		'#define HAVE_BIGENDIAN 0' \
		'#define WITH_CR32 1' \
		'#define WITH_CR32S 0' \
		'#define WITH_CR64 1' \
		'#define WITH_CR64S 0' \
		'#define WITH_VR32 1' \
		'#define WITH_HI_PREC_CLOCK 1' \
		'#define WITH_FLOAT_STD_PREC_CLOCK 0' \
		'#define WITH_DEV_TRACE 0' \
		'#endif' > $@

$(SOXR_OBJ_DIR)/%.o: $(SOXR_DIR)/src/%.c $(SOXR_CONFIG) $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps
	$(MKPATH) $(SOXR_OBJ_DIR)
	$(CC) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< $(SOXR_CFLAGS)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(DYLIB) : $(OBJ_FILES) $(SOXR_LIB) $(DYLIB_EXPORTS) $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	$(CXX) $(SHARED_FLAG) -o $@ $(OBJ_FILES) $(DYLIB_LINK_FLAGS) $(DYLIB_EXPORT_FLAGS)
	@$(TOOLS_DIR)/check_libstdcxx.sh $(CXX) $@

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_STD_FLAG) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC

$(OUT_DIR)/%.o : $(SRC_DIR)/%.mm $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) -x objective-c++ $(CXX_STD_FLAG) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -fobjc-arc

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cu $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NVCCFLAGS)

# No-CUDA-toolkit builds: compile the stub branch of tyr_laguna_moe.cu as C++.
$(OUT_DIR)/tyr_laguna_moe_stub.o : $(SRC_DIR)/tyr_laguna_moe.cu $(BUILD_FLAGS) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	$(CXX) $(CXX_STD_FLAG) -x c++ -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(LEAN_INCLUDE) $(EXTRA_CXX_FLAGS) -fPIC

clean:
	$(RMPATH) $(OUT_DIR)
	$(RMPATH) $(SOXR_BUILD_DIR)
