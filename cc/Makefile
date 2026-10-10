# Detect Lean
# Lake tracks this file hash for extern_lib invalidation.
TYR_MAKEFILE := $(lastword $(MAKEFILE_LIST))
.DEFAULT_GOAL := all

ifndef LEAN_HOME
LEAN ?= lean
LEAN_HOME := $(shell $(LEAN) --print-prefix)
endif

LEAN_INCLUDE := $(LEAN_HOME)/include

# Detect LIBTORCH


# TORCH FLAGS
TORCHDIR=../external/wheels/torch
TORCH_INC_FLAGS=-I$(TORCHDIR)/include -I$(TORCHDIR)/include/torch/csrc/api/include  -I$(TORCHDIR)/include/TH  -I$(TORCHDIR)/include/THC
TORCH_LIBPATH=$(TORCHDIR)/lib
TORCH_CUDA_VENDOR_LIBPATH=../external/wheels/nvidia/cu13/lib
# Python.h is only needed by tk_vendor_mha_h100.cu (it includes torch's pybind.h).
# Prefer CPython 3.12, the version the fetched wheels target (WHEEL_PYTHON in
# deps/lock_wheels.py).
PYTHON ?= $(shell command -v python3.12 2>/dev/null || echo python3)
PYTHON_INCLUDE := $(shell "$(PYTHON)" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include") or "")' 2>/dev/null)
PYTHON_INC_FLAGS := $(if $(strip $(PYTHON_INCLUDE)),-I$(PYTHON_INCLUDE),)
TKDIR=../external/git/ThunderKittens
TK_INC_FLAGS=-I$(TKDIR)/include -I$(TKDIR)/prototype
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
# Fetched by deps/fetch.sh from the pinned pyarrow wheel. Libraries are
# linked by exact file name; bump ARROW_SOVERSION together with the pyarrow pin
# in deps/lock_wheels.py.
UNAME_S := $(shell uname -s)
ARROW_DIR := ../external/wheels/pyarrow
ARROW_SOVERSION := 2500
ifeq ($(UNAME_S),Darwin)
ARROW_LIBS := $(ARROW_DIR)/libarrow.$(ARROW_SOVERSION).dylib $(ARROW_DIR)/libparquet.$(ARROW_SOVERSION).dylib
else
ARROW_LIBS := $(ARROW_DIR)/libarrow.so.$(ARROW_SOVERSION) $(ARROW_DIR)/libparquet.so.$(ARROW_SOVERSION)
endif
ARROW_INC_FLAGS := -I$(ARROW_DIR)/include
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
ifneq ($(wildcard $(CUDA_HOME)/include/cuda_runtime.h),)
CUDA_INC_FLAGS += -I$(CUDA_HOME)/include
endif
HAS_CUDA_TOOLKIT := 0
ifeq ($(HAS_NVCC),1)
ifneq ($(wildcard $(CUDA_HOME)/include/cuda_runtime_api.h),)
HAS_CUDA_TOOLKIT := 1
endif
endif
CPP_CUDA_FLAGS := -DTYR_OPS_HAS_CUDA_TOOLKIT=$(HAS_CUDA_TOOLKIT)
CUDA_LIB_FLAGS :=
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
		 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment
NVCCFLAGS := -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -lineinfo -Xcompiler -fPIC
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# CUDA 12.1 headers do not expose cudaLaunchAttributePreferredClusterDimension.
ifneq ($(strip $(CUDA_MAJOR_VERSION)),)
ifeq ($(shell [ "$(CUDA_MAJOR_VERSION)" -lt 13 ] && echo 1 || echo 0),1)
NVCCFLAGS += -DcudaLaunchAttributePreferredClusterDimension=cudaLaunchAttributeClusterDimension
endif
endif

# Physical GPU target (controls generated SASS) and logical kernel family
# (controls which ThunderKittens/Tyr arch guards are enabled) are split so
# physical Blackwell-family targets are not instruction-compatible: GB10 uses
# SM121 warp MMA, Hopper uses WGMMA, and B200/B300 use architecture-specific
# tcgen05/TMEM instructions.
GPU ?= H100
GPU_FAMILY ?=
GPU_COMPUTE ?=
GPU_CODE ?=

ifeq ($(GPU),GB10)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := BLACKWELL
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_121
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_121
endif
else ifeq ($(GPU),B300)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := BLACKWELL
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_103a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_103a
endif
else ifeq ($(GPU),B200)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := BLACKWELL
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_100a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_100a
endif
else ifeq ($(GPU),A100)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := AMPERE
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_80a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_80a
endif
else ifeq ($(GPU),H100)
ifeq ($(strip $(GPU_FAMILY)),)
GPU_FAMILY := HOPPER
endif
ifeq ($(strip $(GPU_COMPUTE)),)
GPU_COMPUTE := compute_90a
endif
ifeq ($(strip $(GPU_CODE)),)
GPU_CODE := sm_90a
endif
else
$(error Unsupported GPU: $(GPU). Please set GPU to A100, H100, B200, B300, or GB10)
endif

ifeq ($(GPU_FAMILY),BLACKWELL)
NVCCFLAGS += -DKITTENS_BLACKWELL
else ifeq ($(GPU_FAMILY),AMPERE)
NVCCFLAGS += -DKITTENS_AMPERE
else ifeq ($(GPU_FAMILY),HOPPER)
NVCCFLAGS += -DKITTENS_HOPPER
else
$(error Unsupported GPU_FAMILY: $(GPU_FAMILY). Please set GPU_FAMILY to AMPERE, HOPPER, or BLACKWELL)
endif

NVCCFLAGS += -gencode arch=$(GPU_COMPUTE),code=$(GPU_CODE)

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
BENCH_FLASH_ATTN := $(OUT_DIR)/tools/bench_flash_attn
GPU_STUB_SRC := $(SRC_DIR)/generated/tyr_gpu_kernel_stubs.cpp
GPU_STUB_SCRIPT := $(TOOLS_DIR)/generate_gpu_kernel_stubs.py
GPU_STUB_IR_ROOT := ../.lake/build/ir/Tyr/GPU
GPU_KERNEL_SRC_ROOT := ../Tyr/GPU/Kernels
NATIVE_CONFIG := $(OUT_DIR)/native-build.json

SRCS := \
	tyr.cpp\
	tyr_qwen35.cpp\
	tyr_ops.cpp\
	tyr_distributed.cpp\
	tyr_polar.cpp\
	tyr_media_stub.cpp\
	tyr_parquet.cpp\
	tyr_execution.cpp\
	float_buffer.cpp
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
# tk_vendor_mha_h100.cu emits Hopper `wgmma` instructions; ptxas rejects it for
# sm_10x/sm_12x (Blackwell) targets like B200/GB10. The launcher stubs are
# weak so the runtime dispatcher's Hopper-arch gate routes those problems to
# the portable SDPA fallback when the kernels are absent.
ifeq ($(GPU),H100)
STATIC_CU_SRCS := $(SRC_DIR)/tk_vendor_mha_h100.cu
else
STATIC_CU_SRCS :=
endif
# tyr_laguna_moe.cu (fused NVFP4 MoE expert forward) is arch-portable CUDA —
# no Hopper WGMMA or Blackwell tcgen05/TMEM instructions — so it compiles for
# every selected GPU target (including GB10 sm_121), unlike the H100-only
# vendored kernel above.
STATIC_CU_SRCS += $(SRC_DIR)/tyr_laguna_moe.cu
GPU_CODEGEN_MODULES := $(strip $(TYR_GPU_CODEGEN_MODULE))
ifeq ($(GPU_CODEGEN_MODULES),)
CU_SRCS := $(STATIC_CU_SRCS) $(wildcard $(SRC_DIR)/generated/*.cu)
else ifeq ($(GPU_CODEGEN_MODULES),all)
CU_SRCS := $(STATIC_CU_SRCS) $(wildcard $(SRC_DIR)/generated/*.cu)
else
GPU_CODEGEN_CU_NAMES := $(addsuffix .cu,$(subst .,_,$(GPU_CODEGEN_MODULES)))
# Use $(wildcard ...) so that .cu files which haven't been generated yet are
# silently dropped from CU_SRCS rather than producing "no rule to make target"
# errors. The outer libtyr build orchestrates codegen *before* this make runs,
# so all expected .cu files exist by the time we need them. Inner sub-builds
# (those that skip codegen, used to build GenerateGpuKernels itself) don't
# need any kernel symbols and tolerate missing .cu files.
CU_SRCS := $(STATIC_CU_SRCS) $(foreach name,$(GPU_CODEGEN_CU_NAMES),$(wildcard $(SRC_DIR)/generated/$(name)))
endif
# CPU build: no CUDA kernels; the weak launcher stubs stand in for them.
ifneq ($(HAS_NVCC),1)
CU_SRCS :=
endif
ifeq ($(filter $(SRC_DIR)/tk_vendor_mha_h100.cu,$(CU_SRCS)),)
SRCS += tk_vendor_stubs.cpp
endif
# Always include the kernel-launcher stubs in libTyrC.a. The stubs declare
# every `lean_launch_Tyr_GPU_Kernels_...` symbol with `__attribute__((weak))`,
# so when the corresponding generated .cu compiles to a strong definition,
# that wins; otherwise the weak stub provides the symbol so any link
# referencing it (e.g. `tyr_ops.cpp`) succeeds. This breaks the
# chicken-and-egg between `GenerateGpuKernels` (needs libTyrC.a to link, but
# is also what produces the kernels' .cu files) without any conditional.
SRCS += generated/tyr_gpu_kernel_stubs.cpp
REQUIRES_TK := $(if $(strip $(CU_SRCS)),1,0)
# Without NVCC, src/tyr_laguna_moe.cu is not compiled (CU_SRCS is cleared
# above). Compile its non-CUDA stub branch (the #ifndef __CUDACC__ section of
# the same file) as plain C++ so the `lean_torch_laguna_moe_fp4_forward`
# extern symbol still resolves for Lean executables.
ifneq ($(HAS_NVCC),1)
LAGUNA_MOE_STUB_OBJ := $(OUT_DIR)/tyr_laguna_moe_stub.o
else
LAGUNA_MOE_STUB_OBJ :=
endif

LIB_NAME := TyrC
LIB := lib${LIB_NAME}.a
ifeq ($(UNAME_S),Darwin)
DYLIB := lib${LIB_NAME}.dylib
SHARED_FLAG := -dynamiclib
else
DYLIB := lib${LIB_NAME}.so
SHARED_FLAG := -shared
endif

CPP_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(SRCS:.cpp=.o))
# Pull the stubs object out of CPP_OBJ_FILES so we can put it LAST in the
# archive. Static-archive symbol resolution scans members in order; the
# stubs declare weak versions of every kernel launcher, so if they appear
# before the strong definitions emitted from generated CUDA, the stubs
# would be picked first and the real kernels never linked.
GPU_STUB_OBJ := $(OUT_DIR)/generated/tyr_gpu_kernel_stubs.o
CPP_OBJ_FILES_NO_STUBS := $(filter-out $(GPU_STUB_OBJ),$(CPP_OBJ_FILES))
MM_OBJ_FILES := $(addprefix $(OUT_DIR)/,$(MM_SRCS:.mm=.o))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OUT_DIR)/%.o,$(CU_SRCS))
OBJ_FILES := $(CPP_OBJ_FILES_NO_STUBS) $(MM_OBJ_FILES) $(CU_OBJ_FILES) $(GPU_STUB_OBJ) $(LAGUNA_MOE_STUB_OBJ)
DEP_FILES := $(OBJ_FILES:.o=.d) $(SOXR_OBJS:.o=.d) $(BENCH_FLASH_ATTN).d

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
else
DYLIB_LEAN_FLAGS :=
RPATH_ORIGIN := \$$ORIGIN
endif
# Vendored libraries, relative to $(OUT_DIR) so the checkout can move.
DYLIB_RPATH_FLAGS := $(foreach dir,torch/lib pyarrow nvidia/cu13/lib,-Wl,-rpath,$(RPATH_ORIGIN)/../../external/wheels/$(dir))# tk_vendor_mha_h100.cu calls the CUDA driver API (cuTensorMapEncodeTiled).
CUDA_DRIVER_LINK_FLAGS :=
ifneq ($(wildcard $(CUDA_HOME)/lib64/stubs/libcuda.so),)
CUDA_DRIVER_LINK_FLAGS := -L$(CUDA_HOME)/lib64/stubs -lcuda
endif
DYLIB_LINK_FLAGS := $(DYLIB_LEAN_FLAGS) \
	-L$(TORCH_LIBPATH) -ltorch -ltorch_cpu -lc10 $(TORCH_CUDA_LINK_FLAGS) \
	$(ARROW_LINK_FLAGS) \
	-L$(SOXR_LIBPATH) -lsoxr \
	$(CUDA_LIB_FLAGS) $(CUDA_DRIVER_LINK_FLAGS) $(NCCL_LIB_FLAGS) \
	$(DYLIB_RPATH_FLAGS) \
	$(APPLE_FRAMEWORK_FLAGS)

# Build Rules

# Quote configuration values as data, including quotes in compiler flags.
shell_quote = '$(subst ','"'"',$(1))'
NATIVE_CONFIG_KEYS := CC CXX NVCC LEAN_HOME TORCHDIR SOXR_DIR TKDIR \
	CXX_STD_FLAG EXTRA_CXX_FLAGS CPP_CUDA_FLAGS NVCCFLAGS SOXR_CFLAGS \
	TORCH_INC_FLAGS PYTHON_INC_FLAGS CUDA_INC_FLAGS NCCL_INC_FLAGS ARROW_INC_FLAGS \
	GPU GPU_FAMILY GPU_COMPUTE GPU_CODE HAS_NVCC HAS_CUDA_TOOLKIT \
	CUDA_HOME CUDA_MAJOR_VERSION GPU_CODEGEN_MODULES OBJ_FILES DYLIB_LINK_FLAGS \
	MACOSX_DEPLOYMENT_TARGET SDKROOT

.PHONY: check-deps native-config gpu-stubs FORCE
FORCE:

# Lake refreshes these inexpensive manifests before checking its native trace.
# FORCE notices environment and added/removed inputs; write-if-changed preserves
# timestamps when the effective configuration/symbol inventory is unchanged.
native-config: $(NATIVE_CONFIG)
gpu-stubs: $(GPU_STUB_SRC)

$(NATIVE_CONFIG): FORCE $(TOOLS_DIR)/write_build_config.py $(TYR_MAKEFILE)
	@python3 $(TOOLS_DIR)/write_build_config.py --output $@ --dependencies-output $(OUT_DIR)/native-dependencies.txt $(foreach key,$(NATIVE_CONFIG_KEYS),$(call shell_quote,$(key)=$($(key)))) --dependency-files $(wildcard $(DEP_FILES))

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
	@if [ "$(REQUIRES_TK)" = "1" ] && [ ! -d "$(TKDIR)/include" ]; then \
		echo "Missing $(TKDIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	@if [ ! -f "$(TORCHDIR)/include/torch/csrc/api/include/torch/torch.h" ] || [ ! -d "$(TORCH_LIBPATH)" ]; then \
		echo "Missing LibTorch at $(TORCHDIR). Run: deps/fetch.sh"; \
		exit 1; \
	fi
	@for lib in $(ARROW_LIBS); do \
		if [ ! -f "$$lib" ]; then \
			echo "Missing Arrow library $$lib. Run: deps/fetch.sh"; \
			exit 1; \
		fi; \
	done

all: lib dylib

lib: check-deps soxr $(OUT_DIR)/$(LIB)

dylib: check-deps soxr $(OUT_DIR)/$(DYLIB)

bench-flash-attn: check-deps soxr $(OUT_DIR)/$(DYLIB) $(BENCH_FLASH_ATTN)

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

$(SOXR_OBJ_DIR)/%.o: $(SOXR_DIR)/src/%.c $(SOXR_CONFIG) $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps
	$(MKPATH) $(SOXR_OBJ_DIR)
	$(CC) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< $(SOXR_CFLAGS)

$(OUT_DIR):
	$(MKPATH) $@

$(OUT_DIR)/$(LIB) : $(OBJ_FILES) $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	$(RMPATH) $@
	${AR} rcs $@ $(OBJ_FILES)

$(OUT_DIR)/$(DYLIB) : $(OBJ_FILES) $(SOXR_LIB) $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	$(CXX) $(SHARED_FLAG) -o $@ $(OBJ_FILES) $(DYLIB_LINK_FLAGS)
	@$(TOOLS_DIR)/check_libstdcxx.sh $(CXX) $@

$(BENCH_FLASH_ATTN): $(TOOLS_DIR)/bench_flash_attn.cpp $(OUT_DIR)/$(DYLIB) $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_STD_FLAG) -MMD -MP -MF $@.d -MT $@ -o $@ $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(EXTRA_CXX_FLAGS) -L$(OUT_DIR) -l$(LIB_NAME) $(DYLIB_LINK_FLAGS) $(foreach dir,$(OUT_DIR) $(TORCH_LIBPATH) $(ARROW_DIR) $(LEAN_LIB),-Wl,-rpath,$(abspath $(dir))) -L$(LEAN_LIB) -lleanshared

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cpp $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_STD_FLAG) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(CPP_CUDA_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC

$(OUT_DIR)/%.o : $(SRC_DIR)/%.mm $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	$(CXX) -x objective-c++ $(CXX_STD_FLAG) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(CUDA_INC_FLAGS) $(NCCL_INC_FLAGS) $(ARROW_INC_FLAGS) $(SOXR_INC_FLAGS) $(CPP_CUDA_FLAGS) $(EXTRA_CXX_FLAGS) -fPIC -fobjc-arc

$(OUT_DIR)/%.o : $(SRC_DIR)/%.cu $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	@mkdir -p $(dir $@)
	@if [ "$(GPU)" = "GB10" ] && grep -Eq 'tcgen05|tensor_allocator<|warpgroup::(mma_|mm_)|mma_async_wait|mma_commit_group' "$<"; then \
		echo "error: $< requires Hopper WGMMA or SM100a tcgen05/TMEM, neither of which is supported by GB10 sm_121" >&2; \
		exit 2; \
	fi
	$(NVCC) -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(HDR_DIR) -I$(LEAN_INCLUDE) $(TORCH_INC_FLAGS) $(PYTHON_INC_FLAGS) $(CUDA_INC_FLAGS) $(TK_INC_FLAGS) $(NVCCFLAGS)

# No-CUDA-toolkit builds: compile the stub branch of tyr_laguna_moe.cu as C++.
$(OUT_DIR)/tyr_laguna_moe_stub.o : $(SRC_DIR)/tyr_laguna_moe.cu $(NATIVE_CONFIG) $(TYR_MAKEFILE) | check-deps $(OUT_DIR)
	$(CXX) $(CXX_STD_FLAG) -x c++ -MMD -MP -MF $(@:.o=.d) -MT $@ -o $@ -c $< -I$(LEAN_INCLUDE) $(EXTRA_CXX_FLAGS) -fPIC

$(GPU_STUB_SRC): FORCE $(GPU_STUB_SCRIPT)
	@python3 $(GPU_STUB_SCRIPT) --ir-root $(GPU_STUB_IR_ROOT) --kernel-src-root $(GPU_KERNEL_SRC_ROOT) --output $@

clean:
	$(RMPATH) $(OUT_DIR)
	$(RMPATH) $(SOXR_BUILD_DIR)
