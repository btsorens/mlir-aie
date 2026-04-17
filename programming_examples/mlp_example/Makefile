##===- Makefile -----------------------------------------------------------===##
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
#
##===----------------------------------------------------------------------===##

MLIR_AIE_DIR   := $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
PEANO_INSTALL_DIR := $(shell python3 -c \
	"import site, os; \
	 hits = [os.path.join(d,'llvm-aie') for d in site.getsitepackages() \
	         if os.path.isdir(os.path.join(d,'llvm-aie'))]; \
	 print(hits[0] if hits else '')")

KERNELS_DIR    := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNEL_CC:= $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_CFLAGS  := -O2 -std=c++20 --target=aie2-none-unknown-elf \
                  -Wno-parentheses -Wno-attributes -Wno-macro-redefined \
                  -Wno-empty-body -DNDEBUG \
                  -I $(MLIR_AIE_DIR)/include

# MLP uses M=64, K=64, n_tile=16, int16 in/out
KERNEL_DEFINES := -Dbf16_bf16_ONLY -DDIM_M=64 -DDIM_K=64 -DDIM_N=16 -DVECTORIZED_ONLY

BUILD_DIR := build
MM_OBJ    := $(BUILD_DIR)/mm.o
RELU_OBJ  := $(BUILD_DIR)/relu.o
ARCHIVE   := $(BUILD_DIR)/mm.a

.PHONY: all run clean

all: $(ARCHIVE)

$(ARCHIVE): $(MM_OBJ) $(RELU_OBJ)
	ar rcs $@ $^

$(MM_OBJ): $(KERNELS_DIR)/mm.cc $(KERNELS_DIR)/zero.cc | $(BUILD_DIR)
	cd $(BUILD_DIR) && $(KERNEL_CC) $(KERNEL_CFLAGS) $(KERNEL_DEFINES) -c $< -o $(notdir $@)

$(RELU_OBJ): $(KERNELS_DIR)/relu.cc | $(BUILD_DIR)
	cd $(BUILD_DIR) && $(KERNEL_CC) $(KERNEL_CFLAGS) $(KERNEL_DEFINES) -c $< -o $(notdir $@)

$(BUILD_DIR):
	mkdir -p $@

run: all
	python3 mlp.py

clean:
	rm -rf $(BUILD_DIR)
