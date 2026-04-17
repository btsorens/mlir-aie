#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

# mlp.py – 4-layer Linear MLP on 4 AIE compute tiles (one column)
#
# Each MLP layer runs sequentially on column 0 (4 compute tiles).
# Inter-layer activations pass through NPU/host memory between dispatches.
# This avoids cross-column FIFO routing, which is unsupported in this
# version of IRON.
#
# Within each layer:
#   Input activation (M=64, K=64) – broadcast to 4 tiles in column 0
#   Weight slice     (K=64, n=16) – unique per tile (1/4 of the weight matrix)
#   Output activation (M=64, N=64) – joined from 4 tile outputs in MemTile

import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile
import aie.iron as iron

from aie.helpers.taplib import TensorAccessPattern


def _make_layer(with_relu: bool):
    """Return a JIT-compiled single-column MLP layer (with or without relu)."""
    batch_size       = 64
    input_features   = 64
    output_features  = 64
    features_per_tile = 16  # output_features // 4 tiles

    micro_r, micro_s, micro_t = 4, 8, 4  # bf16 micro-kernel dimensions

    activation_layout_dims = [
        (batch_size // micro_r,     micro_r * input_features),
        (input_features // micro_s, micro_s),
        (micro_r,                   input_features),
        (micro_s,                   1),
    ]
    weight_layout_dims = [
        (input_features // micro_s,    micro_s * features_per_tile),
        (features_per_tile // micro_t, micro_t),
        (micro_s,                      features_per_tile),
        (micro_t,                      1),
    ]
    output_layout_dims = [
        (batch_size // micro_r,          micro_r * features_per_tile),
        (micro_r,                        micro_t),
        (features_per_tile // micro_t,   micro_r * micro_t),
        (micro_t,                        1),
    ]

    activation_flat_type = np.ndarray[(batch_size * input_features,),     np.dtype[bfloat16]]
    weight_flat_type     = np.ndarray[(input_features * output_features,), np.dtype[bfloat16]]
    activation_full_type = np.ndarray[(batch_size, input_features),        np.dtype[bfloat16]]
    activation_tile_type = np.ndarray[(batch_size, features_per_tile),     np.dtype[bfloat16]]
    weight_tile_type     = np.ndarray[(input_features, features_per_tile), np.dtype[bfloat16]]

    build_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    mm_archive = os.path.join(build_dir, "mm.a")

    zero_kernel   = Kernel("zero_bf16",        mm_archive, [activation_tile_type])
    matmul_kernel = Kernel("matmul_bf16_bf16", mm_archive, [activation_full_type, weight_tile_type, activation_tile_type])

    # Input activation: host → Shim(0,0) → MemTile(0,1) [layout reformat] → 4 compute tiles
    input_fifo     = ObjectFifo(activation_full_type, depth=2, name="input_fifo")
    input_col_fifo = input_fifo.cons().forward(
        name="input_col_fifo", dims_to_stream=activation_layout_dims, placement=Tile(0, 1)
    )

    # Weights: host → Shim(0,0) → MemTile(0,1) → split into 4 per-tile slices
    weights_fifo = ObjectFifo(weight_flat_type, depth=1, name="weights_fifo")
    weight_tile_split = weights_fifo.cons().split(
        offsets=[0,
                 input_features * features_per_tile,
                 2 * input_features * features_per_tile,
                 3 * input_features * features_per_tile],
        obj_types=[weight_tile_type] * 4,
        names=["weight_tile_row0", "weight_tile_row1", "weight_tile_row2", "weight_tile_row3"],
        dims_to_stream=[weight_layout_dims] * 4,
        placement=Tile(0, 1),
    )

    # Output: 4 compute tiles → join at MemTile(0,1) → host (tiled → row-major)
    output_fifo = ObjectFifo(
        activation_full_type, depth=2, name="output_fifo",
        dims_to_stream=output_layout_dims,
    )
    activation_join = output_fifo.prod().join(
        offsets=[0,
                 batch_size * features_per_tile,
                 2 * batch_size * features_per_tile,
                 3 * batch_size * features_per_tile],
        obj_types=[activation_tile_type] * 4,
        names=["act_tile_row0", "act_tile_row1", "act_tile_row2", "act_tile_row3"],
        placement=Tile(0, 1),
    )

    if with_relu:
        relu_kernel = Kernel("bf16_relu", mm_archive, [activation_tile_type, activation_tile_type])

        def core_fn_relu(zero, matmul, relu, activation_in, weights, activation_out):
            buf_in  = activation_in.acquire(1)
            buf_w   = weights.acquire(1)
            buf_out = activation_out.acquire(1)
            zero(buf_out)
            matmul(buf_in, buf_w, buf_out)
            relu(buf_out, buf_out)
            activation_in.release(1)
            weights.release(1)
            activation_out.release(1)

        workers = [
            Worker(core_fn_relu,
                   [zero_kernel, matmul_kernel, relu_kernel,
                    input_col_fifo.cons(),
                    weight_tile_split[i].cons(),
                    activation_join[i].prod()],
                   placement=Tile(0, 2 + i),
                   stack_size=0x2000)
            for i in range(4)
        ]
    else:
        def core_fn_linear(zero, matmul, activation_in, weights, activation_out):
            buf_in  = activation_in.acquire(1)
            buf_w   = weights.acquire(1)
            buf_out = activation_out.acquire(1)
            zero(buf_out)
            matmul(buf_in, buf_w, buf_out)
            activation_in.release(1)
            weights.release(1)
            activation_out.release(1)

        workers = [
            Worker(core_fn_linear,
                   [zero_kernel, matmul_kernel,
                    input_col_fifo.cons(),
                    weight_tile_split[i].cons(),
                    activation_join[i].prod()],
                   placement=Tile(0, 2 + i),
                   stack_size=0xD00)
            for i in range(4)
        ]

    tap_act = TensorAccessPattern(tensor_dims=[batch_size * input_features],  offset=0, sizes=[1, batch_size * input_features],  strides=[batch_size * input_features,  1])
    tap_w   = TensorAccessPattern(tensor_dims=[input_features * output_features], offset=0, sizes=[1, input_features * output_features], strides=[input_features * output_features, 1])
    tap_out = TensorAccessPattern(tensor_dims=[batch_size * output_features], offset=0, sizes=[1, batch_size * output_features], strides=[batch_size * output_features, 1])

    runtime = Runtime()
    with runtime.sequence(activation_flat_type, weight_flat_type, activation_flat_type) as (
        act_buf, w_buf, out_buf
    ):
        runtime.start(*workers)
        runtime.fill(placement=Tile(0, 0), in_fifo=input_fifo.prod(),  source=act_buf, tap=tap_act)
        runtime.fill(placement=Tile(0, 0), in_fifo=weights_fifo.prod(), source=w_buf,   tap=tap_w)
        runtime.drain(placement=Tile(0, 0), out_fifo=output_fifo.cons(), dest=out_buf,  wait=True, tap=tap_out)

    return Program(iron.get_current_device(), runtime).resolve_program(SequentialPlacer())


@iron.jit(is_placed=False)
def layer_relu(X, W, Y):
    """Single MLP layer: Y = relu(X @ W), 4 tiles in column 0."""
    return _make_layer(with_relu=True)


@iron.jit(is_placed=False)
def layer_linear(X, W, Y):
    """Single MLP layer: Y = X @ W (no activation), 4 tiles in column 0."""
    return _make_layer(with_relu=False)


def main():
    M, K, N = 64, 64, 64

    rng = np.random.default_rng(42)

    X  = iron.zeros(M * K, dtype=bfloat16, device="npu")
    W0 = iron.zeros(K * N, dtype=bfloat16, device="npu")
    W1 = iron.zeros(K * N, dtype=bfloat16, device="npu")
    W2 = iron.zeros(K * N, dtype=bfloat16, device="npu")
    W3 = iron.zeros(K * N, dtype=bfloat16, device="npu")
    H1 = iron.zeros(M * N, dtype=bfloat16, device="npu")
    H2 = iron.zeros(M * N, dtype=bfloat16, device="npu")
    H3 = iron.zeros(M * N, dtype=bfloat16, device="npu")
    Y  = iron.zeros(M * N, dtype=bfloat16, device="npu")

    X[:]  = rng.uniform(-1.0, 1.0, M * K).astype(bfloat16)
    W0[:] = rng.uniform(-1.0, 1.0, K * N).astype(bfloat16)
    W1[:] = rng.uniform(-1.0, 1.0, K * N).astype(bfloat16)
    W2[:] = rng.uniform(-1.0, 1.0, K * N).astype(bfloat16)
    W3[:] = rng.uniform(-1.0, 1.0, K * N).astype(bfloat16)

    # Run 4 layers sequentially; inter-layer activations stay on NPU/host memory.
    # Each dispatch uses column 0 only — no cross-column FIFO routing.
    layer_relu(X,  W0, H1)
    layer_relu(H1, W1, H2)
    layer_relu(H2, W2, H3)
    layer_linear(H3, W3, Y)

    print(Y)

    # Validation: reference in float32, round-trip through bfloat16 each layer.
    def bf16(x): return x.astype(bfloat16).astype(np.float32)
    def bf16_relu(x): return np.maximum(bf16(x), 0.0)

    x_np  = np.array(X, dtype=np.float32).reshape(M, K)
    w0_np = np.array(W0, dtype=np.float32).reshape(K, N)
    w1_np = np.array(W1, dtype=np.float32).reshape(K, N)
    w2_np = np.array(W2, dtype=np.float32).reshape(K, N)
    w3_np = np.array(W3, dtype=np.float32).reshape(K, N)

    h1 = bf16_relu(x_np @ w0_np)
    h2 = bf16_relu(h1   @ w1_np)
    h3 = bf16_relu(h2   @ w2_np)
    h4 = bf16     (h3   @ w3_np)
    expected = h4.astype(bfloat16).flatten()

    actual = np.array(Y, dtype=bfloat16)

    print("Element-by-element comparison (first 8):")
    for i in range(min(8, M * N)):
        print(f"  [{i}] expected={float(expected[i]):.4f}  actual={float(actual[i]):.4f}")

    mismatches = np.where(actual != expected)[0]
    if len(mismatches) == 0:
        print(f"Validation passed: all {M * N} elements match")
    else:
        print(f"Validation failed: {len(mismatches)} mismatches")
        for idx in mismatches[:5]:
            print(f"  [{idx}] expected={float(expected[idx]):.4f}, actual={float(actual[idx]):.4f}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")


if __name__ == "__main__":
    main()
