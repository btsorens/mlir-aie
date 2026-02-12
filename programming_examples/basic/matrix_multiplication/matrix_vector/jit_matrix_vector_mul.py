# jit_matrix_vector_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron import ExternalFunction, jit
from aie.iron.controlflow import range_
import aie.iron as iron


@iron.jit(is_placed=False)
def matrix_vector_mul_jit(inputA, inputB, outputC):
    # Matrix dimensions
    M = 288
    K = 288
    m = 32  # tile rows (matches DIM_M default in mv.cc)
    k = 32  # tile cols (matches DIM_K default in mv.cc)
    M_div_m = M // m
    K_div_k = K // k

    # Define tile types
    inA_ty = np.ndarray[(m, k), np.dtype[np.int16]]
    inB_ty = np.ndarray[(k,), np.dtype[np.int16]]
    outC_ty = np.ndarray[(m,), np.dtype[np.int32]]

    # Sequence buffer types (flat)
    A_seq_ty = np.ndarray[(M * K,), np.dtype[np.int16]]
    B_seq_ty = np.ndarray[(M_div_m * K,), np.dtype[np.int16]]  # B repeated per row block
    C_seq_ty = np.ndarray[(M,), np.dtype[np.int32]]

    # Generate handles to externally defined kernel functions
    matvec = ExternalFunction(
        name="matvec_vectorized_i16_i32",
        source_file="/scratch/IRONSmithTesting/mlir-aie/aie_kernels/aie2/mv.cc",
        arg_types=[inA_ty, inB_ty, outC_ty],
        include_dirs=[
            "/scratch/IRONSmithTesting/mlir-aie/aie_kernels",
            "/scratch/IRONSmithTesting/mlir-aie/aie_kernels/aie2",
            "/scratch/IRONSmithTesting/mlir-aie/aie_runtime_lib/AIE2",
        ],
    )
    zero = ExternalFunction(
        name="zero_vectorized_i32",
        source_file="/scratch/IRONSmithTesting/mlir-aie/aie_kernels/aie2/mv.cc",
        arg_types=[outC_ty],
        include_dirs=[
            "/scratch/IRONSmithTesting/mlir-aie/aie_kernels",
            "/scratch/IRONSmithTesting/mlir-aie/aie_kernels/aie2",
            "/scratch/IRONSmithTesting/mlir-aie/aie_runtime_lib/AIE2",
        ],
    )

    # Dataflow with ObjectFifos
    A_fifo = ObjectFifo(inA_ty, name="inA")
    B_fifo = ObjectFifo(inB_ty, name="inB")
    C_fifo = ObjectFifo(outC_ty, name="outC")
    # Fifo to pass zeroed output tiles from zero worker to matvec worker
    zero_to_matvec_fifo = ObjectFifo(outC_ty, name="zeroC")

    # Zero worker: zeros output tiles on a separate tile
    def zero_fn(of_out, zero):
        for _ in range_(M_div_m):
            elem_out = of_out.acquire(1)
            zero(elem_out)
            of_out.release(1)

    # Matvec worker: accumulates matrix-vector products into zeroed output
    def matvec_fn(of_a, of_b, of_zeroed, of_c, matvec):
        for _ in range_(M_div_m):
            elem_zeroed = of_zeroed.acquire(1)
            elem_out = of_c.acquire(1)
            # Copy zeroed tile to output
            for i in range_(m):
                elem_out[i] = elem_zeroed[i]
            of_zeroed.release(1)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matvec(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    # Create workers on separate tiles
    zero_worker = Worker(
        zero_fn,
        fn_args=[zero_to_matvec_fifo.prod(), zero],
    )
    matvec_worker = Worker(
        matvec_fn,
        fn_args=[A_fifo.cons(), B_fifo.cons(), zero_to_matvec_fifo.cons(), C_fifo.prod(), matvec],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_seq_ty, B_seq_ty, C_seq_ty) as (a_in, b_in, c_out):
        rt.start(zero_worker, matvec_worker)
        rt.fill(A_fifo.prod(), a_in)
        rt.fill(B_fifo.prod(), b_in)
        rt.drain(C_fifo.cons(), c_out, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    M = 288
    K = 288
    m = 32
    k = 32
    M_div_m = M // m
    K_div_k = K // k

    # Create input data
    np.random.seed(42)
    A_data = np.random.randint(-16, 16, size=(M, K)).astype(np.int16)
    b_data = np.random.randint(-16, 16, size=(K,)).astype(np.int16)

    # Tile A into (m, k) blocks in row-block-major order, then apply
    # 32-bit word transposition within each block as required by
    # matvec_vectorized (column-major at 4-byte / 2-element granularity)
    A_blocked = A_data.reshape(M_div_m, m, K_div_k, k).transpose(0, 2, 1, 3)
    A_blocked = A_blocked.reshape(M_div_m, K_div_k, m, k // 2, 2)
    A_blocked = A_blocked.transpose(0, 1, 3, 2, 4)
    A_tiled = A_blocked.reshape(-1).copy()

    # B needs to be repeated M_div_m times since each row block re-reads the full vector
    B_repeated = np.tile(b_data, M_div_m)

    inputA = iron.zeros(M * K, dtype=np.int16, device="npu")
    inputA.data[:] = A_tiled
    inputA._sync_to_device()
    inputB = iron.zeros(M_div_m * K, dtype=np.int16, device="npu")
    inputB.data[:] = B_repeated
    inputB._sync_to_device()
    outputC = iron.zeros(M, dtype=np.int32, device="npu")

    # Run the JIT function
    matrix_vector_mul_jit(inputA, inputB, outputC)

    print(outputC)

    # Validation
    expected = A_data.astype(np.int32) @ b_data.astype(np.int32)
    actual = np.asarray(outputC, dtype=np.int32)

    # Print some sample values
    print("\nSample element-by-element comparison (first 10):")
    for i in range(min(10, M)):
        print(f"  Row {i}: Expected = {expected[i]} : Received = {actual[i]}")

    mismatches = np.where(actual != expected)[0]

    print(f"\nValidation results:")
    print(f"  Elements tested: {M}")

    if len(mismatches) == 0:
        print(f"  Status: PASSED - All {M} values match exactly")
    else:
        print(f"  Status: FAILED - {len(mismatches)} mismatches found")
        print(f"\nFirst few mismatches:")
        for i, idx in enumerate(mismatches[:5]):
            print(f"  Row {idx}: actual={actual[idx]}, expected={expected[idx]}")


if __name__ == "__main__":
    main()
