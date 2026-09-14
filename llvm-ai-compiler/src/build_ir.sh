#!/usr/bin/env bash
# Compile SIMD vectorized LLVM IR into native machine assembly and object file
set -e

echo "=== LLVM Vectorized AI Tensor Kernel Compiler ==="

IR_SRC="src/tensor_kernel.ll"

if command -v llc &> /dev/null; then
    echo "[1/2] Compiling LLVM IR to native x86-64 assembly with AVX2/FMA intrinsics..."
    llc -O3 -mattr=+avx2,+fma "$IR_SRC" -o build/tensor_kernel.s

    echo "[2/2] Generating machine object file..."
    llc -O3 -filetype=obj "$IR_SRC" -o build/tensor_kernel.o
    echo "[SUCCESS] Generated native vectorized binary artifacts."
elif command -v clang &> /dev/null; then
    clang -O3 -c "$IR_SRC" -o build/tensor_kernel.o
    echo "[SUCCESS] Clang compiled LLVM IR to object code."
else
    echo "[INFO] LLVM tools (llc/clang) not in PATH. Running LLVM IR static validator..."
    node runner/run.js
fi
