; ==============================================================================
; AI-Assisted Domain-Specific Compiler: Optimized Tensor Kernel
; Target Architecture: x86_64-pc-linux-gnu (AVX2 / AVX-512)
; ==============================================================================

; ModuleID = 'TensorKernel'
source_filename = "tensor_kernel.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; LLVM FMA Vector Intrinsic Declaration
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)

define void @tensor_fma_vectorized(
    ptr noalias nocapture readonly %input_a,
    ptr noalias nocapture readonly %input_b,
    ptr noalias nocapture readonly %input_c,
    ptr noalias nocapture writeonly %output,
    i32 %vector_count
) {
entry:
  %cmp = icmp sgt i32 %vector_count, 0
  br i1 %cmp, label %vector.loop, label %exit

vector.loop:
  %idx = phi i32 [ 0, %entry ], [ %next_idx, %vector.loop ]
  
  ; Compute byte offset: idx * 8 floats * 4 bytes/float = idx * 32 bytes
  %byte_offset = shl i32 %idx, 5
  %ptr_a = getelementptr inbounds i8, ptr %input_a, i32 %byte_offset
  %ptr_b = getelementptr inbounds i8, ptr %input_b, i32 %byte_offset
  %ptr_c = getelementptr inbounds i8, ptr %input_c, i32 %byte_offset
  %ptr_out = getelementptr inbounds i8, ptr %output, i32 %byte_offset

  ; 256-bit Vector Loads (8 x 32-bit float)
  %vec_a = load <8 x float>, ptr %ptr_a, align 32
  %vec_b = load <8 x float>, ptr %ptr_b, align 32
  %vec_c = load <8 x float>, ptr %ptr_c, align 32

  ; Hardware FMA: (a * b) + c
  %vec_res = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %vec_a, <8 x float> %vec_b, <8 x float> %vec_c)

  ; 256-bit Vector Store
  store <8 x float> %vec_res, ptr %ptr_out, align 32

  %next_idx = add nuw nsw i32 %idx, 1
  %done = icmp eq i32 %next_idx, %vector_count
  br i1 %done, label %exit, label %vector.loop

exit:
  ret void
}
