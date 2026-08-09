# Attention Backend

## Supporting matrix for different attention backends

| **Backend**           | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** |         **Device Support**         |         **Server Args**         |         **Stage**         |
|-----------------------|-------------------|-------------------|---------|--------------------|------------------------------------|---------------------------------|---------------------------|
| **FLASHINFER_TRTLLM_GEN**        | ✅                | ✅                 | ❌      | ❌                 | NV SM100 ✅<br> AMD ❌ | --enable_flashinfer_trtllm_gen        | PREFILL ✅ <br>  DECODE✅  |
| **FLASHINFER_TRT_FMHA_V2**       | ❌                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_flashinfer_trt_fmha_v2       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_FLASHINFER_TRT_FMHA_V2** | ✅                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_paged_flashinfer_trt_fmha_v2 | PREFILL ✅ <br>  DECODE❌  |
| **OPEN_SOURCE**       | ❌                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_open_source_fmha       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_OPEN_SOURCE** | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_paged_open_source_fmha | PREFILL ✅ <br>  DECODE❌  |
| **CKFMHA**            | ❌                | ❌                 | ✅      | ✅                 | NV ❌<br> AMD ✅        | None                            | PREFILL ✅ <br>  DECODE❌  |
| **FLASHINFER_NATIVE** | ✅                | ✅                 | ✅      | ✅                 | NV ✅<br> AMD ✅        | --disable_flashinfer_native     | PREFILL ✅ <br>  DECODE✅  |
| **XQA**               | ✅                | ❌                 | ❌      | ❌                 | NV Hopper ✅<br> AMD ❌ | --enable_xqa                    | PREFILL ❌ <br>  DECODE✅  |
| **FlashMLA**          | ✅                | ✅                 | ✅      | ❌                 | NV Hopper ✅<br> AMD ❌ | None                            | PREFILL ❌ <br>  DECODE✅  |
| **MMHA**              | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |
| **AiterPA**           | ✅                | ❌                 | ❌      | ❌                 | NV ❌<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |

## ROCm KV-cache V layout and PA flag combinations

Prefill and decode share the paged KV pool, so both must use the same V layout: linear `head_dim × page`
or vectorized `page/width × head_dim × width`. The factory checks full-attention MHA with a rope KV
cache; MLA uses another factory, and models without a rope KV cache skip the check. Since PD roles
validate independently, both must use the same PA flags and `--kernel_seq_size_per_block`. Prefill-only
checks geometry; decode also checks the layout pair and non-ASM partition. Here `head_dim` is
`size_per_head`, `page` is `--kernel_seq_size_per_block` (falling back to `--seq_size_per_block` and
then 16 on ROCm), and `width` is 8 for BASE or 16 for FP8. Both `head_dim` and `page` must be multiples
of `width`; `page` must also divide 256 for non-ASM decode because the runtime may switch from its
512-token path to a 256-token partition as `max_seq_len` changes.

The table rows are the `--use_aiter_pa`/`--use_asm_pa`/`--use_triton_pa` flags, columns are
`size_per_head` × KV cache dtype, and each cell shows `[prefill] (decode)` plus the implementation
and written V layout (`V` vectorized, `L` linear) at `page=16`. `❌` means the layout check rejects the
pair; `⛔` means the factory has no implementation for that phase. `--use_aiter_pa 0` only removes
implementations, ASM decode supports `size_per_head=128` only, and `page=width` is an accepted
mismatch only for FP8; a BASE mismatch requires `--use_asm_pa 0`. The table assumes no prefix reuse;
with prefix reuse, `--use_asm_pa 1` selects the capturable `AiterPrefillImplPaged` path, which disabling
ASM removes.

| aiter/asm/triton | 128 BASE | 128 FP8 | 256 BASE | 256 FP8 | fix for ❌ ⛔ |
|---|---|---|---|---|---|
| `1/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `1/1/0` **(default)** | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ❌ `[Asm:V] (NonAsm:L)` | ✅ `[Asm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1`, or `--use_asm_pa 0` on BASE |
| `1/0/1` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | — |
| `1/0/0` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1` |
| `0/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `0/1/0` | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ⛔ `[Asm:V] (none)` | ⛔ `[Asm:V] (none)` | `--use_triton_pa 1` |
| `0/0/1` | ⛔ `[none] (TritonLin:L)` | ⛔ `[none] (Triton:V)` | ⛔ `[none] (TritonLin:L)` | ⛔ `[none] (Triton:V)` | `--use_asm_pa 1`, or `--use_aiter_pa 1` |
| `0/0/0` | ⛔ `[none] (none)` | ⛔ `[none] (none)` | ⛔ `[none] (none)` | ⛔ `[none] (none)` | `--use_aiter_pa 1` |
