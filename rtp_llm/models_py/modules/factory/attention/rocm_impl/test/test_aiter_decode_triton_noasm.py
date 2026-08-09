"""Layout-aware parity test for AITER Triton vs Non-ASM decode.

This test isolates decode FMHA kernels at the paged-attention boundary and
feeds each kernel the physical KV layout it expects, while preserving the same
semantic K/V values.
"""

import math
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    AiterDecodeAttnOpNonAsm,
    AiterDecodeAttnOpTriton,
    AiterDecodeImplNonAsm,
    AiterDecodeImplTriton,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta

HEAD_NUM = 24
KV_HEAD_NUM = 4
HEAD_DIM = 256
BLOCK_SIZE = 16
CONTEXT_LENGTH = 6359
NUM_BLOCKS = math.ceil(CONTEXT_LENGTH / BLOCK_SIZE)


def make_config() -> AttentionConfigs:
    config = AttentionConfigs()
    config.head_num = HEAD_NUM
    config.kv_head_num = KV_HEAD_NUM
    config.size_per_head = HEAD_DIM
    config.tokens_per_block = BLOCK_SIZE
    config.kernel_tokens_per_block = BLOCK_SIZE
    config.max_seq_len = 40960
    config.kv_cache_dtype = KvCacheDataType.BASE
    config.dtype = torch.bfloat16
    config.need_rope_kv_cache = False
    return config


def make_inputs(device: torch.device) -> PyAttentionInputs:
    inputs = PyAttentionInputs()
    inputs.is_prefill = False
    inputs.is_cuda_graph = False
    # Decode sees full context after current token is inserted into KV cache.
    inputs.sequence_lengths = torch.tensor([CONTEXT_LENGTH - 1], dtype=torch.int32)
    inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device).view(1, -1)
    inputs.kv_cache_kernel_block_id_device = block_table
    inputs.kv_cache_block_id_device = block_table
    inputs.dtype = get_typemeta(torch.empty((), dtype=torch.bfloat16))
    return inputs


def run_decode(impl_class, op, inputs, query, kv_cache):
    # Bypass RoPE/cache insertion and test decode FMHA kernels directly.
    impl = impl_class.__new__(impl_class)
    impl.need_rope_kv_cache = False
    impl.fmha_impl = op
    impl.attn_inputs = inputs
    impl.fmha_params = op.prepare(inputs)
    impl.write_cache_store_impl = None

    cache = LayerKVCache()
    cache.kv_cache_base = kv_cache.clone()
    cache.kv_scale_base = torch.empty(0, device=query.device)
    return impl.forward(query.clone(), cache, layer_idx=3)


def pack_cache(key_phys: torch.Tensor, value_phys: torch.Tensor) -> torch.Tensor:
    return torch.stack([key_phys, value_phys], dim=1)


def physical_key_for_decode(semantic_key: torch.Tensor) -> torch.Tensor:
    # Both decode paths reinterpret K as vectorized [hd//x, ps, x] via a view.
    # Keeping K in canonical [ps, hd] memory order is enough.
    return semantic_key.contiguous()


def physical_value_for_nonasm(semantic_value: torch.Tensor) -> torch.Tensor:
    # Non-ASM paged_attention_rocm reads BASE V as linear [hd, ps].
    return (
        semantic_value.permute(0, 1, 3, 2)
        .contiguous()
        .view(NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM)
    )


def physical_value_for_triton(semantic_value: torch.Tensor) -> torch.Tensor:
    # Triton pa_decode_gluon (VALUE_TRANSPOSED=True) reads V as [ps//x, hd, x].
    x_vec = 16 // semantic_value.element_size()
    assert BLOCK_SIZE % x_vec == 0
    return (
        semantic_value.view(
            NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE // x_vec, x_vec, HEAD_DIM
        )
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view(NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM)
    )


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    flat_reference = reference.float().flatten()
    diff = actual.float().flatten() - flat_reference
    return (diff.norm() / flat_reference.norm()).item()


def make_query(generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        (1, HEAD_NUM, HEAD_DIM), generator=generator, dtype=torch.bfloat16
    ).cuda()


def make_semantic(generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        (NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
    ).cuda()


class AiterDecodeLayoutParityTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.skipTest("requires a ROCm GPU")
        self.config = make_config()

    def _triton(self, query, kv_cache, *, linear_v):
        return run_decode(
            AiterDecodeImplTriton,
            AiterDecodeAttnOpTriton(self.config, linear_v=linear_v),
            make_inputs(query.device),
            query,
            kv_cache,
        )

    def _nonasm(self, query, kv_cache):
        return run_decode(
            AiterDecodeImplNonAsm,
            AiterDecodeAttnOpNonAsm(self.config),
            make_inputs(query.device),
            query,
            kv_cache,
        )

    def test_layout_mismatch_reproduces_large_error(self):
        generator = torch.Generator().manual_seed(0)
        query = make_query(generator)
        # A single shared physical V layout is not comparable across kernels.
        shared_cache = torch.randn(
            (NUM_BLOCKS, 2, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
        ).cuda()

        error = relative_l2(
            self._triton(query, shared_cache, linear_v=False),
            self._nonasm(query, shared_cache),
        )
        self.assertGreater(
            error, 0.5, f"unexpectedly small mismatch: relative_l2={error:.6f}"
        )

    def test_triton_matches_nonasm_with_layout_aware_cache(self):
        generator = torch.Generator().manual_seed(0)
        query = make_query(generator)
        semantic_key = make_semantic(generator)
        semantic_value = make_semantic(generator)

        key_phys = physical_key_for_decode(semantic_key)
        error = relative_l2(
            self._triton(
                query,
                pack_cache(key_phys, physical_value_for_triton(semantic_value)),
                linear_v=False,
            ),
            self._nonasm(
                query,
                pack_cache(key_phys, physical_value_for_nonasm(semantic_value)),
            ),
        )
        self.assertLess(
            error,
            0.01,
            f"layout-aware comparison still mismatches: relative_l2={error:.6f}",
        )

    def test_triton_linear_matches_nonasm_on_one_physical_cache(self):
        generator = torch.Generator().manual_seed(1)
        query = make_query(generator)
        semantic_key = make_semantic(generator)
        semantic_value = make_semantic(generator)
        # Both readers address linear V, so one physical cache serves both.
        linear_cache = pack_cache(
            physical_key_for_decode(semantic_key),
            physical_value_for_nonasm(semantic_value),
        )

        error = relative_l2(
            self._triton(query, linear_cache, linear_v=True),
            self._nonasm(query, linear_cache),
        )
        self.assertLess(
            error, 0.01, f"linear Triton and NonAsm mismatch: relative_l2={error:.6f}"
        )


if __name__ == "__main__":
    unittest.main()
