"""ROCm paged-attention V-layout contract tests."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type

# aiter imports off-device too, so availability cannot stand in for the device: only
# ROCm registers the impls selected between here. The imports stay unguarded so a ROCm
# image with a broken aiter goes red rather than skipping green.
ROCM_IMPLS_REGISTERED = get_device_type() == DeviceType.ROCm
if ROCM_IMPLS_REGISTERED:
    from rtp_llm.models_py.modules.factory.attention import attn_factory
    from rtp_llm.models_py.modules.factory.attention.rocm_impl import aiter as rocm
    from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, RopeStyle
    from rtp_llm.ops.compute_ops import PyAttentionInputs

    BASE = KvCacheDataType.BASE
    FP8 = KvCacheDataType.FP8


def _config(cache_dtype, head, page=32):
    config = AttentionConfigs()
    config.dtype = torch.float16
    config.kv_cache_dtype = cache_dtype
    config.head_num = 8
    config.kv_head_num = 2
    config.size_per_head = head
    config.kernel_tokens_per_block = page
    config.need_rope_kv_cache = True
    return config


def _flags(values):
    config = FMHAConfig()
    config.use_aiter_pa, config.use_asm_pa, config.use_triton_pa = map(bool, values)
    return config


def _inputs(is_prefill=True, prefix_len=0):
    inputs = PyAttentionInputs()
    inputs.is_prefill = is_prefill
    inputs.prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32)
    inputs.kv_cache_kernel_block_id = torch.tensor([[0]], dtype=torch.int32)
    return inputs


def _select(
    dtype,
    head,
    flags,
    is_prefill=True,
    page=32,
    prefix_len=0,
    is_cuda_graph=False,
):
    return attn_factory.get_fmha_impl(
        _config(dtype, head, page),
        None,
        _inputs(is_prefill, prefix_len),
        fmha_config=_flags(flags),
        is_cuda_graph=is_cuda_graph,
    )


@unittest.skipUnless(ROCM_IMPLS_REGISTERED, "not a ROCm device")
class TestVLayoutContract(unittest.TestCase):
    """Depends on the ROCm branch of the impl registration order."""

    def test_configurations(self):
        for dtype, head, page, flags, is_prefill, rejection in (
            # dtype, head, page, aiter/asm/triton, role, rejection (None if valid)
            (BASE, 128, 32, (1, 1, 0), True, None),
            # Geometry. The page divides the head here, so misprinting the width check
            # as head % page goes red.
            (BASE, 100, 4, (1, 1, 0), True, "head_dim=100 is invalid for 8-element"),
            (FP8, 120, 8, (1, 0, 0), True, "head_dim=120 is invalid for 16-element"),
            (BASE, 256, 0, (1, 1, 0), True, "page=0 is invalid for 8-element"),
            (BASE, 128, 12, (1, 1, 0), True, "page=12 is invalid for 8-element"),
            (BASE, 256, 12, (0, 1, 0), True, "page=12 is invalid for 8-element"),
            # BASE can align by dropping ASM; FP8 always writes vectorized, so folding
            # the page onto one vector is the only remedy left for it.
            (BASE, 256, 32, (1, 1, 0), False, r"--use_asm_pa 0"),
            (FP8, 128, 32, (1, 0, 0), False, r"--kernel_seq_size_per_block=16"),
            # page == width folds the two formulas together, but only FP8's fold has
            # a reader; BASE's page=8 is rejected for that, not for its geometry.
            (BASE, 256, 8, (1, 1, 0), False, r"--use_asm_pa 0"),
            (FP8, 256, 16, (1, 1, 0), False, None),
            # Triton reads whichever layout prefill writes, so both dtypes pair up.
            (BASE, 256, 32, (1, 0, 1), False, None),
            (FP8, 128, 1024, (1, 0, 1), False, None),
            # The width check outranks the partition one even where both phases are
            # linear, because every prefill reader still vectorizes V.
            (BASE, 256, 4, (1, 0, 0), False, "page=4 is invalid for 8-element"),
            (BASE, 256, 512, (1, 0, 0), False, "must divide the 256-token partition"),
            # A phase the flags leave empty is the factory's error, not a layout one.
            (BASE, 256, 32, (0, 1, 0), True, None),
        ):
            with self.subTest(
                dtype=dtype, head=head, page=page, flags=flags, is_prefill=is_prefill
            ):
                args = (
                    _config(dtype, head, page),
                    _inputs(is_prefill=is_prefill),
                    _flags(flags),
                )
                if rejection is None:
                    rocm.validate_v_layout(*args)
                else:
                    with self.assertRaisesRegex(ValueError, rejection):
                        rocm.validate_v_layout(*args)

    def test_shipped_defaults_pick_a_valid_pair(self):
        # Default-constructed and absent configs rather than explicit triples, so a
        # flipped default goes red here. No config counts every flag as on.
        for config in (FMHAConfig(), None):
            rocm.validate_v_layout(_config(BASE, 128), _inputs(), config)
        rocm.validate_v_layout(
            _config(BASE, 256), _inputs(is_prefill=True), FMHAConfig()
        )
        with self.assertRaisesRegex(ValueError, "align the PA flags"):
            rocm.validate_v_layout(
                _config(BASE, 256), _inputs(is_prefill=False), FMHAConfig()
            )

    def test_triton_readers_declare_matching_writers(self):
        self.assertEqual(
            (rocm.AiterDecodeImplTriton.LINEAR_V, rocm.AiterDecodeImplTriton.WRITER),
            (False, rocm.FusedRopeKVCacheDecodeOpAsm),
        )
        self.assertEqual(
            (
                rocm.AiterDecodeImplTritonLinear.LINEAR_V,
                rocm.AiterDecodeImplTritonLinear.WRITER,
            ),
            (True, rocm.FusedRopeKVCacheDecodeOpNonAsm),
        )

    def test_no_kv_cache_skips_the_check(self):
        # Keeps the embedding deployments startable: their head dims have no ASM
        # decode reader, so the shipped defaults would otherwise be rejected.
        no_rope = _config(BASE, 256)
        no_rope.need_rope_kv_cache = False
        empty = _inputs(is_prefill=False)
        empty.kv_cache_kernel_block_id = torch.empty((0, 0), dtype=torch.int32)
        for configs, inputs in (
            (no_rope, _inputs(is_prefill=False)),
            (_config(BASE, 256), empty),
        ):
            rocm.validate_v_layout(configs, inputs, _flags((1, 1, 0)))

    def test_unsupported_mrope_keeps_its_own_error(self):
        # support() empties both phases here, so the layout check has no pair to judge
        # and its remedy cannot help; the factory's MRoPE error must win.
        configs = _config(BASE, 256)
        configs.rope_config.style = RopeStyle.Mrope
        configs.rope_config.mrope_interleaved = False
        rocm.validate_v_layout(configs, _inputs(), _flags((1, 1, 0)))
        with self.assertRaisesRegex(ValueError, "non-interleaved MRoPE"):
            attn_factory.get_fmha_impl(
                configs, None, _inputs(), fmha_config=_flags((1, 1, 0))
            )

    def test_factory_priority_matches_the_contract(self):
        for dtype, head, page, flags, prefill, expected in (
            (BASE, 128, 32, (1, 1, 0), True, rocm.AiterPrefillImplAsm),
            (BASE, 128, 32, (1, 1, 0), False, rocm.AiterDecodeImplAsm),
            # Triton outranks ASM decode, the only rule asm=1 with triton=1 reaches:
            # the ASM writer keeps prefill and its reader becomes the vectorized one.
            (BASE, 128, 32, (1, 1, 1), True, rocm.AiterPrefillImplAsm),
            (BASE, 128, 32, (1, 1, 1), False, rocm.AiterDecodeImplTriton),
            (BASE, 256, 32, (1, 0, 0), True, rocm.AiterPrefillImplNonAsm),
            (BASE, 256, 32, (1, 0, 0), False, rocm.AiterDecodeImplNonAsm),
            (FP8, 256, 32, (1, 0, 1), True, rocm.AiterPrefillImplNonAsm),
            (FP8, 256, 32, (1, 0, 1), False, rocm.AiterDecodeImplTriton),
            # BASE non-ASM prefill writes linear V, so Triton reads it linearly.
            (BASE, 128, 32, (1, 0, 1), True, rocm.AiterPrefillImplNonAsm),
            (BASE, 128, 32, (1, 0, 1), False, rocm.AiterDecodeImplTritonLinear),
            # asm decode asked for, head_dim != 128, so it degrades to non-ASM. Only
            # FP8 at page=width lets that pair through; on BASE it is a layout error.
            (FP8, 256, 16, (1, 1, 0), False, rocm.AiterDecodeImplNonAsm),
        ):
            with self.subTest(flags=flags, prefill=prefill, head=head):
                # No warning means no candidate ahead of `expected` was constructed
                # and swallowed; that silent degrade is what this PR removes.
                with self.assertNoLogs(level="WARNING"), patch.object(
                    expected, "__init__", return_value=None
                ):
                    selected = _select(dtype, head, flags, prefill, page)
                self.assertIs(type(selected), expected)

    def test_prefix_reuse_swaps_the_impl_but_not_the_writer(self):
        # Paged is priority 1 but gated on a prefix, so every other selection case
        # here reaches Asm instead; both share the ASM writer, so one validation of
        # this config has to accept either impl.
        for prefix_len, expected in (
            (17, rocm.AiterPrefillImplPaged),
            (0, rocm.AiterPrefillImplAsm),
        ):
            with self.subTest(prefix_len=prefix_len), self.assertNoLogs(
                level="WARNING"
            ), patch.object(expected, "__init__", return_value=None):
                selected = _select(BASE, 128, (1, 1, 0), prefix_len=prefix_len)
            self.assertIs(type(selected), expected)

    def test_paged_degrade_lands_on_the_same_writer(self):
        # The one prefill degrade left open, safe only because Asm writes the same
        # vectorized V. It has to be audible, not silent.
        with self.assertLogs(level="WARNING") as logs, patch.object(
            rocm.AiterPrefillImplPaged, "__init__", side_effect=RuntimeError("boom")
        ), patch.object(rocm.AiterPrefillImplAsm, "__init__", return_value=None):
            selected = _select(BASE, 128, (1, 1, 0), prefix_len=17)
        self.assertIs(type(selected), rocm.AiterPrefillImplAsm)
        self.assertIn("AiterPrefillImplPaged", "".join(logs.output))

    def test_no_enabled_impl_is_not_reported_as_a_layout_error(self):
        # Enablement is the factory's error; the layout check runs first and must
        # stay out of it, or the remedy it prints cannot help.
        with self.assertRaisesRegex(Exception, "can not find mha type"):
            _select(BASE, 256, (0, 0, 0))

    def test_factory_rejects_mismatch_before_construction(self):
        with patch.object(rocm.AiterDecodeImplAsm, "__init__") as init:
            with self.assertRaisesRegex(ValueError, "align the PA flags"):
                _select(BASE, 256, (1, 1, 0), is_prefill=False)
        init.assert_not_called()

    def test_constructor_error_does_not_change_layout(self):
        # Each phase picks its first candidate and the next one reads the other
        # layout, so a construction failure must propagate, not degrade onto it.
        for dtype, head, prefill, failing, unreachable in (
            (BASE, 128, True, rocm.AiterPrefillImplAsm, rocm.AiterPrefillImplNonAsm),
            (
                FP8,
                256,
                False,
                rocm.AiterDecodeImplTriton,
                rocm.AiterDecodeImplNonAsm,
            ),
        ):
            # asm carries prefill, triton carries decode; both are the head of a list.
            flags = (1, 1, 0) if prefill else (1, 0, 1)
            with self.subTest(failing=failing.__name__), patch.object(
                failing, "__init__", side_effect=RuntimeError("boom")
            ), patch.object(unreachable, "__init__") as fallback:
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    _select(dtype, head, flags, prefill)
            fallback.assert_not_called()

    def test_cuda_graph_does_not_fall_through_to_another_writer(self):
        with patch.object(
            rocm.AiterPrefillImplAsm, "__init__", return_value=None
        ), patch.object(rocm.AiterPrefillImplNonAsm, "__init__") as fallback:
            with self.assertRaisesRegex(
                ValueError, "cannot be captured in a CUDA graph"
            ):
                _select(BASE, 128, (1, 1, 0), is_cuda_graph=True)
        fallback.assert_not_called()

    def test_aiter_consumer_unpacks_a_packed_layer_cache(self):
        # The op's own geometry drives the unpacking: a 5D cache passes through, a 2D
        # buffer keeps only its full-attention prefix, and a short one is rejected
        # instead of being read past its end.
        config = _config(BASE, 128, page=16)
        op = rocm.AiterDecodeAttnOpBase(config)
        hk, page = config.kv_head_num, config.kernel_tokens_per_block
        elems = 2 * hk * page * 128

        paged = torch.zeros(2, 2, hk, page, 128)
        self.assertIs(op.reshape_kv_cache(paged), paged)

        packed = torch.arange(2 * (elems + 32), dtype=torch.float32).view(2, elems + 32)
        unpacked = op.reshape_kv_cache(packed)
        self.assertEqual(unpacked.shape, (2, 2, hk, page, 128))
        torch.testing.assert_close(
            unpacked[1], packed[1, :elems].reshape(2, hk, page, 128)
        )

        exact = op.reshape_kv_cache(packed[:, :elems])
        self.assertEqual(exact.shape, (2, 2, hk, page, 128))

        with self.assertRaisesRegex(ValueError, "insufficient stride"):
            op.reshape_kv_cache(packed[:, : elems - 1])


if __name__ == "__main__":
    unittest.main()
