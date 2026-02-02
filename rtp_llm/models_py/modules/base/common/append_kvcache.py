from typing import Any

import flashinfer
import torch

from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops


def dbg_msg(msg):
    print(f"DBG: {msg}", flush=True)


def dbg_print_tsr(name: str, tsr: torch.Tensor, full_content: bool = True):
    dbg_msg(f"{name}.shape={tsr.shape}")
    dbg_msg(f"{name}.dtype={tsr.dtype}")
    dbg_msg(f"{name}.device={tsr.device}")
    if tsr.numel() > 0:
        dbg_msg(f"{name}.min()={tsr.min()}")
        dbg_msg(f"{name}.max()={tsr.max()}")
        dbg_msg(f"{name}.mean()={tsr.float().mean()}")
    if full_content:
        dbg_msg(f"{name}={tsr}")


class AppendKVCacheOpBase:
    def __init__(self, config: AttentionConfigs):
        self.token_per_block = config.tokens_per_block

    def create_params(
        self, attn_inputs: PyAttentionInputs
    ) -> rtp_llm_ops.FlashInferMlaAttnParams:
        params = rtp_llm_ops.fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )
        return params

    def prepare(
        self,
        params: rtp_llm_ops.FlashInferMlaAttnParams,
        attn_inputs: PyAttentionInputs,
    ):
        new_params = self.create_params(attn_inputs)
        params.batch_indice_d.copy_(new_params.batch_indice_d, non_blocking=True)
        params.positions_d.copy_(new_params.positions_d, non_blocking=True)
        params.page_indice_d.copy_(new_params.page_indice_d, non_blocking=True)
        if attn_inputs.is_prefill:
            params.prefill_page_indptr_d.copy_(
                new_params.prefill_page_indptr_d, non_blocking=True
            )
        else:
            params.decode_page_indptr_d.copy_(
                new_params.decode_page_indptr_d, non_blocking=True
            )
        params.paged_kv_last_page_len_d.copy_(
            new_params.paged_kv_last_page_len_d, non_blocking=True
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache_base: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        params: rtp_llm_ops.FlashInferMlaAttnParams,
    ):
        # dbg_msg(f"k.shape={k.shape}")
        # dbg_msg(f"v.shape={v.shape}")
        # dbg_msg(f"kv_cache_base.shape={kv_cache_base.shape}")
        # dbg_msg(f"k.dtype={k.dtype}")
        # dbg_msg(f"v.dtype={v.dtype}")
        # dbg_msg(f"kv_cache_base.dtype={kv_cache_base.dtype}")

        # dbg_print_tsr("params.batch_indice_d", params.batch_indice_d)
        # dbg_print_tsr("params.positions_d", params.positions_d)
        # dbg_print_tsr("params.page_indice_d", params.page_indice_d)
        # if attn_inputs.is_prefill:
        #     dbg_print_tsr("params.prefill_page_indptr_d", params.prefill_page_indptr_d)
        #     dbg_print_tsr("params.decode_page_indptr_d", params.decode_page_indptr_d)
        # dbg_print_tsr(
        #     "params.paged_kv_last_page_len_d", params.paged_kv_last_page_len_d
        # )

        flashinfer.append_paged_kv_cache(
            k,
            v,
            params.batch_indice_d,
            params.positions_d,
            kv_cache_base,
            params.page_indice_d,
            (
                params.prefill_page_indptr_d
                if attn_inputs.is_prefill
                else params.decode_page_indptr_d
            ),
            params.paged_kv_last_page_len_d,
            "HND",
        )


class AppendKVCacheOp(AppendKVCacheOpBase):
    def __init__(self, config: AttentionConfigs):
        super().__init__(config)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache,
        attn_inputs: PyAttentionInputs,
        params: rtp_llm_ops.FlashInferMlaAttnParams,
    ):
        super().forward(k, v, kv_cache.kv_cache_base, attn_inputs, params)
