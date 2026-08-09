# Tests for ROCm prefill/decode attention helpers.
#
# - test_attn_utils: geometry helpers, plain torch math; on the runner only because
#   the imports reach compute_ops, not for any GPU compute.
# - test_aiter_prefill_op: numerical regression for AiterPrefillAttnOp; runs
#   on the ROCm CI runner because it calls aiter.flash_attn_varlen_func.
# - test_v_layout_contract: factory selection, V-layout validation and KV-cache
#   unpacking; patches constructors, so it needs a ROCm device but does no GPU compute.
# - test_aiter_decode_triton_noasm: numerical parity for both Triton V layouts
#   against NonAsm decode on a ROCm GPU.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

model_test_deps = [
    "//rtp_llm/models_py:models",
    "//rtp_llm:config",
    "//rtp_llm:testlib",
]

py_test(
    name = "test_attn_utils",
    srcs = ["test_attn_utils.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_v_layout_contract",
    srcs = ["test_v_layout_contract.py"],
    deps = model_test_deps,
    # Import-bound, not kernel-bound: eternal would hold a runner for an hour.
    timeout = "long",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_aiter_prefill_op",
    srcs = ["test_aiter_prefill_op.py"],
    deps = model_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_fused_qkv_transpose_v3",
    srcs = ["test_fused_qkv_transpose_v3.py"],
    deps = model_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_aiter_decode_triton_noasm",
    srcs = ["test_aiter_decode_triton_noasm.py"],
    deps = model_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
