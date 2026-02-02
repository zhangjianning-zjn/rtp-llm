
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/AttentionOpTest.hpp"
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace rtp_llm;

// TEST_F(AttentionOpTest, SelfAttentionOpTest) {
//     // batch size > 8 may exceed cache manager buffer size.
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = true;
//     device_init_params.fmha_config.enable_xqa                   = false;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch  = {2, 4, 8};
//     std::vector<size_t> seq    = {1};
//     std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len : kv_seq) {
//                 size_t num_heads           = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim            = 64;
//                 selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, MultiBlockSelfAttentionOpTest) {
//     // batch size > 8 may exceed cache manager buffer size.
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = true;
//     device_init_params.fmha_config.enable_xqa                   = false;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = true;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch  = {2, 4, 8};
//     std::vector<size_t> seq    = {1};
//     std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len : kv_seq) {
//                 size_t num_heads           = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim            = 64;
//                 selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, ContextAttentionOpTest) {
//     auto device_init_params                                = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha         = false;
//     device_init_params.fmha_config.enable_trtv1_fmha       = false;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer     = true;
//     device_init_params.fmha_config.enable_xqa              = false;
//     device_                                                = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads           = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim            = 64;
//             contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, ContextAttentionOpMultiGroupTest) {
//     auto device_init_params                                = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha         = false;
//     device_init_params.fmha_config.enable_trtv1_fmha       = false;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer     = true;
//     device_init_params.fmha_config.enable_xqa              = false;
//     device_                                                = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads           = 64;
//             size_t num_key_value_heads = 4;
//             size_t head_dim            = 64;
//             contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, OpenSourceFMHAContextAttentionOpTest) {
//     auto device_init_params                                = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha         = false;
//     device_init_params.fmha_config.enable_trtv1_fmha       = false;
//     device_init_params.fmha_config.enable_open_source_fmha = true;
//     device_init_params.fmha_config.disable_flash_infer     = true;
//     device_init_params.fmha_config.enable_xqa              = false;
//     device_                                                = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_open_source_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads           = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim            = 64;
//             contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, TrtV2ContextAttentionOpTest) {
//     auto device_init_params                                = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha         = true;
//     device_init_params.fmha_config.enable_trtv1_fmha       = false;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer     = true;
//     device_init_params.fmha_config.enable_xqa              = false;
//     device_                                                = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads           = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim            = 64;
//             contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, TrtV1ContextAttentionOpTest) {
//     auto device_init_params                                = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha         = false;
//     device_init_params.fmha_config.enable_trtv1_fmha       = true;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer     = true;
//     device_init_params.fmha_config.enable_xqa              = false;
//     device_                                                = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads           = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim            = 64;
//             contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, LongSeqMultiBlockSelfAttentionOpTest) {
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = true;
//     device_init_params.fmha_config.enable_xqa                   = false;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = true;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch  = {4};
//     std::vector<size_t> seq    = {1};
//     std::vector<size_t> kv_seq = {16000};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len : kv_seq) {
//                 size_t num_heads           = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim            = 64;
//                 selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, LongSeqSelfAttentionOpTest) {
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = true;
//     device_init_params.fmha_config.enable_xqa                   = false;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch  = {4};
//     std::vector<size_t> seq    = {1};
//     std::vector<size_t> kv_seq = {16000};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len : kv_seq) {
//                 size_t num_heads           = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim            = 64;
//                 selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

#ifdef USING_CUDA12
// TEST_F(AttentionOpTest, XqaAttentionOpTest) {
//     auto device_init_params                                     = DeviceInitParams();
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = true;
//     device_init_params.fmha_config.enable_xqa                   = true;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
//     ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     size_t              batch_size      = 3;
//     std::vector<size_t> head_dim        = {64, 128, 256};
//     size_t              seq_q           = 1;
//     size_t              seq_kv          = 129;
//     size_t              head_q          = 64;
//     std::vector<size_t> head_kv         = {4, 8, 16, 32, 64};
//     std::vector<size_t> page_size       = {16, 32, 64, 128};
//     std::vector<bool>   is_kv_cache_fp8 = {true, false};
//     for (auto hd : head_dim) {
//         for (auto hkv : head_kv) {
//             for (auto ps : page_size) {
//                 for (auto is_kv_fp8 : is_kv_cache_fp8) {
//                     xqaAttentionOpTest(batch_size, seq_q, seq_kv, head_q, hkv, hd, ps, is_kv_fp8);
//                 }
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, FlashinferContextAttentionOpTest) {
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha              = false;
//     device_init_params.fmha_config.enable_trtv1_fmha            = false;
//     device_init_params.fmha_config.enable_open_source_fmha      = false;
//     device_init_params.fmha_config.disable_flash_infer          = false;
//     device_init_params.fmha_config.enable_xqa                   = false;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_                                                     = new CudaDevice(device_init_params);
//     device_->init();
//     std::vector<size_t> batch  = {3};
//     std::vector<size_t> seq    = {1};
//     std::vector<size_t> kv_seq = {2049};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len : kv_seq) {
//                 size_t num_heads           = 64;
//                 size_t num_key_value_heads = 4;
//                 size_t head_dim            = 128;
//                 flashinferPrefillOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, XqaContextAttentionOpTest) {
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha = true;
//     device_init_params.fmha_config.enable_trtv1_fmha = false;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer = false;
//     device_init_params.fmha_config.enable_xqa = true;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_ = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
//     ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch = {3};
//     std::vector<size_t> seq   = {1};
//     std::vector<size_t> kv_seq = {2049};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len: kv_seq) {
//                 size_t num_heads = 64;
//                 size_t num_key_value_heads = 4;
//                 size_t head_dim = 128;
//                 xqaPrefillOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }

TEST_F(AttentionOpTest, XqaAttentionOpTest) {
    auto device_init_params                                     = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = true;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);

#define CHK(msg, error)                                                                                                \
    do {                                                                                                               \
        RTP_LLM_CHECK_WITH_INFO(error == cudaSuccess,                                                                  \
                                msg " failed with error %s(%d): %s",                                                   \
                                cudaGetErrorName(error),                                                               \
                                error,                                                                                 \
                                cudaGetErrorString(error));                                                            \
    } while (0)

    {
        torch::serialize::InputArchive archive;
        archive.load_from("/home/zhangjianning.zjn/RTP-LLM/github-opensource/xqa_input.pt");
#define READ_NAMED_DUMP(type, name, val)                                                                               \
    torch::IValue val##_ival__;                                                                                        \
    archive.read(name, val##_ival__);                                                                                  \
    auto val = val##_ival__.to##type()
#define READ_DUMP(type, name) READ_NAMED_DUMP(type, #name, name)

        READ_DUMP(Tensor, input);
        READ_DUMP(Bool, is_input_bf16);
        READ_DUMP(IntVector, output_shape);
        READ_DUMP(Tensor, output_options);
        READ_DUMP(Int, head_num);
        READ_DUMP(Int, head_num_kv);
        READ_DUMP(Int, head_dim);
        READ_DUMP(Int, batch_size);
        READ_DUMP(Int, max_blocks_per_seq);
        READ_DUMP(Int, max_seq_len);
        READ_DUMP(Int, page_size);
        READ_DUMP(IntVector, kv_cache_pool_shape);
        READ_DUMP(Tensor, kv_cache_pool_options);
        READ_DUMP(Tensor, kv_cache_page_list);
        READ_DUMP(Bool, is_kv_cache_fp8);
        READ_DUMP(Tensor, sequence_lengths);

        auto output        = torch::zeros(output_shape, output_options.options());
        auto kv_cache_pool = torch::zeros(kv_cache_pool_shape, kv_cache_pool_options.options());

#define LOG_DBG(val)                                                                                                   \
    do {                                                                                                               \
        std::stringstream ss;                                                                                          \
        std::string       msg;                                                                                         \
        ss.str("");                                                                                                    \
        ss << val;                                                                                                     \
        ss.flush();                                                                                                    \
        msg = ss.str();                                                                                                \
        RTP_LLM_LOG_INFO(#val " = %s", msg.c_str());                                                                   \
    } while (0)

        LOG_DBG(input.device().str());
        LOG_DBG(input.dtype().name());
        LOG_DBG(input.sizes());
        LOG_DBG(input);
        LOG_DBG(is_input_bf16);
        LOG_DBG(output_shape);
        LOG_DBG(output_options);
        LOG_DBG(head_num);
        LOG_DBG(head_num_kv);
        LOG_DBG(head_dim);
        LOG_DBG(batch_size);
        LOG_DBG(max_blocks_per_seq);
        LOG_DBG(max_seq_len);
        LOG_DBG(page_size);
        LOG_DBG(kv_cache_pool_shape);
        LOG_DBG(kv_cache_pool_options);
        LOG_DBG(kv_cache_page_list.sizes());
        LOG_DBG(kv_cache_page_list);
        LOG_DBG(is_kv_cache_fp8);
        LOG_DBG(sequence_lengths.sizes());
        LOG_DBG(sequence_lengths);

        {
            auto error = cudaDeviceSynchronize();
            CHK("XQA prepare sync", error);
            error = cudaGetLastError();
            CHK("XQA prepare", error);
        }

        runXqa(input.data_ptr(),
               is_input_bf16,
               output.data_ptr(),
               head_num,
               head_num_kv,
               head_dim,
               batch_size,
               max_blocks_per_seq,
               max_seq_len,
               page_size,
               kv_cache_pool.data_ptr(),
               reinterpret_cast<int32_t*>(kv_cache_page_list.data_ptr()),
               is_kv_cache_fp8,
               reinterpret_cast<uint32_t*>(sequence_lengths.data_ptr()));

        {
            auto error = cudaDeviceSynchronize();
            CHK("XQA sync", error);
            error = cudaGetLastError();
            CHK("XQA", error);
        }
    }

    //     torch::serialize::InputArchive archive;
    //     archive.load_from("/home/zhangjianning.zjn/RTP-LLM/github-opensource/xqa_input.pt");
    // #define LOAD(type, name)
    //     torch::IValue name##_ival__;
    //     archive.read(#name, name##_ival__); \ auto name = name##_ival__.to##type()

    //     LOAD(Tensor, input);
    //     // LOAD(Bool, is_input_bf16);
    //     LOAD(Int, head_num);
    //     LOAD(Int, head_num_kv);
    //     LOAD(Int, head_dim);
    //     LOAD(Int, batch_size);
    //     // LOAD(Int, max_blocks_per_seq);
    //     // LOAD(Int, max_seq_len);
    //     LOAD(Int, page_size);
    //     LOAD(Tensor, kv_cache_offset);
    //     LOAD(Bool, is_kv_cache_fp8);
    //     LOAD(Tensor, sequence_lengths);

    //     xqaAttentionOpTest(
    //         batch_size, 1, sequence_lengths[0].item<int>(), head_num, head_num_kv, head_dim, page_size,
    //         is_kv_cache_fp8);
}

#endif
