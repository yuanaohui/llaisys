#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
void linear_impl(T *out_ptr, const T *in_ptr, const T *weight_ptr, const T *bias_ptr,
                 size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // 转换为float进行计算，避免精度损失
                sum += llaisys::utils::cast<float>(in_ptr[m * K + k]) * llaisys::utils::cast<float>(weight_ptr[n * K + k]);
            }
            if (bias_ptr) {
                sum += llaisys::utils::cast<float>(bias_ptr[n]);
            }
            // 转换回目标类型
            out_ptr[m * N + n] = llaisys::utils::cast<T>(sum);
        }
    }
}
} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 检查输入输出张量是否在同一设备上
    CHECK_SAME_DEVICE(out, in, weight, bias);
    // 检查数据类型是否匹配
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    // 检查张量是否是连续存储的
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
           "Linear: all tensors must be contiguous.");
    // 检查形状是否符合要求
    CHECK_ARGUMENT(out->ndim() == 2, "Linear: out must be 2D.");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: in must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D.");
    CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D.");
    // 此时weight还没转置，故in的第二维度应等于weight的第二维度
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must equal weight.shape[1].");
    // 输出张量的第一维度应等于输入张量的第一维度
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear: out.shape[0] must equal in.shape[0].");
    // 输出张量的第二维度应等于weight的第一维度
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: out.shape[1] must equal weight.shape[0].");
    // bias可为空
    // 若不为空，bias的大小应等于输出张量的第二维度
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1], "Linear: bias.shape[0] must equal out.shape[1].");
    }
    // 目前仅支持CPU设备
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    size_t M = in->shape()[0];     // in行数
    size_t K = in->shape()[1];     // in列数
    size_t N = weight->shape()[0]; // weight行数
    auto type = in->dtype();

    // 根据数据类型调用相应的实现函数
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_impl(reinterpret_cast<float *>(out->data()),
                           reinterpret_cast<const float *>(in->data()),
                           reinterpret_cast<const float *>(weight->data()),
                           bias ? reinterpret_cast<const float *>(bias->data()) : nullptr,
                           M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_impl(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                           bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr,
                           M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_impl(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                           bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr,
                           M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
