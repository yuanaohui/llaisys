#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
void argmax_impl(int64_t *out_idx, T *out_val, const T *vals, size_t numel) {
    size_t max_i = 0;
    float max_v = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > max_v) {
            max_v = v;
            max_i = i;
        }
    }
    // store results
    // out_idx is of type int64_t*
    // out_val is of type T*
    *out_idx = static_cast<int64_t>(max_i);
    out_val[0] = llaisys::utils::cast<T>(max_v);
}
} // namespace

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 获取vals的最大值及其索引，分别存储在max_val和max_idx中
    // 判断max_idx, max_val, vals是否在同一设备上
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 判断数据类型是否符合要求
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    // 判断张量是否是连续存储的
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be of dtype int64.");
    // 判断形状是否符合要求
    CHECK_SAME_SHAPE(max_idx->shape(), max_val->shape());
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be 1D.");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1,
                   "Argmax: max_idx and max_val must have exactly one element.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals must not be empty.");

    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    auto numel = vals->numel();
    auto type = vals->dtype();
    // 获取数据指针，并调用相应的数据类型处理函数， 使max_idx->data()的数据类型强制转换为int64_t*
    auto idx_ptr = reinterpret_cast<int64_t *>(max_idx->data());

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl(idx_ptr,
                           reinterpret_cast<float *>(max_val->data()),
                           reinterpret_cast<const float *>(vals->data()),
                           numel);
    case LLAISYS_DTYPE_F16:
        return argmax_impl(idx_ptr,
                           reinterpret_cast<llaisys::fp16_t *>(max_val->data()),
                           reinterpret_cast<const llaisys::fp16_t *>(vals->data()),
                           numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl(idx_ptr,
                           reinterpret_cast<llaisys::bf16_t *>(max_val->data()),
                           reinterpret_cast<const llaisys::bf16_t *>(vals->data()),
                           numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
