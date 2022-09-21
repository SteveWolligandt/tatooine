#ifndef TATOOINE_TENSOR_TYPEDEFS_H
#define TATOOINE_TENSOR_TYPEDEFS_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t... Dims>
using Tensor = tensor<real_number, Dims...>;

template <std::size_t... Dims>
using TensorF  = tensor<float, Dims...>;

template <std::size_t... Dims>
using TensorD  = tensor<double, Dims...>;

template <std::size_t... Dims>
using TensorI64  = tensor<std::int64_t, Dims...>;

template <typename T, std::size_t... Dims>
using complex_tensor = tensor<std::complex<T>, Dims...>;
template <std::size_t ...Dims>
using ComplexTensor   = tensor<std::complex<real_number>, Dims...>;
template <std::size_t ...Dims>
using ComplexTensorD   = tensor<std::complex<double>, Dims...>;
template <std::size_t ...Dims>
using ComplexTensorF   = tensor<std::complex<float>, Dims...>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
