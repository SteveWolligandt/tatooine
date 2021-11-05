#ifndef TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
#define TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/tensor_operations/binary_operation.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename TensorLhs, typename TensorRhs, typename T,
          std::size_t... Dims>
constexpr auto operator==(base_tensor<TensorLhs, T, Dims...> const& lhs,
                          base_tensor<TensorRhs, T, Dims...> const& rhs) {
  bool equal = true;
  for_loop(
      [&](auto const... is) {
        if (lhs(is...) != rhs(is...)) {
          equal = false;
          return;
        }
      },
      Dims...);
  return equal;
}
//------------------------------------------------------------------------------
template <typename TensorLhs, typename TensorRhs, typename T,
          std::size_t... Dims>
constexpr auto operator!=(base_tensor<TensorLhs, T, Dims...> const& lhs,
                          base_tensor<TensorRhs, T, Dims...> const& rhs) {
  return !(lhs == rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, std::size_t... Dims>
constexpr auto operator-(base_tensor<Tensor, T, Dims...> const& t) {
  return unary_operation([](auto const& c) { return -c; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename T1, std::size_t... Dims,
          enable_if<is_arithmetic_or_complex<T1>> = true>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         T1                                       scalar) {
  return unary_operation([scalar](auto const& c) { return c + scalar; }, lhs);
}
//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t M, std::size_t N, std::size_t O>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N, O> const& rhs) {
  mat<common_type<T0, T1>, M, O> product;
  for (std::size_t r = 0; r < M; ++r) {
    for (std::size_t c = 0; c < O; ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// component-wise multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t... Dims, enable_if<(sizeof...(Dims) != 2)> = true>
constexpr auto operator*(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::multiplies<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t... Dims>
constexpr auto operator/(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::divides<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t... Dims>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::plus<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true,
          std::size_t... Dims>
constexpr auto operator*(base_tensor<Tensor, TensorT, Dims...> const& t,
                         Scalar const                                 scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true,
          std::size_t... Dims>
constexpr auto operator*(Scalar const                                 scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true,
          std::size_t... Dims>
constexpr auto operator/(base_tensor<Tensor, TensorT, Dims...> const& t,
                         Scalar const                                 scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true,
          std::size_t... Dims>
constexpr auto operator/(Scalar const                                 scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return scalar / component; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t... Dims>
constexpr auto operator-(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::minus<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t M, std::size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N> const&    rhs) {
  vec<common_type<T0, T1>, M> product;
  for (std::size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          std::size_t M, std::size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M> const&    lhs,
                         base_tensor<Tensor1, T1, M, N> const& rhs) {
  vec<common_type<T0, T1>, N> product;
  for (std::size_t i = 0; i < N; ++i) {
    product(i) = dot(lhs, rhs.template slice<1>(i));
  }
  return product;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <typename LhsTensor, typename RhsTensor>
requires is_dynamic_tensor<LhsTensor> && is_dynamic_tensor<RhsTensor>
#else
template <typename LhsTensor, typename RhsTensor,
          enable_if<is_dynamic_tensor<LhsTensor>,
                    is_dynamic_tensor<RhsTensor>> = true>
#endif
auto operator*(LhsTensor const& lhs, RhsTensor const& rhs)
    -> tensor<std::common_type_t<typename LhsTensor::value_type,
                                 typename RhsTensor::value_type>> {
  using out_t = tensor<std::common_type_t<typename LhsTensor::value_type,
                                          typename RhsTensor::value_type>>;
  out_t out;
  // matrix-matrix-multiplication
  if (lhs.num_dimensions() == 2 && rhs.num_dimensions() == 2 &&
      lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0), rhs.size(1));
    for (std::size_t r = 0; r < lhs.size(0); ++r) {
      for (std::size_t c = 0; c < rhs.size(1); ++c) {
        for (std::size_t i = 0; i < lhs.size(1); ++i) {
          out(r, c) += lhs(r, i) * rhs(i, c);
        }
      }
    }
    return out;
  }
  // matrix-vector-multiplication
  else if (lhs.num_dimensions() == 2 && rhs.num_dimensions() == 1 &&
           lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0));
    for (std::size_t r = 0; r < lhs.size(0); ++r) {
      for (std::size_t i = 0; i < lhs.size(1); ++i) {
        out(r) += lhs(r, i) * rhs(i);
      }
    }
    return out;
  }

  std::stringstream A;
  A << "[ " << lhs.size(0);
  for (std::size_t i = 1; i < lhs.num_dimensions(); ++i) {
    A << " x " << lhs.size(i);
  }
  A << " ]";
  std::stringstream B;
  B << "[ " << rhs.size(0);
  for (std::size_t i = 1; i < rhs.num_dimensions(); ++i) {
    B << " x " << rhs.size(i);
  }
  B << " ]";
  throw std::runtime_error{"Cannot contract given dynamic tensors. (A:" +
                           A.str() + "; B" + B.str() + ")"};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
