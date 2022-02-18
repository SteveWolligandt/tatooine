#ifndef TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
#define TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
//==============================================================================
#include <tatooine/tensor_operations/binary_operation.h>
#include <tatooine/tensor_operations/same_dimensions.h>
//==============================================================================
namespace tatooine {
//==============================================================================
auto constexpr operator==(
    static_tensor auto const& lhs,
    static_tensor auto const& rhs) requires(same_dimensions(lhs, rhs)) {
  bool equal = true;
  for_loop(
      [&](auto const... is) {
        if (lhs(is...) != rhs(is...)) {
          equal = false;
          return;
        }
      },
      lhs.dimensions());
  return equal;
}
//------------------------------------------------------------------------------
auto constexpr operator!=(
    static_tensor auto const& lhs,
    static_tensor auto const& rhs)
  requires(same_dimensions(lhs, rhs)) {
  return !(lhs == rhs);
}
//------------------------------------------------------------------------------
auto constexpr operator-(static_tensor auto const& t) {
  return unary_operation([](auto const& c) { return -c; }, t);
}
//------------------------------------------------------------------------------
auto constexpr operator+(static_tensor auto const&        lhs,
                         arithmetic_or_complex auto const scalar) {
  return unary_operation([scalar](auto const& c) { return c + scalar; }, lhs);
}
//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <static_mat Lhs, static_mat Rhs>
requires(Lhs::dimension(1) == Rhs::dimension(0)) auto constexpr operator*(
    Lhs const& lhs, Rhs const& rhs) {
  auto product =
      mat<common_type<typename Lhs::value_type, typename Rhs::value_type>,
          Lhs::dimension(0), Rhs::dimension(1)>{};
  for (std::size_t r = 0; r < Lhs::dimension(0); ++r) {
    for (std::size_t c = 0; c < Rhs::dimension(1); ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <static_mat Lhs, static_vec Rhs>
requires(Lhs::dimension(1) == Rhs::dimension(0)) auto constexpr operator*(
    Lhs const& lhs, Rhs const& rhs) {
  auto product =
      vec<common_type<typename Lhs::value_type, typename Rhs::value_type>,
          Lhs::dimension(0)>{};
  for (std::size_t i = 0; i < Lhs::dimension(0); ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <static_vec Lhs, static_mat Rhs>
requires(Lhs::dimension(0) == Rhs::dimension(0)) auto constexpr operator*(
    Lhs const& lhs, Rhs const& rhs) {
  vec<common_type<
          common_type<typename Lhs::value_type, typename Rhs::value_type>>,
      Rhs::dimension(1)>
      product;
  for (std::size_t i = 0; i < Rhs::dimension(1); ++i) {
    product(i) = dot(lhs, rhs.template slice<1>(i));
  }
  return product;
}
//------------------------------------------------------------------------------
/// component-wise multiplication
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>() && Lhs::rank() != 2 &&
         Rhs::rank() != 2) auto constexpr
operator*(Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(
      std::multiplies<
          common_type<typename Lhs::value_type, typename Rhs::value_type>>{},
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise division
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>() && Lhs::rank() != 2 &&
         Rhs::rank() != 2) auto constexpr
operator/(Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(
      std::divides<
          common_type<typename Lhs::value_type, typename Rhs::value_type>>{},
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise addition
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>() && Lhs::rank() != 2 &&
         Rhs::rank() != 2) auto constexpr
operator+(Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(
      std::plus<
          common_type<typename Lhs::value_type, typename Rhs::value_type>>{},
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise subtraction
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>() && Lhs::rank() != 2 &&
         Rhs::rank() != 2) auto constexpr
operator-(Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(
      std::minus<
          common_type<typename Lhs::value_type, typename Rhs::value_type>>{},
      lhs, rhs);
}
//------------------------------------------------------------------------------
auto constexpr operator*(static_tensor auto const&        t,
                         arithmetic_or_complex auto const scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto constexpr operator*(arithmetic_or_complex auto const scalar,
                         static_tensor auto const&        t) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
//------------------------------------------------------------------------------
auto constexpr operator/(static_tensor auto const&        t,
                         arithmetic_or_complex auto const scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto constexpr operator/(arithmetic_or_complex auto const scalar,
                         static_tensor auto const&        t) {
  return unary_operation(
      [scalar](auto const& component) { return component / scalar; }, t);
}
//------------------------------------------------------------------------------
template <dynamic_tensor Lhs, dynamic_tensor Rhs>
auto operator*(Lhs const& lhs, Rhs const& rhs) {
  using out_t = tensor<
      std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>>;
  auto out = out_t{};
  // matrix-matrix-multiplication
  if (lhs.rank() == 2 && rhs.rank() == 2 &&
      lhs.dimension(1) == rhs.dimension(0)) {
    auto out = out_t::zeros(lhs.dimension(0), rhs.dimension(1));
    for (std::size_t r = 0; r < lhs.dimension(0); ++r) {
      for (std::size_t c = 0; c < rhs.dimension(1); ++c) {
        for (std::size_t i = 0; i < lhs.dimension(1); ++i) {
          out(r, c) += lhs(r, i) * rhs(i, c);
        }
      }
    }
    return out;
  }
  // matrix-vector-multiplication
  else if (lhs.rank() == 2 && rhs.rank() == 1 &&
           lhs.dimension(1) == rhs.dimension(0)) {
    auto out = out_t::zeros(lhs.dimension(0));
    for (std::size_t r = 0; r < lhs.dimension(0); ++r) {
      for (std::size_t i = 0; i < lhs.dimension(1); ++i) {
        out(r) += lhs(r, i) * rhs(i);
      }
    }
    return out;
  }

  std::stringstream A;
  A << "[ " << lhs.dimension(0);
  for (std::size_t i = 1; i < lhs.rank(); ++i) {
    A << " x " << lhs.dimension(i);
  }
  A << " ]";
  std::stringstream B;
  B << "[ " << rhs.dimension(0);
  for (std::size_t i = 1; i < rhs.rank(); ++i) {
    B << " x " << rhs.dimension(i);
  }
  B << " ]";
  throw std::runtime_error{"Cannot contract given dynamic tensors. (A:" +
                           A.str() + "; B" + B.str() + ")"};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
