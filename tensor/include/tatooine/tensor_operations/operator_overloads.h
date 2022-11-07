#ifndef TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
#define TATOOINE_TENSOR_OPERATIONS_OPERATOR_OVERLOADS_H
//==============================================================================
#include <tatooine/tensor_operations/binary_operation.h>
#include <tatooine/tensor_operations/same_dimensions.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) auto constexpr operator==(
    Lhs const& lhs, Rhs const& rhs) {
  bool equal = true;
  for_loop_unpacked(
      [&](auto const... is) {
        if (lhs(is...) != rhs(is...)) {
          equal = false;
          return;
        }
      },
      tensor_dimensions<Lhs>);

  return equal;
}
//------------------------------------------------------------------------------
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) auto constexpr operator!=(
    Lhs const& lhs, Rhs const& rhs) {
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
requires(Lhs::dimension(1) == Rhs::dimension(0))
auto constexpr operator*(Lhs const& lhs, Rhs const& rhs) {
  auto product =
      mat<common_type<typename Lhs::value_type, typename Rhs::value_type>,
          Lhs::dimension(0), Rhs::dimension(1)>{};
  for (std::size_t r = 0; r < Lhs::dimension(0); ++r) {
    for (std::size_t c = 0; c < Rhs::dimension(1); ++c) {
      for (std::size_t i = 0; i < Lhs::dimension(1); ++i) {
        product(r, c) += lhs(r, i) * rhs(i, c);
      }
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <static_mat Lhs, static_vec Rhs>
requires(Lhs::dimension(1) == Rhs::dimension(0))
auto constexpr operator*(Lhs const& lhs, Rhs const& rhs) {
  auto product =
      vec<common_type<typename Lhs::value_type, typename Rhs::value_type>,
          Lhs::dimension(0)>{};
  for (std::size_t j = 0; j < Lhs::dimension(0); ++j) {
    for (std::size_t i = 0; i < Lhs::dimension(1); ++i) {
      product(j) += lhs(j, i) * rhs(i);
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <static_vec Lhs, static_mat Rhs>
requires(Lhs::dimension(0) == Rhs::dimension(0))
auto constexpr operator*(Lhs const& lhs, Rhs const& rhs) {
  auto product =
      vec<common_type<
              common_type<typename Lhs::value_type, typename Rhs::value_type>>,
          Rhs::dimension(1)>{};
  for (std::size_t j = 0; j < Rhs::dimension(0); ++j) {
    for (std::size_t i = 0; i < Rhs::dimension(1); ++i) {
      product += lhs(i) * rhs(i, j);
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// component-wise multiplication
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>() && Lhs::rank() != 2 && Rhs::rank() != 2)
auto constexpr operator*(Lhs const& lhs, Rhs const& rhs) {
  return binary_operation(
      [](auto&& l, auto&& r) {
        using out_type =
            common_type<std::decay_t<decltype(l)>, std::decay_t<decltype(r)>>;
        return static_cast<out_type>(l) * static_cast<out_type>(r);
      },
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise division
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) auto constexpr operator/(Lhs const& lhs,
                                                               Rhs const& rhs) {
  return binary_operation(
      [](auto&& l, auto&& r) {
        using out_type =
            common_type<std::decay_t<decltype(l)>, std::decay_t<decltype(r)>>;
        return static_cast<out_type>(l) / static_cast<out_type>(r);
      },
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise addition
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) auto constexpr operator+(Lhs const& lhs,
                                                               Rhs const& rhs) {
  return binary_operation(
      [](auto&& l, auto&& r) {
        using out_type =
            common_type<std::decay_t<decltype(l)>, std::decay_t<decltype(r)>>;
        return static_cast<out_type>(l) + static_cast<out_type>(r);
      },
      lhs, rhs);
}
//------------------------------------------------------------------------------
/// component-wise subtraction
template <static_tensor Lhs, static_tensor Rhs>
requires(same_dimensions<Lhs, Rhs>()) auto constexpr operator-(Lhs const& lhs,
                                                               Rhs const& rhs) {
  return binary_operation(
      [](auto&& l, auto&& r) {
        using out_type =
            common_type<std::decay_t<decltype(l)>, std::decay_t<decltype(r)>>;
        return static_cast<out_type>(l) - static_cast<out_type>(r);
      },
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
  return t * (1 / scalar);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
auto constexpr operator/(arithmetic_or_complex auto const scalar,
                         static_tensor auto const&        t) {
  return unary_operation(
      [scalar](auto const& component) { return scalar / component; }, t);
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
