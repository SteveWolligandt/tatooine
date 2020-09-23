#include <tatooine/tensor.h>
//==============================================================================
#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_or_complex_number T, size_t M, size_t N>
struct mat : tensor<T, M, N> {
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_rows() { return M; }
  static constexpr auto num_columns() { return N; }

  //============================================================================
  // typedefs
  //============================================================================
  using this_t   = mat<T, M, N>;
  using parent_t = tensor<T, M, N>;

  //============================================================================
  // inherited methods
  //============================================================================
  using parent_t::parent_t;

  //============================================================================
  // constructors
  //============================================================================
  constexpr mat(mat const&) = default;
  //----------------------------------------------------------------------------
  constexpr mat(mat&& other) noexcept = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  constexpr mat(base_tensor<Tensor, TensorReal, M, N> const& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
  template <real_number... Rows>
  constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {  // NOLINT
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](auto const& row) mutable {
      for (size_t c = 0; c < parent_t::dimension(1); ++c) {
        this->at(r, c) = static_cast<T>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //----------------------------------------------------------------------------
  constexpr mat(tag::eye_t /*flag*/) : parent_t{tag::zeros} {
    for (size_t i = 0; i < std::min(M, N); ++i) { this->at(i, i) = 1; }
  }

  //============================================================================
  // assign operators
  //============================================================================
  constexpr auto operator=(mat const&) -> mat& = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(mat&& other) noexcept -> mat& = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  constexpr auto operator=(
      base_tensor<Tensor, TensorReal, M, N> const& other) noexcept -> mat& {
    parent_t::operator=(other);
    return *this;
  }
  //============================================================================
  // destructor
  //============================================================================
  ~mat() = default;

  //============================================================================
  // factory functions
  //============================================================================
  static constexpr auto eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename Tensor>
  static constexpr auto vander(base_tensor<Tensor, T, N> const & v) {
    this_t V;
    auto   factor_up_row = [row = 0ul, &V](auto x) mutable {
      V(row, 0) = 1;
      for (std::size_t col = 1; col < N; ++col) {
        V(row, col) = V(row, col - 1) * x;
      }
      ++row;
    };
    for (size_t i = 0; i < N; ++i) { factor_up_row(v(i)); }
    return V;
  }
  //----------------------------------------------------------------------------
  static constexpr auto vander(convertible_to<T> auto&&... xs) {
    static_assert(sizeof...(xs) == num_columns());
    this_t V;
    auto   factor_up_row = [row = 0ul, &V](auto x) mutable {
      V(row, 0) = 1;
      for (std::size_t col = 1; col < N; ++col) {
        V(row, col) = V(row, col - 1) * x;
      }
      ++row;
    };
    (factor_up_row(xs), ...);
    return V;
  }

  //============================================================================
  // methods
  //============================================================================
  constexpr auto row(size_t i) { return this->template slice<0>(i); }
  constexpr auto row(size_t i) const { return this->template slice<0>(i); }
  //------------------------------------------------------------------------------
  constexpr auto col(size_t i) { return this->template slice<1>(i); }
  constexpr auto col(size_t i) const { return this->template slice<1>(i); }
};
//==============================================================================
// type traits
//==============================================================================
template <typename Real, size_t M, size_t N>
struct is_tensor<mat<Real, M, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
struct is_mat<mat<Real, M, N>> : std::true_type {};
//==============================================================================
// deduction guide
//==============================================================================
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C])  // NOLINT
    ->mat<promote_t<Rows...>, sizeof...(Rows), C>;
//==============================================================================
// typedefs
//==============================================================================
using mat2 = mat<double, 2, 2>;
using mat3 = mat<double, 3, 3>;
using mat4 = mat<double, 4, 4>;
using mat5 = mat<double, 5, 5>;
using mat6 = mat<double, 6, 6>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
