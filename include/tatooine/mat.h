#include <tatooine/tensor.h>
//==============================================================================
#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t M, size_t N>
struct mat : tensor<Real, M, N> {  // NOLINT
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_rows() { return M; }
  static constexpr auto num_columns() { return N; }

  //============================================================================
  // typedefs
  //============================================================================
  using this_t    = mat<Real, M, N>;
  using parent_t  = tensor<Real, M, N>;

  //============================================================================
  // inherited methods
  //============================================================================
  using parent_t::parent_t;

  //============================================================================
  // constructors
  //============================================================================
  constexpr mat(const mat&) = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            ,
            enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr mat(const base_tensor<Tensor, TensorReal, M, N>& other)
      : parent_t{other} {
  }
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr mat(mat&& other) noexcept : parent_t{std::move(other)} {}
#else
  constexpr mat(mat&& other) noexcept = default;
#endif
  //----------------------------------------------------------------------------
  constexpr mat(tag::eye_t /*flag*/) : parent_t{tag::zeros} {
    for (size_t i = 0; i < std::min(M, N); ++i) { this->at(i, i) = 1; }
  }

  //============================================================================
  // assign operators
  //============================================================================
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            ,
            enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr auto operator=(
      const base_tensor<Tensor, TensorReal, M, N>& other) noexcept -> mat& {
    parent_t::operator=(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(const mat&) -> mat& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#if TATOOINE_GINAC_AVAILABLE
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  constexpr auto operator=(mat&& other) noexcept -> mat& {
    parent_t::operator=(std::move(other));
    return *this;
  }
#else
  constexpr auto operator=(mat&& other) noexcept -> mat& = default;
#endif
  //============================================================================
  // destructor
  //============================================================================
  ~mat() = default;

  //============================================================================
  // factory functions
  //============================================================================
  static constexpr auto eye() { return this_t{tag::eye}; }
  //------------------------------------------------------------------------------
  static constexpr auto vander(std::convertible_to<Real> auto&&... xs) {
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

#if TATOOINE_GINAC_AVAILABLE
  template <typename... Rows, enable_if_arithmetic_or_symbolic<Rows...> = true>
#else
  template <typename... Rows, enable_if_arithmetic<Rows...> = true>
#endif
  constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {  // NOLINT
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](const auto& row) mutable {
      for (size_t c = 0; c < parent_t::dimension(1); ++c) {
        this->at(r, c) = static_cast<Real>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //------------------------------------------------------------------------------
  constexpr auto row(size_t i) { return this->template slice<0>(i); }
  constexpr auto row(size_t i) const { return this->template slice<0>(i); }
  //------------------------------------------------------------------------------
  constexpr auto col(size_t i) { return this->template slice<1>(i); }
  constexpr auto col(size_t i) const { return this->template slice<1>(i); }
};
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
