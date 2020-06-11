#ifndef TATOOINE_TENSOR_MAT_H
#define TATOOINE_TENSOR_MAT_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t M, size_t N>
struct mat : tensor<Real, M, N> {  // NOLINT
  using this_t   = mat<Real, M, N>;
  using parent_t = tensor<Real, M, N>;
  using parent_t::parent_t;
  //----------------------------------------------------------------------------
  constexpr mat(const mat&) = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            , enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr mat(const base_tensor<Tensor, TensorReal, M, N>& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr mat(mat&& other) noexcept : parent_t{std::move(other)} {}
#else
  constexpr mat(mat&& other) noexcept = default;
#endif
  //----------------------------------------------------------------------------
  constexpr auto operator=(const mat&) -> mat& = default;
  //----------------------------------------------------------------------------
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
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            , enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr auto operator=(
      const base_tensor<Tensor, TensorReal, M, N>& other) noexcept -> mat& {
    parent_t::operator=(other);
    return *this;
  }

  //----------------------------------------------------------------------------
  constexpr mat(tag::eye_t /*flag*/) : parent_t{tag::zeros} {
    for (size_t i = 0; i < std::min(M, N); ++i) { this->at(i, i) = 1; }
  }
  //----------------------------------------------------------------------------
  ~mat() = default;
  //----------------------------------------------------------------------------
  static constexpr auto eye() {
    return this_t{tag::eye};
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C])  // NOLINT
    ->mat<promote_t<Rows...>, sizeof...(Rows), C>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using mat2 = mat<double, 2, 2>;
using mat3 = mat<double, 3, 3>;
using mat4 = mat<double, 4, 4>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
