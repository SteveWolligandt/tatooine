#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_or_complex_number T, size_t M, size_t N>
struct mat : tensor<T, M, N> {  // NOLINT
  using this_t   = mat<T, M, N>;
  using parent_t = tensor<T, M, N>;
  using parent_t::parent_t;
  //----------------------------------------------------------------------------
  constexpr mat(const mat&) = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  constexpr mat(const base_tensor<Tensor, TensorReal, M, N>& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
  constexpr mat(mat&& other) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(const mat&) -> mat& = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(mat&& other) noexcept -> mat& = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
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
  static constexpr auto eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <real_number... Rows>
  constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {  // NOLINT
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](const auto& row) mutable {
      for (size_t c = 0; c < parent_t::dimension(1); ++c) {
        this->at(r, c) = static_cast<T>(row[c]);
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
