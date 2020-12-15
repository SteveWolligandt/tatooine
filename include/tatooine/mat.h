#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
#include <tatooine/real.h>
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
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
  using parent_t::is_quadratic_mat;
  using parent_t::parent_t;
  //============================================================================
  // factories
  //============================================================================
  static constexpr auto zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::fill<T>{1}}; }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires (is_quadratic_mat())
  static constexpr auto eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal<T>{eng, mean, stddev}};
  }

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
// deduction guide
//==============================================================================
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C])  // NOLINT
    ->mat<promote_t<Rows...>, sizeof...(Rows), C>;
//==============================================================================
// typedefs
//==============================================================================

template <real_or_complex_number T>
using Mat2 = mat<T, 2, 2>;
template <real_or_complex_number T>
using Mat23 = mat<T, 2, 3>;
template <real_or_complex_number T>
using Mat24 = mat<T, 2, 4>;

template <real_or_complex_number T>
using Mat32 = mat<T, 3, 2>;
template <real_or_complex_number T>
using Mat3 = mat<T, 3, 3>;
template <real_or_complex_number T>
using Mat34 = mat<T, 3, 4>;

template <real_or_complex_number T>
using Mat42 = mat<T, 4, 2>;
template <real_or_complex_number T>
using Mat43 = mat<T, 4, 3>;
template <real_or_complex_number T>
using Mat4 = mat<T, 4, 4>;

template <real_or_complex_number T>
using Mat5 = mat<T, 5, 5>;
template <real_or_complex_number T>
using Mat6 = mat<T, 6, 6>;


using mat2d = Mat2<double>;
using mat23d = Mat23<double>;
using mat24d = Mat24<double>;

using mat32d = Mat32<double>;
using mat3d = Mat3<double>;
using mat34d = Mat34<double>;

using mat42d = Mat42<double>;
using mat43d = Mat43<double>;
using mat4d = Mat4<double>;

using mat5d = Mat5<double>;
using mat6d = Mat6<double>;

using mat2f = Mat2<float>;
using mat23f = Mat23<float>;
using mat24f = Mat24<float>;

using mat32f = Mat32<float>;
using mat3f = Mat3<float>;
using mat34f = Mat34<float>;

using mat42f = Mat42<float>;
using mat43f = Mat43<float>;
using mat4f = Mat4<float>;

using mat5f = Mat5<float>;
using mat6f = Mat6<float>;

using mat2 = Mat2<real_t>;
using mat23 = Mat23<real_t>;
using mat24 = Mat24<real_t>;

using mat32 = Mat32<real_t>;
using mat3 = Mat3<real_t>;
using mat34 = Mat34<real_t>;

using mat42 = Mat42<real_t>;
using mat43 = Mat43<real_t>;
using mat4 = Mat4<real_t>;

using mat5 = Mat5<real_t>;
using mat6 = Mat6<real_t>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
