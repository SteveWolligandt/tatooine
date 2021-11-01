#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
#include <tatooine/real.h>
#include <tatooine/tensor.h>
#include <tatooine/hdf5.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic_or_complex T, size_t M, size_t N>
#else
template <typename T, size_t M, size_t N>
#endif
struct mat : tensor<T, M, N> {
#ifndef __cpp_concepts
  static_assert(is_arithmetic<T> || is_complex<T>);
#endif
  //============================================================================
  // static methods
  //============================================================================
  static auto constexpr num_rows() { return M; }
  static auto constexpr num_columns() { return N; }

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
  using parent_t::operator();
  using parent_t::at;

  //============================================================================
  // factories
  //============================================================================
  static auto constexpr zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static auto constexpr ones() { return this_t{tag::fill<T>{1}}; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires (is_quadratic_mat())
#else
  template <typename = void, enable_if<is_quadratic_mat()> = true>
#endif
  static auto constexpr eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::normal<T>{eng, mean, stddev}};
  }

  //============================================================================
  // constructors
  //============================================================================
  constexpr mat(mat const&) = default;
  //----------------------------------------------------------------------------
  constexpr mat(mat&& other) noexcept = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  explicit constexpr mat(base_tensor<Tensor, TensorReal, M, N> const& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
  template <typename... Rows>
  explicit constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {
    static_assert(((is_arithmetic<Rows> || is_complex<Rows>)&&...));
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = std::size_t(0), this](auto const& row) mutable {
      for (size_t c = 0; c < parent_t::dimension(1); ++c) {
        at(r, c) = static_cast<T>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //----------------------------------------------------------------------------
  explicit constexpr mat(tag::eye_t /*flag*/) : parent_t{tag::zeros} {
    for (size_t i = 0; i < std::min(M, N); ++i) { at(i, i) = 1; }
  }

  //============================================================================
  // assign operators
  //============================================================================
  auto constexpr operator=(mat const&) -> mat& = default;
  //----------------------------------------------------------------------------
  auto constexpr operator=(mat&& other) noexcept -> mat& = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  auto constexpr operator=(
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
  static auto constexpr eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename Tensor>
  static auto constexpr vander(base_tensor<Tensor, T, N> const & v) {
    this_t V;
    auto   factor_up_row = [row = std::size_t(0), &V](auto x) mutable {
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
#ifdef __cpp_concepts
  template <convertible_to<T> ...Xs>
#else
  template <typename... Xs, enable_if<(is_convertible<Xs, T> && ...)> = true>
#endif
  static auto constexpr vander(Xs&&... xs) {
    static_assert(sizeof...(xs) == num_columns());
    this_t V;
    auto   factor_up_row = [row = std::size_t(0), &V](auto x) mutable {
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
  auto constexpr row(size_t i) { return this->template slice<0>(i); }
  auto constexpr row(size_t i) const { return this->template slice<0>(i); }
  //------------------------------------------------------------------------------
  auto constexpr col(size_t i) { return this->template slice<1>(i); }
  auto constexpr col(size_t i) const { return this->template slice<1>(i); }
};
//==============================================================================
// deduction guide
//==============================================================================
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C]) -> mat<common_type<Rows...>, sizeof...(Rows), C>;
//==============================================================================
namespace reflection {
template <typename T, size_t M, size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (mat<T, M, N>), TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/mat_typedefs.h>
//==============================================================================
#endif
