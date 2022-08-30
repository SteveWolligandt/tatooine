#ifndef TATOOINE_MAT_H
#define TATOOINE_MAT_H
//==============================================================================
#include <tatooine/hdf5.h>
#include <tatooine/real.h>
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex ValueType, std::size_t M, std::size_t N>
struct mat : tensor<ValueType, M, N> {
  //============================================================================
  // static methods
  //============================================================================
  static auto constexpr num_rows() { return M; }
  static auto constexpr num_columns() { return N; }

  //============================================================================
  // typedefs
  //============================================================================
  using this_type   = mat<ValueType, M, N>;
  using parent_type = tensor<ValueType, M, N>;

  //============================================================================
  // inherited methods
  //============================================================================
  using parent_type::parent_type;
  using parent_type::operator();
  using parent_type::at;

  //============================================================================
  // factories
  //============================================================================
  static auto constexpr zeros() { return this_type{tag::fill<ValueType>{0}}; }
  //----------------------------------------------------------------------------
  static auto constexpr ones() { return this_type{tag::fill<ValueType>{1}}; }
  //----------------------------------------------------------------------------
  static auto constexpr nans() requires floating_point<ValueType> {
    return this_type{tag::fill<ValueType>{ValueType{} / ValueType{}}};
  }
  //----------------------------------------------------------------------------
  static auto constexpr eye() { return this_type{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randu(ValueType min = 0, ValueType max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randn(ValueType mean = 0, ValueType stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{random::normal<ValueType>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  static auto constexpr vander(fixed_size_vec<N> auto const& v) {
    auto V             = this_type{};
    auto factor_up_row = [row = std::size_t(0), &V](auto x) mutable {
      V(row, 0) = 1;
      for (std::size_t col = 1; col < N; ++col) {
        V(row, col) = V(row, col - 1) * x;
      }
      ++row;
    };
    for (std::size_t i = 0; i < N; ++i) {
      factor_up_row(v(i));
    }
    return V;
  }
  //----------------------------------------------------------------------------
  static auto constexpr vander(convertible_to<ValueType> auto&&... xs) {
    static_assert(sizeof...(xs) == num_columns());
    auto V             = this_type{};
    auto factor_up_row = [row = std::size_t(0), &V](auto x) mutable {
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
  // constructors
  //============================================================================
  constexpr mat(mat const&) = default;
  //----------------------------------------------------------------------------
  constexpr mat(mat&& other) noexcept = default;
  //----------------------------------------------------------------------------
  /// Copies any other tensor with same dimensions.
  template <static_tensor Other>
  requires(same_dimensions<this_type, Other>()) explicit constexpr mat(
      Other&& other)
      : parent_type{std::forward<Other>(other)} {}
  //----------------------------------------------------------------------------
  template <typename... Rows>
  explicit constexpr mat(Rows(&&... rows)[parent_type::dimension(1)]) {
    static_assert(((is_arithmetic<Rows> || is_complex<Rows>)&&...));
    static_assert(
        sizeof...(rows) == parent_type::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = std::size_t(0), this](auto const& row) mutable {
      for (std::size_t c = 0; c < parent_type::dimension(1); ++c) {
        at(r, c) = static_cast<ValueType>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //----------------------------------------------------------------------------
  /// Constructs an identity matrix.
  explicit constexpr mat(tag::eye_t /*flag*/) : parent_type{tag::zeros} {
    for (std::size_t i = 0; i < std::min(M, N); ++i) {
      at(i, i) = 1;
    }
  }
  //============================================================================
  // assign operators
  //============================================================================
  auto constexpr operator=(mat const&) -> mat& = default;
  //----------------------------------------------------------------------------
  auto constexpr operator=(mat&& other) noexcept -> mat& = default;
  //----------------------------------------------------------------------------
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <static_tensor Other>
  requires(same_dimensions<this_type, Other>())
  auto constexpr operator=(Other const& other) noexcept -> mat& {
    parent_type::operator=(other);
    return *this;
  }
  //============================================================================
  // destructor
  //============================================================================
  ~mat() = default;
  //============================================================================
  // methods
  //============================================================================
  auto constexpr row(std::size_t i) { return this->template slice<0>(i); }
  auto constexpr row(std::size_t i) const { return this->template slice<0>(i); }
  //------------------------------------------------------------------------------
  auto constexpr col(std::size_t i) { return this->template slice<1>(i); }
  auto constexpr col(std::size_t i) const { return this->template slice<1>(i); }
};
//==============================================================================
// deduction guides
//==============================================================================
template <std::size_t C, typename... Rows>
mat(Rows const(&&... rows)[C]) -> mat<common_type<Rows...>, sizeof...(Rows), C>;
//------------------------------------------------------------------------------
template <typename Mat, typename ValueType, std::size_t M, std::size_t N>
mat(base_tensor<Mat, ValueType, M, N>) -> mat<ValueType, M, N>;
//==============================================================================
namespace reflection {
template <typename ValueType, std::size_t M, std::size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (mat<ValueType, M, N>), TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/mat_typedefs.h>
//==============================================================================
#endif
