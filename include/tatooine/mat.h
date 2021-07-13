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
template <typename T, size_t M, size_t N>
struct mat : tensor<T, M, N> {
  static_assert(is_arithmetic<T> || is_complex<T>);
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
  using parent_t::operator();

  //============================================================================
  // factories
  //============================================================================
  static constexpr auto zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::fill<T>{1}}; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires (is_quadratic_mat())
#else
  template <typename = void, enable_if<is_quadratic_mat()> = true>
#endif
  static constexpr auto eye() { return this_t{tag::eye}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(T mean = 0, T stddev = 1,
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
  constexpr mat(base_tensor<Tensor, TensorReal, M, N> const& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
  template <typename... Rows>
  constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {  // NOLINT
    static_assert(((is_arithmetic<Rows> || is_complex<Rows>)&&...));
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
#ifdef __cpp_concepts
  template <convertible_to<T> ...Xs>
#else
  template <typename... Xs, enable_if<(is_convertible<Xs, T> && ...)> = true>
#endif
  static constexpr auto vander(Xs&&... xs) {
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
    ->mat<common_type<Rows...>, sizeof...(Rows), C>;
//==============================================================================
// typedefs
//==============================================================================
template <size_t M, size_t N>
using Mat = mat<real_t, M, N>;
template <size_t M, size_t N>
using MatD = mat<double, M, N>;
template <size_t M, size_t N>
using MatF = mat<float, M, N>;

using mat22 = Mat<2, 2>;
using mat23 = Mat<2, 3>;
using mat24 = Mat<2, 4>;
using mat25 = Mat<2, 5>;
using mat26 = Mat<2, 6>;
using mat27 = Mat<2, 7>;
using mat28 = Mat<2, 8>;
using mat29 = Mat<2, 9>;
using mat32 = Mat<3, 2>;
using mat33 = Mat<3, 3>;
using mat34 = Mat<3, 4>;
using mat35 = Mat<3, 5>;
using mat36 = Mat<3, 6>;
using mat37 = Mat<3, 7>;
using mat38 = Mat<3, 8>;
using mat39 = Mat<3, 9>;
using mat42 = Mat<4, 2>;
using mat43 = Mat<4, 3>;
using mat44 = Mat<4, 4>;
using mat45 = Mat<4, 5>;
using mat46 = Mat<4, 6>;
using mat47 = Mat<4, 7>;
using mat48 = Mat<4, 8>;
using mat49 = Mat<4, 9>;
using mat52 = Mat<5, 2>;
using mat53 = Mat<5, 3>;
using mat54 = Mat<5, 4>;
using mat55 = Mat<5, 5>;
using mat56 = Mat<5, 6>;
using mat57 = Mat<5, 7>;
using mat58 = Mat<5, 8>;
using mat59 = Mat<5, 9>;
using mat62 = Mat<6, 2>;
using mat63 = Mat<6, 3>;
using mat64 = Mat<6, 4>;
using mat65 = Mat<6, 5>;
using mat66 = Mat<6, 6>;
using mat67 = Mat<6, 7>;
using mat68 = Mat<6, 8>;
using mat69 = Mat<6, 9>;
using mat72 = Mat<7, 2>;
using mat73 = Mat<7, 3>;
using mat74 = Mat<7, 4>;
using mat75 = Mat<7, 5>;
using mat76 = Mat<7, 6>;
using mat77 = Mat<7, 7>;
using mat78 = Mat<7, 8>;
using mat79 = Mat<7, 9>;
using mat82 = Mat<8, 2>;
using mat83 = Mat<8, 3>;
using mat84 = Mat<8, 4>;
using mat85 = Mat<8, 5>;
using mat86 = Mat<8, 6>;
using mat87 = Mat<8, 7>;
using mat88 = Mat<8, 8>;
using mat89 = Mat<8, 9>;
using mat92 = Mat<9, 2>;
using mat93 = Mat<9, 3>;
using mat94 = Mat<9, 4>;
using mat95 = Mat<9, 5>;
using mat96 = Mat<9, 6>;
using mat97 = Mat<9, 7>;
using mat98 = Mat<9, 8>;
using mat99 = Mat<9, 9>;
using mat2  = mat22;
using mat3  = mat33;
using mat4  = mat44;
using mat5  = mat55;
using mat6  = mat66;
using mat7  = mat77;
using mat8  = mat88;
using mat9  = mat99;

using mat22f  = MatF<2, 2>;
using mat23f = MatF<2, 3>;
using mat24f = MatF<2, 4>;
using mat25f = MatF<2, 5>;
using mat26f = MatF<2, 6>;
using mat27f = MatF<2, 7>;
using mat28f = MatF<2, 8>;
using mat29f = MatF<2, 9>;
using mat32f = MatF<3, 2>;
using mat33f = MatF<3, 3>;
using mat34f = MatF<3, 4>;
using mat35f = MatF<3, 5>;
using mat36f = MatF<3, 6>;
using mat37f = MatF<3, 7>;
using mat38f = MatF<3, 8>;
using mat39f = MatF<3, 9>;
using mat42f = MatF<4, 2>;
using mat43f = MatF<4, 3>;
using mat44f = MatF<4, 4>;
using mat45f = MatF<4, 5>;
using mat46f = MatF<4, 6>;
using mat47f = MatF<4, 7>;
using mat48f = MatF<4, 8>;
using mat49f = MatF<4, 9>;
using mat52f = MatF<5, 2>;
using mat53f = MatF<5, 3>;
using mat54f = MatF<5, 4>;
using mat55f = MatF<5, 5>;
using mat56f = MatF<5, 6>;
using mat57f = MatF<5, 7>;
using mat58f = MatF<5, 8>;
using mat59f = MatF<5, 9>;
using mat62f = MatF<6, 2>;
using mat63f = MatF<6, 3>;
using mat64f = MatF<6, 4>;
using mat65f = MatF<6, 5>;
using mat66f = MatF<6, 6>;
using mat67f = MatF<6, 7>;
using mat68f = MatF<6, 8>;
using mat69f = MatF<6, 9>;
using mat72f = MatF<7, 2>;
using mat73f = MatF<7, 3>;
using mat74f = MatF<7, 4>;
using mat75f = MatF<7, 5>;
using mat76f = MatF<7, 6>;
using mat77f = MatF<7, 7>;
using mat78f = MatF<7, 8>;
using mat79f = MatF<7, 9>;
using mat82f = MatF<8, 2>;
using mat83f = MatF<8, 3>;
using mat84f = MatF<8, 4>;
using mat85f = MatF<8, 5>;
using mat86f = MatF<8, 6>;
using mat87f = MatF<8, 7>;
using mat88f = MatF<8, 8>;
using mat89f = MatF<8, 9>;
using mat92f = MatF<9, 2>;
using mat93f = MatF<9, 3>;
using mat94f = MatF<9, 4>;
using mat95f = MatF<9, 5>;
using mat96f = MatF<9, 6>;
using mat97f = MatF<9, 7>;
using mat98f = MatF<9, 8>;
using mat99f = MatF<9, 9>;
using mat2f  = mat22f;
using mat3f  = mat33f;
using mat4f  = mat44f;
using mat5f  = mat55f;
using mat6f  = mat66f;
using mat7f  = mat77f;
using mat8f  = mat88f;
using mat9f  = mat99f;

using mat22d = MatD<2, 2>;
using mat23d = MatD<2, 3>;
using mat24d = MatD<2, 4>;
using mat25d = MatD<2, 5>;
using mat26d = MatD<2, 6>;
using mat27d = MatD<2, 7>;
using mat28d = MatD<2, 8>;
using mat29d = MatD<2, 9>;
using mat32d = MatD<3, 2>;
using mat33d = MatD<3, 3>;
using mat34d = MatD<3, 4>;
using mat35d = MatD<3, 5>;
using mat36d = MatD<3, 6>;
using mat37d = MatD<3, 7>;
using mat38d = MatD<3, 8>;
using mat39d = MatD<3, 9>;
using mat42d = MatD<4, 2>;
using mat43d = MatD<4, 3>;
using mat44d = MatD<4, 4>;
using mat45d = MatD<4, 5>;
using mat46d = MatD<4, 6>;
using mat47d = MatD<4, 7>;
using mat48d = MatD<4, 8>;
using mat49d = MatD<4, 9>;
using mat52d = MatD<5, 2>;
using mat53d = MatD<5, 3>;
using mat54d = MatD<5, 4>;
using mat55d = MatD<5, 5>;
using mat56d = MatD<5, 6>;
using mat57d = MatD<5, 7>;
using mat58d = MatD<5, 8>;
using mat59d = MatD<5, 9>;
using mat62d = MatD<6, 2>;
using mat63d = MatD<6, 3>;
using mat64d = MatD<6, 4>;
using mat65d = MatD<6, 5>;
using mat66d = MatD<6, 6>;
using mat67d = MatD<6, 7>;
using mat68d = MatD<6, 8>;
using mat69d = MatD<6, 9>;
using mat72d = MatD<7, 2>;
using mat73d = MatD<7, 3>;
using mat74d = MatD<7, 4>;
using mat75d = MatD<7, 5>;
using mat76d = MatD<7, 6>;
using mat77d = MatD<7, 7>;
using mat78d = MatD<7, 8>;
using mat79d = MatD<7, 9>;
using mat82d = MatD<8, 2>;
using mat83d = MatD<8, 3>;
using mat84d = MatD<8, 4>;
using mat85d = MatD<8, 5>;
using mat86d = MatD<8, 6>;
using mat87d = MatD<8, 7>;
using mat88d = MatD<8, 8>;
using mat89d = MatD<8, 9>;
using mat92d = MatD<9, 2>;
using mat93d = MatD<9, 3>;
using mat94d = MatD<9, 4>;
using mat95d = MatD<9, 5>;
using mat96d = MatD<9, 6>;
using mat97d = MatD<9, 7>;
using mat98d = MatD<9, 8>;
using mat99d = MatD<9, 9>;
using mat2d  = mat22d;
using mat3d  = mat33d;
using mat4d  = mat44d;
using mat5d  = mat55d;
using mat6d  = mat66d;
using mat7d  = mat77d;
using mat8d  = mat88d;
using mat9d  = mat99d;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
