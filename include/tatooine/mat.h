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

template <typename T, size_t M, size_t N>
using complex_mat = mat<std::complex<T>, M, N>;
template <size_t M, size_t N>
using ComplexMat = complex_mat<real_t, M, N>;
template <size_t M, size_t N>
using ComplexMatD = complex_mat<double, M, N>;
template <size_t M, size_t N>
using ComplexMatF = complex_mat<float, M, N>;

using complex_mat22 = ComplexMat<2, 2>;
using complex_mat23 = ComplexMat<2, 3>;
using complex_mat24 = ComplexMat<2, 4>;
using complex_mat25 = ComplexMat<2, 5>;
using complex_mat26 = ComplexMat<2, 6>;
using complex_mat27 = ComplexMat<2, 7>;
using complex_mat28 = ComplexMat<2, 8>;
using complex_mat29 = ComplexMat<2, 9>;
using complex_mat32 = ComplexMat<3, 2>;
using complex_mat33 = ComplexMat<3, 3>;
using complex_mat34 = ComplexMat<3, 4>;
using complex_mat35 = ComplexMat<3, 5>;
using complex_mat36 = ComplexMat<3, 6>;
using complex_mat37 = ComplexMat<3, 7>;
using complex_mat38 = ComplexMat<3, 8>;
using complex_mat39 = ComplexMat<3, 9>;
using complex_mat42 = ComplexMat<4, 2>;
using complex_mat43 = ComplexMat<4, 3>;
using complex_mat44 = ComplexMat<4, 4>;
using complex_mat45 = ComplexMat<4, 5>;
using complex_mat46 = ComplexMat<4, 6>;
using complex_mat47 = ComplexMat<4, 7>;
using complex_mat48 = ComplexMat<4, 8>;
using complex_mat49 = ComplexMat<4, 9>;
using complex_mat52 = ComplexMat<5, 2>;
using complex_mat53 = ComplexMat<5, 3>;
using complex_mat54 = ComplexMat<5, 4>;
using complex_mat55 = ComplexMat<5, 5>;
using complex_mat56 = ComplexMat<5, 6>;
using complex_mat57 = ComplexMat<5, 7>;
using complex_mat58 = ComplexMat<5, 8>;
using complex_mat59 = ComplexMat<5, 9>;
using complex_mat62 = ComplexMat<6, 2>;
using complex_mat63 = ComplexMat<6, 3>;
using complex_mat64 = ComplexMat<6, 4>;
using complex_mat65 = ComplexMat<6, 5>;
using complex_mat66 = ComplexMat<6, 6>;
using complex_mat67 = ComplexMat<6, 7>;
using complex_mat68 = ComplexMat<6, 8>;
using complex_mat69 = ComplexMat<6, 9>;
using complex_mat72 = ComplexMat<7, 2>;
using complex_mat73 = ComplexMat<7, 3>;
using complex_mat74 = ComplexMat<7, 4>;
using complex_mat75 = ComplexMat<7, 5>;
using complex_mat76 = ComplexMat<7, 6>;
using complex_mat77 = ComplexMat<7, 7>;
using complex_mat78 = ComplexMat<7, 8>;
using complex_mat79 = ComplexMat<7, 9>;
using complex_mat82 = ComplexMat<8, 2>;
using complex_mat83 = ComplexMat<8, 3>;
using complex_mat84 = ComplexMat<8, 4>;
using complex_mat85 = ComplexMat<8, 5>;
using complex_mat86 = ComplexMat<8, 6>;
using complex_mat87 = ComplexMat<8, 7>;
using complex_mat88 = ComplexMat<8, 8>;
using complex_mat89 = ComplexMat<8, 9>;
using complex_mat92 = ComplexMat<9, 2>;
using complex_mat93 = ComplexMat<9, 3>;
using complex_mat94 = ComplexMat<9, 4>;
using complex_mat95 = ComplexMat<9, 5>;
using complex_mat96 = ComplexMat<9, 6>;
using complex_mat97 = ComplexMat<9, 7>;
using complex_mat98 = ComplexMat<9, 8>;
using complex_mat99 = ComplexMat<9, 9>;
using complex_mat2  = complex_mat22;
using complex_mat3  = complex_mat33;
using complex_mat4  = complex_mat44;
using complex_mat5  = complex_mat55;
using complex_mat6  = complex_mat66;
using complex_mat7  = complex_mat77;
using complex_mat8  = complex_mat88;
using complex_mat9  = complex_mat99;

using complex_mat22f = ComplexMatF<2, 2>;
using complex_mat23f = ComplexMatF<2, 3>;
using complex_mat24f = ComplexMatF<2, 4>;
using complex_mat25f = ComplexMatF<2, 5>;
using complex_mat26f = ComplexMatF<2, 6>;
using complex_mat27f = ComplexMatF<2, 7>;
using complex_mat28f = ComplexMatF<2, 8>;
using complex_mat29f = ComplexMatF<2, 9>;
using complex_mat32f = ComplexMatF<3, 2>;
using complex_mat33f = ComplexMatF<3, 3>;
using complex_mat34f = ComplexMatF<3, 4>;
using complex_mat35f = ComplexMatF<3, 5>;
using complex_mat36f = ComplexMatF<3, 6>;
using complex_mat37f = ComplexMatF<3, 7>;
using complex_mat38f = ComplexMatF<3, 8>;
using complex_mat39f = ComplexMatF<3, 9>;
using complex_mat42f = ComplexMatF<4, 2>;
using complex_mat43f = ComplexMatF<4, 3>;
using complex_mat44f = ComplexMatF<4, 4>;
using complex_mat45f = ComplexMatF<4, 5>;
using complex_mat46f = ComplexMatF<4, 6>;
using complex_mat47f = ComplexMatF<4, 7>;
using complex_mat48f = ComplexMatF<4, 8>;
using complex_mat49f = ComplexMatF<4, 9>;
using complex_mat52f = ComplexMatF<5, 2>;
using complex_mat53f = ComplexMatF<5, 3>;
using complex_mat54f = ComplexMatF<5, 4>;
using complex_mat55f = ComplexMatF<5, 5>;
using complex_mat56f = ComplexMatF<5, 6>;
using complex_mat57f = ComplexMatF<5, 7>;
using complex_mat58f = ComplexMatF<5, 8>;
using complex_mat59f = ComplexMatF<5, 9>;
using complex_mat62f = ComplexMatF<6, 2>;
using complex_mat63f = ComplexMatF<6, 3>;
using complex_mat64f = ComplexMatF<6, 4>;
using complex_mat65f = ComplexMatF<6, 5>;
using complex_mat66f = ComplexMatF<6, 6>;
using complex_mat67f = ComplexMatF<6, 7>;
using complex_mat68f = ComplexMatF<6, 8>;
using complex_mat69f = ComplexMatF<6, 9>;
using complex_mat72f = ComplexMatF<7, 2>;
using complex_mat73f = ComplexMatF<7, 3>;
using complex_mat74f = ComplexMatF<7, 4>;
using complex_mat75f = ComplexMatF<7, 5>;
using complex_mat76f = ComplexMatF<7, 6>;
using complex_mat77f = ComplexMatF<7, 7>;
using complex_mat78f = ComplexMatF<7, 8>;
using complex_mat79f = ComplexMatF<7, 9>;
using complex_mat82f = ComplexMatF<8, 2>;
using complex_mat83f = ComplexMatF<8, 3>;
using complex_mat84f = ComplexMatF<8, 4>;
using complex_mat85f = ComplexMatF<8, 5>;
using complex_mat86f = ComplexMatF<8, 6>;
using complex_mat87f = ComplexMatF<8, 7>;
using complex_mat88f = ComplexMatF<8, 8>;
using complex_mat89f = ComplexMatF<8, 9>;
using complex_mat92f = ComplexMatF<9, 2>;
using complex_mat93f = ComplexMatF<9, 3>;
using complex_mat94f = ComplexMatF<9, 4>;
using complex_mat95f = ComplexMatF<9, 5>;
using complex_mat96f = ComplexMatF<9, 6>;
using complex_mat97f = ComplexMatF<9, 7>;
using complex_mat98f = ComplexMatF<9, 8>;
using complex_mat99f = ComplexMatF<9, 9>;
using complex_mat2f  = complex_mat22f;
using complex_mat3f  = complex_mat33f;
using complex_mat4f  = complex_mat44f;
using complex_mat5f  = complex_mat55f;
using complex_mat6f  = complex_mat66f;
using complex_mat7f  = complex_mat77f;
using complex_mat8f  = complex_mat88f;
using complex_mat9f  = complex_mat99f;

using complex_mat22d = ComplexMatD<2, 2>;
using complex_mat23d = ComplexMatD<2, 3>;
using complex_mat24d = ComplexMatD<2, 4>;
using complex_mat25d = ComplexMatD<2, 5>;
using complex_mat26d = ComplexMatD<2, 6>;
using complex_mat27d = ComplexMatD<2, 7>;
using complex_mat28d = ComplexMatD<2, 8>;
using complex_mat29d = ComplexMatD<2, 9>;
using complex_mat32d = ComplexMatD<3, 2>;
using complex_mat33d = ComplexMatD<3, 3>;
using complex_mat34d = ComplexMatD<3, 4>;
using complex_mat35d = ComplexMatD<3, 5>;
using complex_mat36d = ComplexMatD<3, 6>;
using complex_mat37d = ComplexMatD<3, 7>;
using complex_mat38d = ComplexMatD<3, 8>;
using complex_mat39d = ComplexMatD<3, 9>;
using complex_mat42d = ComplexMatD<4, 2>;
using complex_mat43d = ComplexMatD<4, 3>;
using complex_mat44d = ComplexMatD<4, 4>;
using complex_mat45d = ComplexMatD<4, 5>;
using complex_mat46d = ComplexMatD<4, 6>;
using complex_mat47d = ComplexMatD<4, 7>;
using complex_mat48d = ComplexMatD<4, 8>;
using complex_mat49d = ComplexMatD<4, 9>;
using complex_mat52d = ComplexMatD<5, 2>;
using complex_mat53d = ComplexMatD<5, 3>;
using complex_mat54d = ComplexMatD<5, 4>;
using complex_mat55d = ComplexMatD<5, 5>;
using complex_mat56d = ComplexMatD<5, 6>;
using complex_mat57d = ComplexMatD<5, 7>;
using complex_mat58d = ComplexMatD<5, 8>;
using complex_mat59d = ComplexMatD<5, 9>;
using complex_mat62d = ComplexMatD<6, 2>;
using complex_mat63d = ComplexMatD<6, 3>;
using complex_mat64d = ComplexMatD<6, 4>;
using complex_mat65d = ComplexMatD<6, 5>;
using complex_mat66d = ComplexMatD<6, 6>;
using complex_mat67d = ComplexMatD<6, 7>;
using complex_mat68d = ComplexMatD<6, 8>;
using complex_mat69d = ComplexMatD<6, 9>;
using complex_mat72d = ComplexMatD<7, 2>;
using complex_mat73d = ComplexMatD<7, 3>;
using complex_mat74d = ComplexMatD<7, 4>;
using complex_mat75d = ComplexMatD<7, 5>;
using complex_mat76d = ComplexMatD<7, 6>;
using complex_mat77d = ComplexMatD<7, 7>;
using complex_mat78d = ComplexMatD<7, 8>;
using complex_mat79d = ComplexMatD<7, 9>;
using complex_mat82d = ComplexMatD<8, 2>;
using complex_mat83d = ComplexMatD<8, 3>;
using complex_mat84d = ComplexMatD<8, 4>;
using complex_mat85d = ComplexMatD<8, 5>;
using complex_mat86d = ComplexMatD<8, 6>;
using complex_mat87d = ComplexMatD<8, 7>;
using complex_mat88d = ComplexMatD<8, 8>;
using complex_mat89d = ComplexMatD<8, 9>;
using complex_mat92d = ComplexMatD<9, 2>;
using complex_mat93d = ComplexMatD<9, 3>;
using complex_mat94d = ComplexMatD<9, 4>;
using complex_mat95d = ComplexMatD<9, 5>;
using complex_mat96d = ComplexMatD<9, 6>;
using complex_mat97d = ComplexMatD<9, 7>;
using complex_mat98d = ComplexMatD<9, 8>;
using complex_mat99d = ComplexMatD<9, 9>;
using complex_mat2d  = complex_mat22d;
using complex_mat3d  = complex_mat33d;
using complex_mat4d  = complex_mat44d;
using complex_mat5d  = complex_mat55d;
using complex_mat6d  = complex_mat66d;
using complex_mat7d  = complex_mat77d;
using complex_mat8d  = complex_mat88d;
using complex_mat9d  = complex_mat99d;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
