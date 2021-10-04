#ifndef TATOOINE_VEC_H
#define TATOOINE_VEC_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic_or_complex T, size_t N>
#else
template <typename T, size_t N>
#endif
struct vec : tensor<T, N> {  // NOLINT
#ifndef __cpp_concepts
  static_assert(is_arithmetic<T> || is_complex<T>);
#endif
  using this_t = vec<T, N>;
  using parent_t = tensor<T, N>;
  using parent_t::parent_t;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::operator();

  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename... Ts>
      requires((is_convertible<Ts, T> && ...)) &&
      (parent_t::dimension(0) == sizeof...(Ts))
#else
  template <typename... Ts,
            enable_if<(is_convertible<std::decay_t<Ts>, T> && ...)> = true,
            enable_if<parent_t::dimension(0) == sizeof...(Ts)>      = true>
#endif
          constexpr vec(Ts const&... ts)
      : parent_t{ts...} {
  }

  using iterator = typename parent_t::array_parent_t::container_t::iterator;
  using const_iterator =
      typename parent_t::array_parent_t::container_t::const_iterator;
  //============================================================================
  static constexpr auto zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::fill<T>{1}}; }
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
  //----------------------------------------------------------------------------
  constexpr vec(vec const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr vec(vec&& other) noexcept : parent_t{std::move(other)} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(vec const&) -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(vec&& other) noexcept -> vec& {
    parent_t::operator=(std::move(other));
    return *this;
  }
  template <typename OtherTensor, typename OtherReal>
  constexpr vec(base_tensor<OtherTensor, OtherReal, N> const& other) {
    for (size_t i = 0; i < N; ++i) { at(i) = other(i); }
  }
  //----------------------------------------------------------------------------
  ~vec() = default;

  auto begin() const { return this->data().begin(); }
  auto begin() { return this->data().begin(); }
  auto end() const { return this->data().end(); }
  auto end() { return this->data().end(); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 1)
#else
  template <size_t _N = N, enable_if<(_N >= 1)> = true>
#endif
  auto x() const -> auto const& {return this->at(0);}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 1)
#else
  template <size_t _N = N, enable_if<(_N >= 1)> = true>
#endif
  auto x() -> auto& {return this->at(0);}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 2)
#else
  template <size_t _N = N, enable_if<(_N >= 2)> = true>
#endif
  auto y() const -> auto const& {return this->at(1);}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 2)
#else
  template <size_t _N = N, enable_if<(_N >= 2)> = true>
#endif
  auto y() -> auto& {return this->at(1);}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 3)
#else
  template <size_t _N = N, enable_if<(_N >= 3)> = true>
#endif
  auto z() const -> auto const& {return this->at(2);}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 3)
#else
  template <size_t _N = N, enable_if<(_N >= 3)> = true>
#endif
  auto z() -> auto& {return this->at(2);}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 4)
#else
  template <size_t _N = N, enable_if<(_N >= 4)> = true>
#endif
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto w() const -> auto const& {return this->at(3);}
#ifdef __cpp_concepts
  template <typename = void> requires (N >= 4)
#else
  template <size_t _N = N, enable_if<(_N >= 4)> = true>
#endif
  auto w() -> auto& {return this->at(3);}

  template <typename Archive>
  auto serialize(Archive& ar, unsigned int const /*version*/) -> void {
    for (size_t i = 0; i < N; ++i) {
      ar& at(i);
    }
  }
};
//==============================================================================
// type traits
//==============================================================================
template <typename T, size_t N>
auto begin(vec<T, N> const& v) {return v.begin();}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto end(vec<T, N> const& v) {return v.ned();}
//==============================================================================
template <typename... Ts>
vec(const Ts&...) -> vec<common_type<Ts...>, sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t N>
using Vec = vec<real_t, N>;

using vec2 = Vec<2>;
using vec3 = Vec<3>;
using vec4 = Vec<4>;
using vec5 = Vec<5>;
using vec6 = Vec<6>;
using vec7 = Vec<7>;
using vec8 = Vec<8>;
using vec9 = Vec<9>;

template <size_t N>
using VecF = vec<float, N>;
using vec2f = VecF<2>;
using vec3f = VecF<3>;
using vec4f = VecF<4>;
using vec5f = VecF<5>;
using vec6f = VecF<6>;
using vec7f = VecF<7>;
using vec8f = VecF<8>;
using vec9f = VecF<9>;

template <size_t N>
using VecD = vec<double, N>;
using vec2d = VecD<2>;
using vec3d = VecD<3>;
using vec4d = VecD<4>;
using vec5d = VecD<5>;
using vec6d = VecD<6>;
using vec7d = VecD<7>;
using vec8d = VecD<8>;
using vec9d = VecD<9>;

template <size_t N>
using VecI64 = vec<std::int64_t, N>;
using vec2i64 = VecI64<2>;
using vec3i64 = VecI64<3>;
using vec4i64 = VecI64<4>;
using vec5i64 = VecI64<5>;
using vec6i64 = VecI64<6>;
using vec7i64 = VecI64<7>;
using vec8i64 = VecI64<8>;
using vec9i64 = VecI64<9>;

template <typename T, size_t N>
using complex_vec = vec<std::complex<T>, N>;
template <size_t N>
using ComplexVec = vec<std::complex<real_t>, N>;
using complex_vec2 = ComplexVec<2>;
using complex_vec3 = ComplexVec<3>;
using complex_vec4 = ComplexVec<4>;
using complex_vec5 = ComplexVec<5>;
using complex_vec6 = ComplexVec<6>;
using complex_vec7 = ComplexVec<7>;
using complex_vec8 = ComplexVec<8>;
using complex_vec9 = ComplexVec<9>;
template <size_t N>
using ComplexVecD = vec<std::complex<double>, N>;
using complex_vec2d = ComplexVecD<2>;
using complex_vec3d = ComplexVecD<3>;
using complex_vec4d = ComplexVecD<4>;
using complex_vec5d = ComplexVecD<5>;
using complex_vec6d = ComplexVecD<6>;
using complex_vec7d = ComplexVecD<7>;
using complex_vec8d = ComplexVecD<8>;
using complex_vec9d = ComplexVecD<9>;
template <size_t N>
using ComplexVecF   = vec<std::complex<float>, N>;
using complex_vec2f = ComplexVecF<2>;
using complex_vec3f = ComplexVecF<3>;
using complex_vec4f = ComplexVecF<4>;
using complex_vec5f = ComplexVecF<5>;
using complex_vec6f = ComplexVecF<6>;
using complex_vec7f = ComplexVecF<7>;
using complex_vec8f = ComplexVecF<8>;
using complex_vec9f = ComplexVecF<9>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
