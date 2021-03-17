#ifndef TATOOINE_VEC_H
#define TATOOINE_VEC_H
//==============================================================================
#include <tatooine/real.h>
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t N>
struct vec : tensor<T, N> {  // NOLINT
  static_assert(is_arithmetic<T> || is_complex<T>);
  using this_t = vec<T, N>;
  using parent_t = tensor<T, N>;
  using parent_t::parent_t;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::operator();

  static constexpr auto num_dimensions() { return N; }
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
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal<T>{eng, mean, stddev}};
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
template <typename T>
using Vec2 = vec<T, 2>;
template <typename T>
using Vec3 = vec<T, 3>;
template <typename T>
using Vec4 = vec<T, 4>;

using vec2f = Vec2<float>;
using vec3f = Vec3<float>;
using vec4f = Vec4<float>;

using vec2d = Vec2<double>;
using vec3d = Vec3<double>;
using vec4d = Vec4<double>;

using vec2 = Vec2<real_t>;
using vec3 = Vec3<real_t>;
using vec4 = Vec4<real_t>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
