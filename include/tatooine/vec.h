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
template <real_or_complex_number T, size_t N>
struct vec : tensor<T, N> {  // NOLINT
  using this_t = vec<T, N>;
  using parent_t = tensor<T, N>;
  using parent_t::parent_t;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::operator();

  template <real_or_complex_number... Ts, size_t _Dim0 = parent_t::dimension(0),
            std::enable_if_t<_Dim0 == sizeof...(Ts), bool> = true>
  constexpr vec(Ts const&... ts) : parent_t{ts...} {}

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
  constexpr vec(const vec&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr vec(vec&& other) noexcept : parent_t{std::move(other)} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(const vec&) -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(vec&& other) noexcept -> vec& {
    parent_t::operator=(std::move(other));
    return *this;
  }
  template <typename OtherTensor, typename OtherReal>
  constexpr vec(const base_tensor<OtherTensor, OtherReal, N>& other) {
    for (size_t i = 0; i < N; ++i) { at(i) = other(i); }
  }
  //----------------------------------------------------------------------------
  ~vec() = default;

  auto begin() const { return this->data().begin(); }
  auto begin() { return this->data().begin(); }
  auto end() const { return this->data().end(); }
  auto end() { return this->data().end(); }
  //----------------------------------------------------------------------------
  template <typename = void> requires (N >= 1)
  auto x() const -> auto const& {return this->at(0);}
  template <typename = void> requires (N >= 1)
  auto x() -> auto& {return this->at(0);}
  //----------------------------------------------------------------------------
  template <typename = void> requires (N >= 2)
  auto y() const -> auto const& {return this->at(1);}
  template <typename = void> requires (N >= 2)
  auto y() -> auto& {return this->at(1);}
  //----------------------------------------------------------------------------
  template <typename = void> requires (N >= 3)
  auto z() const -> auto const& {return this->at(2);}
  template <typename = void> requires (N >= 3)
  auto z() -> auto& {return this->at(2);}
  //----------------------------------------------------------------------------
  template <typename = void> requires (N >= 4)
  auto w() const -> auto const& {return this->at(3);}
  template <typename = void> requires (N >= 4)
  auto w() -> auto& {return this->at(3);}
};
//==============================================================================
// type traits
//==============================================================================
template <real_or_complex_number T, size_t N>
auto begin(vec<T, N> const& v) {return v.begin();}
//------------------------------------------------------------------------------
template <real_or_complex_number T, size_t N>
auto end(vec<T, N> const& v) {return v.ned();}
//==============================================================================
template <typename... Ts>
vec(const Ts&...) -> vec<promote_t<Ts...>, sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_or_complex_number T>
using Vec2 = vec<T, 2>;
template <real_or_complex_number T>
using Vec3 = vec<T, 3>;
template <real_or_complex_number T>
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
