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
template <arithmetic_or_complex ValueType, std::size_t N>
struct vec : tensor<ValueType, N> {
  using this_type   = vec<ValueType, N>;
  using parent_type = tensor<ValueType, N>;
  using parent_type::at;
  using parent_type::dimension;
  using parent_type::parent_type;
  using parent_type::operator();
  //============================================================================
  using iterator =
      typename parent_type::array_parent_type::container_t::iterator;
  //============================================================================
  using const_iterator =
      typename parent_type::array_parent_type::container_t::const_iterator;
  //============================================================================
  static auto constexpr zeros() { return this_type{tag::fill<ValueType>{0}}; }
  //----------------------------------------------------------------------------
  static auto constexpr ones() { return this_type{tag::fill<ValueType>{1}}; }
  //----------------------------------------------------------------------------
  static auto constexpr fill(ValueType const& t) {
    return this_type{tag::fill<ValueType>{t}};
  }
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
  constexpr vec(vec const&)           = default;
  constexpr vec(vec&& other) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr vec(convertible_to<ValueType> auto&&... ts) requires(
      parent_type::dimension(0) == sizeof...(ts))
      : parent_type{std::forward<decltype(ts)>(ts)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr vec(base_tensor<OtherTensor, OtherReal, N> const& other)
      : parent_type{other} {}
  //----------------------------------------------------------------------------
  auto constexpr operator=(vec const&) -> vec& = default;
  auto constexpr operator=(vec&& other) noexcept -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr operator=(static_tensor auto&& other) -> vec& {
    this->assign(std::forward<decltype(other)>(other));
    return *this;
  }
  //----------------------------------------------------------------------------
  ~vec() = default;
  //----------------------------------------------------------------------------
  auto begin() const { return this->internal_container().begin(); }
  auto begin() { return this->internal_container().begin(); }
  auto end() const { return this->internal_container().end(); }
  auto end() { return this->internal_container().end(); }
  auto size() const { return parent_type::size().front(); }
  //----------------------------------------------------------------------------
  auto constexpr x() const -> auto const& requires(N >= 1) { return at(0); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr x() -> auto& requires(N >= 1) { return at(0); }
  //----------------------------------------------------------------------------
  auto constexpr y() const -> auto const& requires(N >= 2) { return at(1); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr xy() const requires(N >= 2) {
    return vec<ValueType, 2>{at(0), at(1)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr yx() const requires(N >= 2) {
    return vec<ValueType, 2>{at(1), at(0)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr y() -> auto& requires(N >= 2) { return at(1); }
  //----------------------------------------------------------------------------
  auto constexpr xyz() const requires(N >= 3) {
    return vec<ValueType, 3>{at(0), at(1), at(2)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr z() const -> auto const& requires(N >= 3) { return at(2); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr z() -> auto& requires(N >= 3) { return at(2); }
  //----------------------------------------------------------------------------
  auto constexpr xyzw() const requires(N >= 3) {
    return vec<ValueType, 4>{at(0), at(1), at(2), at(3)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr w() const -> auto const& requires(N >= 4) { return at(3); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr w() -> auto& requires(N >= 4) { return at(3); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Archive>
  auto serialize(Archive& ar, unsigned int const /*version*/) -> void {
    for (std::size_t i = 0; i < N; ++i) {
      ar& at(i);
    }
  }
};
//==============================================================================
// type traits
//==============================================================================
template <typename ValueType, std::size_t N>
auto begin(vec<ValueType, N> const& v) {
  return v.begin();
}
//------------------------------------------------------------------------------
template <typename ValueType, std::size_t N>
auto end(vec<ValueType, N> const& v) {
  return v.end();
}
//------------------------------------------------------------------------------
template <typename ValueType, std::size_t N>
auto size(vec<ValueType, N> const& v) {
  return v.size();
}
//==============================================================================
template <typename... Ts>
vec(const Ts&...) -> vec<common_type<Ts...>, sizeof...(Ts)>;
//------------------------------------------------------------------------------
template <typename V, typename ValueType, std::size_t N>
vec(base_tensor<V, ValueType, N> const&) -> vec<ValueType, N>;
//==============================================================================
namespace reflection {
template <typename ValueType, std::size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (vec<ValueType, N>), TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/vec_typedefs.h>
//==============================================================================
#endif
