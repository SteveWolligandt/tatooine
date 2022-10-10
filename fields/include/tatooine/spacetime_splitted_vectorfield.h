#ifndef TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
#define TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/concepts.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V>
struct spacetime_splitted_vectorfield
    : vectorfield<spacetime_splitted_vectorfield<V>,
                  typename std::remove_pointer_t<std::decay_t<V>>::real_type,
                  std::remove_pointer_t<std::decay_t<V>>::num_dimensions() -
                      1> {
  using this_type = spacetime_splitted_vectorfield<V>;
  using parent_type =
      vectorfield<this_type,
                  typename std::remove_pointer_t<std::decay_t<V>>::real_type,
                  std::remove_pointer_t<std::decay_t<V>>::num_dimensions() - 1>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //============================================================================
  V m_v;
  //============================================================================
  constexpr auto internal_field() const -> auto const& {
    if constexpr (std::is_pointer_v<std::decay_t<V>>) {
      return *m_v;
    } else {
      return m_v;
    };
  }
  //----------------------------------------------------------------------------
  template <typename W>
  auto set_field(vectorfield<W, real_type, num_dimensions() + 1> const& v) 
  requires std::is_pointer_v<V> {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_field(
      polymorphic::vectorfield<real_type, num_dimensions() + 1> const& v)
  requires std::is_pointer_v<V> {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_field(
      polymorphic::vectorfield<real_type, num_dimensions() + 1> const* v)
  requires std::is_pointer_v<V> {
    m_v = v;
  }
  //============================================================================
  constexpr explicit spacetime_splitted_vectorfield(convertible_to<V> auto&& v)
      : m_v{std::forward<decltype(v)>(v)} {}
  //----------------------------------------------------------------------------
  constexpr spacetime_splitted_vectorfield(spacetime_splitted_vectorfield const&) =
      default;
  constexpr spacetime_splitted_vectorfield(spacetime_splitted_vectorfield&&) noexcept =
      default;
  //----------------------------------------------------------------------------
  ~spacetime_splitted_vectorfield() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type {
    auto spatio_temporal_position = vec<real_type, num_dimensions() + 1>{};
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      spatio_temporal_position(i) = x(i);
    }
    spatio_temporal_position(num_dimensions()) = t;
    auto vt = internal_field()(spatio_temporal_position, t);
    if constexpr (!same_as<decltype(vt), tensor_type>) {
      auto v = tensor_type{};
      for (std::size_t i = 0; i < num_dimensions(); ++i) {
        v(i) = vt(i);
      }
      return v;
    } else {
      return vt;
    }
  }
};
template <typename V>
spacetime_splitted_vectorfield(V&& v) -> spacetime_splitted_vectorfield<V&&>;
template <typename V>
spacetime_splitted_vectorfield(V const& v)
    -> spacetime_splitted_vectorfield<V const&>;
template <typename V>
spacetime_splitted_vectorfield(V& v) -> spacetime_splitted_vectorfield<V&>;
template <typename V>
spacetime_splitted_vectorfield(V* v) -> spacetime_splitted_vectorfield<V*>;
template <typename V>
spacetime_splitted_vectorfield(V const* v)
    -> spacetime_splitted_vectorfield<V const*>;
//==============================================================================
template <typename V, typename VReal, std::size_t N, std::size_t NV>
auto split_spacetime(vectorfield<V, VReal, N, NV> const& v) {
  return spacetime_splitted_vectorfield<V const&>{v.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, std::size_t N, std::size_t NV>
auto split_spacetime(vectorfield<V, VReal, N, NV>& v) {
  return spacetime_splitted_vectorfield<V&>{v.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, std::size_t N, std::size_t NV>
auto split_spacetime(vectorfield<V, VReal, N, NV>&& v) {
  return spacetime_splitted_vectorfield<V>{std::move(v.as_derived())};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename VReal, std::size_t N, std::size_t NV>
auto split_spacetime(polymorphic::vectorfield<VReal, N, NV>* v) {
  return spacetime_splitted_vectorfield<polymorphic::vectorfield<VReal, N, NV>*>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename VReal, std::size_t N, std::size_t NV>
auto split_spacetime(polymorphic::vectorfield<VReal, N, NV> const* v) {
  return spacetime_splitted_vectorfield<
      polymorphic::vectorfield<VReal, N, NV> const*>{v};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
