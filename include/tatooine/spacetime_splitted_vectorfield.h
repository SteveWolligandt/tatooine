#ifndef TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
#define TATOOINE_SPACETIME_SPLITTED_VECTORFIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V>
struct spacetime_splitted_vectorfield
    : vectorfield<spacetime_splitted_vectorfield<V>,
                  typename std::remove_pointer_t<std::decay_t<V>>::real_t,
                  std::remove_pointer_t<std::decay_t<V>>::num_dimensions() -
                      1> {
  using this_t = spacetime_splitted_vectorfield<V>;
  using parent_t =
      vectorfield<this_t,
                  typename std::remove_pointer_t<std::decay_t<V>>::real_t,
                  std::remove_pointer_t<std::decay_t<V>>::num_dimensions() - 1>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  //============================================================================
  V m_v;
  //============================================================================
  auto internal_field() const -> auto const& {
    if constexpr (std::is_pointer_v<std::decay_t<V>>) {
      return *m_v;
    } else {
      return m_v;
    };
  }
  //----------------------------------------------------------------------------
  template <typename W>
  requires std::is_pointer_v<V>
  void set_field(vectorfield<W, real_t, num_dimensions() + 1> const& v) {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires std::is_pointer_v<V>
  void set_field(polymorphic::vectorfield<real_t, num_dimensions() + 1> const& v) {
    m_v = &v;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires std::is_pointer_v<V>
  void set_field(polymorphic::vectorfield<real_t, num_dimensions() + 1> const* v) {
    m_v = v;
  }
  //============================================================================
  template <typename V_>
  explicit spacetime_splitted_vectorfield(V_&& v) : m_v{std::forward<V_>(v)} {}
  //----------------------------------------------------------------------------
  spacetime_splitted_vectorfield(spacetime_splitted_vectorfield const&) =
      default;
  spacetime_splitted_vectorfield(spacetime_splitted_vectorfield&&) noexcept =
      default;
  //----------------------------------------------------------------------------
  ~spacetime_splitted_vectorfield() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    auto const vt        = internal_field()(pt, t);
    tensor_t   v;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      v(i) = vt(i);
    }
    return v;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t t) const -> bool final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    return internal_field().in_domain(pt, t);
  }
};
//==============================================================================
template <typename V, typename VReal, size_t N>
auto split_spacetime(vectorfield<V, VReal, N> const& v) {
  return spacetime_splitted_vectorfield<V const&>{v.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, size_t N>
auto split_spacetime(vectorfield<V, VReal, N>& v) {
  return spacetime_splitted_vectorfield<V&>{v.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, size_t N>
auto split_spacetime(vectorfield<V, VReal, N>&& v) {
  return spacetime_splitted_vectorfield<V>{std::move(v.as_derived())};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename VReal, size_t N>
auto split_spacetime(polymorphic::vectorfield<VReal, N>* v) {
  return spacetime_splitted_vectorfield<polymorphic::vectorfield<VReal, N>*>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename VReal, size_t N>
auto split_spacetime(polymorphic::vectorfield<VReal, N> const* v) {
  return spacetime_splitted_vectorfield<polymorphic::vectorfield<VReal, N> const*>{
      v};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
