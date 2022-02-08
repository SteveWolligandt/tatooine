#ifndef TATOOINE_FLOWMAP_GRADIENT_CENTRAL_DIFFERENCES_H
#define TATOOINE_FLOWMAP_GRADIENT_CENTRAL_DIFFERENCES_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <flowmap_c Flowmap>
struct flowmap_gradient_central_differences {
  //============================================================================
 public:
  using flowmap_t = std::decay_t<Flowmap>;
  using this_type    = flowmap_gradient_central_differences<flowmap_t>;
  using real_type    = typename flowmap_t::real_type;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using vec_t      = typename flowmap_t::vec_t;
  using pos_type      = typename flowmap_t::pos_type;
  using mat_t      = mat<real_type, num_dimensions(), num_dimensions()>;
  using gradient_t = mat_t;

  //============================================================================
 private:
  Flowmap m_flowmap;
  vec_t   m_epsilon;

  //============================================================================
 public:
  template <fixed_dims_flowmap_c<num_dimensions()> _Flowmap>
  flowmap_gradient_central_differences(_Flowmap flowmap, real_type const epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_epsilon{tag::fill{epsilon}} {}
  //----------------------------------------------------------------------------
  template <fixed_dims_flowmap_c<num_dimensions()> _Flowmap>
  flowmap_gradient_central_differences(_Flowmap flowmap, vec_t const& epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)}, m_epsilon{epsilon} {}
  //============================================================================
  auto evaluate(pos_type const& y0, real_type const t0, real_type const tau) const {
    gradient_t derivative;

    auto offset = pos_type::zeros();
    for (size_t i = 0; i < num_dimensions(); ++i) {
      offset(i)     = m_epsilon(i);
      auto const dx = 2 * m_epsilon(i);
      derivative.col(i) =
          (m_flowmap(y0 + offset, t0, tau) - m_flowmap(y0 - offset, t0, tau)) /
          dx;
      offset(i) = 0;
    }
    return derivative;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(pos_type const& y0, real_type const t0, real_type const tau) const {
    return evaluate(y0, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto epsilon() const -> const auto& { return m_epsilon; }
  auto epsilon() -> auto& { return m_epsilon; }
  auto epsilon(size_t i) const { return m_epsilon(i); }
  auto epsilon(size_t i) -> auto& { return m_epsilon(i); }
  void set_epsilon(const vec_t& epsilon) { m_epsilon = epsilon; }
  void set_epsilon(vec_t&& epsilon) { m_epsilon = std::move(epsilon); }
  auto flowmap() const -> auto const& { return m_flowmap; }
  auto flowmap() -> auto& { return m_flowmap; }
};
// copy when having rvalue
template <flowmap_c Flowmap>
flowmap_gradient_central_differences(Flowmap &&)
    -> flowmap_gradient_central_differences<Flowmap>;

// keep reference when having lvalue
template <flowmap_c Flowmap>
flowmap_gradient_central_differences(Flowmap const&)
    -> flowmap_gradient_central_differences<const Flowmap&>;
//==============================================================================
template <flowmap_c Flowmap>
constexpr bool is_flowmap_gradient_central_differences(
    const flowmap_gradient_central_differences<Flowmap>&) {
  return true;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename AnyOther>
constexpr bool is_flowmap_gradient_central_differences(AnyOther&&) {
  return false;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
