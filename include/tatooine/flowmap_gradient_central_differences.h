#ifndef TATOOINE_FLOWMAP_GRADIENT_CENTRAL_DIFFERENCES_H
#define TATOOINE_FLOWMAP_GRADIENT_CENTRAL_DIFFERENCES_H
//==============================================================================
#include "tensor.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Flowmap>
struct flowmap_gradient_central_differences {
  //============================================================================
 public:
  using flowmap_t = std::decay_t<Flowmap>;
  using this_t    = flowmap_gradient_central_differences<flowmap_t>;
  using real_t    = typename flowmap_t::real_t;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using vec_t      = typename flowmap_t::vec_t;
  using pos_t      = typename flowmap_t::pos_t;
  using mat_t      = mat<real_t, num_dimensions(), num_dimensions()>;
  using gradient_t = mat_t;

  //============================================================================
 private:
  Flowmap m_flowmap;
  vec_t   m_epsilon;

  //============================================================================
 public:
  template <typename _Flowmap>
  flowmap_gradient_central_differences(_Flowmap flowmap, real_t const epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_epsilon{tag::fill{epsilon}} {}
  //----------------------------------------------------------------------------
  template <typename _Flowmap>
  flowmap_gradient_central_differences(Flowmap flowmap, vec_t const& epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)}, m_epsilon{epsilon} {}
  //============================================================================
  auto evaluate(pos_t const& y0, real_t const t0, real_t const tau) const {
    gradient_t derivative;

    auto offset = pos_t::zeros();
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
  auto operator()(pos_t const& y0, real_t const t0, real_t const tau) const {
    return evaluate(y0, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto epsilon() const -> const auto& { return m_epsilon; }
  auto epsilon() -> auto& { return m_epsilon; }
  auto epsilon(size_t i) const { return m_epsilon(i); }
  auto epsilon(size_t i) -> auto& { return m_epsilon(i); }
  void set_epsilon(const vec_t& epsilon) { m_epsilon = epsilon; }
  void set_epsilon(vec_t&& epsilon) { m_epsilon = std::move(epsilon); }
};
// copy when having rvalue
template <typename Flowmap>
flowmap_gradient_central_differences(Flowmap &&)
    -> flowmap_gradient_central_differences<Flowmap>;

// keep reference when having lvalue
template <typename Flowmap>
flowmap_gradient_central_differences(Flowmap const&)
    -> flowmap_gradient_central_differences<const Flowmap&>;
//==============================================================================
template <typename Flowmap>
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
