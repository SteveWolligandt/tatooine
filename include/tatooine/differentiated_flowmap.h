#ifndef TATOOINE_DIFFERENTIATED_FLOWMAP_H
#define TATOOINE_DIFFERENTIATED_FLOWMAP_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Flowmap>
struct differentiated_flowmap {
  //============================================================================
 public:
  using flowmap_type = std::decay_t<Flowmap>;
  using this_type    = differentiated_flowmap<flowmap_type>;
  using real_type    = typename flowmap_type::real_type;
  static constexpr auto num_dimensions() {
    return flowmap_type::num_dimensions();
  }
  using vec_type      = typename flowmap_type::vec_type;
  using pos_type      = typename flowmap_type::pos_type;
  using mat_type      = mat<real_type, num_dimensions(), num_dimensions()>;
  using gradient_type = mat_type;

  //============================================================================
 private:
  flowmap_type m_flowmap;
  vec_type     m_epsilon;

  //============================================================================
 public:
  template <typename _Flowmap>
  differentiated_flowmap(_Flowmap flowmap, real_type const epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_epsilon{tag::fill{epsilon}} {}
  //----------------------------------------------------------------------------
  template <typename _Flowmap>
  differentiated_flowmap(_Flowmap flowmap, vec_type const& epsilon)
      : m_flowmap{std::forward<_Flowmap>(flowmap)}, m_epsilon{epsilon} {}
  //============================================================================
  auto evaluate(pos_type const& y0, real_type const t0,
                real_type const tau) const {
    auto derivative = gradient_type{};

    auto offset = pos_type::zeros();
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
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
  auto operator()(pos_type const& y0, real_type const t0,
                  real_type const tau) const {
    return evaluate(y0, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto epsilon() const -> const auto& { return m_epsilon; }
  auto epsilon() -> auto& { return m_epsilon; }
  auto epsilon(std::size_t i) const { return m_epsilon(i); }
  auto epsilon(std::size_t i) -> auto& { return m_epsilon(i); }
  void set_epsilon(const vec_type& epsilon) { m_epsilon = epsilon; }
  void set_epsilon(vec_type&& epsilon) { m_epsilon = std::move(epsilon); }
  auto flowmap() const -> auto const& { return m_flowmap; }
  auto flowmap() -> auto& { return m_flowmap; }
};
// copy when having rvalue
template <typename Flowmap>
differentiated_flowmap(Flowmap&&) -> differentiated_flowmap<Flowmap>;

// keep reference when having lvalue
template <typename Flowmap>
differentiated_flowmap(Flowmap const&)
    -> differentiated_flowmap<const Flowmap&>;
//==============================================================================
template <typename Flowmap>
constexpr bool is_differentiated_flowmap(
    const differentiated_flowmap<Flowmap>&) {
  return true;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename AnyOther>
constexpr bool is_differentiated_flowmap(AnyOther&&) {
  return false;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
