#ifndef TATOOINE_FLOWMAP_H
#define TATOOINE_FLOWMAP_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V,
          template <typename, size_t, template <typename> typename>
          typename Integrator,
          template <typename> typename InterpolationKernel>
struct flowmap
    : tatooine::vectorfield<flowmap<V, Integrator, InterpolationKernel>,
                            typename V::real_t, V::num_dimensions()> {
  using this_t = flowmap<V, Integrator, InterpolationKernel>;
  using parent_t =
      tatooine::vectorfield<this_t, typename V::real_t, V::num_dimensions()>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using parent_t::operator();
  using integrator_t =
      Integrator<real_t, parent_t::num_dimensions(), InterpolationKernel>;
  //============================================================================
 private:
  V m_vectorfield;
  std::shared_ptr<integrator_t> m_integrator;
  real_t m_tau;
  //============================================================================
 public:
  template <typename FieldReal, typename TauReal, size_t N>
  constexpr flowmap(const vectorfield<V, FieldReal, N>& vf,
                    const integrator_t& integrator, TauReal tau)
      : m_vectorfield{vf.as_derived()},
        m_integrator{std::make_shared<integrator_t>(integrator)},
        m_tau{static_cast<real_t>(tau)} {}
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, real_t t0, real_t tau) const {
    return m_integrator->integrate(m_vectorfield, x, t0, tau)(tau);
  }
  //----------------------------------------------------------------------------
  constexpr tensor_t operator()(const pos_t& x, real_t t0, real_t tau) const {
    return evaluate(x, t0, tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& x, real_t t) const
      -> tensor_t final {
    if (m_tau == 0) { return x; }
    const auto& integral = m_integrator->integrate(m_vectorfield, x, t, m_tau);
    if (integral.empty()) { return x; }
    return integral(t + m_tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& x, real_t t) const
      -> bool final {
    return m_vectorfield.in_domain(x, t);
  }
  //============================================================================
  auto tau() const { return m_tau; }
  //----------------------------------------------------------------------------
  void set_tau(const real_t tau) { m_tau = tau; }
  //----------------------------------------------------------------------------
  auto vectorfield() const -> const auto& { return m_vectorfield; }
  auto vectorfield()       -> auto&       { return m_vectorfield; }
  //----------------------------------------------------------------------------
  auto integrator() const -> const auto& { return *m_integrator; }
  auto integrator()       ->       auto& { return *m_integrator; }
  //----------------------------------------------------------------------------
  auto shared_integrator() const -> const auto& { return m_integrator; }
  auto shared_integrator()       ->       auto& { return m_integrator; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
