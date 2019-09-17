#ifndef TATOOINE_FLOWMAP_H
#define TATOOINE_FLOWMAP_H

#include "field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, template <typename, size_t> typename Integrator>
struct flowmap : field<flowmap<V, Integrator>, typename V::real_t,
                       V::num_dimensions(), V::num_dimensions()> {
  using this_t   = flowmap<V, Integrator>;
  using parent_t = field<this_t, typename V::real_t, V::num_dimensions(),
                         V::num_dimensions()>;
  using parent_t::num_dimensions;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using parent_t::operator();
  using integrator_t = Integrator<real_t, parent_t::num_dimensions()>;

  //============================================================================
 private:
  V m_vectorfield;
  std::shared_ptr<integrator_t> m_integrator;
  real_t m_tau;

  //============================================================================
 public:
  template <typename FieldReal, typename TauReal, size_t N>
  constexpr flowmap(const field<V, FieldReal, N, N>& vf,
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
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    auto integral =
        m_integrator->integrate_uncached(m_vectorfield, x, t, m_tau);
    if (integral.empty()) {
      std::cerr <<"empty!!! " << x << '\n';
      return x;
    }
    if (m_tau > 0) {
      std::cerr << "back " << integral.back_position() << '\n';
      return integral.back_position();
    }
    std::cerr << integral.front_position() << '\n';
    return integral.front_position();
  }

  //============================================================================
  constexpr decltype(auto) in_domain(const pos_t& x, real_t t) const {
    return m_vectorfield.in_domain(x, t);
  }

  //============================================================================
  auto tau() const { return m_tau; }
  void set_tau(const real_t tau) { m_tau = tau; }

  //----------------------------------------------------------------------------
  const auto& integrator() const { return m_integrator; }
  auto&       integrator() { return m_integrator; }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
