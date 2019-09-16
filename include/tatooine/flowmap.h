#ifndef TATOOINE_FLOWMAP_H
#define TATOOINE_FLOWMAP_H

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, template <typename, size_t> typename Integrator>
struct flowmap : field<flowmap<V, Integrator>, typename V::real_t,
                       V::num_dimensions() + 2, V::num_dimensions()> {
  using this_t   = flowmap<V, Integrator>;
  using parent_t = field<this_t, typename V::real_t, V::num_dimensions() + 2,
                         V::num_dimensions()>;
  using parent_t::num_dimensions();
  static constexpr auto num_spatial_dimensions() {
    return num_dimensions() - 2;
  }
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using integrator_t = Integrator<real_t, num_spatial_dimensions()>;

  //============================================================================
 private:
  V m_vectorfield;
  std::shared_ptr<integrator_t> m_integrator;

  //============================================================================
 public:
  constexpr flowmap(const field<V, real_t, num_spatial_dimensions(),
                                num_spatial_dimensions()>& vf,
                    const integrator_t&                    integrator)
      : m_vectorfield{vf},
        m_integrator{std::make_shared<integrator_t>(integrator)} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const vec<Real, num_spatial_dimensions()>& x,
                              real_t t0, real_t tau) const {
    return m_integrator.integrate(p, t0, tau)(tau);
  }
  //----------------------------------------------------------------------------
  constexpr tensor_t operator()(const vec<Real, num_spatial_dimensions()> & x,
                                real_t t0, real_t tau) const {
    evaluate(x, t0, tau);
  }
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, real_t t = 0) const {
    vec<Real, num_dimensions() - 2> p;
    for (size_t i = 0; i < num_dimensions() - 2; ++i) { p(i) = x(i); }
    Real t0 = x(num_dimensions() - 2), tau(num_dimensions() - 1);
    return m_integrator.integrate(p, t0, tau)(tau);
  }

  ////----------------------------------------------------------------------------
  //constexpr bool in_domain(const pos_t& x, real_t) const {
  //  m_vectorfield.in_domain();
  //}
};

flowmap()->flowmap<double>;

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
