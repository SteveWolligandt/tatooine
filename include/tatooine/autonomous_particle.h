#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/diff.h>
#include <tatooine/flowmap.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/spacetime_field.h>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using real_t = Real;
  using vec_t  = vec<Real, N>;
  using mat_t  = mat<Real, N, N>;
  using pos_t  = vec_t;
  using this_t  = autonomous_particle<Real, N>;

  static constexpr real_t max_cond = 10;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 //private:
  pos_t  m_x0, m_x1, m_fromx;
  real_t m_t1, m_fromt;
  mat_t  m_flowmap_gradient;
  size_t m_level;
  real_t m_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename RealX0, typename RealTS, typename RealRadius,
            enable_if_arithmetic<RealX0, RealTS, RealRadius> = true>
  autonomous_particle(vec<RealX0, N> const& x0, RealTS const ts,
                      RealRadius const radius)
      : m_x0{x0},
        m_x1{x0},
        m_fromx{x0},
        m_t1{static_cast<real_t>(ts)},
        m_fromt{ts},
        m_flowmap_gradient{mat_t::eye()},
        m_level{0},
        m_radius{radius} {}

  autonomous_particle(pos_t const& x0, pos_t const& x1, pos_t const& from,
                      real_t const t1, real_t const fromt, const mat_t& flowmap_gradient,
                      size_t const level, real_t const radius)
      : m_x0{x0},
        m_x1{x1},
        m_fromx{from},
        m_t1{t1},
        m_fromt{fromt},
        m_flowmap_gradient{flowmap_gradient},
        m_level{level},
        m_radius{radius} {}
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(const autonomous_particle&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const autonomous_particle&)
    -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&&) noexcept
    -> autonomous_particle& = default;
  
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::hermite,
            typename V>
  auto integrate(vectorfield<V, Real, N> const& v, real_t tau_step,
             real_t const max_t) const {
    std::vector<this_t> particles{*this};

    size_t n = 0;
    size_t start_idx = 0;
    do {
      n = size(particles);
      auto const cur_size = size(particles);
      for (size_t i = start_idx; i < cur_size; ++i) {
        particles[i].integrate_until_split(v, tau_step, particles, max_t);
      }
      start_idx = cur_size;
    } while (n != size(particles));
    return particles;
  }
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::hermite,
            typename V>
  void integrate_until_split(vectorfield<V, Real, N> const& v, real_t tau_step,
                             std::vector<this_t>& particles,
                             real_t const         max_t) const {
    if (m_t1 >= max_t) {
      return;
    }
    using integrator_t = Integrator<Real, N, InterpolationKernel>;
    static real_t const twothirds = real_t(2) / real_t(3);
    static real_t const sqrt3     = std::sqrt(real_t(3));

    Real         tau          = 0;
    auto         fmgrad_field = diff(flowmap{v, integrator_t{}, tau}, m_radius);
    auto&        fm           = fmgrad_field.internal_field();
    Real   cond   = 1;
    std::pair<mat<Real, N, N>, vec<Real, N>> eig;
    auto const&                              eigvecs = eig.first;
    auto const&                              eigvals = eig.second;

    typename decltype(fmgrad_field)::tensor_t fmgrad;
    while (cond < 9 && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }
      fm.set_tau(tau);
      fmgrad = fmgrad_field(m_x1, m_t1);
      eig = eigenvectors_sym(fmgrad * transpose(fmgrad));

      cond = eigvals(1) / eigvals(0);
      if (cond > max_cond) {
        cond = 0;
        tau -= tau_step;
        tau_step *= 0.5;
      }
    }
    real_t const r3       = m_radius / real_t(3);
    mat_t const  fmg2fmg1 = fmgrad * m_flowmap_gradient;
    pos_t const  x2       = fm(m_x1, m_t1);
    real_t const t2       = m_t1 + tau;
    if (cond > 9) {
      vec_t const offset2 = twothirds * sqrt3 * eigvecs.col(1);
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      particles.emplace_back(m_x0 - offset0, x2 - offset2, m_x1, t2, m_t1,
                             fmg2fmg1, m_level + 1, r3);
      particles.emplace_back(m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             r3);
      particles.emplace_back(m_x0 + offset0, x2 + offset2, m_x1, t2,m_t1, fmg2fmg1,
                             m_level + 1, r3);
    } else {
      particles.emplace_back(m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             r3);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename RealX0, size_t N, typename RealTS, typename RealRadius>
autonomous_particle(vec<RealX0, N> const& x0, RealTS const ts,
                    RealRadius const radius)
    -> autonomous_particle<promote_t<RealX0, RealTS, RealRadius>, N>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
