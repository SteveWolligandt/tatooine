#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include "tensor.h"
#include "diff.h"
#include "flowmap.h"
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

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  pos_t  m_x0, m_x1;
  real_t m_t1;
  mat_t  m_flowmap_gradient;
  size_t m_level;
  real_t m_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename RealX0, typename RealT0, typename RealRadius,
            enable_if_arithmetic<RealX0, RealT0, RealRadius> = true>
  autonomous_particle(vec<RealX0, N> const& x0, RealT0 const t0,
                      RealRadius const radius)
      : m_x0{x0},
        m_x1{x0},
        m_t1{static_cast<real_t>(t0)},
        m_flowmap_gradient{mat_t::eye()},
        m_level{0},
        m_radius{radius} {}

 private:
  autonomous_particle(vec<real_t, N> const& x0, vec<real_t, N> const& x1,
                      real_t const t1, mat_t&& flowmap_gradient,
                      real_t const level, real_t const radius)
      : m_x0{x0},
        m_x1{x1},
        m_t1{t1},
        m_flowmap_gradient{std::move(flowmap_gradient)},
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
  template <typename V, typename VReal,
            template <typename, size_t> typename Integrator>
  auto evolve(field<V, VReal, N> const&         v,
              Integrator<Real, N> const& integrator,
              real_t const                      tau) const {
    
    flowmap fm{v, integrator, tau};
    auto    fmgrad = diff(fm, 1e-6);

    this_t{x0, m_t1 + tau, fm(m_x0, m_t1),
           m_flowmap_gradient * fmgrad(m_x0, m_t1), m_level + 1, m_radius};
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal,
            template <typename, size_t> typename Integrator>
  auto split(field<V, VReal, N> const&         v,
              Integrator<Real, N> const& integrator, real_t const tau) const {
    flowmap fm{v, integrator, tau};
    auto    fmgrad = diff(fm, 1e-6);

    this_t {x0, m_t1 + tau, fm(m_x0, m_t1),
           m_flowmap_gradient * fmgrad(m_x0, m_t1), m_level + 1, m_radius};
    //return std::array<this_t, 3>{left, center, right};
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename RealX0, size_t N, typename RealT0, typename RealRadius>
autonomous_particle(vec<RealX0, N> const& x0, RealT0 const t0,
                    RealRadius const radius)
    -> autonomous_particle<promote_t<RealX0, RealT0, RealRadius>, N>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
