#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/diff.h>
#include <tatooine/flowmap.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
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
  real_t m_t0, m_t1;
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
        m_t0{static_cast<real_t>(t0)},
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
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::hermite,
            typename V>
  auto split(vectorfield<V, Real, N> const& v, real_t tau_step) const {
    using integrator_t = Integrator<Real, N, InterpolationKernel>;

    Real         tau          = 0;
    auto         fmgrad_field = diff(flowmap{v, integrator_t{}, tau}, m_radius);
    auto&        fm           = fmgrad_field.internal_field();
    Real   cond   = 0;
    std::pair<mat<Real, N, N>, vec<Real, N>> eig;
    auto& eigvecs = eig.first;
    auto& eigvals = eig.second;

    typename decltype(fmgrad_field)::tensor_t fmgrad;
    std::cerr << "tau_step = " << tau_step << "\n";
    while (cond < 9) {
      tau += tau_step;
      fm.set_tau(tau);
      std::cerr << "tau      = " << fm.tau() << '\n';

      fmgrad = fmgrad_field(m_x0, m_t0);
      eig = eigenvectors_sym(fmgrad * transpose(fmgrad));
      cond = max(eigvals) / min(eigvals);
      std::cerr << "cond     = " << cond << '\n';
      if (cond > 9.05) {
        std::cerr << "cond too high, decreasing tau_step\n";
        cond = 0;
        tau -= tau_step;
        tau_step *= 0.5;
        std::cerr << "tau_step = " << tau_step << "\n";
      }
      std::cerr << "=====\n";
    }
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
