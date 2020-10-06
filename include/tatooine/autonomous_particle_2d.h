#ifndef TATOOINE_AUTONOMOUS_PARTICLES_2D_H
#define TATOOINE_AUTONOMOUS_PARTICLES_2D_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/field.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/tensor.h>

#include "concepts.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <fixed_dims_flowmap_c<2> Flowmap>
struct autonomous_particle<Flowmap> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t    = autonomous_particle<Flowmap>;
  using flowmap_t = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() {
    return flowmap_t::num_dimensions();
  }
  using real_t  = typename flowmap_t::real_t;
  using vec_t   = vec<real_t, num_dimensions()>;
  using mat_t   = mat<real_t, num_dimensions(), num_dimensions()>;
  using diff1_t = mat_t;
  using pos_t   = vec_t;

  static constexpr real_t objective_cond = 4;
  static constexpr real_t max_cond       = objective_cond + 0.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  flowmap_t m_phi;
  pos_t     m_x0, m_x1;
  real_t    m_t1;
  mat_t     m_nabla_phi1;
  diff1_t   m_S;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename V, std::floating_point VReal, real_number RealX0>
  autonomous_particle(const vectorfield<V, VReal, 2>&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
      : autonomous_particle{flowmap(v), x0, t0, r0} {}
  //----------------------------------------------------------------------------
  template <real_number RealX0>
  autonomous_particle(flowmap_t phi, vec<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x0},
        m_t1{static_cast<real_t>(t0)},
        m_nabla_phi1{mat_t::eye()},
        m_S{mat_t::eye() * r0} {}
  //----------------------------------------------------------------------------
  autonomous_particle(flowmap_t phi, pos_t const& x0, pos_t const& x1,
                      real_t const t1, mat_t const& nabla_phi1, mat_t const& S)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x1},
        m_t1{t1},
        m_nabla_phi1{nabla_phi1},
        m_S{S} {}
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(const autonomous_particle&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const autonomous_particle&) -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&&) noexcept
      -> autonomous_particle& = default;

  //----------------------------------------------------------------------------
  // getter
  //----------------------------------------------------------------------------
  auto x0() const -> auto const& {
    return m_x0;
  }
  auto x0(size_t i) const {
    return m_x0(i);
  }
  auto x1() const -> auto const& {
    return m_x1;
  }
  auto x1(size_t i) const {
    return m_x1(i);
  }
  auto t1() const {
    return m_t1;
  }
  auto nabla_phi1() const -> auto const& {
    return m_nabla_phi1;
  }
  auto S() const -> auto const& {
    return m_S;
  }
  auto phi() const -> auto const& {
    return m_phi;
  }
  auto phi() -> auto& {
    return m_phi;
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto integrate(real_t tau_step, real_t const max_t) const {
    std::vector<this_t>                             particles{*this};
    std::vector<line<real_t, num_dimensions() + 1>> ellipses;

    size_t size_before = size(particles);
    size_t start_idx   = 0;
    do {
      // this is necessary because after a resize of particles the currently
      // used adresses are not valid anymore
      particles.reserve(size(particles) + size(particles) * 3);
      size_before = size(particles);
      for (size_t i = start_idx; i < size_before; ++i) {
        particles[i].integrate_until_split(tau_step, particles, ellipses,
                                           max_t);
      }
      start_idx = size_before;
    } while (size_before != size(particles));
    return std::pair{std::move(particles), std::move(ellipses)};
  }
  //----------------------------------------------------------------------------
  void integrate_until_split(
      real_t tau_step, std::vector<this_t>& particles,
      std::vector<line<real_t, num_dimensions() + 1>>& ellipses,
      real_t const                                     max_t) const {
    // add initial circle
    auto const discretized = discretize(geometry::sphere<real_t, 2>{1}, 100);
    auto&      ellipse     = ellipses.emplace_back();
    for (auto const& x : discretized.vertices()) {
      auto const p = m_S * x + m_x1;
      ellipse.push_back(vec{p(0), p(1), m_t1});
    }
    ellipse.push_back(ellipse.front_vertex());

    if (m_t1 >= max_t) {
      return;
    }

    real_t tau  = 0;
    real_t cond = 1;

    auto const [Q, lambdas] = eigenvectors_sym(m_S);
    auto const sigma        = diag(lambdas);
    auto const B            = Q * sigma;

    // n stands for negative offset, p for positive offset
    auto const o_p0 = B * vec_t{ 1,  0};
    auto const o_n0 = B * vec_t{-1,  0};
    auto const o_0p = B * vec_t{ 0,  1};
    auto const o_0n = B * vec_t{ 0, -1};

    pos_t                   p_00, p_p0, p_n0, p_0p, p_0n;
    mat_t                   H, nabla_phi2;
    std::pair<mat_t, vec_t> eig_HHt;
    auto const&             eigvecs_HHt = eig_HHt.first;
    auto const&             eigvals_HHt = eig_HHt.second;

    while (cond < objective_cond && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) {
        tau = max_t - m_t1;
      }
      // integrate center points
      p_00 = m_phi(m_x1, m_t1, tau);

      // integrate ghost particles
      p_p0 = m_phi(m_x1 + o_p0, m_t1, tau);
      p_n0 = m_phi(m_x1 + o_n0, m_t1, tau);

      p_0p = m_phi(m_x1 + o_0p, m_t1, tau);
      p_0n = m_phi(m_x1 + o_0n, m_t1, tau);

      // construct ∇φ2
      H.col(0) = p_p0 - p_n0;
      H.col(1) = p_0p - p_0n;
      H /= 2;

      nabla_phi2 = H * inv(sigma) * transposed(Q);

      auto const HHt = H * transposed(H);
      eig_HHt        = eigenvectors_sym(HHt);

      cond = eigvals_HHt(1) / eigvals_HHt(0);

      if (cond >= objective_cond && cond <= max_cond) {
        auto& ellipse = ellipses.emplace_back();

        for (auto x : discretized.vertices()) {
          auto const p = H * x + p_00;
          ellipse.push_back(vec{p(0), p(1), m_t1 + tau});
        }
        ellipse.push_back(ellipse.front_vertex());
      }

      if (cond > max_cond) {
        tau -= tau_step;
        tau_step *= 0.5;
        cond = 0;
      }
    }
    if (cond >= objective_cond) {
      mat_t const  fmg2fmg1 = nabla_phi2 * m_nabla_phi1;
      real_t const t2       = m_t1 + tau;
      // auto [eigvecs_np2, eigvals_np2] = eigenvectors_sym(nabla_phi2);
      // std::cerr << "split at tau = " << tau << '\n';
      // std::cerr << "S:\n" << m_S << '\n';
      // std::cerr << "H:\n" << H << '\n';
      // std::cerr << "H*Ht:\n" << H * transposed(H) << '\n';
      // std::cerr << "eigenvalues HHt: " << eigvals_HHt << '\n';
      // std::cerr << "eigenvectors HHt: \n" << eigvecs_HHt << '\n';
      // std::cerr << "∇φ2:\n" << nabla_phi2 << '\n';
      // std::cerr << "eigenvalues ∇φ2: " << eigvals_np2 << '\n';
      // std::cerr << "eigenvectors ∇φ2: \n" << eigvecs_np2 << '\n';
      vec const   new_eigvals_inner{std::sqrt(eigvals_HHt(0)),
                                  std::sqrt(eigvals_HHt(1)) / 2};
      auto const  new_eigvals_outer = new_eigvals_inner / 2;
      vec_t const offset2 =
          std::sqrt(eigvals_HHt(1)) * eigvecs_HHt.col(1) * 3 / 4;
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      auto const new_S_inner =
          eigvecs_HHt * diag(new_eigvals_inner) * transposed(eigvecs_HHt);
      auto const new_S_outer =
          eigvecs_HHt * diag(new_eigvals_outer) * transposed(eigvecs_HHt);

      particles.emplace_back(m_phi, m_x0 - offset0, p_00 - offset2, t2,
                             fmg2fmg1, new_S_outer);
      particles.emplace_back(m_phi, m_x0, p_00, t2, fmg2fmg1, new_S_inner);
      particles.emplace_back(m_phi, m_x0 + offset0, p_00 + offset2, t2,
                             fmg2fmg1, new_S_outer);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, real_number RealX0, size_t N>
autonomous_particle(const vectorfield<V, VReal, 2>& v, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<decltype(flowmap(v))>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_dims_flowmap_c<2> Flowmap, real_number RealX0, size_t N>
autonomous_particle(const Flowmap& flowmap, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<Flowmap>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
