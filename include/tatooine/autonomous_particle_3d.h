#ifndef TATOOINE_AUTONOMOUS_PARTICLES_3D_H
#define TATOOINE_AUTONOMOUS_PARTICLES_3D_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/field.h>
#include <tatooine/tensor.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/geometry/sphere.h>

#include "concepts.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <fixed_dims_flowmap_c<3> Flowmap>
struct autonomous_particle<Flowmap> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t    = autonomous_particle<Flowmap>;
  using flowmap_t = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using real_t  = typename flowmap_t::real_t;
  using vec_t   = vec<real_t, num_dimensions()>;
  using mat_t   = mat<real_t, num_dimensions(), num_dimensions()>;
  using diff1_t = mat_t;
  using diff2_t =
      tensor<real_t, num_dimensions(), num_dimensions(), num_dimensions()>;
  using pos_t = vec_t;

  static constexpr real_t max_cond = 9.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  flowmap_t m_phi;
  pos_t     m_x0, m_x1;
  real_t    m_t1;
  diff1_t   m_nabla_phi1;
  mat_t     m_S;
  size_t    m_level;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  /// creates an initial particle
  template <typename V, std::floating_point VReal, arithmetic RealX0>
  autonomous_particle(const vectorfield<V, VReal, 3>&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const t0, arithmetic auto const r0)
      : autonomous_particle{flowmap(v), x0, t0, r0} {}
  //----------------------------------------------------------------------------
  /// creates an initial particle
  template <arithmetic RealX0>
  autonomous_particle(flowmap_t phi, vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const t0, arithmetic auto const r0)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x0},
        m_t1{static_cast<real_t>(t0)},
        m_nabla_phi1{mat_t::eye()},
        m_S{mat_t::eye() * r0},
        m_level{0} {}
  //----------------------------------------------------------------------------
  /// defines all attributes
  autonomous_particle(flowmap_t phi, pos_t x0, pos_t x1, real_t const t1,
                      diff1_t nabla_phi1, mat_t S, size_t const level)
      : m_phi{std::move(phi)},
        m_x0{std::move(x0)},
        m_x1{std::move(x1)},
        m_t1{t1},
        m_nabla_phi1{std::move(nabla_phi1)},
        m_S{std::move(S)},
        m_level{level} {}
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
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return m_x0(i); }
  auto x1() const -> auto const& { return m_x1; }
  auto x1(size_t i) const { return m_x1(i); }
  auto t1() const { return m_t1; }
  auto nabla_phi1() const -> auto const& { return m_nabla_phi1; }
  auto phi() const -> auto const& { return m_phi; }
  auto phi() -> auto& { return m_phi; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto integrate(real_t tau_step, real_t const max_t) const {
    std::vector<this_t> particles{*this};
    std::vector<simple_tri_mesh<real_t, 3>> ellipsoids;

    size_t size_before = size(particles);
    size_t start_idx   = 0;
    do {
      // this is necessary because after a resize of particles the currently
      // used adresses are not valid anymore
      particles.reserve(size(particles) + size(particles) * 3);
      size_before = size(particles);
      for (size_t i = start_idx; i < size_before; ++i) {
        particles[i].integrate_until_split(tau_step, particles, ellipsoids,
                                           max_t);
      }
      start_idx = size_before;
    } while (size_before != size(particles));
    return std::pair{std::move(particles), std::move(ellipsoids)};
  }
  //----------------------------------------------------------------------------
  void integrate_until_split(real_t tau_step, std::vector<this_t>& particles,
                             std::vector<simple_tri_mesh<real_t, 3>>& ellipsoids,
                             real_t const max_t) const {
    // add initial sphere
    ellipsoids.push_back(discretize(geometry::sphere<real_t, 3>{}, 2));

    for (auto const v: ellipsoids.back().vertices()) {
      ellipsoids.back()[v] = m_S * ellipsoids.back()[v];
      ellipsoids.back()[v] += m_x1;
    }

    size_t const n = 100;

    if (m_t1 >= max_t) { return; }
    static real_t const threequarters = real_t(3) / real_t(4);
    static real_t const sqrt2         = std::sqrt(real_t(2));
    static real_t const sqrt3         = std::sqrt(real_t(3));

    real_t tau  = 0;
    real_t cond = 1;

    auto const [Q, lambdas] = eigenvectors_sym(m_S);
    auto const sigma = diag(lambdas);
    auto const B            = Q * sigma;

    // n stands for negative offset, p for positive offset
    auto const o_p00 = B * vec_t{ 1,  0,  0};
    auto const o_n00 = B * vec_t{-1,  0,  0};
    auto const o_0p0 = B * vec_t{ 0,  1,  0};
    auto const o_0n0 = B * vec_t{ 0, -1,  0};
    auto const o_00p = B * vec_t{ 0,  0,  1};
    auto const o_00n = B * vec_t{ 0,  0, -1};

    //pos_t p_nnn;
    //pos_t p_0nn;
    //pos_t p_pnn;
    //pos_t p_n0n;
    pos_t p_00n;
    //pos_t p_p0n;
    //pos_t p_npn;
    //pos_t p_0pn;
    //pos_t p_ppn;

    //pos_t p_nn0;
    pos_t p_0n0;
    //pos_t p_pn0;
    pos_t p_n00;
    pos_t p_000;
    pos_t p_p00;
    //pos_t p_np0;
    pos_t p_0p0;
    //pos_t p_pp0;

    //pos_t p_nnp;
    //pos_t p_0np;
    //pos_t p_pnp;
    //pos_t p_n0p;
    pos_t p_00p;
    //pos_t p_p0p;
    //pos_t p_npp;
    //pos_t p_0pp;
    //pos_t p_ppp;

    std::pair<mat_t, vec_t> eig_HHt;
    auto&   eigvecs_HHt = eig_HHt.first;
    auto&   eigvals_HHt = eig_HHt.second;
    diff1_t H, nabla_phi2;
    //diff2_t                      nabla_nabla_phi2;

    while (cond < 9 && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }

      // integrate ghost particles
      // p_nnn = m_phi(m_x1 + o_nnn, m_t1, tau);
      // p_0nn = m_phi(m_x1 + o_0nn, m_t1, tau);
      // p_pnn = m_phi(m_x1 + o_pnn, m_t1, tau);
      // p_n0n = m_phi(m_x1 + o_n0n, m_t1, tau);
      p_00n = m_phi(m_x1 + o_00n, m_t1, tau);
      // p_p0n = m_phi(m_x1 + o_p0n, m_t1, tau);
      // p_npn = m_phi(m_x1 + o_npn, m_t1, tau);
      // p_0pn = m_phi(m_x1 + o_0pn, m_t1, tau);
      // p_ppn = m_phi(m_x1 + o_ppn, m_t1, tau);

      // p_nn0 = m_phi(m_x1 + o_nn0, m_t1, tau);
      p_0n0 = m_phi(m_x1 + o_0n0, m_t1, tau);
      // p_pn0 = m_phi(m_x1 + o_pn0, m_t1, tau);
      p_n00 = m_phi(m_x1 + o_n00, m_t1, tau);
      p_000 = m_phi(m_x1, m_t1, tau);
      p_p00 = m_phi(m_x1 + o_p00, m_t1, tau);
      // p_np0 = m_phi(m_x1 + o_np0, m_t1, tau);
      p_0p0 = m_phi(m_x1 + o_0p0, m_t1, tau);
      // p_pp0 = m_phi(m_x1 + o_pp0, m_t1, tau);

      // p_nnp = m_phi(m_x1 + o_nnp, m_t1, tau);
      // p_0np = m_phi(m_x1 + o_0np, m_t1, tau);
      // p_pnp = m_phi(m_x1 + o_pnp, m_t1, tau);
      // p_n0p = m_phi(m_x1 + o_n0p, m_t1, tau);
      p_00p = m_phi(m_x1 + o_00p, m_t1, tau);
      // p_p0p = m_phi(m_x1 + o_p0p, m_t1, tau);
      // p_npp = m_phi(m_x1 + o_npp, m_t1, tau);
      // p_0pp = m_phi(m_x1 + o_0pp, m_t1, tau);
      // p_ppp = m_phi(m_x1 + o_ppp, m_t1, tau);

      // construct ∇φ2
      H.col(0) = (p_p00 - p_n00) / 2;  // h1
      H.col(1) = (p_0p0 - p_0n0) / 2;  // h2
      H.col(2) = (p_00p - p_00n) / 2;  // h3
      nabla_phi2 = H * inverse(sigma) * transpose(Q);

      auto const HHt = H * transpose(H);
      eig_HHt = eigenvectors_sym(HHt);

      cond = eigvals_HHt(2) / eigvals_HHt(0);

      if (cond >= 9 && cond <= max_cond){
        ellipsoids.push_back(discretize(geometry::sphere<real_t, 3>{}, 4));

        for (auto const v : ellipsoids.back().vertices()) {
          ellipsoids.back()[v] = H * ellipsoids.back()[v];
          ellipsoids.back()[v] += p_000;
        }
      }

      if (cond > max_cond) {
        tau -= tau_step;
        tau_step *= 0.5;
        cond = 0;
      }
    }
    mat_t const  fmg2fmg1 = nabla_phi2 * m_nabla_phi1;
    pos_t const  x2       = m_phi(m_x1, m_t1, tau);
    real_t const t2       = m_t1 + tau;
    if (cond >= 9) {
      vec const eigvals_HHt_sqrt{std::sqrt(eigvals_HHt(0)),
                                 std::sqrt(eigvals_HHt(1)),
                                 std::sqrt(eigvals_HHt(2)) / 3};
      vec_t const offset2 = std::sqrt(eigvals_HHt(2)) * eigvecs_HHt.col(2) / 3;
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      particles.emplace_back(
          m_phi, m_x0 - offset0, x2 - offset2, t2, fmg2fmg1,
          eigvecs_HHt * diag(eigvals_HHt_sqrt) * transpose(eigvecs_HHt),
          m_level + 1);
      particles.emplace_back(
          m_phi, m_x0, x2, t2, fmg2fmg1,
          eigvecs_HHt * diag(eigvals_HHt_sqrt) * transpose(eigvecs_HHt),
          m_level + 1);
      particles.emplace_back(
          m_phi, m_x0 + offset0, x2 + offset2, t2, fmg2fmg1,
          eigvecs_HHt * diag(eigvals_HHt_sqrt) * transpose(eigvecs_HHt),
          m_level + 1);

    } else {
      // particles.emplace_back(m_phi, m_x0, x2, m_x1, t2, m_t1, fmg2fmg1,
      //                       m_current_radius);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, arithmetic RealX0, size_t N>
autonomous_particle(const vectorfield<V, VReal, 3>& v, vec<RealX0, N> const&,
                    arithmetic auto const, arithmetic auto const)
    -> autonomous_particle<decltype(flowmap(v))>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_dims_flowmap_c<3> Flowmap, arithmetic RealX0, size_t N>
autonomous_particle(const Flowmap& flowmap, vec<RealX0, N> const&,
                    arithmetic auto const, arithmetic auto const)
    -> autonomous_particle<Flowmap>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
