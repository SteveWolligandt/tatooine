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

  static constexpr real_t objective_cond = 4;
  static constexpr real_t max_cond       = objective_cond + 0.01;

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
  template <typename V, std::floating_point VReal, real_number RealX0>
  autonomous_particle(const vectorfield<V, VReal, 3>&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
      : autonomous_particle{flowmap(v), x0, t0, r0} {}
  //----------------------------------------------------------------------------
  /// creates an initial particle
  template <real_number RealX0>
  autonomous_particle(flowmap_t phi, vec<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
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
  void integrate_until_split(
      real_t tau_step, std::vector<this_t>& particles,
      std::vector<simple_tri_mesh<real_t, 3>>& ellipsoids,
      real_t const                             max_t) const {
    // add initial sphere
    ellipsoids.push_back(discretize(geometry::sphere<real_t, 3>{1}, 3));

    for (auto const v : ellipsoids.back().vertices()) {
      ellipsoids.back()[v] = m_S * ellipsoids.back()[v];
      ellipsoids.back()[v] += m_x1;
    }

    if (m_t1 >= max_t) { return; }
    real_t tau  = 0;
    real_t cond = 1;

    auto const [Q, lambdas] = eigenvectors_sym(m_S);
    auto const sigma        = diag(lambdas);
    auto const B            = Q * sigma;

    // n stands for negative offset, p for positive offset
    auto const o_p00 = B * vec_t{ 1,  0,  0};
    auto const o_n00 = B * vec_t{-1,  0,  0};
    auto const o_0p0 = B * vec_t{ 0,  1,  0};
    auto const o_0n0 = B * vec_t{ 0, -1,  0};
    auto const o_00p = B * vec_t{ 0,  0,  1};
    auto const o_00n = B * vec_t{ 0,  0, -1};

    pos_t p_p00, p_0p0, p_00p, p_n00, p_0n0, p_00n, p_000;

    std::pair<mat_t, vec_t> eig_HHt;
    auto&   eigvecs_HHt = eig_HHt.first;
    auto&   eigvals_HHt = eig_HHt.second;
    diff1_t H, nabla_phi2;

    while (cond < objective_cond && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }
      // integrate ghost particles
      p_000 = m_phi(m_x1, m_t1, tau);
      
      p_p00 = m_phi(m_x1 + o_p00, m_t1, tau);
      p_n00 = m_phi(m_x1 + o_n00, m_t1, tau);

      p_0p0 = m_phi(m_x1 + o_0p0, m_t1, tau);
      p_0n0 = m_phi(m_x1 + o_0n0, m_t1, tau);

      p_00p = m_phi(m_x1 + o_00p, m_t1, tau);
      p_00n = m_phi(m_x1 + o_00n, m_t1, tau);

      // construct ∇φ2
      H.col(0) = (p_p00 - p_n00) / 2;  // h1
      H.col(1) = (p_0p0 - p_0n0) / 2;  // h2
      H.col(2) = (p_00p - p_00n) / 2;  // h3

      nabla_phi2 = H * inverse(sigma) * transposed(Q);

      auto const HHt = H * transposed(H);
      eig_HHt = eigenvectors_sym(HHt);

      cond = eigvals_HHt(2) / eigvals_HHt(0);

      if (cond >= objective_cond && cond <= max_cond){
        ellipsoids.push_back(discretize(geometry::sphere<real_t, 3>{1}, 3));

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
    real_t const t2       = m_t1 + tau;
    if (cond >= objective_cond) {
      auto [eigvecs_np2, eigvals_np2] = eigenvectors_sym(nabla_phi2);
      std::cerr << "split at tau = " << tau << '\n';
      std::cerr << "S:\n" << m_S << '\n';
      std::cerr << "H:\n" << H << '\n'; 
      std::cerr << "H*Ht:\n" << H * transposed(H) << '\n';
      std::cerr << "eigenvalues HHt: " << eigvals_HHt << '\n';
      std::cerr << "eigenvectors HHt: \n" << eigvecs_HHt << '\n';
      std::cerr << "∇φ2:\n" << nabla_phi2 << '\n';
      std::cerr << "eigenvalues ∇φ2: " << eigvals_np2 << '\n';
      std::cerr << "eigenvectors ∇φ2: \n" << eigvecs_np2 << '\n';
      vec const eigvals_HHt_sqrt{std::sqrt(eigvals_HHt(0)),
                                 std::sqrt(eigvals_HHt(1)),
                                 std::sqrt(eigvals_HHt(2)) / 2};
      vec_t const offset2 =
          std::sqrt(eigvals_HHt(2)) * eigvecs_HHt.col(2) * 3 / 4;
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      auto const new_S = eigvecs_HHt * diag(eigvals_HHt_sqrt) * transposed(eigvecs_HHt);
      auto const half_new_S = new_S / 2;

      particles.emplace_back(m_phi, m_x0 - offset0, p_000 - offset2, t2,
                             fmg2fmg1, half_new_S, m_level + 1);
      particles.emplace_back(m_phi, m_x0, p_000, t2, fmg2fmg1, new_S,
                             m_level + 1);
      particles.emplace_back(m_phi, m_x0 + offset0, p_000 + offset2, t2,
                             fmg2fmg1, half_new_S, m_level + 1);

    } else {
      //vec const eigvals_HHt_sqrt{std::sqrt(eigvals_HHt(0)),
      //                           std::sqrt(eigvals_HHt(1)),
      //                           std::sqrt(eigvals_HHt(2)) / 3};
      //auto const new_S = eigvecs_HHt * diag(eigvals_HHt_sqrt) * transposed(eigvecs_HHt);
      //particles.emplace_back(m_phi, m_x0, p_000, t2, fmg2fmg1, new_S,
      //                       m_level + 1);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, real_number RealX0, size_t N>
autonomous_particle(const vectorfield<V, VReal, 3>& v, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<decltype(flowmap(v))>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_dims_flowmap_c<3> Flowmap, real_number RealX0, size_t N>
autonomous_particle(const Flowmap& flowmap, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<Flowmap>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
