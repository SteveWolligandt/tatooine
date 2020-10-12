#ifndef TATOOINE_AUTONOMOUS_PARTICLES_2D_H
#define TATOOINE_AUTONOMOUS_PARTICLES_2D_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/field.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/tensor.h>

#include <ranges>

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
  autonomous_particle(autonomous_particle const&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  auto operator=(autonomous_particle const&) -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&&) noexcept
      -> autonomous_particle& = default;
  template <typename = void>
  requires is_numerical_flowmap_v<Flowmap> autonomous_particle()
      : autonomous_particle{Flowmap{}, pos_t::zeros(), real_t(0), real_t(0)} {}
  //----------------------------------------------------------------------------
  template <typename V, std::floating_point VReal, real_number RealX0>
  autonomous_particle(vectorfield<V, VReal, 2> const&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
      : autonomous_particle{flowmap(v), x0, t0, r0} {}
  //----------------------------------------------------------------------------
  template <real_number RealX0>
  autonomous_particle(flowmap_t phi, tensor<RealX0, num_dimensions()> const& x0,
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
  //----------------------------------------------------------------------------
  // getter
  //----------------------------------------------------------------------------
  auto x0() -> auto& {
    return m_x0;
  }
  auto x0() const -> auto const& {
    return m_x0;
  }
  auto x0(size_t i) const {
    return m_x0(i);
  }
  auto x1() -> auto& {
    return m_x1;
  }
  auto x1() const -> auto const& {
    return m_x1;
  }
  auto x1(size_t i) const {
    return m_x1(i);
  }
  auto t1() -> auto& {
    return m_t1;
  }
  auto t1() const {
    return m_t1;
  }
  auto nabla_phi1() const -> auto const& {
    return m_nabla_phi1;
  }
  auto S() -> auto& {
    return m_S;
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
  auto advect_with_2_splits(real_t tau_step, real_t const max_t,
                            bool const& stop = false) const {
    static real_t const sqrt2 = std::sqrt(real_t(2));
    return advect(tau_step, max_t, 2,
                  std::array<real_t, 1>{sqrt2 / real_t(2)}, false, stop);
  }
  //----------------------------------------------------------------------------
  auto advect_with_3_splits(real_t tau_step, real_t const max_t,
                            bool const& stop = false) const {
    return advect(tau_step, max_t, 4, std::array<real_t, 1>{real_t(1) / 2},
                  true, stop);
  }
  //----------------------------------------------------------------------------
  auto advect_with_5_splits(real_t tau_step, real_t const max_t,
                            bool const& stop = false) const {
    static real_t const sqrt5 = std::sqrt(real_t(5));
    return advect(tau_step, max_t, 6 + sqrt5 * 2,
                  std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
                  true, stop);
  }
  //----------------------------------------------------------------------------
  auto advect_with_7_splits(real_t tau_step, real_t const max_t,
                            bool const& stop = false) const {
    return advect(tau_step, max_t, 4.493959210 * 4.493959210,
                  std::array{real_t(.9009688678), real_t(.6234898004),
                             real_t(.2225209338)},
                  true, stop);
  }
  //----------------------------------------------------------------------------
  auto advect(real_t tau_step, real_t const max_t, real_t const objective_cond,
              std::ranges::range auto const radii, bool const add_center,
              bool const& stop = false) const {
    std::vector<this_t> finished_particles;
    std::array particles{std::vector<this_t>{*this}, std::vector<this_t>{}};
    size_t     active = 0;
    real_t     tau    = 0;
    while (!particles[active].empty()) {
      if (stop) {
        break;
      }
      if (tau + tau_step > max_t) {
        tau_step = max_t - tau;
      }
      tau += tau_step;
      particles[1 - active].clear();
      for (auto const& particle : particles[active]) {
        if (stop) {
          break;
        }
        auto new_particles = particle.step_until_split(tau_step, max_t, objective_cond,
                                                       add_center, radii);
        if (size(new_particles) == 1) {
          std::move(begin(new_particles), end(new_particles),
                    std::back_inserter(finished_particles));
        } else {
          std::move(begin(new_particles), end(new_particles),
                    std::back_inserter(particles[1 - active]));
        }
      }
      active = 1 - active;
    }
    return finished_particles;
  }

 private:
  //----------------------------------------------------------------------------
  auto step_until_split(real_t const tau_step, real_t const max_t,
                        real_t const objective_cond, bool const add_center,
                        std::ranges::range auto const radii) const
      -> std::vector<this_t> {
    auto const [Q, lambdas]       = eigenvectors_sym(m_S);
    auto const              sigma = diag(lambdas);
    auto const              B     = Q * sigma;
    mat_t                   H, HHt, nabla_phi2, fmg2fmg1;
    std::pair<mat_t, vec_t> eig_HHt;
    real_t                  old_cond_HHt = 1, cond_HHt = 1;
    auto const&             eigvecs_HHt = eig_HHt.first;
    auto const&             eigvals_HHt = eig_HHt.second;
    auto const              o_p0        = B * vec_t{1, 0};
    auto const              o_n0        = B * vec_t{-1, 0};
    auto const              o_0p        = B * vec_t{0, 1};
    auto const              o_0n        = B * vec_t{0, -1};

    vec_t  advected_center, p_p0, p_n0, p_0p, p_0n;
    real_t t2 = m_t1;

    while (cond_HHt < objective_cond || t2 != max_t) {
      old_cond_HHt = cond_HHt;
      t2 += tau_step;
      t2 = std::min(t2, max_t);

      p_p0 = m_phi(m_x1 + o_p0, m_t1, t2 - m_t1);
      p_n0 = m_phi(m_x1 + o_n0, m_t1, t2 - m_t1);
      p_0p = m_phi(m_x1 + o_0p, m_t1, t2 - m_t1);
      p_0n = m_phi(m_x1 + o_0n, m_t1, t2 - m_t1);

      H.col(0) = p_p0 - p_n0;
      H.col(1) = p_0p - p_0n;
      H /= 2;
      advected_center = m_phi(m_x1, m_t1, t2 - m_t1);
      HHt             = H * transposed(H);
      eig_HHt         = eigenvectors_sym(HHt);
      cond_HHt        = eigvals_HHt(1) / eigvals_HHt(0);

      nabla_phi2 = H * inv(sigma) * transposed(Q);
      fmg2fmg1   = nabla_phi2 * m_nabla_phi1;
      if (t2 == max_t) {
        vec const new_eig_vals{std::sqrt(eigvals_HHt(0)),
                               std::sqrt(eigvals_HHt(1))};
        return {{m_phi, m_x0, advected_center, t2, fmg2fmg1,
                 eigvecs_HHt * diag(new_eig_vals) * transposed(eigvecs_HHt)}};
      } else if (cond_HHt >= objective_cond) {
        auto const best_tau_step_lin_interp_fac =
            (objective_cond - old_cond_HHt) / (cond_HHt - old_cond_HHt);
        t2 -= (1 - best_tau_step_lin_interp_fac) * tau_step;
        p_p0 = m_phi(m_x1 + o_p0, m_t1, t2 - m_t1);
        p_n0 = m_phi(m_x1 + o_n0, m_t1, t2 - m_t1);
        p_0p = m_phi(m_x1 + o_0p, m_t1, t2 - m_t1);
        p_0n = m_phi(m_x1 + o_0n, m_t1, t2 - m_t1);

        H.col(0) = p_p0 - p_n0;
        H.col(1) = p_0p - p_0n;
        H /= 2;
        advected_center = m_phi(m_x1, m_t1, t2 - m_t1);
        HHt             = H * transposed(H);
        eig_HHt         = eigenvectors_sym(HHt);
        cond_HHt        = eigvals_HHt(1) / eigvals_HHt(0);

        nabla_phi2 = H * inv(sigma) * transposed(Q);
        fmg2fmg1   = nabla_phi2 * m_nabla_phi1;
        std::vector<this_t> splits;
        auto const          relative_1        = std::sqrt(eigvals_HHt(0));
        auto const          relative_unit_vec = eigvecs_HHt.col(1) * relative_1;
        auto                current_offset    = vec_t::zeros();

        if (add_center) {
          auto const new_eigvals = vec_t::ones() * relative_1;
          auto const new_S =
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt);
          splits.emplace_back(m_phi, m_x0, advected_center, t2, fmg2fmg1,
                              new_S);
          current_offset = relative_unit_vec;
        }

        for (auto const radius : radii) {
          auto const new_eigvals = vec_t::ones() * relative_1 * radius;
          auto const offset2     = current_offset + relative_unit_vec * radius;
          auto const offset0     = inv(fmg2fmg1) * offset2;
          auto const new_S =
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt);
          splits.emplace_back(m_phi, m_x0 - offset0, advected_center - offset2,
                              t2, fmg2fmg1, new_S);
          splits.emplace_back(m_phi, m_x0 + offset0, advected_center + offset2,
                              t2, fmg2fmg1, new_S);
          current_offset += relative_unit_vec * 2 * radius;
        }
        return splits;
      }
    }
    return {};
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
