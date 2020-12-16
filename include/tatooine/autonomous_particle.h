#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/random.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <flowmap_c Flowmap>
struct autonomous_particle {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t    = autonomous_particle<Flowmap>;
  using flowmap_t = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using real_t = typename flowmap_t::real_t;
  using vec_t  = vec<real_t, num_dimensions()>;
  using mat_t  = mat<real_t, num_dimensions(), num_dimensions()>;
  using pos_t  = vec_t;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  flowmap_t m_phi;
  pos_t     m_x0, m_x1;
  real_t    m_t1;
  mat_t     m_nabla_phi1;
  mat_t     m_S;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(autonomous_particle const&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(autonomous_particle const&) -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&&) noexcept
      -> autonomous_particle& = default;
  //----------------------------------------------------------------------------
  ~autonomous_particle() = default;
  //----------------------------------------------------------------------------
  template <typename = void>
  requires is_numerical_flowmap_v<Flowmap> autonomous_particle()
      : autonomous_particle{Flowmap{}, pos_t::zeros(), real_t(0), real_t(0)} {
    if constexpr (is_cacheable_v<decltype(m_phi)>) {
      m_phi.use_caching(false);
    }
  }
  //----------------------------------------------------------------------------
  template <typename V, std::floating_point VReal, real_number RealX0>
  autonomous_particle(vectorfield<V, VReal, num_dimensions()> const& v,
                      vec<RealX0, num_dimensions()> const&           x0,
                      real_number auto const t0, real_number auto const r0)
      : autonomous_particle{flowmap(v), x0, t0, r0} {
    if constexpr (is_cacheable_v<decltype(m_phi)>) {
      m_phi.use_caching(false);
    }
  }
  //----------------------------------------------------------------------------
  template <real_number RealX0>
  autonomous_particle(flowmap_t phi, tensor<RealX0, num_dimensions()> const& x0,
                      real_number auto const t0, real_number auto const r0)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x0},
        m_t1{static_cast<real_t>(t0)},
        m_nabla_phi1{mat_t::eye()},
        m_S{mat_t::eye() * r0} {
    if constexpr (is_cacheable_v<decltype(m_phi)>) {
      m_phi.use_caching(false);
    }
  }
  //----------------------------------------------------------------------------
  autonomous_particle(flowmap_t phi, pos_t const& x0, pos_t const& x1,
                      real_t const t1, mat_t const& nabla_phi1, mat_t const& S)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x1},
        m_t1{t1},
        m_nabla_phi1{nabla_phi1},
        m_S{S} {
    if constexpr (is_cacheable_v<decltype(m_phi)>) {
      m_phi.use_caching(false);
    }
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  // getter
  //----------------------------------------------------------------------------
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return m_x0(i); }
  auto x1() -> auto& { return m_x1; }
  auto x1() const -> auto const& { return m_x1; }
  auto x1(size_t i) const { return m_x1(i); }
  auto t1() -> auto& { return m_t1; }
  auto t1() const { return m_t1; }
  auto nabla_phi1() const -> auto const& { return m_nabla_phi1; }
  auto S() -> auto& { return m_S; }
  auto S() const -> auto const& { return m_S; }
  auto phi() const -> auto const& { return m_phi; }
  auto phi() -> auto& { return m_phi; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  //auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false, stop);
  //}
  ////----------------------------------------------------------------------------
  //auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false, stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles), stop);
  //}
  //----------------------------------------------------------------------------
  auto advect_with_3_splits(real_t const tau_step, real_t const max_t,
                            bool const&  stop = false) const {
    return advect_with_3_splits(tau_step, max_t, 0, std::vector{*this}, stop);
  }
  //----------------------------------------------------------------------------
  auto advect_with_3_splits(real_t const tau_step, real_t const max_t,
                            size_t const max_num_particles,
                            bool const&  stop = false) const {
    return advect_with_3_splits(
        tau_step, max_t, max_num_particles, std::vector{*this}, stop);
  }
  //----------------------------------------------------------------------------
  static auto advect_with_3_splits(real_t const tau_step, real_t const max_t,
                                   std::vector<this_t> particles,
                                   bool const&         stop = false) {
    return advect_with_3_splits(tau_step, max_t, 0, std::move(particles), stop);
  }
  //----------------------------------------------------------------------------
  static auto advect_with_3_splits(real_t const tau_step, real_t const max_t,
                                   size_t const        max_num_particles,
                                   std::vector<this_t> particles,
                                   bool const&         stop = false) {
    static real_t const x5 = 0.4830517593887872;
    return advect(tau_step, max_t, 4, max_num_particles,
                  //std::array{vec_t{real_t(1),             real_t(1) / real_t(2)},
                  //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                  //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
                  //std::array{vec_t{0, 0},
                  //           vec_t{real_t(3) / 4, 0},
                  //           vec_t{-real_t(3) / 4, 0}},
                  std::array{vec_t{x5, x5/2},
                             vec_t{x5, x5/2},
                             vec_t{x5, x5/2},
                             vec_t{x5, x5/2},
                             vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                             vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
                  std::array{vec_t{-x5/2, -x5},
                             vec_t{x5/2, -x5},
                             vec_t{-x5/2, x5},
                             vec_t{x5/2, x5},
                             vec_t{real_t(3) / 4, 0},
                             vec_t{-real_t(3) / 4, 0}},
                  std::move(particles), stop);
  }
  ////----------------------------------------------------------------------------
  //auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  //auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  //auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  //auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  //static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 std::vector<this_t> particles,
  //                                 bool const&         stop = false) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles), stop);
  //}
  //----------------------------------------------------------------------------
  auto advect(real_t tau_step, real_t const max_t, real_t const objective_cond,
              size_t const                  max_num_particles,
              std::ranges::range auto const radii, std::ranges::range auto const& offsets,
              bool const& stop = false) const {
    return advect(tau_step, max_t, objective_cond, max_num_particles, radii,
                  offsets, {*this}, stop);
  }
  //----------------------------------------------------------------------------
  static auto advect(real_t tau_step, real_t const max_t,
                     real_t const                   objective_cond,
                     size_t const                   max_num_particles,
                     std::ranges::range auto const  radii,
                     std::ranges::range auto const& offsets,
                     std::vector<this_t>            input_particles,
                     bool const&                    stop = false) {
    std::mutex          finished_particles_mutex;
    std::mutex          inactive_particles_mutex;
    std::vector<this_t> finished_particles;
    std::array particles{std::move(input_particles), std::vector<this_t>{}};
    auto     active = &particles[0];
    auto     inactive = &particles[1];
    real_t     tau    = 0;
    while (!active->empty()) {
      if (stop) {
        break;
      }
      if (tau + tau_step > max_t) {
        tau_step = max_t - tau;
      }
      tau += tau_step;
      inactive->clear();
#pragma omp parallel for
      for (size_t i = 0; i < active->size(); ++i) {
        if (!stop) {
          auto const& particle      = active->at(i);
          auto        new_particles = particle.advect_until_split(
              tau_step, max_t, objective_cond, radii, offsets);

          //if (size(new_particles) == 1) {
          //  std::lock_guard lock{finished_particles_mutex};
          //  std::copy(begin(new_particles), end(new_particles),
          //            std::back_inserter(finished_particles));
          //} else {
          //  std::lock_guard lock{inactive_particles_mutex};
          //  std::copy(begin(new_particles), end(new_particles),
          //            std::back_inserter(*inactive));
          //}
          {
            std::lock_guard lock{finished_particles_mutex};
            std::copy(begin(new_particles), end(new_particles),
                      std::back_inserter(finished_particles));
          }
          if (size(new_particles) > 1) {
            std::lock_guard lock{inactive_particles_mutex};
            std::copy(begin(new_particles), end(new_particles),
                      std::back_inserter(*inactive));
          }
        }
      }

      std::swap(active, inactive);
      // if (max_num_particles > 0 && active->size() > max_num_particles) {
      //  size_t const num_particles_to_delete =
      //      active->size() - max_num_particles;
      //
      //  for (size_t i = 0; i < num_particles_to_delete; ++i) {
      //    random_uniform<size_t> rand{0, active->size() - 1};
      //    active->rand()] = std::move(active->back());
      //    active->pop_back();
      //  }
      //}
    }
    while (max_num_particles > 0 && size(finished_particles) > max_num_particles) {
      random_uniform<size_t> rand{0, size(finished_particles) - 1};
      finished_particles[rand()] =
          std::move(finished_particles.back());
      finished_particles.pop_back();
    }
    return finished_particles;
  }

  //----------------------------------------------------------------------------
  auto advect_until_split(real_t tau_step, real_t const max_t,
                          real_t const                   objective_cond,
                          std::ranges::range auto const  radii,
                          std::ranges::range auto const& offsets) const
      -> std::vector<this_t> {
    auto const [Q, lambdas] = eigenvectors_sym(m_S);
    auto const Sigma        = diag(lambdas);
    auto const B            = Q * Sigma;

    mat_t      H, HHt, nabla_phi2, fmg2fmg1;
    std::pair<mat_t, vec_t> eig_HHt;
    real_t                  old_cond_HHt = 1, cond_HHt = 1;
    auto const&             eigvecs_HHt = eig_HHt.first;
    auto const&             eigvals_HHt = eig_HHt.second;
    vec_t                   advected_center;
    real_t                  t2 = m_t1;

    while (cond_HHt < objective_cond || t2 < max_t) {
      old_cond_HHt = cond_HHt;
      t2 += tau_step;
      t2 = std::min(t2, max_t);

      advected_center = m_phi(m_x1, m_t1, t2 - m_t1);
      for (size_t i = 0; i < num_dimensions(); ++i) {
        H.col(i) = m_phi(m_x1 + B.col(i), m_t1, t2 - m_t1) -
                   m_phi(m_x1 - B.col(i), m_t1, t2 - m_t1);
      }
      H /= 2;
      HHt      = H * transposed(H);
      eig_HHt  = eigenvectors_sym(HHt);
      cond_HHt = eigvals_HHt(num_dimensions() - 1) / eigvals_HHt(0);

      nabla_phi2 = H * *solve(Sigma, transposed(Q));
      fmg2fmg1   = nabla_phi2 * m_nabla_phi1;

      auto const current_radii = sqrt(eigvals_HHt);
      auto const cur_S =
          eigvecs_HHt * diag(current_radii) * transposed(eigvecs_HHt);

        std::vector<this_t> splits;
      if (t2 == max_t) {
        splits.emplace_back(m_phi, m_x0, advected_center, t2, fmg2fmg1, cur_S);
        return splits;
      } else if (cond_HHt >= objective_cond &&
                 (cond_HHt - objective_cond < 0.0001 || tau_step < 1e-13)) {

        for (size_t i = 0; i < size(radii); ++i) {
          auto const new_eigvals = current_radii * radii[i];
          auto const new_S =
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt);
          auto const offset2     = cur_S * offsets[i];
          auto const offset0     = solve(fmg2fmg1, offset2);
          splits.emplace_back(m_phi, m_x0 - offset0, advected_center - offset2,
                              t2, fmg2fmg1, new_S);
        }
        return splits;
      } else if (cond_HHt >= objective_cond &&
                 cond_HHt - objective_cond >= 0.00001) {
        t2 -= tau_step;
        tau_step /= 2;
      }
    }
    return {};
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, real_number RealX0, size_t N>
autonomous_particle(vectorfield<V, VReal, N> const& v, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<decltype(flowmap(v))>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <flowmap_c Flowmap, real_number RealX0, size_t N>
autonomous_particle(Flowmap const& flowmap, vec<RealX0, N> const&,
                    real_number auto const, real_number auto const)
    -> autonomous_particle<Flowmap>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
