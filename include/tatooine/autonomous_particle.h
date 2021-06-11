#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/random.h>
#include <tatooine/tensor.h>

#include <ranges>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle {
  static constexpr auto num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  constexpr static auto half = Real(0.5);
  using this_t      = autonomous_particle;
  using real_t      = Real;
  using vec_t       = vec<real_t, N>;
  using mat_t       = mat<real_t, N, N>;
  using pos_t       = vec_t;
  using container_t = std::deque<autonomous_particle<Real, N>>;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  pos_t  m_x0, m_x1;
  real_t m_t1;
  mat_t  m_nabla_phi1;
  mat_t  m_S;
  real_t m_propability_to_be_deleted;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(autonomous_particle const& other);
  autonomous_particle(autonomous_particle&& other) noexcept;
  //----------------------------------------------------------------------------
  auto operator=(autonomous_particle const& other) -> autonomous_particle&;
  auto operator=(autonomous_particle&& other) noexcept -> autonomous_particle&;
  //----------------------------------------------------------------------------
  ~autonomous_particle() = default;
  //----------------------------------------------------------------------------
  autonomous_particle() : m_nabla_phi1{mat_t::eye()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle(pos_t const& x0, real_t t0, real_t r0, real_t propability_to_be_deleted);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle(pos_t const& x0, pos_t const& x1, real_t t1,
                      mat_t const& nabla_phi1, mat_t const& S,
                      real_t propability_to_be_deleted);
  //----------------------------------------------------------------------------
  // getters / setters
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

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  // auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false, stop);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false, stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles), stop);
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                            real_t const max_t,
                            bool const&  stop = false) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t, 0,
                                container_t{*this}, stop);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                            real_t const max_t, size_t const max_num_particles,
                            bool const& stop = false) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t,
                                max_num_particles, container_t{*this}, stop);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                                   real_t const max_t, container_t particles,
                                   bool const& stop = false) {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t, 0,
                                std::move(particles), stop);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                                   real_t const max_t,
                                   size_t const max_num_particles,
                                   container_t  particles,
                                   bool const&  stop = false) {
    [[maybe_unused]] static real_t const x5 = 0.4830517593887872;
    if constexpr (N == 2) {
      return advect(
          phi, tau_step, max_t, 4, max_num_particles,
          std::array{vec_t{real_t(1), real_t(1) / real_t(2)},
                     vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                     vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
          std::array{vec_t{0, 0}, vec_t{0, real_t(3) / 4},
                     vec_t{0, -real_t(3) / 4}},
          // std::array{vec_t{x5, x5 / 2}, vec_t{x5, x5 / 2}, vec_t{x5, x5 / 2},
          //           vec_t{x5, x5 / 2},
          //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
          //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
          // std::array{vec_t{-x5, -x5 / 2}, vec_t{x5, -x5 / 2}, vec_t{-x5, x5 /
          // 2},
          //           vec_t{x5, x5 / 2}, vec_t{0, real_t(3) / 4},
          //           vec_t{0, -real_t(3) / 4}},
          std::move(particles), stop);
    } else if constexpr (N == 3) {
      return advect(
          phi, tau_step, max_t, 4, max_num_particles,
          std::array{
              vec_t{real_t(1), real_t(0), real_t(1) / real_t(2)},
              vec_t{real_t(1) / real_t(2), real_t(0), real_t(1) / real_t(4)},
              vec_t{real_t(1) / real_t(2), real_t(0), real_t(1) / real_t(4)}},
          std::array{vec_t{real_t(0), real_t(0), real_t(0)},
                     vec_t{real_t(0), real_t(0), real_t(3) / real_t(4)},
                     vec_t{real_t(0), real_t(0), -real_t(3) / real_t(4)}},
          // std::array{vec_t{x5, x5 / 2}, vec_t{x5, x5 / 2}, vec_t{x5, x5 / 2},
          //           vec_t{x5, x5 / 2},
          //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
          //           vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
          // std::array{vec_t{-x5, -x5 / 2}, vec_t{x5, -x5 / 2}, vec_t{-x5, x5 /
          // 2},
          //           vec_t{x5, x5 / 2}, vec_t{0, real_t(3) / 4},
          //           vec_t{0, -real_t(3) / 4}},
          std::move(particles), stop);
    }
  }
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles,
  //                          bool const&  stop = false) const {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                          bool const&  stop = false) const {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles), stop);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles,
  //                                 bool const&         stop = false) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles), stop);
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect(Flowmap& phi, real_t const tau_step, real_t const max_t,
              real_t const objective_cond, size_t const max_num_particles,
              std::ranges::range auto const  radii,
              std::ranges::range auto const& offsets,
              bool const&                    stop = false) const {
    return advect(phi, tau_step, max_t, objective_cond, max_num_particles,
                  radii, offsets, {*this}, stop);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect(Flowmap& phi, real_t const tau_step, real_t const max_t,
                     real_t const                   objective_cond,
                     size_t const                   max_num_particles,
                     std::ranges::range auto const  radii,
                     std::ranges::range auto const& offsets,
                     container_t input_particles, bool const& stop = false) {
    auto finished_particles_mutex = std::mutex{};
    auto advected_particles_mutex = std::mutex{};
    auto finished_particles       = container_t{};
    auto particles = std::array{std::move(input_particles), container_t{}};
    auto particles_to_be_advected = &particles[0];
    auto advected_particles       = &particles[1];
    while (!particles_to_be_advected->empty()) {
      if (stop) {
        break;
      }
      advected_particles->clear();
//#pragma omp parallel for
      for (size_t i = 0; i < particles_to_be_advected->size(); ++i) {
        if (!stop) {
          particles_to_be_advected->at(i).advect_until_split(
              phi, tau_step, max_t, objective_cond, radii, offsets,
              *advected_particles, advected_particles_mutex, finished_particles,
              finished_particles_mutex);
        }
      }

      std::swap(particles_to_be_advected, advected_particles);
      if (max_num_particles > 0 &&
          particles_to_be_advected->size() > max_num_particles) {
        size_t const num_particles_to_delete =
            particles_to_be_advected->size() - max_num_particles;

        for (size_t i = 0; i < num_particles_to_delete; ++i) {
          random_uniform<size_t> rand{0, particles_to_be_advected->size() - 1};
          particles_to_be_advected->at(rand()) =
              std::move(particles_to_be_advected->back());
          particles_to_be_advected->pop_back();
        }
      }
      particles_to_be_advected->shrink_to_fit();
    }
    while (max_num_particles > 0 &&
           size(finished_particles) > max_num_particles) {
      random_uniform<size_t> rand{0, size(finished_particles) - 1};
      finished_particles[rand()] = std::move(finished_particles.back());
      finished_particles.pop_back();
    }
    return finished_particles;
  }

  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_until_split(Flowmap& phi, real_t const tau_step,
                          real_t const max_t, real_t const objective_cond,
                          std::ranges::range auto const  radii,
                          std::ranges::range auto const& offsets) const
      -> container_t {
    auto advected = container_t{};
    auto mut      = std::mutex{};

    advect_until_split(phi, tau_step, max_t, objective_cond, radii, offsets,
                       advected, mut, advected, mut);

    return advected;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  auto advect_until_split(Flowmap& phi, real_t tau_step, real_t const max_t,
                          real_t const                   objective_cond,
                          std::ranges::range auto const  radii,
                          std::ranges::range auto const& offsets,
                          std::ranges::range auto& out, auto& out_mutex,
                          std::ranges::range auto& finished_particles,
                          auto& finished_particles_mutex) const {
    bool tau_should_have_changed_but_did_not = false;
    static constexpr real_t min_tau_step = 1e-8;
    static constexpr real_t max_cond_overshoot = 1e-6;
    auto const [Q, lambdas]                    = eigenvectors_sym(m_S);
    auto const Sigma                           = diag(lambdas);
    auto const B = Q * Sigma;  // current main axes

    mat_t                   H, HHt, nabla_phi2, fmg2fmg1, cur_B, cur_S;
    vec_t                   current_radii;
    std::pair<mat_t, vec_t> eig_HHt;
    real_t                  cond_HHt        = 1;
    auto const&             cur_Q           = eig_HHt.first;
    auto const&             cur_lambdas     = eig_HHt.second;
    vec_t                   advected_center = m_x1;
    auto                    ghosts = make_array<vec_t, num_dimensions() * 2>();
    for (size_t i = 0; i < num_dimensions(); ++i) {
      ghosts[i * 2]     = m_x1 + B.col(i);
      ghosts[i * 2 + 1] = m_x1 - B.col(i);
    }
    real_t t2                  = m_t1;
    auto   old_advected_center = advected_center;
    auto   old_t2              = t2;
    auto   old_ghosts          = ghosts;
    auto   old_cond_HHt        = cond_HHt;
    bool   first               = true;
    if constexpr (is_cacheable_v<Flowmap>) {
      phi.use_caching(false);
    }

    while (cond_HHt < objective_cond || t2 < max_t) {
      if (!tau_should_have_changed_but_did_not) {
        if (!first) {
          old_ghosts          = ghosts;
          old_advected_center = advected_center;
          old_cond_HHt        = cond_HHt;
          old_t2              = t2;
        } else {
          first = false;
        }

        if (t2 + tau_step > max_t) {
          tau_step = max_t - t2;
          t2       = max_t;
        } else {
          t2 += tau_step;
        }

        advected_center = phi(advected_center, old_t2, t2 - old_t2);
        for (size_t i = 0; i < num_dimensions(); ++i) {
          ghosts[i * 2]     = phi(ghosts[i * 2], old_t2, t2 - old_t2);
          ghosts[i * 2 + 1] = phi(ghosts[i * 2 + 1], old_t2, t2 - old_t2);
          H.col(i)          = ghosts[i * 2] - ghosts[i * 2 + 1];
        }
        H *= half;

        HHt      = H * transposed(H);
        eig_HHt  = eigenvectors_sym(HHt);
        cond_HHt = cur_lambdas(num_dimensions() - 1) / cur_lambdas(0);

        nabla_phi2 = H * *inv(Sigma) * transposed(Q);
        fmg2fmg1   = nabla_phi2 * m_nabla_phi1;

        current_radii = sqrt(cur_lambdas);
        cur_B         = cur_Q * diag(current_radii);
        cur_S         = cur_B * transposed(cur_Q);
      }

      if (t2 == max_t && cond_HHt <= objective_cond + max_cond_overshoot) {
        auto lock = std::lock_guard{finished_particles_mutex};
        finished_particles.emplace_back(m_x0, advected_center, t2, fmg2fmg1,
                                        cur_S, m_propability_to_be_deleted);
        return;
      } 
      if ((cond_HHt >= objective_cond &&
                  cond_HHt <= objective_cond + max_cond_overshoot) ||
                 tau_should_have_changed_but_did_not) {
        for (size_t i = 0; i < size(radii); ++i) {
          auto const      new_eigvals = current_radii * radii[i];
          auto const      new_S = cur_Q * diag(new_eigvals) * transposed(cur_Q);
          auto const      offset2 = cur_B * offsets[i];
          auto const      offset0 = *inv(fmg2fmg1) * offset2;
          std::lock_guard lock{out_mutex};
          out.emplace_back(m_x0 + offset0, advected_center + offset2, t2,
                           fmg2fmg1, new_S, m_propability_to_be_deleted);
          // std::lock_guard lock2{finished_particles_mutex};
          // finished_particles.push_back(out.back());
        }
        // std::lock_guard lock{finished_particles_mutex};
        // finished_particles.emplace_back(m_x0, advected_center, t2,
        // fmg2fmg1, cur_S, m_propability_to_be_deleted);
        return;
      }
      if (cond_HHt > objective_cond + max_cond_overshoot) {
        // if (old_cond_HHt < objective_cond) {
        //  auto const _t =
        //      (old_cond_HHt - objective_cond) / (old_cond_HHt - cond_HHt);
        //  assert(_t >= 0 && _t <= 1);
        //  tau_step *= _t;
        //}
        auto const old_tau_step = tau_step;
        tau_step *= half;
        tau_should_have_changed_but_did_not = tau_step == old_tau_step;
        if (tau_step < min_tau_step) {
          tau_should_have_changed_but_did_not = true;
        }

        if (!tau_should_have_changed_but_did_not) {
          cond_HHt        = old_cond_HHt;
          ghosts          = old_ghosts;
          t2              = old_t2;
          advected_center = old_advected_center;
        }
        //} else {
        //   auto const _t =
        //      (old_cond_HHt - objective_cond) / (old_cond_HHt - cond_HHt);
        //   assert(_t >= 1);
        //   tau_step *= _t;
      }
    }
  }
};
//==============================================================================
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle const& other)
    : m_x0{other.m_x0},
      m_x1{other.m_x1},
      m_t1{other.m_t1},
      m_nabla_phi1{other.m_nabla_phi1},
      m_S{other.m_S} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle&& other) noexcept
    : m_x0{std::move(other.m_x0)},
      m_x1{std::move(other.m_x1)},
      m_t1{other.m_t1},
      m_nabla_phi1{std::move(other.m_nabla_phi1)},
      m_S{std::move(other.m_S)} {}

//----------------------------------------------------------------------------
template <typename Real, size_t N>
auto autonomous_particle<Real, N>::operator=(autonomous_particle const& other)
    -> autonomous_particle& {
  if (&other == this) {
    return *this;
  };
  m_x0         = other.m_x0;
  m_x1         = other.m_x1;
  m_t1         = other.m_t1;
  m_nabla_phi1 = other.m_nabla_phi1;
  m_S          = other.m_S;
  return *this;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
auto autonomous_particle<Real, N>::operator=(
    autonomous_particle&& other) noexcept -> autonomous_particle& {
  m_x0         = std::move(other.m_x0);
  m_x1         = std::move(other.m_x1);
  m_t1         = other.m_t1;
  m_nabla_phi1 = std::move(other.m_nabla_phi1);
  m_S          = std::move(other.m_S);
  return *this;
}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    pos_t const& x0, real_t const t0, real_t const r0,
    real_t const propability_to_be_deleted)
    : m_x0{x0},
      m_x1{x0},
      m_t1{t0},
      m_nabla_phi1{mat_t::eye()},
      m_S{mat_t::eye() * r0},
      m_propability_to_be_deleted{propability_to_be_deleted} {}

//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(pos_t const& x0,
                                                  pos_t const& x1,
                                                  real_t const t1,
                                                  mat_t const& nabla_phi1,
                                                  mat_t const& S,
    real_t const propability_to_be_deleted)
    : m_x0{x0}, m_x1{x1}, m_t1{t1}, m_nabla_phi1{nabla_phi1}, m_S{S},
      m_propability_to_be_deleted{propability_to_be_deleted} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// template <typename Flowmap, arithmetic RealX0, size_t N>
// autonomous_particle(Flowmap& phi, vec<RealX0, N> const&, arithmetic auto
// const,
//                    arithmetic auto const)
//    -> autonomous_particle<std::decay_t<V>,
//                           std::decay_t<decltype(flowmap(v))>>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Flowmap, arithmetic RealX0, size_t N>
// autonomous_particle(V& v, vec<RealX0, N> const&, arithmetic auto const,
//                    arithmetic auto const)
//    -> autonomous_particle<std::decay_t<V>,
//                           std::decay_t<decltype(flowmap(v))>>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Flowmap, size_t N, arithmetic RealX0>
// autonomous_particle(V&& v, vec<RealX0, N> const&, arithmetic auto const,
//                    arithmetic auto const)
//    -> autonomous_particle<std::decay_t<V>,
//    std::decay_t<decltype(flowmap(v))>>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Flowmap, size_t N, arithmetic RealX0>
// autonomous_particle(V* v, vec<RealX0, N> const&, arithmetic auto const,
//                    arithmetic auto const)
//    -> autonomous_particle<std::decay_t<V>*,
//                           std::decay_t<decltype(flowmap(v))>>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Flowmap, size_t N, arithmetic RealX0>
// autonomous_particle(V const* v, vec<RealX0, N> const&, arithmetic auto const,
//                    arithmetic auto const)
//    -> autonomous_particle<std::decay_t<V> const*,
//                           std::decay_t<decltype(flowmap(v))>>;
using autonomous_particle_2 = autonomous_particle<real_t, 2>;
using autonomous_particle_3 = autonomous_particle<real_t, 3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
