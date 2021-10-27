#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/cache_alignment.h>
#include <tatooine/concepts.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/random.h>
#include <tatooine/reflection.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle_sampler {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using vec_t     = vec<Real, N>;
  using pos_t     = vec_t;
  using mat_t     = mat<Real, N, N>;
  using ellipse_t = geometry::hyper_ellipse<Real, N>;

 private:
  //============================================================================
  // MEMBERS
  //============================================================================
  ellipse_t m_ellipse0, m_ellipse1;
  mat_t     m_nabla_phi, m_nabla_phi_inv;

 public:
  //============================================================================
  // CTORS
  //============================================================================
  autonomous_particle_sampler(autonomous_particle_sampler const&)     = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle_sampler(autonomous_particle_sampler&&) noexcept = default;
  //============================================================================
  auto operator=(autonomous_particle_sampler const&)
      -> autonomous_particle_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(autonomous_particle_sampler&&) noexcept
      -> autonomous_particle_sampler& = default;
  //============================================================================
  autonomous_particle_sampler(ellipse_t const& e0, ellipse_t const& e1,
                              mat_t const& nabla_phi)
      : m_ellipse0{e0},
        m_ellipse1{e1},
        m_nabla_phi{nabla_phi},
        m_nabla_phi_inv{*inv(nabla_phi)} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto ellipse(tag::forward_t) const -> auto const& { return m_ellipse0; }
  auto ellipse(tag::backward_t) const -> auto const& { return m_ellipse1; }
  auto ellipse0() const -> auto const& { return m_ellipse0; }
  auto ellipse1() const -> auto const& { return m_ellipse1; }
  //============================================================================
  // METHODS
  //============================================================================
  auto sample_forward(pos_t const& x) const {
    return ellipse1().center() + m_nabla_phi * (x - ellipse0().center());
  }
  auto operator()(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  auto sample(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  auto sample_backward(pos_t const& x) const {
    return ellipse0().center() + m_nabla_phi_inv * (x - ellipse1().center());
  }
  auto sample(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
  auto operator()(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
  auto is_inside0(pos_t const& x) const { return m_ellipse0.is_inside(x); }
  auto is_inside(pos_t const& x, tag::forward_t /*tag*/) const {
    return is_inside0(x);
  }
  auto is_inside1(pos_t const& x) const { return m_ellipse1.is_inside(x); }
  auto is_inside(pos_t const& x, tag::backward_t /*tag*/) const {
    return is_inside1(x);
  }
  auto center(tag::forward_t /*tag*/) const -> auto const& {
    return m_ellipse0.center();
  }
  auto center(tag::backward_t /*tag*/) const -> auto const& {
    return m_ellipse1.center();
  }
  auto distance_sqr(pos_t const& x, tag::forward_t tag) const {
    return tatooine::length(m_nabla_phi * (x - center(tag)));
  }
  auto distance_sqr(pos_t const& x, tag::backward_t tag) const {
    return tatooine::length(solve(m_nabla_phi, (x - center(tag))));
  }
  template <typename Tag>
  auto distance(pos_t const& x, Tag tag) const {
    return distance_sqr(x, tag);
  }
};
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle : geometry::hyper_ellipse<Real, N> {
  static constexpr auto num_dimensions() { return N; }
  //============================================================================
  // TYPEDEFS
  //============================================================================
 public:
  constexpr static auto half = Real(0.5);
  using this_t               = autonomous_particle<Real, N>;
  using real_t               = Real;
  using vec_t                = vec<real_t, N>;
  using mat_t                = mat<real_t, N, N>;
  using pos_t                = vec_t;
  using container_t          = std::vector<this_t>;
  using ellipse_t            = geometry::hyper_ellipse<Real, N>;
  using parent_t             = ellipse_t;
  using sampler_t            = autonomous_particle_sampler<Real, N>;
  //============================================================================
  // members
  //============================================================================
 private:
  pos_t  m_x0;
  real_t m_t1;
  mat_t  m_nabla_phi1;

  //============================================================================
  // CTORS
  //============================================================================
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
  autonomous_particle(pos_t const& x0, real_t const t0, real_t const r0);
  autonomous_particle(ellipse_t const& ell, real_t const t0);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle(pos_t const& x0, real_t const t1, mat_t const& nabla_phi1,
                      ellipse_t const& ell);
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return m_x0(i); }
  //----------------------------------------------------------------------------
  auto x1() -> auto& { return parent_t::center(); }
  auto x1() const -> auto const& { return parent_t::center(); }
  auto x1(size_t i) const { return parent_t::center()(i); }
  //----------------------------------------------------------------------------
  auto t1() -> auto& { return m_t1; }
  auto t1() const { return m_t1; }
  //----------------------------------------------------------------------------
  auto nabla_phi1() const -> auto const& { return m_nabla_phi1; }
  //----------------------------------------------------------------------------
  auto S1() -> auto& { return parent_t::S(); }
  auto S1() const -> auto const& { return parent_t::S(); }
  //----------------------------------------------------------------------------
  auto S0() const {
    auto sqrS =
        *inv(nabla_phi1()) * S1() * S1() * *inv(transposed(nabla_phi1()));
    auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
    for (size_t i = 0; i < N; ++i) {
      eig_vals(i) = std::sqrt(eig_vals(i));
    }
    return eig_vecs * diag(eig_vals) * transposed(eig_vecs);
  }
  //----------------------------------------------------------------------------
  auto initial_ellipse() const { return ellipse_t{m_x0, S0()}; }
  //============================================================================
  // METHODS
  //============================================================================
  // auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_2_splits(real_t const tau_step, real_t const max_t) const
  // {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(tau_step, max_t, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles));
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                            real_t const max_t) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t, 0,
                                container_t{*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                            real_t const max_t,
                            size_t const max_num_particles) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t,
                                max_num_particles, container_t{*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                                   real_t const max_t, container_t particles) {
    return advect_with_3_splits(std::forward<Flowmap>(phi), tau_step, max_t, 0,
                                std::move(particles));
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const tau_step,
                                   real_t const max_t,
                                   size_t const max_num_particles,
                                   container_t  particles) {
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
          std::move(particles));
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
          std::move(particles));
    }
  }
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const tau_step, real_t const max_t) const
  // {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(tau_step, max_t, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                          size_t const max_num_particles) const {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const tau_step, real_t const max_t) const
  // {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
 //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const tau_step, real_t const max_t,
  //                                 container_t particles) {
  //  return advect(tau_step, max_t, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles));
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect(Flowmap& phi, real_t const tau_step, real_t const max_t,
              real_t const objective_cond, size_t const max_num_particles,
              range auto const radii, range auto const& offsets) const {
    return advect(phi, tau_step, max_t, objective_cond, max_num_particles,
                  radii, offsets, {*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect(Flowmap& phi, real_t const tau_step, real_t const max_t,
                     real_t const objective_cond,
                     size_t const max_num_particles, range auto const radii,
                     range auto const& offsets, container_t const& particles) {
    //auto       finished_particles = container_t{};
    auto const num_threads =
        static_cast<size_t>(std::thread::hardware_concurrency());
    auto const path = filesystem::path{"temp.h5"};
    if (filesystem::exists(path)) {
      filesystem::remove(path);
    }
    auto file     = hdf5::file{path};
    auto hdd_data =
        std::array{file.add_dataset<typename container_t::value_type>(
                       "ping", hdf5::unlimited),
                   file.add_dataset<typename container_t::value_type>(
                       "pong", hdf5::unlimited)};
    auto finished = file.add_dataset<typename container_t::value_type>(
        "finished", hdf5::unlimited);
    size_t reader   = 0;
    size_t writer   = 1;
    hdd_data[reader].write(particles);

    while (hdd_data[reader].dataspace().current_resolution()[0] > 0) {
      auto const num_particles =
          hdd_data[reader].dataspace().current_resolution()[0];
      auto thread_ranges     = std::vector<aligned<std::pair<size_t, size_t>>>(num_threads);
      {
        // distribute particles
        auto thread_pool = std::vector<std::thread>{};
        thread_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                size_t const begin = thr_id * num_particles / num_threads;
                size_t const end   = (thr_id + 1) * num_particles / num_threads;

                *thread_ranges[thr_id] = std::pair{begin, end};
              },
              i);
        }
        for (auto& thread : thread_pool) {
          thread.join();
        }
      }
      {
        // advect particle pools
        auto thread_pool = std::vector<std::thread>{};
        thread_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                auto const& range = *thread_ranges[thr_id];
                for (size_t i = range.first; i < range.second; ++i) {
                  auto const particle = hdd_data[reader][i];
                  particle.advect_until_split(
                      phi, tau_step, max_t, objective_cond, radii, offsets,
                      hdd_data[writer],
                      finished);
                }
              },
              i);
        }
        for (auto& thread : thread_pool) {
          thread.join();
        }
      }

      reader = 1 - reader;
      writer = 1 - writer;
      hdd_data[writer].clear();
    }
    hdd_data[reader].clear();
  }
  //----------------------------------------------------------------------------
  // template <typename Flowmap>
  // auto advect_until_split(Flowmap& phi, real_t const tau_step,
  //                        real_t const max_t, real_t const objective_cond,
  //                        range auto const  radii,
  //                        range auto const& offsets) const -> container_t {
  //  auto advected = container_t{};
  //  advect_until_split(phi, tau_step, max_t, objective_cond, radii, offsets,
  //                     advected, advected);
  //
  //  return advected;
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  auto advect_until_split(Flowmap phi, real_t tau_step, real_t const max_t,
                          real_t const objective_cond, range auto const radii,
                          range auto const& offsets, hdf5::dataset<this_t>& out,
                          hdf5::dataset<this_t>& finished_particles) const {
    bool                    tau_should_have_changed_but_did_not = false;
    static constexpr real_t min_tau_step                        = 1e-8;
    static constexpr real_t max_cond_overshoot                  = 1e-6;
    auto const [Q, lambdas] = eigenvectors_sym(S1());
    auto const Sigma        = diag(lambdas);
    auto const B            = Q * Sigma;  // current main axes

    mat_t                   H, HHt, nabla_phi2, fmg2fmg1, cur_B;
    ellipse_t               advected_ellipse = *this;
    vec_t                   current_radii;
    std::pair<mat_t, vec_t> eig_HHt;
    real_t                  cond_HHt    = 1;
    auto const&             cur_Q       = eig_HHt.first;
    auto const&             cur_lambdas = eig_HHt.second;
    auto                    ghosts = make_array<vec_t, num_dimensions() * 2>();
    for (size_t i = 0; i < num_dimensions(); ++i) {
      ghosts[i * 2]     = x1() + B.col(i);
      ghosts[i * 2 + 1] = x1() - B.col(i);
    }
    real_t t2                  = m_t1;
    auto   old_advected_center = advected_ellipse.center();
    auto   old_t2              = t2;
    auto   old_ghosts          = ghosts;
    auto   old_cond_HHt        = cond_HHt;
    bool   first               = true;
    if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
      phi.use_caching(false);
    }

    while (cond_HHt < objective_cond || t2 < max_t) {
      if (!tau_should_have_changed_but_did_not) {
        if (!first) {
          old_ghosts          = ghosts;
          old_advected_center = advected_ellipse.center();
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

        advected_ellipse.center() =
            phi(advected_ellipse.center(), old_t2, t2 - old_t2);
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

        current_radii        = sqrt(cur_lambdas);
        cur_B                = cur_Q * diag(current_radii);
        advected_ellipse.S() = cur_B * transposed(cur_Q);
      }

      static std::mutex mutex;
      if (t2 == max_t && cond_HHt <= objective_cond + max_cond_overshoot) {
        auto lock = std::lock_guard{mutex};
        finished_particles.push_back(this_t{m_x0, t2, fmg2fmg1, advected_ellipse});
        return;
      }
      if ((cond_HHt >= objective_cond &&
           cond_HHt <= objective_cond + max_cond_overshoot) ||
          tau_should_have_changed_but_did_not) {
        real_t acc_radii = 0;
        for (auto const& radius : radii) {
          acc_radii += radius(0) * radius(1);
        }
        {
          auto lock = std::lock_guard{mutex};
          for (size_t i = 0; i < size(radii); ++i) {
            auto const new_eigvals = current_radii * radii[i];
            auto const offset2     = cur_B * offsets[i];
            auto const offset0     = *inv(fmg2fmg1) * offset2;
            auto       offset_ellipse =
                ellipse_t{advected_ellipse.center() + offset2,
                          cur_Q * diag(new_eigvals) * transposed(cur_Q)};
            out.push_back(this_t{m_x0 + offset0, t2, fmg2fmg1, offset_ellipse});
          }
        }
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
          cond_HHt                  = old_cond_HHt;
          ghosts                    = old_ghosts;
          t2                        = old_t2;
          advected_ellipse.center() = old_advected_center;
        }
        //} else {
        //   auto const _t =
        //      (old_cond_HHt - objective_cond) / (old_cond_HHt - cond_HHt);
        //   assert(_t >= 1);
        //   tau_step *= _t;
      }
    }
  }
  auto sampler() const {
    return sampler_t{initial_ellipse(), *this, m_nabla_phi1};
  }
};
//==============================================================================
namespace reflection {
template <typename Real, size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (autonomous_particle<Real, N>),
    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
    TATOOINE_REFLECTION_INSERT_METHOD(S, S()),
    TATOOINE_REFLECTION_INSERT_METHOD(x0, x0()),
    TATOOINE_REFLECTION_INSERT_METHOD(t1, t1()),
    TATOOINE_REFLECTION_INSERT_METHOD(nabla_phi1, nabla_phi1()))
}  // namespace reflection
//==============================================================================
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle const& other)
    : parent_t{other},
      m_x0{other.m_x0},
      m_t1{other.m_t1},
      m_nabla_phi1{other.m_nabla_phi1} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle&& other) noexcept
    : parent_t{std::move(other)},
      m_x0{std::move(other.m_x0)},
      m_t1{other.m_t1},
      m_nabla_phi1{std::move(other.m_nabla_phi1)} {}

//----------------------------------------------------------------------------
template <typename Real, size_t N>
auto autonomous_particle<Real, N>::operator=(autonomous_particle const& other)
    -> autonomous_particle& {
  if (&other == this) {
    return *this;
  };
  parent_t::operator=(other);
  m_x0              = other.m_x0;
  m_t1              = other.m_t1;
  m_nabla_phi1      = other.m_nabla_phi1;
  return *this;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
auto autonomous_particle<Real, N>::operator=(
    autonomous_particle&& other) noexcept -> autonomous_particle& {
  parent_t::operator=(std::move(other));
  m_x0              = std::move(other.m_x0);
  m_t1              = other.m_t1;
  m_nabla_phi1      = std::move(other.m_nabla_phi1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(ellipse_t const& ell,
                                                  real_t const     t0)
    : parent_t{ell}, m_x0{ell.center()}, m_t1{t0}, m_nabla_phi1{mat_t::eye()} {}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(pos_t const& x0,
                                                  real_t const t0,
                                                  real_t const r0)
    : parent_t{x0, r0}, m_x0{x0}, m_t1{t0}, m_nabla_phi1{mat_t::eye()} {}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(pos_t const&     x0,
                                                  real_t const     t1,
                                                  mat_t const&     nabla_phi1,
                                                  ellipse_t const& ell)
    : parent_t{ell}, m_x0{x0}, m_t1{t1}, m_nabla_phi1{nabla_phi1} {}
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
//==============================================================================
template <size_t N>
using AutonomousParticle    = autonomous_particle<real_t, N>;
using autonomous_particle_2 = AutonomousParticle<2>;
using autonomous_particle_3 = AutonomousParticle<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
