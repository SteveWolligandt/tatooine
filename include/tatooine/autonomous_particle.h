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
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t N>
struct sampler {
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
  sampler(sampler const&)     = default;
  sampler(sampler&&) noexcept = default;
  //============================================================================
  auto operator=(sampler const&) -> sampler& = default;
  auto operator=(sampler&&) noexcept -> sampler& = default;
  //============================================================================
  sampler()  = default;
  ~sampler() = default;
  //----------------------------------------------------------------------------
  sampler(ellipse_t const& e0, ellipse_t const& e1, mat_t const& nabla_phi)
      : m_ellipse0{e0},
        m_ellipse1{e1},
        m_nabla_phi{nabla_phi},
        m_nabla_phi_inv{*inv(nabla_phi)} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto ellipse(tag::forward_t /*tag*/) const -> auto const& {
    return m_ellipse0;
  }
  auto ellipse(tag::backward_t /*tag*/) const -> auto const& {
    return m_ellipse1;
  }
  auto ellipse0() const -> auto const& { return m_ellipse0; }
  auto ellipse1() const -> auto const& { return m_ellipse1; }
  //============================================================================
  // METHODS
  //============================================================================
  auto sample_forward(pos_t const& x) const {
    return ellipse1().center() + nabla_phi() * (x - ellipse0().center());
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
    return tatooine::length(nabla_phi() * (x - center(tag)));
  }
  auto distance_sqr(pos_t const& x, tag::backward_t tag) const {
    return tatooine::length(solve(nabla_phi(), (x - center(tag))));
  }
  auto distance(pos_t const& x, auto const tag) const {
    return gcem::sqrt(distance_sqr(x, tag));
  }
  auto nabla_phi() const -> auto const& { return m_nabla_phi; }
  auto nabla_phi_inv() const -> auto const& { return m_nabla_phi_inv; }
};
//==============================================================================
template <floating_point Real, std::size_t N>
struct split_behaviors;
//==============================================================================
template <floating_point Real>
struct split_behaviors<Real, 2> {
  static auto constexpr one            = Real(1);
  static auto constexpr half           = one / Real(2);
  static auto constexpr quarter        = one / Real(4);
  static auto constexpr three_quarters = 3 * quarter;
  static auto constexpr sqrt5          = gcem::sqrt<Real>(5);
  using vec_t                          = vec<Real, 2>;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct two_splits {
    static auto constexpr sqrt2    = gcem::sqrt(real_t(2));
    static auto constexpr sqr_cond = 2;
    static auto constexpr half     = Real(1) / Real(2);
    static auto constexpr quarter  = Real(1) / Real(4);
    static constexpr auto radii =
        std::array{vec_t{1 / sqrt2, half}, vec_t{1 / sqrt2, half}};
    static constexpr auto offsets = std::array{vec_t{0, -half}, vec_t{0, half}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct three_splits {
    static auto constexpr sqr_cond = Real(4);
    static constexpr auto radii    = std::array{
        vec_t{half, quarter}, vec_t{one, half}, vec_t{half, quarter}};
    static constexpr auto offsets = std::array{
        vec_t{0, -three_quarters}, vec_t{0, 0}, vec_t{0, three_quarters}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct five_splits {
    static auto constexpr sqr_cond = Real(6 + sqrt5 * 2);
    static auto constexpr radii    = std::array{
        vec_t{1, 1 / (sqrt5 + 1)}, vec_t{1, (sqrt5 + 3) / (sqrt5 * 2 + 2)},
        vec_t{1, 1}, vec_t{1, (sqrt5 + 3) / (sqrt5 * 2 + 2)},
        vec_t{1, 1 / (sqrt5 + 1)}};
    static auto constexpr offsets = std::array{
        vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct seven_splits {
    static auto constexpr sqr_cond = 4.493959210 * 4.493959210;
    static auto constexpr radii    = std::array{
        real_t(.9009688678), real_t(.6234898004), real_t(.2225209338)};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct centered_four {
    static auto constexpr x5       = Real(0.4830517593887872);
    static auto constexpr sqr_cond = Real{4};
    static auto constexpr radii =
        std::array{vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                   vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}};
    static auto constexpr offsets = std::array{
        vec_t{-x5, -x5 / 2}, vec_t{x5, -x5 / 2},      vec_t{-x5, x5 / 2},
        vec_t{x5, x5 / 2},   vec_t{0, real_t(3) / 4}, vec_t{0, -real_t(3) / 4}};
  };
};
//==============================================================================
template <floating_point Real>
struct split_behaviors<Real, 3> {
  static auto constexpr half           = 1 / Real(2);
  static auto constexpr quarter        = 1 / Real(4);
  static auto constexpr three_quarters = 3 * quarter;
  using vec_t                          = vec<Real, 3>;
  struct three_splits {
    static auto constexpr sqr_cond = Real{4};
    static constexpr auto radii =
        std::array{vec_t{half, half, quarter}, vec_t{1, 1, half},
                   vec_t{half, half, quarter}};
    static constexpr auto offsets =
        std::array{vec_t{0, 0, -three_quarters}, vec_t{0, 0, 0},
                   vec_t{0, 0, three_quarters}};
  };
};
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename B>
concept split_behavior = requires {
  floating_point<decltype(B::sqr_cond)>;
  range<decltype(B::radii)>;
  range<decltype(B::offsets)>;
  is_vec<typename decltype(B::radii)::value_type>;
  is_vec<typename decltype(B::offsets)::value_type>;
};
//==============================================================================
template <floating_point Real, std::size_t N>
struct autonomous_particle : geometry::hyper_ellipse<Real, N> {
  using split_behaviors = detail::autonomous_particle::split_behaviors<Real, N>;
  static constexpr auto num_dimensions() { return N; }
  //============================================================================
  // TYPEDEFS
  //============================================================================
 public:
  constexpr static auto half = 1 / Real(2);

  using this_t      = autonomous_particle<Real, N>;
  using real_t      = Real;
  using vec_t       = vec<real_t, N>;
  using mat_t       = mat<real_t, N, N>;
  using pos_t       = vec_t;
  using container_t = std::vector<this_t>;
  using ellipse_t   = geometry::hyper_ellipse<Real, N>;
  using parent_t    = ellipse_t;
  using sampler_t   = detail::autonomous_particle::sampler<Real, N>;
  using parent_t::center;
  using parent_t::discretize;
  using parent_t::S;
  //============================================================================
  // members
  //============================================================================
 private:
  pos_t  m_x0;
  real_t m_t;
  mat_t  m_nabla_phi;

  static auto mutex() -> auto& {
    static auto m = std::mutex{};
    return m;
  }
  //============================================================================
  // CTORS
  //============================================================================
 public:
  autonomous_particle(autonomous_particle const& other)     = default;
  autonomous_particle(autonomous_particle&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator               =(autonomous_particle const& other)
      -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&& other) noexcept
      -> autonomous_particle& = default;
  //----------------------------------------------------------------------------
  ~autonomous_particle() = default;
  //----------------------------------------------------------------------------
  autonomous_particle() : m_nabla_phi{mat_t::eye()} {}
  autonomous_particle(ellipse_t const& ell, real_t const t)
      : parent_t{ell}, m_x0{ell.center()}, m_t{t}, m_nabla_phi{mat_t::eye()} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_t const& x, real_t const t, real_t const r)
      : parent_t{x, r}, m_x0{x}, m_t{t}, m_nabla_phi{mat_t::eye()} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_t const& ell, real_t const t, pos_t const& x0,
                      mat_t const& nabla_phi)
      : parent_t{ell}, m_x0{x0}, m_t{t}, m_nabla_phi{nabla_phi} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(std::size_t i) const { return x0()(i); }
  //----------------------------------------------------------------------------
  auto x() -> auto& { return parent_t::center(); }
  auto x() const -> auto const& { return parent_t::center(); }
  auto x(std::size_t i) const { return parent_t::center()(i); }
  //----------------------------------------------------------------------------
  auto t() -> auto& { return m_t; }
  auto t() const { return m_t; }
  //----------------------------------------------------------------------------
  auto nabla_phi() const -> auto const& { return m_nabla_phi; }
  //----------------------------------------------------------------------------
  auto S0() const {
    auto sqrS = *inv(nabla_phi()) * S() * S() * *inv(transposed(nabla_phi()));
    auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
    for (std::size_t i = 0; i < N; ++i) {
      eig_vals(i) = gcem::sqrt(eig_vals(i));
    }
    return eig_vecs * diag(eig_vals) * transposed(eig_vecs);
  }
  //----------------------------------------------------------------------------
  auto initial_ellipse() const { return ellipse_t{x0(), S0()}; }
  //============================================================================
  // METHODS
  //============================================================================
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  auto advect(Flowmap&& phi, real_t const step_size, real_t const t_end,
              filesystem::path const& path) const {
    return advect(std::forward<Flowmap>(phi), step_size, t_end, {*this}, path);
  }
  //----------------------------------------------------------------------------
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  static auto advect(Flowmap&& phi, real_t const step_size, real_t const t_end,
                     container_t const&      particles,
                     filesystem::path const& path) {
    // auto       finished_particles = container_t{};
    auto const num_threads =
        static_cast<std::size_t>(std::thread::hardware_concurrency());
    if (filesystem::exists(path)) {
      filesystem::remove(path);
    }
    auto file = hdf5::file{path};
    auto hdd_data =
        std::array{file.create_dataset<typename container_t::value_type>(
                       "ping", hdf5::unlimited),
                   file.create_dataset<typename container_t::value_type>(
                       "pong", hdf5::unlimited)};
    auto finished = file.create_dataset<typename container_t::value_type>(
        "finished", hdf5::unlimited);
    std::size_t reader = 0;
    std::size_t writer = 1;
    hdd_data[reader].write(particles);

    while (hdd_data[reader].dataspace().current_resolution()[0] > 0) {
      auto const num_particles =
          hdd_data[reader].dataspace().current_resolution()[0];
      auto thread_ranges =
          std::vector<aligned<std::pair<std::size_t, std::size_t>>>(
              num_threads);
      auto advected_particles = std::vector<aligned<container_t>>(num_threads);
      auto finished_particles = std::vector<aligned<container_t>>(num_threads);
      auto loaded_particles   = std::vector<aligned<container_t>>(num_threads);
      std::size_t const num_particles_at_once = 10000000;
      for (auto& l : loaded_particles) {
        l->reserve(num_particles_at_once);
      }
      {
        // distribute particles
        auto thread_pool = std::vector<std::thread>{};
        thread_pool.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                std::size_t const begin = thr_id * num_particles / num_threads;
                std::size_t const end =
                    (thr_id + 1) * num_particles / num_threads;

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
        for (std::size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                auto const& range     = *thread_ranges[thr_id];
                auto&       particles = *loaded_particles[thr_id];
                std::size_t chunk_idx = 0;
                for (std::size_t i = range.first; i < range.second;
                     i += num_particles_at_once, ++chunk_idx) {
                  auto const cur_num_particles =
                      std::min(num_particles_at_once, range.second - i);
                  particles.resize(cur_num_particles);
                  {
                    auto lock = std::lock_guard{mutex()};
                    hdd_data[reader].read(i, cur_num_particles, particles);
                  }
                  for (auto const& particle : particles) {
                    particle.template advect_until_split<SplitBehavior>(
                        std::forward<Flowmap>(phi), step_size, t_end,
                        *advected_particles[thr_id],
                        *finished_particles[thr_id]);
                    if (advected_particles[thr_id]->size() > 10000000) {
                      {
                        auto lock = std::lock_guard{mutex()};
                        hdd_data[writer].push_back(*advected_particles[thr_id]);
                      }
                      advected_particles[thr_id]->clear();
                    }
                    if (finished_particles[thr_id]->size() > 10000000) {
                      {
                        auto lock = std::lock_guard{mutex()};
                        finished.push_back(*finished_particles[thr_id]);
                      }
                      finished_particles[thr_id]->clear();
                    }
                    if (!advected_particles[thr_id]->empty()) {
                      {
                        auto lock = std::lock_guard{mutex()};
                        hdd_data[writer].push_back(*advected_particles[thr_id]);
                      }
                      advected_particles[thr_id]->clear();
                    }
                    if (!finished_particles[thr_id]->empty()) {
                      {
                        auto lock = std::lock_guard{mutex()};
                        finished.push_back(*finished_particles[thr_id]);
                      }
                      finished_particles[thr_id]->clear();
                    }
                  }
                  particles.clear();
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
    return path;
  }
  //----------------------------------------------------------------------------
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  auto advect(Flowmap&& phi, real_t const step_size, real_t const t_end) const {
    return advect<SplitBehavior>(std::forward<Flowmap>(phi), step_size, t_end,
                                 {*this});
  }
  //----------------------------------------------------------------------------
  static auto num_threads() {
    auto num_threads = std::size_t{};
#pragma omp parallel
    {
      if (omp_get_thread_num() == 0) {
        num_threads = omp_get_num_threads();
      }
    }
    return num_threads;
  }
  //----------------------------------------------------------------------------
 private:
  static auto distribute_particles_to_thread_containers(
      std::size_t const num_threads, container_t& particles,
      auto& particles_per_thread) {
#pragma omp parallel
    {
      auto const        thr_id = omp_get_thread_num();
      std::size_t const begin  = thr_id * size(particles) / num_threads;
      std::size_t const end    = (thr_id + 1) * size(particles) / num_threads;
      auto&             cont   = *particles_per_thread[thr_id][0];
      cont.reserve(end - begin);
      std::copy(particles.begin() + begin, particles.begin() + end,
                std::back_inserter(cont));
    }
    particles.clear();
  }
  //----------------------------------------------------------------------------
  template <split_behavior SplitBehavior, typename Flowmap>
  static auto advect_particle_pools(std::size_t const num_threads,
                                    Flowmap&& phi, real_t const step_size,
                                    real_t const      t_end,
                                    auto&             particles_per_thread,
                                    std::atomic_bool& stop) {
#pragma omp parallel
    {
      auto const thr_id          = omp_get_thread_num();
      auto&      particles_at_t0 = *particles_per_thread[thr_id][0];
      auto&      particles_at_t1 = *particles_per_thread[thr_id][1];
      auto&      finished        = *particles_per_thread[thr_id][2];
      for (auto const& particle : particles_at_t0) {
        //try {
          particle.template advect_until_split<SplitBehavior>(
              std::forward<Flowmap>(phi), step_size, t_end, particles_at_t1,
              finished);
        //} catch (...) {
        //  stop = true;
        //}
        if (stop) {
          break;
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  static auto gather_particles(container_t& particles,
                               container_t& finished_particles,
                               auto&        particles_per_thread) {
    for (auto& ps : particles_per_thread) {
      using namespace std::ranges;
      auto& base     = *ps[0];
      auto& advected = *ps[1];
      auto& finished = *ps[2];
      copy(advected, std::back_inserter(particles));
      copy(finished, std::back_inserter(finished_particles));
      base.clear();
      advected.clear();
      finished.clear();
    }
  }
  //----------------------------------------------------------------------------
 public:
  /// Advects all particles in particles container in the flowmap phi until
  /// time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param step_size Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  /// \param particles Particles to be advected.
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  static auto advect(Flowmap&& phi, real_t const step_size, real_t const t_end,
                     container_t particles) {
    std::atomic_bool stop        = false;
    auto const       num_threads = this_t::num_threads();

    auto finished_particles = container_t{};

    // {particles_to_be_advected, advected_particles, finished_particles}
    auto particles_per_thread =
        std::vector<std::array<aligned<container_t>, 3>>(num_threads);
    while (!stop && particles.size() > 0) {
      distribute_particles_to_thread_containers(num_threads, particles,
                                                particles_per_thread);
      advect_particle_pools<SplitBehavior>(
          num_threads, std::forward<Flowmap>(phi), step_size, t_end,
          particles_per_thread, stop);
      gather_particles(particles, finished_particles, particles_per_thread);
    }
    return finished_particles;
  }
  //----------------------------------------------------------------------------
  /// Advectes the particle in the flowmap phi until either a split needs to be
  /// performed or time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param step_size Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  /// \param splitted_particles Splitted particles (Their time is smaller than
  ///                           t_end.)
  /// \param finished_particles Finished particles (Their time is equal to
  ///                           t_end.)
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  auto advect_until_split(Flowmap phi, real_t step_size, real_t const t_end,
                          container_t&            splitted_particles,
                          container_t&            finished_particles,
                          std::atomic_bool const& stop = false) const {
    if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
      phi.use_caching(false);
    }
    static constexpr real_t min_tau_step          = 1e-8;
    static constexpr real_t max_cond_overshoot    = 1e-6;
    bool                    min_step_size_reached = false;
    static constexpr auto   split_sqr_cond        = SplitBehavior::sqr_cond;
    static constexpr auto   split_radii           = SplitBehavior::radii;
    static constexpr auto   split_offsets         = SplitBehavior::offsets;
    auto const [eigvecs_S, eigvals_S]             = eigenvectors_sym(S());
    auto const B = eigvecs_S * diag(eigvals_S);  // current main axes

    mat_t H, HHt, advected_nabla_phi, assembled_nabla_phi, advected_B;
    mat_t ghosts_forward, ghosts_backward, prev_ghosts_forward,
        prev_ghosts_backward;
    auto        advected_ellipse = ellipse_t{*this};
    auto        current_radii    = vec_t{};
    auto        eig_HHt          = std::pair<mat_t, vec_t>{};
    auto        sqr_cond_H       = real_t(1);
    auto const& eigvecs_HHt      = eig_HHt.first;
    auto const& eigvals_HHt      = eig_HHt.second;

    // initialize ghosts
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      ghosts_forward.col(i) = x();
    }
    ghosts_backward = ghosts_forward;

    ghosts_forward += B;
    ghosts_backward -= B;

    // fields for backup
    auto t_prev          = t();
    auto prev_center     = advected_ellipse.center();
    prev_ghosts_forward  = ghosts_forward;
    prev_ghosts_backward = ghosts_backward;
    auto prev_cond_HHt   = sqr_cond_H;

    // repeat as long as particle's ellipse is not wide enough or t_end is not
    // reached
    auto t_advected      = t();
    while (!stop || sqr_cond_H < split_sqr_cond || t_advected < t_end) {
      if (!min_step_size_reached) {
        // backup state before advection
        prev_ghosts_forward  = ghosts_forward;
        prev_ghosts_backward = ghosts_backward;
        prev_center          = advected_ellipse.center();
        prev_cond_HHt        = sqr_cond_H;
        t_prev               = t_advected;

        // increase time
        if (t_advected + step_size > t_end) {
          step_size  = t_end - t_advected;
          t_advected = t_end;
        } else {
          t_advected += step_size;
        }
        auto const cur_tau = t_advected - t_prev;

        // advect center and ghosts
        advected_ellipse.center() =
            phi(advected_ellipse.center(), t_advected, cur_tau);
        ghosts_forward  = phi(ghosts_forward, t_advected, cur_tau);
        ghosts_backward = phi(ghosts_backward, t_advected, cur_tau);

        // make computations
        H          = (ghosts_forward - ghosts_backward) * half;
        HHt        = H * transposed(H);
        eig_HHt    = eigenvectors_sym(HHt);
        sqr_cond_H = eigvals_HHt(num_dimensions() - 1) / eigvals_HHt(0);

        advected_nabla_phi = H * *solve(diag(eigvals_S), transposed(eigvecs_S));
        assembled_nabla_phi = advected_nabla_phi * m_nabla_phi;

        current_radii        = sqrt(eigvals_HHt);
        advected_B           = eigvecs_HHt * diag(current_radii);
        advected_ellipse.S() = advected_B * transposed(eigvecs_HHt);
        if (std::isnan(sqr_cond_H)) {
          std::cout << "===========================\n";
          std::cout << "center(): \n" << center() << '\n';
          std::cout << "S: \n" << S() << "\n\n";

          std::cout << "advected_ellipse.center(): \n"
                    << advected_ellipse.center() << '\n';
          std::cout << "advected_ellipse.S(): \n"
                    << advected_ellipse.S() << "\n\n";

          std::cout << "B: \n" << B << '\n';
          std::cout << "H: \n" << H << '\n';
          std::cout << "sqr_cond_H: " << sqr_cond_H << '\n';
          std::cout << "ghosts_forward: \n" << ghosts_forward << '\n';
          std::cout << "ghosts_backward: \n" << ghosts_backward << '\n';
          std::cout << "eigvals_HHt: " << eigvals_HHt << '\n';
          std::cout << "eigvecs_HHt:\n" << eigvecs_HHt << '\n';
          std::cout << "nabla_phi: \n" << m_nabla_phi << '\n';
          std::cout << "advected_nabla_phi: \n" << advected_nabla_phi << '\n';
          std::cout << "advected_B: \n" << advected_B << '\n';
          discretize().write_vtp("malicious_base_particle.vtp");
          advected_ellipse.discretize().write_vtp(
              "malicious_advected_particle.vtp");
          throw std::runtime_error{"dadsadsa"};
        }
      }

      // check if particle has reached t_end
      if (t_advected == t_end &&
          sqr_cond_H <= split_sqr_cond + max_cond_overshoot) {
        finished_particles.emplace_back(advected_ellipse, t_advected, x0(),
                                        assembled_nabla_phi);
        return;
      }

      // check if particle's ellipse has reached its splitting wideness
      if ((sqr_cond_H >= split_sqr_cond &&
           sqr_cond_H <= split_sqr_cond + max_cond_overshoot) ||
          min_step_size_reached) {
        for (std::size_t i = 0; i < size(split_radii); ++i) {
          auto const new_eigvals    = current_radii * split_radii[i];
          auto const offset2        = advected_B * split_offsets[i];
          auto const offset0        = solve(assembled_nabla_phi, offset2);
          auto       offset_ellipse = ellipse_t{
              advected_ellipse.center() + offset2,
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt)};


        //auto const [eigvecs_S, eigvals_S]             = eigenvectors_sym(S());
        //auto const B = eigvecs_S * diag(eigvals_S);  // current main axes

          if (stop) {
            finished_particles.emplace_back(offset_ellipse, t_advected,
                                            x0() + offset0,
                                            assembled_nabla_phi);
          } else {
            splitted_particles.emplace_back(offset_ellipse, t_advected,
                                            x0() + offset0,
                                            assembled_nabla_phi);
          }
        }
        return;
      }
      // check if particle's ellipse is wider than its splitting wideness
      if (sqr_cond_H > split_sqr_cond + max_cond_overshoot) {
        auto const prev_step_size = step_size;
        step_size *= half;
        min_step_size_reached = step_size == prev_step_size;
        if (step_size < min_tau_step) {
          min_step_size_reached = true;
        }
        if (!min_step_size_reached) {
          sqr_cond_H                = prev_cond_HHt;
          ghosts_forward            = prev_ghosts_forward;
          ghosts_backward           = prev_ghosts_backward;
          t_advected                = t_prev;
          advected_ellipse.center() = prev_center;
        }
        if (stop) {
          finished_particles.emplace_back(advected_ellipse, t_advected, x0(),
                                          assembled_nabla_phi);
          return;
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto sampler() const {
    return sampler_t{initial_ellipse(), *this, m_nabla_phi};
  }
};
//==============================================================================
// typedefs
//==============================================================================
template <std::size_t N>
using AutonomousParticle = autonomous_particle<real_t, N>;
template <floating_point Real>
using AutonomousParticle2 = autonomous_particle<Real, 2>;
template <floating_point Real>
using AutonomousParticle3  = autonomous_particle<Real, 3>;
using autonomous_particle2 = AutonomousParticle<2>;
using autonomous_particle3 = AutonomousParticle<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
template <floating_point Real, std::size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (autonomous_particle<Real, N>),
    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
    TATOOINE_REFLECTION_INSERT_METHOD(S, S()),
    TATOOINE_REFLECTION_INSERT_METHOD(x, x()),
    TATOOINE_REFLECTION_INSERT_METHOD(t, t()),
    TATOOINE_REFLECTION_INSERT_METHOD(nabla_phi, nabla_phi()))
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
