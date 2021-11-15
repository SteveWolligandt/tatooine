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
  autonomous_particle_sampler(autonomous_particle_sampler const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle_sampler(autonomous_particle_sampler&&) noexcept = default;
  //============================================================================
  auto operator                       =(autonomous_particle_sampler const&)
      -> autonomous_particle_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator                       =(autonomous_particle_sampler&&) noexcept
      -> autonomous_particle_sampler& = default;
  //============================================================================
  autonomous_particle_sampler() = default;
  //----------------------------------------------------------------------------
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
  template <typename Tag>
  auto distance(pos_t const& x, Tag tag) const {
    return distance_sqr(x, tag);
  }
  auto nabla_phi() const -> auto const& { return m_nabla_phi; }
  auto nabla_phi_inv() const -> auto const& { return m_nabla_phi_inv; }
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
  real_t m_t;
  mat_t  m_nabla_phi1;

  static auto mutex() -> auto& {
    static auto m = std::mutex{};
    return m;
  }
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
  autonomous_particle(pos_t const& x, real_t const t, real_t const r);
  autonomous_particle(ellipse_t const& ell, real_t const t);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle(pos_t const& x, real_t const t, mat_t const& nabla_phi1,
                      ellipse_t const& ell);
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return x0()(i); }
  //----------------------------------------------------------------------------
  auto x() -> auto& { return parent_t::center(); }
  auto x() const -> auto const& { return parent_t::center(); }
  auto x(size_t i) const { return parent_t::center()(i); }
  //----------------------------------------------------------------------------
  auto t() -> auto& { return m_t; }
  auto t() const { return m_t; }
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
  auto initial_ellipse() const { return ellipse_t{x0(), S0()}; }
  //============================================================================
  // METHODS
  //============================================================================
  // auto advect_with_2_splits(real_t const step_size, real_t const t_end,
  //                          size_t const max_num_particles) const {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(step_size, t_end, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_2_splits(real_t const step_size, real_t const t_end) const
  // {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(step_size, t_end, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const step_size, real_t const
  // t_end,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(step_size, t_end, 2, max_num_particles,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_2_splits(real_t const step_size, real_t const
  // t_end,
  //                                 container_t particles) {
  //  static real_t const sqrt2 = std::sqrt(real_t(2));
  //  return advect(step_size, t_end, 2, 0,
  //                std::array<real_t, 1>{sqrt2 / real_t(2)}, false,
  //                std::move(particles));
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                            real_t const t_end) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end, 0,
                                container_t{*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                            real_t const t_end,
                            size_t const max_num_particles) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end,
                                max_num_particles, container_t{*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                                   real_t const t_end, container_t particles) {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end, 0,
                                std::move(particles));
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                                   real_t const t_end,
                                   size_t const max_num_particles,
                                   container_t  particles) {
    [[maybe_unused]] static real_t const x5 = 0.4830517593887872;
    if constexpr (N == 2) {
      return advect(
          phi, step_size, t_end, 4, max_num_particles,
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
          phi, step_size, t_end, 4, max_num_particles,
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
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                            real_t const            t_end,
                            filesystem::path const& path) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end, 0,
                                container_t{*this}, path);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                            real_t const t_end, size_t const max_num_particles,
                            filesystem::path const& path) const {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end,
                                max_num_particles, container_t{*this}, path);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                                   real_t const t_end, container_t particles,
                                   filesystem::path const& path) {
    return advect_with_3_splits(std::forward<Flowmap>(phi), step_size, t_end, 0,
                                std::move(particles), path);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_3_splits(Flowmap&& phi, real_t const step_size,
                                   real_t const            t_end,
                                   size_t const            max_num_particles,
                                   container_t             particles,
                                   filesystem::path const& path) {
    [[maybe_unused]] static real_t const x5 = 0.4830517593887872;
    if constexpr (N == 2) {
      return advect(
          phi, step_size, t_end, 4, max_num_particles,
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
          std::move(particles), path);
    } else if constexpr (N == 3) {
      return advect(
          phi, step_size, t_end, 4, max_num_particles,
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
          std::move(particles), path);
    }
  }
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const step_size, real_t const t_end,
  //                          size_t const max_num_particles) const {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(step_size, t_end, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_5_splits(real_t const step_size, real_t const t_end) const
  // {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(step_size, t_end, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const step_size, real_t const
  // t_end,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(step_size, t_end, 6 + sqrt5 * 2, max_num_particles,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_5_splits(real_t const step_size, real_t const
  // t_end,
  //                                 container_t particles) {
  //  static real_t const sqrt5 = std::sqrt(real_t(5));
  //  return advect(step_size, t_end, 6 + sqrt5 * 2, 0,
  //                std::array{(sqrt5 + 3) / (sqrt5 * 2 + 2), 1 / (sqrt5 + 1)},
  //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const step_size, real_t const t_end,
  //                          size_t const max_num_particles) const {
  //  return advect(step_size, t_end, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // auto advect_with_7_splits(real_t const step_size, real_t const t_end) const
  // {
  //  return advect(step_size, t_end, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true);
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const step_size, real_t const
  // t_end,
  //                                 size_t const        max_num_particles,
  //                                 container_t particles) {
  //  return advect(step_size, t_end, 4.493959210 * 4.493959210,
  //  max_num_particles,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles));
  //}
  ////----------------------------------------------------------------------------
  // static auto advect_with_7_splits(real_t const step_size, real_t const
  // t_end,
  //                                 container_t particles) {
  //  return advect(step_size, t_end, 4.493959210 * 4.493959210, 0,
  //                std::array{real_t(.9009688678), real_t(.6234898004),
  //                           real_t(.2225209338)},
  //                true, std::move(particles));
  //}
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect(Flowmap& phi, real_t const step_size, real_t const t_end,
              real_t const objective_cond, size_t const max_num_particles,
              range auto const radii, range auto const& offsets,
              filesystem::path const& path) const {
    return advect(phi, step_size, t_end, objective_cond, max_num_particles,
                  radii, offsets, {*this}, path);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect(Flowmap& phi, real_t const step_size, real_t const t_end,
                     real_t const objective_cond,
                     size_t const max_num_particles, range auto const radii,
                     range auto const& offsets, container_t const& particles,
                     filesystem::path const& path) {
    // auto       finished_particles = container_t{};
    auto const num_threads =
        static_cast<size_t>(std::thread::hardware_concurrency());
    if (filesystem::exists(path)) {
      filesystem::remove(path);
    }
    auto file = hdf5::file{path};
    auto hdd_data =
        std::array{file.add_dataset<typename container_t::value_type>(
                       "ping", hdf5::unlimited),
                   file.add_dataset<typename container_t::value_type>(
                       "pong", hdf5::unlimited)};
    auto finished = file.add_dataset<typename container_t::value_type>(
        "finished", hdf5::unlimited);
    size_t reader = 0;
    size_t writer = 1;
    hdd_data[reader].write(particles);

    while (hdd_data[reader].dataspace().current_resolution()[0] > 0) {
      auto const num_particles =
          hdd_data[reader].dataspace().current_resolution()[0];
      auto thread_ranges =
          std::vector<aligned<std::pair<size_t, size_t>>>(num_threads);
      auto advected_particles = std::vector<aligned<container_t>>(num_threads);
      auto finished_particles = std::vector<aligned<container_t>>(num_threads);
      auto loaded_particles   = std::vector<aligned<container_t>>(num_threads);
      size_t const num_particles_at_once = 10000000;
      for (auto& l : loaded_particles) {
        l->reserve(num_particles_at_once);
      }
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
                auto const& range     = *thread_ranges[thr_id];
                auto&       particles = *loaded_particles[thr_id];
                size_t      chunk_idx = 0;
                for (size_t i = range.first; i < range.second;
                     i += num_particles_at_once, ++chunk_idx) {
                  auto const cur_num_particles =
                      std::min(num_particles_at_once, range.second - i);
                  particles.resize(cur_num_particles);
                  {
                    auto lock = std::lock_guard{mutex()};
                    hdd_data[reader].read(i, cur_num_particles, particles);
                  }
                  for (auto const& particle : particles) {
                    particle.advect_until_split(phi, step_size, t_end,
                                                objective_cond, radii, offsets,
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
  template <typename Flowmap>
  auto advect(Flowmap& phi, real_t const step_size, real_t const t_end,
              real_t const objective_cond, size_t const max_num_particles,
              range auto const radii, range auto const& offsets) const {
    return advect(phi, step_size, t_end, objective_cond, max_num_particles,
                  radii, offsets, {*this});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect(Flowmap& phi, real_t const step_size, real_t const t_end,
                     real_t const objective_cond,
                     size_t const max_num_particles, range auto const radii,
                     range auto const& offsets, container_t particles) {
    auto const num_threads =
        static_cast<size_t>(std::thread::hardware_concurrency());

    auto finished_particles = container_t{};

    // {particles_to_be_advected, advected_particles, finished_particles}
    auto particles_per_thread =
        std::vector<std::array<aligned<container_t>, 3>>(num_threads);
    while (particles.size() > 0) {
      auto const num_particles = particles.size();
      {
        // distribute particles
        auto thread_pool = std::vector<std::thread>{};
        thread_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                size_t const begin = thr_id * num_particles / num_threads;
                size_t const end   = (thr_id + 1) * num_particles / num_threads;
                auto&        cont  = *particles_per_thread[thr_id][0];
                cont.reserve(end - begin);
                std::copy(particles.begin() + begin, particles.begin() + end,
                          std::back_inserter(cont));
              },
              i);
        }
        for (auto& thread : thread_pool) {
          thread.join();
        }
        particles.clear();
      }
      {
        // advect particle pools
        auto thread_pool = std::vector<std::thread>{};
        thread_pool.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
          thread_pool.emplace_back(
              [&](auto const thr_id) {
                auto& particles_at_t0 = *particles_per_thread[thr_id][0];
                auto& particles_at_t1 = *particles_per_thread[thr_id][1];
                auto& finished        = *particles_per_thread[thr_id][2];
                for (auto const& particle : particles_at_t0) {
                  particle.advect_until_split(phi, step_size, t_end,
                                              objective_cond, radii, offsets,
                                              particles_at_t1, finished);
                }
              },
              i);
        }
        for (auto& thread : thread_pool) {
          thread.join();
        }
      }
      for (auto& ps : particles_per_thread) {
        auto& base     = *ps[0];
        auto& advected = *ps[1];
        auto& finished = *ps[2];
        std::copy(begin(advected), end(advected),
                  std::back_inserter(particles));
        std::copy(begin(finished), end(finished),
                  std::back_inserter(finished_particles));
        base.clear();
        advected.clear();
        finished.clear();
      }
    }
    return finished_particles;
  }
  //----------------------------------------------------------------------------
  // template <typename Flowmap>
  // auto advect_until_split(Flowmap& phi, real_t const step_size,
  //                        real_t const t_end, real_t const objective_cond,
  //                        range auto const  radii,
  //                        range auto const& offsets) const -> container_t {
  //  auto advected = container_t{};
  //  advect_until_split(phi, step_size, t_end, objective_cond, radii, offsets,
  //                     advected, advected);
  //
  //  return advected;
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  auto advect_until_split(Flowmap phi, real_t step_size, real_t const t_end,
                          real_t const objective_cond, range auto const radii,
                          range auto const& offsets, container_t& out,
                          container_t& finished_particles) const {
    bool                    tau_should_have_changed_but_did_not = false;
    static constexpr real_t min_tau_step                        = 1e-8;
    static constexpr real_t max_cond_overshoot                  = 1e-6;
    auto const [eigvecs_S, eigvals_S] = eigenvectors_sym(S1());
    auto const B = eigvecs_S * diag(eigvals_S);  // current main axes

    mat_t                   H, HHt, nabla_phi2, fmg2fmg1, cur_B;
    ellipse_t               advected_ellipse = *this;
    vec_t                   current_radii;
    std::pair<mat_t, vec_t> eig_HHt;
    real_t                  cond_HHt    = 1;
    auto const&             eigvecs_HHt = eig_HHt.first;
    auto const&             eigvals_HHt = eig_HHt.second;
    auto                    ghosts = make_array<vec_t, num_dimensions() * 2>();
    for (size_t i = 0; i < num_dimensions(); ++i) {
      ghosts[i * 2] = x();
      ghosts[i * 2] += B.col(i);
      ghosts[i * 2 + 1] = x();
      ghosts[i * 2 + 1] -= B.col(i);
    }
    real_t advected_t          = t();
    auto   old_advected_center = advected_ellipse.center();
    auto   old_advected_time   = advected_t;
    auto   old_ghosts          = ghosts;
    auto   old_cond_HHt        = cond_HHt;
    bool   first               = true;
    if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
      phi.use_caching(false);
    }

    while (cond_HHt < objective_cond || advected_t < t_end) {
      if (!tau_should_have_changed_but_did_not) {
        if (!first) {
          old_ghosts          = ghosts;
          old_advected_center = advected_ellipse.center();
          old_cond_HHt        = cond_HHt;
          old_advected_time   = advected_t;
        } else {
          first = false;
        }

        if (advected_t + step_size > t_end) {
          step_size  = t_end - advected_t;
          advected_t = t_end;
        } else {
          advected_t += step_size;
        }

        advected_ellipse.center() =
            phi(advected_ellipse.center(), old_advected_time,
                advected_t - old_advected_time);
        for (size_t i = 0; i < num_dimensions(); ++i) {
          ghosts[i * 2]     = phi(ghosts[i * 2], old_advected_time,
                              advected_t - old_advected_time);
          ghosts[i * 2 + 1] = phi(ghosts[i * 2 + 1], old_advected_time,
                                  advected_t - old_advected_time);
          H.col(i)          = ghosts[i * 2] - ghosts[i * 2 + 1];
        }
        H *= half;

        HHt      = H * transposed(H);
        eig_HHt  = eigenvectors_sym(HHt);
        cond_HHt = eigvals_HHt(num_dimensions() - 1) / eigvals_HHt(0);

        nabla_phi2 = H * *solve(diag(eigvals_S), transposed(eigvecs_S));
        fmg2fmg1   = nabla_phi2 * m_nabla_phi1;

        current_radii        = sqrt(eigvals_HHt);
        cur_B                = eigvecs_HHt * diag(current_radii);
        advected_ellipse.S() = cur_B * transposed(eigvecs_HHt);
      }

      if (advected_t == t_end &&
          cond_HHt <= objective_cond + max_cond_overshoot) {
        finished_particles.emplace_back(x0(), advected_t, fmg2fmg1,
                                        advected_ellipse);
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
          for (size_t i = 0; i < size(radii); ++i) {
            auto const new_eigvals    = current_radii * radii[i];
            auto const offset2        = cur_B * offsets[i];
            auto const offset0        = solve(fmg2fmg1, offset2);
            auto       offset_ellipse = ellipse_t{
                advected_ellipse.center() + offset2,
                eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt)};
            out.emplace_back(x0() + offset0, advected_t, fmg2fmg1,
                             offset_ellipse);
          }
        }
        return;
      }
      if (cond_HHt > objective_cond + max_cond_overshoot) {
        // if (old_cond_HHt < objective_cond) {
        //  auto const _t =
        //      (old_cond_HHt - objective_cond) / (old_cond_HHt - cond_HHt);
        //  assert(_t >= 0 && _t <= 1);
        //  step_size *= _t;
        //}
        auto const old_tau_step = step_size;
        step_size *= half;
        tau_should_have_changed_but_did_not = step_size == old_tau_step;
        if (step_size < min_tau_step) {
          tau_should_have_changed_but_did_not = true;
        }

        if (!tau_should_have_changed_but_did_not) {
          cond_HHt                  = old_cond_HHt;
          ghosts                    = old_ghosts;
          advected_t                = old_advected_time;
          advected_ellipse.center() = old_advected_center;
        }
        //} else {
        //   auto const _t =
        //      (old_cond_HHt - objective_cond) / (old_cond_HHt - cond_HHt);
        //   assert(_t >= 1);
        //   step_size *= _t;
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
    TATOOINE_REFLECTION_INSERT_METHOD(t, t()),
    TATOOINE_REFLECTION_INSERT_METHOD(nabla_phi1, nabla_phi1()))
}  // namespace reflection
//==============================================================================
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle const& other)
    : parent_t{other},
      m_x0{other.m_x0},
      m_t{other.m_t},
      m_nabla_phi1{other.m_nabla_phi1} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(
    autonomous_particle&& other) noexcept
    : parent_t{std::move(other)},
      m_x0{std::move(other.m_x0)},
      m_t{other.m_t},
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
  m_t               = other.m_t;
  m_nabla_phi1      = other.m_nabla_phi1;
  return *this;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
auto autonomous_particle<Real, N>::operator=(
    autonomous_particle&& other) noexcept -> autonomous_particle& {
  parent_t::operator=(std::move(other));
  m_x0              = std::move(other.m_x0);
  m_t               = other.m_t;
  m_nabla_phi1      = std::move(other.m_nabla_phi1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(ellipse_t const& ell,
                                                  real_t const     t)
    : parent_t{ell}, m_x0{ell.center()}, m_t{t}, m_nabla_phi1{mat_t::eye()} {}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(pos_t const& x,
                                                  real_t const t,
                                                  real_t const r)
    : parent_t{x, r}, m_x0{x}, m_t{t}, m_nabla_phi1{mat_t::eye()} {}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
autonomous_particle<Real, N>::autonomous_particle(pos_t const&     x0,
                                                  real_t const     t,
                                                  mat_t const&     nabla_phi1,
                                                  ellipse_t const& ell)
    : parent_t{ell}, m_x0{x0}, m_t{t}, m_nabla_phi1{nabla_phi1} {}
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
using AutonomousParticle   = autonomous_particle<real_t, N>;
using autonomous_particle2 = AutonomousParticle<2>;
using autonomous_particle3 = AutonomousParticle<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
