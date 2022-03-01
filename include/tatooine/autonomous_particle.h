#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/cache_alignment.h>
#include <tatooine/concepts.h>
#include <tatooine/detail/autonomous_particle/post_triangulation.h>
#include <tatooine/detail/autonomous_particle/sampler.h>
#include <tatooine/detail/autonomous_particle/split_behavior.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/particle.h>
#include <tatooine/random.h>
#include <tatooine/reflection.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename B>
concept split_behavior = requires {
  floating_point<decltype(B::split_cond)>;
  range<decltype(B::radii)>;
  range<decltype(B::offsets)>;
  static_vec<typename decltype(B::radii)::value_type>;
  static_vec<typename decltype(B::offsets)::value_type>;
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct autonomous_particle : geometry::hyper_ellipse<Real, NumDimensions> {
  using split_behaviors =
      detail::autonomous_particle::split_behaviors<Real, NumDimensions>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
  // TYPEDEFS
  //============================================================================
 public:
  //----------------------------------------------------------------------------
  constexpr static auto half = 1 / Real(2);

  using this_type         = autonomous_particle<Real, NumDimensions>;
  using simple_particle_t = particle<Real, NumDimensions>;
  using real_type         = Real;
  using vec_t             = vec<real_type, NumDimensions>;
  using mat_t             = mat<real_type, NumDimensions, NumDimensions>;
  using pos_type          = vec_t;
  using container_type    = std::vector<this_type>;
  using simple_particle_container_t = std::vector<simple_particle_t>;
  using ellipse_type = geometry::hyper_ellipse<Real, NumDimensions>;
  using parent_type  = ellipse_type;
  using sampler_type =
      detail::autonomous_particle::sampler<Real, NumDimensions>;
  using hierarchy_pair = detail::autonomous_particle::hierarchy_pair;
  using parent_type::center;
  using parent_type::discretize;
  using parent_type::S;
  //============================================================================
  // members
  //============================================================================
 private:
  //----------------------------------------------------------------------------
  pos_type      m_x0;
  real_type     m_t;
  mat_t         m_nabla_phi;
  std::uint64_t m_id = std::numeric_limits<std::uint64_t>::max();

  static auto mutex() -> auto& {
    static auto m = std::mutex{};
    return m;
  }
  //============================================================================
  // CTORS
  //============================================================================
 public:
  //----------------------------------------------------------------------------
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
  explicit autonomous_particle(std::uint64_t const id)
      : m_nabla_phi{mat_t::eye()}, m_id{id} {}
  explicit autonomous_particle(std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::uint64_t const id)
      : parent_type{ell},
        m_x0{ell.center()},
        m_t{t},
        m_nabla_phi{mat_t::eye()},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{ell, t, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::uint64_t const id)
      : parent_type{x, r},
        m_x0{x},
        m_t{t},
        m_nabla_phi{mat_t::eye()},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{x, t, r, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      pos_type const& x0, mat_t const& nabla_phi,
                      std::uint64_t const id)
      : parent_type{ell}, m_x0{x0}, m_t{t}, m_nabla_phi{nabla_phi}, m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      pos_type const& x0, mat_t const& nabla_phi,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{ell, t, x0, nabla_phi, uuid_generator++} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(std::size_t i) const { return x0()(i); }
  //----------------------------------------------------------------------------
  auto x() -> auto& { return parent_type::center(); }
  auto x() const -> auto const& { return parent_type::center(); }
  auto x(std::size_t const i) const { return parent_type::center()(i); }
  //----------------------------------------------------------------------------
  auto t() -> auto& { return m_t; }
  auto t() const { return m_t; }
  //----------------------------------------------------------------------------
  auto nabla_phi() const -> auto const& { return m_nabla_phi; }
  //----------------------------------------------------------------------------
  auto S0() const {
    auto sqrS = *inv(nabla_phi()) * S() * S() * *inv(transposed(nabla_phi()));
    auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      eig_vals(i) = gcem::sqrt(eig_vals(i));
    }
    return eig_vecs * diag(eig_vals) * transposed(eig_vecs);
  }
  //----------------------------------------------------------------------------
  auto initial_ellipse() const { return ellipse_type{x0(), S0()}; }
  //----------------------------------------------------------------------------
  auto id() const { return m_id; }
  //============================================================================
  // METHODS
  //============================================================================
  // template <
  //    split_behavior SplitBehavior = typename split_behaviors::three_splits,
  //    typename Flowmap>
  // auto advect(Flowmap&& phi, real_type const stepwidth, real_type const
  // t_end,
  //            filesystem::path const& path) const {
  //  return advect(std::forward<Flowmap>(phi), stepwidth, t_end, {*this},
  //  path);
  //}
  ////----------------------------------------------------------------------------
  // template <
  //     split_behavior SplitBehavior = typename split_behaviors::three_splits,
  //     typename Flowmap>
  // static auto advect(Flowmap&& phi, real_type const stepwidth, real_type
  // const t_end,
  //                    container_type const&      particles,
  //                    filesystem::path const& path) {
  //   // auto       finished_particles = container_type{};
  //   auto const num_threads =
  //       static_cast<std::size_t>(std::thread::hardware_concurrency());
  //   if (filesystem::exists(path)) {
  //     filesystem::remove(path);
  //   }
  //   auto file = hdf5::file{path};
  //   auto hdd_data =
  //       std::array{file.create_dataset<typename container_type::value_type>(
  //                      "ping", hdf5::unlimited),
  //                  file.create_dataset<typename container_type::value_type>(
  //                      "pong", hdf5::unlimited)};
  //   auto finished = file.create_dataset<typename container_type::value_type>(
  //       "finished", hdf5::unlimited);
  //   std::size_t reader = 0;
  //   std::size_t writer = 1;
  //   hdd_data[reader].write(particles);
  //
  //   while (hdd_data[reader].dataspace().current_resolution()[0] > 0) {
  //     auto const num_particles =
  //         hdd_data[reader].dataspace().current_resolution()[0];
  //     auto thread_ranges =
  //         std::vector<aligned<std::pair<std::size_t, std::size_t>>>(
  //             num_threads);
  //     auto advected_particles =
  //     std::vector<aligned<container_type>>(num_threads); auto
  //     finished_particles = std::vector<aligned<container_type>>(num_threads);
  //     auto loaded_particles =
  //     std::vector<aligned<container_type>>(num_threads); std::size_t const
  //     num_particles_at_once = 10000000; for (auto& l : loaded_particles) {
  //       l->reserve(num_particles_at_once);
  //     }
  //     {
  //       // distribute particles
  //       auto thread_pool = std::vector<std::thread>{};
  //       thread_pool.reserve(num_threads);
  //       for (std::size_t i = 0; i < num_threads; ++i) {
  //         thread_pool.emplace_back(
  //             [&](auto const thr_id) {
  //               std::size_t const begin = thr_id * num_particles /
  //               num_threads; std::size_t const end =
  //                   (thr_id + 1) * num_particles / num_threads;
  //
  //               *thread_ranges[thr_id] = std::pair{begin, end};
  //             },
  //             i);
  //       }
  //       for (auto& thread : thread_pool) {
  //         thread.join();
  //       }
  //     }
  //     {
  //       // advect particle pools
  //       auto thread_pool = std::vector<std::thread>{};
  //       thread_pool.reserve(num_threads);
  //       for (std::size_t i = 0; i < num_threads; ++i) {
  //         thread_pool.emplace_back(
  //             [&](auto const thr_id) {
  //               auto const& range     = *thread_ranges[thr_id];
  //               auto&       particles = *loaded_particles[thr_id];
  //               std::size_t chunk_idx = 0;
  //               for (std::size_t i = range.first; i < range.second;
  //                    i += num_particles_at_once, ++chunk_idx) {
  //                 auto const cur_num_particles =
  //                     std::min(num_particles_at_once, range.second - i);
  //                 particles.resize(cur_num_particles);
  //                 {
  //                   auto lock = std::lock_guard{mutex()};
  //                   hdd_data[reader].read(i, cur_num_particles, particles);
  //                 }
  //                 for (auto const& particle : particles) {
  //                   particle.template advect_until_split<SplitBehavior>(
  //                       std::forward<Flowmap>(phi), stepwidth, t_end,
  //                       *advected_particles[thr_id],
  //                       *finished_particles[thr_id]);
  //                   if (advected_particles[thr_id]->size() > 10000000) {
  //                     {
  //                       auto lock = std::lock_guard{mutex()};
  //                       hdd_data[writer].push_back(*advected_particles[thr_id]);
  //                     }
  //                     advected_particles[thr_id]->clear();
  //                   }
  //                   if (finished_particles[thr_id]->size() > 10000000) {
  //                     {
  //                       auto lock = std::lock_guard{mutex()};
  //                       finished.push_back(*finished_particles[thr_id]);
  //                     }
  //                     finished_particles[thr_id]->clear();
  //                   }
  //                   if (!advected_particles[thr_id]->empty()) {
  //                     {
  //                       auto lock = std::lock_guard{mutex()};
  //                       hdd_data[writer].push_back(*advected_particles[thr_id]);
  //                     }
  //                     advected_particles[thr_id]->clear();
  //                   }
  //                   if (!finished_particles[thr_id]->empty()) {
  //                     {
  //                       auto lock = std::lock_guard{mutex()};
  //                       finished.push_back(*finished_particles[thr_id]);
  //                     }
  //                     finished_particles[thr_id]->clear();
  //                   }
  //                 }
  //                 particles.clear();
  //               }
  //             },
  //             i);
  //       }
  //       for (auto& thread : thread_pool) {
  //         thread.join();
  //       }
  //     }
  //
  //     reader = 1 - reader;
  //     writer = 1 - writer;
  //     hdd_data[writer].clear();
  //   }
  //   hdd_data[reader].clear();
  //   return path;
  // }
  //----------------------------------------------------------------------------
  template <split_behavior SplitBehavior, typename Flowmap>
  auto advect(Flowmap&& phi, real_type const stepwidth, real_type const t_end,
              std::atomic_uint64_t& uuid_generator) const {
    using namespace detail::autonomous_particle;
    auto hierarchy_mutex = std::mutex{};
    auto hierarchy_pairs = std::vector{hierarchy_pair{m_id, m_id}};
    auto [advected_particles, advected_simple_particles] =
        advect<SplitBehavior>(std::forward<Flowmap>(phi), stepwidth, t_end,
                              {*this}, hierarchy_pairs, hierarchy_mutex,
                              uuid_generator);
    auto edges = edgeset<Real, NumDimensions>{};
    auto map   = std::unordered_map<
        std::size_t, typename edgeset<Real, NumDimensions>::vertex_handle>{};
    for (auto const& p : advected_particles) {
      map[p.id()] = edges.insert_vertex(p.center());
    }
    triangulate(edges, hierarchy{hierarchy_pairs, map, edges});
    return std::tuple{std::move(advected_particles),
                      std::move(advected_simple_particles), std::move(edges)};
  }
  //----------------------------------------------------------------------------
  /// Advects single particle.
  template <typename Flowmap>
  auto advect_with_two_splits(Flowmap&& phi, real_type const stepwidth,
                              real_type const       t_end,
                              std::atomic_uint64_t& uuid_generator) const {
    return advect<typename split_behaviors::two_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, uuid_generator);
  }
  //----------------------------------------------------------------------------
  /// Advects single particle.
  template <typename Flowmap>
  auto advect_with_three_splits(Flowmap&& phi, real_type const stepwidth,
                                real_type const       t_end,
                                std::atomic_uint64_t& uuid_generator) const {
    return advect<typename split_behaviors::three_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, uuid_generator);
  }
  //----------------------------------------------------------------------------
  /// Advects single particle.
  template <typename Flowmap>
  auto advect_with_three_and_four_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t_end,
      std::atomic_uint64_t& uuid_generator) const {
    return advect<typename split_behaviors::three_and_four_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, uuid_generator);
  }
  //----------------------------------------------------------------------------
  /// Advects single particle.
  template <typename Flowmap>
  auto advect_with_five_splits(Flowmap&& phi, real_type const stepwidth,
                               real_type const       t_end,
                               std::atomic_uint64_t& uuid_generator) const {
    return advect<typename split_behaviors::five_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, uuid_generator);
  }
  //----------------------------------------------------------------------------
  /// Advects single particle.
  template <typename Flowmap>
  auto advect_with_seven_splits(Flowmap&& phi, real_type const stepwidth,
                                real_type const       t_end,
                                std::atomic_uint64_t& uuid_generator) const {
    return advect<typename split_behaviors::seven_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, uuid_generator);
  }
  //----------------------------------------------------------------------------
  /// Advects all particles in particles container in the flowmap phi until
  /// time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param stepwidth Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  /// \param particles Particles to be advected.
  template <split_behavior SplitBehavior, typename Flowmap>
  static auto advect(Flowmap&& phi, real_type const stepwidth,
                     real_type const       t_end,
                     container_type const& initial_particles,
                     std::atomic_uint64_t& uuid_generator) {
    using namespace detail::autonomous_particle;
    auto particles       = container_type{};
    auto hierarchy_mutex = std::mutex{};
    auto hierarchy_pairs = std::vector<hierarchy_pair>{};
    hierarchy_pairs.reserve(particles.size());
    for (auto const& p : particles) {
      hierarchy_pairs.emplace_back(p.id(), p.id());
    }
    auto [advected_particles, advected_simple_particles] =
        advect<SplitBehavior>(std::forward<Flowmap>(phi), stepwidth, t_end,
                              initial_particles, hierarchy_pairs,
                              hierarchy_mutex, uuid_generator);

    auto edges = edgeset<Real, NumDimensions>{};
    auto map   = std::unordered_map<
        std::size_t, typename edgeset<Real, NumDimensions>::vertex_handle>{};
    for (auto const& p : advected_particles) {
      map[p.id()] = edges.insert_vertex(p.center());
    }
    // auto const h = hierarchy{hierarchy_pairs, map, edges};
    // triangulate(edges, h);
    //
    // auto const s = initial_particle_distribution.size();
    // for (std::size_t j = 0; j < s[1]; ++j) {
    //   for (std::size_t i = 0; i < s[0] - 1; ++i) {
    //     auto const id0 = i + j * s[0];
    //     auto const id1 = (i + 1) + j * s[0];
    //     triangulate(edges, h.find_by_id(id0), h.find_by_id(id1));
    //   }
    // }
    // for (std::size_t i = 0; i < s[0]; ++i) {
    //   for (std::size_t j = 0; j < s[1] - 1; ++j) {
    //     auto const id0 = i + j * s[0];
    //     auto const id1 = i + (j + 1) * s[0];
    //     triangulate(edges, h.find_by_id(id0), h.find_by_id(id1));
    //   }
    // }

    return std::tuple{std::move(advected_particles),
                      std::move(advected_simple_particles), std::move(edges)};
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_two_splits(Flowmap&& phi, real_type const stepwidth,
                                     real_type const       t_end,
                                     container_type const& initial_particles,
                                     std::atomic_uint64_t& uuid_generator) {
    return advect<typename split_behaviors::two_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, initial_particles,
        uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_three_splits(Flowmap&& phi, real_type const stepwidth,
                                       real_type const       t_end,
                                       container_type const& initial_particles,
                                       std::atomic_uint64_t& uuid_generator) {
    return advect<typename split_behaviors::three_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, initial_particles,
        uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_three_and_four_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t_end,
      container_type const& initial_particles,
      std::atomic_uint64_t& uuid_generator) {
    return advect<typename split_behaviors::three_and_four_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, initial_particles,
        uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_five_splits(Flowmap&& phi, real_type const stepwidth,
                                      real_type const       t_end,
                                      container_type const& initial_particles,
                                      std::atomic_uint64_t& uuid_generator) {
    return advect<typename split_behaviors::five_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, initial_particles,
        uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_seven_splits(Flowmap&& phi, real_type const stepwidth,
                                       real_type const       t_end,
                                       container_type const& initial_particles,
                                       std::atomic_uint64_t& uuid_generator) {
    return advect<typename split_behaviors::seven_splits>(
        std::forward<Flowmap>(phi), stepwidth, t_end, initial_particles,
        uuid_generator);
  }
  static auto particles_from_grid(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::atomic_uint64_t&                                uuid_generator) {
    auto particles                     = container_type{};
    auto initial_particle_distribution = g.copy_without_properties();
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2,
              uuid_generator);
        });
    return particles;
  }
  //----------------------------------------------------------------------------
  /// Advects all particles in particles container in the flowmap phi until
  /// time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param stepwidth Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  template <split_behavior SplitBehavior, typename Flowmap>
  static auto advect(Flowmap&& phi, real_type const stepwidth,
                     real_type const t0, real_type const t_end,
                     uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    using namespace detail::autonomous_particle;
    auto uuid_generator  = std::atomic_uint64_t{};
    auto particles       = particles_from_grid(t0, g, uuid_generator);
    auto hierarchy_mutex = std::mutex{};
    auto hierarchy_pairs = std::vector<hierarchy_pair>{};
    hierarchy_pairs.reserve(particles.size());
    for (auto const& p : particles) {
      hierarchy_pairs.emplace_back(p.id(), p.id());
    }
    auto [advected_particles, advected_simple_particles] =
        advect<SplitBehavior>(std::forward<Flowmap>(phi), stepwidth, t_end,
                              particles, hierarchy_pairs, hierarchy_mutex,
                              uuid_generator);

    auto edges = edgeset<Real, NumDimensions>{};
    auto map   = std::unordered_map<
        std::size_t, typename edgeset<Real, NumDimensions>::vertex_handle>{};
    for (auto const& p : advected_particles) {
      map[p.id()] = edges.insert_vertex(p.center());
    }
    auto const h = hierarchy{hierarchy_pairs, map, edges};
    triangulate(edges, h);

    auto const s = g.size();
    --s[0];
    --s[1];
    for (std::size_t j = 0; j < s[1]; ++j) {
      for (std::size_t i = 0; i < s[0] - 1; ++i) {
        auto const id0 = i + j * s[0];
        auto const id1 = (i + 1) + j * s[0];
        triangulate(edges, h.find_by_id(id0), h.find_by_id(id1));
      }
    }
    for (std::size_t i = 0; i < s[0]; ++i) {
      for (std::size_t j = 0; j < s[1] - 1; ++j) {
        auto const id0 = i + j * s[0];
        auto const id1 = i + (j + 1) * s[0];
        triangulate(edges, h.find_by_id(id0), h.find_by_id(id1));
      }
    }

    return std::tuple{std::move(advected_particles),
                      std::move(advected_simple_particles), std::move(edges)};
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_two_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t0,
      real_type const                                      t_end,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    return advect<typename split_behaviors::two_splits>(
        std::forward<Flowmap>(phi), stepwidth, t0, t_end, g);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_three_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t0,
      real_type const                                      t_end,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    return advect<typename split_behaviors::three_splits>(
        std::forward<Flowmap>(phi), stepwidth, t0, t_end, g);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_three_and_four_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t0,
      real_type const                                      t_end,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    return advect<typename split_behaviors::three_and_four_splits>(
        std::forward<Flowmap>(phi), stepwidth, t0, t_end, g);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_five_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t0,
      real_type const                                      t_end,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    return advect<typename split_behaviors::five_splits>(
        std::forward<Flowmap>(phi), stepwidth, t0, t_end, g);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  static auto advect_with_seven_splits(
      Flowmap&& phi, real_type const stepwidth, real_type const t0,
      real_type const                                      t_end,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    return advect<typename split_behaviors::seven_splits>(
        std::forward<Flowmap>(phi), stepwidth, t0, t_end, g);
  }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  /// Advects all particles in particles container in the flowmap phi until
  /// time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param stepwidth Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  /// \param initial_particles Particles to be advected.
  template <split_behavior SplitBehavior, typename Flowmap>
  static auto advect(Flowmap&& phi, real_type const stepwidth,
                     real_type const t_end, container_type particles,
                     std::vector<hierarchy_pair>& hierarchy_pairs,
                     std::mutex&                  hierarchy_mutex,
                     std::atomic_uint64_t&        uuid_generator) {
    auto const num_threads = this_type::num_threads();

    auto finished_particles        = container_type{};
    auto finished_simple_particles = simple_particle_container_t{};

    // {particles_to_be_advected, advected_particles, finished_particles}
    auto particles_per_thread = std::vector<
        aligned<std::tuple<container_type, container_type, container_type,
                           simple_particle_container_t>>>(num_threads);
    while (particles.size() > 0) {
      std::cout << "advecting " << size(particles) << " particles...\n";
      std::cout << "distributing... \n";
      distribute_particles_to_thread_containers(num_threads, particles,
                                                particles_per_thread);
      std::cout << "advecting in parallel... \n";
      advect_particle_pools<SplitBehavior>(
          num_threads, std::forward<Flowmap>(phi), stepwidth, t_end,
          particles_per_thread, hierarchy_pairs, hierarchy_mutex,
          uuid_generator);
      std::cout << "gathering... \n";
      gather_particles(particles, finished_particles, finished_simple_particles,
                       particles_per_thread);
      std::cout << size(finished_particles) << " particles are finished\n";
    }
    std::cout << "ready!\n";
    return std::tuple{std::move(finished_particles),
                      std::move(finished_simple_particles)};
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
  static auto distribute_particles_to_thread_containers(
      std::size_t const num_threads, container_type& particles,
      auto& particles_per_thread) {
    using namespace std::ranges;
#pragma omp parallel
    {
      auto const thr_id = omp_get_thread_num();
      auto const begin  = std::size_t(thr_id * size(particles) / num_threads);
      auto const end =
          std::size_t((thr_id + 1) * size(particles) / num_threads);
      auto& cont = std::get<0>(*particles_per_thread[thr_id]);
      cont.reserve(end - begin);
      copy(particles.begin() + begin, particles.begin() + end,
           std::back_inserter(cont));
    }
    particles.clear();
  }
  //----------------------------------------------------------------------------
  template <split_behavior SplitBehavior, typename Flowmap>
  static auto advect_particle_pools(
      std::size_t const num_threads, Flowmap&& phi, real_type const stepwidth,
      real_type const t_end, auto& particles_per_thread,
      std::vector<hierarchy_pair>& hierarchy_pairs, std::mutex& hierarchy_mutex,
      std::atomic_uint64_t& uuid_generator) {
#pragma omp parallel
    {
      auto const thr_id = omp_get_thread_num();
      auto& [particles_at_t0, particles_at_t1, finished, simple_particles] =
          *particles_per_thread[thr_id];
      for (auto const& particle : particles_at_t0) {
        particle.template advect_until_split<SplitBehavior>(
            std::forward<Flowmap>(phi), stepwidth, t_end, particles_at_t1,
            finished, simple_particles, hierarchy_pairs, hierarchy_mutex,
            uuid_generator);
      }
    }
  }
  //----------------------------------------------------------------------------
  static auto gather_particles(container_type&              particles,
                               container_type&              finished_particles,
                               simple_particle_container_t& simple_particles,
                               auto& particles_per_thread) {
    using namespace std::ranges;
    for (auto& ps : particles_per_thread) {
      auto& [base, advected, finished, simple] = *ps;
      base.clear();
      copy(advected, std::back_inserter(particles));
      advected.clear();
      copy(finished, std::back_inserter(finished_particles));
      finished.clear();
      copy(simple, std::back_inserter(simple_particles));
      simple.clear();
    }
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  /// Advectes the particle in the flowmap phi until either a split needs to be
  /// performed or time `t_end` is reached.
  ///
  /// The split behavior is defined in the type SplitBehavior.
  /// \param phi Flow map of a vector field.
  /// \param stepwidth Step size of advection. (This is independent of the
  ///                  numerical integrators's step width.)
  /// \param t_end End of time of advetion.
  /// \param splitted_particles Splitted particles (Their time is smaller than
  ///                           t_end.)
  /// \param finished_particles Finished particles (Their time is equal to /
  /// t_end.)
  template <
      split_behavior SplitBehavior = typename split_behaviors::three_splits,
      typename Flowmap>
  auto advect_until_split(Flowmap phi, real_type stepwidth,
                          real_type const              t_end,
                          container_type&              splitted_particles,
                          container_type&              finished_particles,
                          simple_particle_container_t& simple_particles,
                          std::vector<hierarchy_pair>& hierarchy_pairs,
                          std::mutex&                  hierarchy_mutex,
                          std::atomic_uint64_t&        uuid_generator) const {

    if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
      phi.use_caching(false);
    }
    static constexpr real_type min_tau_step       = 1e-10;
    static constexpr real_type max_cond_overshoot = 1e-8;
    static constexpr auto      split_cond         = SplitBehavior::split_cond;
    static constexpr auto      split_sqr_cond     = split_cond * split_cond;
    static constexpr auto      split_radii        = SplitBehavior::radii;
    static constexpr auto      split_offsets      = SplitBehavior::offsets;
    auto const [eigvecs_S, eigvals_S]             = this->main_axes();
    auto const B = eigvecs_S * diag(eigvals_S);  // current main axes
    auto const K = solve(diag(eigvals_S), transposed(eigvecs_S));

    mat_t H, HHt, advected_nabla_phi, assembled_nabla_phi, advected_B,
        ghosts_forward, ghosts_backward, prev_ghosts_forward,
        prev_ghosts_backward;
    auto        min_step_size_reached = false;
    auto        advected_ellipse      = ellipse_type{*this};
    auto        current_radii         = vec_t{};
    auto        eig_HHt               = std::pair<mat_t, vec_t>{};
    auto        sqr_cond_H            = real_type(1);
    auto const& eigvecs_HHt           = eig_HHt.first;
    auto const& eigvals_HHt           = eig_HHt.second;

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
    // reached or the ellipse gets too small. If the latter happens make it a
    // simple massless particle
    auto t_advected = t();
    while (sqr_cond_H < split_sqr_cond || t_advected < t_end) {
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
        auto const cur_stepwidth = t_advected - t_prev;

        // advect center and ghosts
        advected_ellipse.center() =
            phi(advected_ellipse.center(), t_advected, cur_stepwidth);
        ghosts_forward  = phi(ghosts_forward, t_advected, cur_stepwidth);
        ghosts_backward = phi(ghosts_backward, t_advected, cur_stepwidth);

        // make computations
        H          = (ghosts_forward - ghosts_backward) * half;
        HHt        = H * transposed(H);
        eig_HHt    = eigenvectors_sym(HHt);
        sqr_cond_H = eigvals_HHt(num_dimensions() - 1) / eigvals_HHt(0);

        if (std::isnan(sqr_cond_H)) {
          simple_particles.emplace_back(x0(), advected_ellipse.center(),
                                        t_advected);
        }
        advected_nabla_phi  = H * K;
        assembled_nabla_phi = advected_nabla_phi * m_nabla_phi;

        current_radii        = sqrt(eigvals_HHt);
        advected_B           = eigvecs_HHt * diag(current_radii);
        advected_ellipse.S() = advected_B * transposed(eigvecs_HHt);
      }

      // check if particle has reached t_end
      if (t_advected == t_end &&
          sqr_cond_H <= split_sqr_cond + max_cond_overshoot) {
        finished_particles.emplace_back(advected_ellipse, t_advected, x0(),
                                        assembled_nabla_phi, m_id);
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
          auto       offset_ellipse = ellipse_type{
              advected_ellipse.center() + offset2,
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt)};

          splitted_particles.emplace_back(offset_ellipse, t_advected,
                                          x0() + offset0, assembled_nabla_phi,
                                          uuid_generator);
          auto lock = std::lock_guard{hierarchy_mutex};
          hierarchy_pairs.emplace_back(splitted_particles.back().m_id, m_id);
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
      }
    }
  }
  //----------------------------------------------------------------------------
  auto sampler() const {
    return sampler_type{initial_ellipse(), *this, m_nabla_phi};
  }
};
//==============================================================================
// typedefs
//==============================================================================
template <std::size_t NumDimensions>
using AutonomousParticle = autonomous_particle<real_number, NumDimensions>;
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
template <floating_point Real, std::size_t NumDimensions>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (autonomous_particle<Real, NumDimensions>),
    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
    TATOOINE_REFLECTION_INSERT_METHOD(S, S()),
    TATOOINE_REFLECTION_INSERT_METHOD(x, x()),
    TATOOINE_REFLECTION_INSERT_METHOD(t, t()),
    TATOOINE_REFLECTION_INSERT_METHOD(nabla_phi, nabla_phi()))
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
