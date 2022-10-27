#ifndef TATOOINE_FIELDS_AUTONOMOUS_PARTICLE_H
#define TATOOINE_FIELDS_AUTONOMOUS_PARTICLE_H
//==============================================================================
#include <tatooine/cache_alignment.h>
#include <tatooine/functional.h>
#include <tatooine/concepts.h>
#include <tatooine/detail/autonomous_particle/post_triangulation.h>
#include <tatooine/detail/autonomous_particle/sampler.h>
#include <tatooine/detail/autonomous_particle/split_behavior.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/tensor_operations.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/particle.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename B>
concept split_behavior = requires {
  requires floating_point<decltype(B::split_cond)>;
  requires range<decltype(B::radii)>;
  requires range<decltype(B::offsets)>;
  requires static_vec<typename decltype(B::radii)::value_type>;
  requires static_vec<typename decltype(B::offsets)::value_type>;
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct autonomous_particle : geometry::hyper_ellipse<Real, NumDimensions> {
  using split_behaviors =
      detail::autonomous_particle::split_behaviors<Real, NumDimensions>;
  static constexpr auto num_dimensions() -> std::size_t { return NumDimensions; }
  //============================================================================
  // TYPEDEFS
  //============================================================================
 public:
  //----------------------------------------------------------------------------
  constexpr static auto half = 1 / Real(2);

  using this_type            = autonomous_particle<Real, NumDimensions>;
  using simple_particle_type = particle<Real, NumDimensions>;
  using real_type            = Real;
  using vec_type             = vec<real_type, NumDimensions>;
  using mat_type             = mat<real_type, NumDimensions, NumDimensions>;
  using pos_type             = vec_type;
  using container_type       = std::vector<this_type>;
  using simple_particle_container_type = std::vector<simple_particle_type>;
  using ellipse_type = geometry::hyper_ellipse<Real, NumDimensions>;
  using parent_type  = ellipse_type;
  using sampler_type =
      detail::autonomous_particle::sampler<Real, NumDimensions>;
  using hierarchy_pair = detail::autonomous_particle::hierarchy_pair;
  using parent_type::center;
  using parent_type::discretize;
  using parent_type::S;
  //============================================================================
  // static members
  //============================================================================
  static constexpr auto default_max_split_depth = 6;
  //============================================================================
  // members
  //============================================================================
 private:
  pos_type      m_x0              = {};
  real_type     m_t               = {};
  mat_type      m_nabla_phi       = {};
  std::uint8_t  m_split_depth     = 0;
  std::uint8_t  m_max_split_depth = default_max_split_depth;
  std::uint64_t m_id              = std::numeric_limits<std::uint64_t>::max();

  static auto mutex() -> auto& {
    static auto m = std::mutex{};
    return m;
  }
  //============================================================================
  // CTORS
  //============================================================================
 public:
  //----------------------------------------------------------------------------
  /// {
  autonomous_particle(autonomous_particle const& other)     = default;
  autonomous_particle(autonomous_particle&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator =(autonomous_particle const& other)
      -> autonomous_particle& = default;
  auto operator=(autonomous_particle&& other) noexcept
      -> autonomous_particle& = default;
  //----------------------------------------------------------------------------
  ~autonomous_particle() = default;
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::uint64_t const id)
      : parent_type{ell},
        m_x0{ell.center()},
        m_t{t},
        m_nabla_phi{mat_type::eye()},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{ell, t, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::uint8_t max_split_depth, std::uint64_t const id)
      : parent_type{ell},
        m_x0{ell.center()},
        m_t{t},
        m_nabla_phi{mat_type::eye()},
        m_max_split_depth{max_split_depth},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      std::uint8_t          max_split_depth,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{ell, t, max_split_depth, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::uint64_t const id)
      : parent_type{x, r},
        m_x0{x},
        m_t{t},
        m_nabla_phi{mat_type::eye()},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{x, t, r, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::uint8_t max_split_depth, std::uint64_t const id)
      : parent_type{x, r},
        m_x0{x},
        m_t{t},
        m_nabla_phi{mat_type::eye()},
        m_max_split_depth{max_split_depth},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(pos_type const& x, real_type const t, real_type const r,
                      std::uint8_t          max_split_depth,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{x, t, r, max_split_depth, uuid_generator++} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      pos_type const& x0, mat_type const& nabla_phi,
                      std::uint8_t const  split_depth,
                      std::uint8_t const  max_split_depth,
                      std::uint64_t const id)
      : parent_type{ell},
        m_x0{x0},
        m_t{t},
        m_nabla_phi{nabla_phi},
        m_split_depth{split_depth},
        m_max_split_depth{max_split_depth},
        m_id{id} {}
  //----------------------------------------------------------------------------
  autonomous_particle(ellipse_type const& ell, real_type const t,
                      pos_type const& x0, mat_type const& nabla_phi,
                      std::uint8_t const    split_depth,
                      std::uint8_t const    max_split_depth,
                      std::atomic_uint64_t& uuid_generator)
      : autonomous_particle{ell,
                            t,
                            x0,
                            nabla_phi,
                            split_depth,
                            max_split_depth,
                            uuid_generator++} {}
  /// \}
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
  auto split_depth() const { return m_split_depth; }
  auto max_split_depth() const { return m_max_split_depth; }
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
    // auto map   = std::unordered_map<
    //     std::size_t, typename edgeset<Real, NumDimensions>::vertex_handle>{};
    // for (auto const& p : advected_particles) {
    //   map[p.id()] = edges.insert_vertex(p.center());
    // }
    // triangulate(edges, hierarchy{hierarchy_pairs, map, edges});
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
    // hierarchy_pairs.reserve(particles.size());
    // for (auto const& p : particles) {
    //   hierarchy_pairs.emplace_back(p.id(), p.id());
    // }
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
  //----------------------------------------------------------------------------
  static auto particles_from_grid(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::atomic_uint64_t&                                uuid_generator) {
    return particles_from_grid(t0, g, default_max_split_depth, uuid_generator);
  }
  //----------------------------------------------------------------------------
  static auto particles_from_grid(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::uint8_t const                                   max_split_depth,
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
              max_split_depth, uuid_generator);
        });
    return particles;
  }
  //------------------------------------------------------------------------------
  static auto particles_from_grid_small_filling_gaps(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::atomic_uint64_t&                                uuid_generator) {
    return particles_from_grid_small_filling_gaps(
        t0, g, default_max_split_depth, uuid_generator);
  }
  //----------------------------------------------------------------------------
  static auto particles_from_grid_small_filling_gaps(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::uint8_t const                                   max_split_depth,
      std::atomic_uint64_t&                                uuid_generator) {
    auto       particles                     = container_type{};
    auto       initial_particle_distribution = g.copy_without_properties();
    auto const radius =
        initial_particle_distribution.dimension(0).spacing() / 2;
    for (std::size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(initial_particle_distribution.vertex_at(is...),
                                 t0, radius, max_split_depth, uuid_generator);
        });
    auto const small_radius =
        (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
                   initial_particle_distribution.dimension(0).spacing()) -
         initial_particle_distribution.dimension(0).spacing()) /
        2;

    for (std::size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(initial_particle_distribution.vertex_at(is...),
                                 t0, small_radius, max_split_depth,
                                 uuid_generator);
        });
    return particles;
  }
  //------------------------------------------------------------------------------
  static auto particles_from_grid_filling_gaps(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::atomic_uint64_t&                                uuid_generator) {
    return particles_from_grid_filling_gaps(t0, g, default_max_split_depth,
                                            uuid_generator);
  }
  //------------------------------------------------------------------------------
  static auto particles_from_grid_filling_gaps(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::uint8_t const                                   max_split_depth,
      std::atomic_uint64_t&                                uuid_generator) {
    return particles_from_grid_filling_gaps(
        t0, g, max_split_depth, uuid_generator,
        std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  static auto particles_from_grid_filling_gaps(
      real_type const                                      t0,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      std::uint8_t const max_split_depth, std::atomic_uint64_t& uuid_generator,
      std::index_sequence<Is...> /*idx_seq*/) {
    auto       particles                     = container_type{};
    auto       initial_particle_distribution = g.copy_without_properties();
    auto const radius =
        initial_particle_distribution.dimension(0).spacing() / 2;
    invoke([&] {
      auto       dim = initial_particle_distribution.template dimension<Is>();
      auto const half_spacing = dim.spacing() / 2;
      dim.pop_front();
      dim.front() -= half_spacing;
      dim.back() -= half_spacing;
      initial_particle_distribution.template set_dimension<Is>(dim);
    }...);
    initial_particle_distribution.vertices().iterate_positions(
        [&](auto const& x) {
          particles.emplace_back(x, t0, radius, max_split_depth,
                                 uuid_generator);
        });

    invoke([&] {
      auto       dim = initial_particle_distribution.template dimension<Is>();
      auto const half_spacing = dim.spacing() / 2;
      dim.pop_front();
      dim.front() -= half_spacing;
      dim.back() -= half_spacing;
      initial_particle_distribution.template set_dimension<Is>(dim);
    }...);
    initial_particle_distribution.vertices().iterate_positions(
        [&](auto const& x) {
          particles.emplace_back(x, t0, radius, max_split_depth,
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
    //hierarchy_pairs.reserve(particles.size());
    //for (auto const& p : particles) {
    //  hierarchy_pairs.emplace_back(p.id(), p.id());
    //}
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
    // triangulate(edges, h);
    //
    // auto const s = g.size();
    //--s[0];
    //--s[1];
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
    auto finished_simple_particles = simple_particle_container_type{};

    // {particles_to_be_advected, advected_particles, finished_particles}
    auto particles_per_thread = std::vector<
        aligned<std::tuple<container_type, container_type, container_type,
                           simple_particle_container_type>>>(num_threads);
    while (particles.size() > 0) {
      distribute_particles_to_thread_containers(num_threads, particles,
                                                particles_per_thread);
      advect_particle_pools<SplitBehavior>(
          num_threads, std::forward<Flowmap>(phi), stepwidth, t_end,
          particles_per_thread, hierarchy_pairs, hierarchy_mutex,
          uuid_generator);
      gather_particles(particles, finished_particles, finished_simple_particles,
                       particles_per_thread);
    }
    return std::tuple{std::move(finished_particles),
                      std::move(finished_simple_particles)};
  }
  //----------------------------------------------------------------------------
  static auto num_threads() {
    auto num_threads = std::size_t{};
#pragma omp parallel
    {
      if (omp_get_thread_num() == 0) {
        num_threads = static_cast<std::size_t>(omp_get_num_threads());
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
      std::size_t const /*num_threads*/, Flowmap&& phi, real_type const stepwidth,
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
  static auto gather_particles(container_type& particles,
                               container_type& finished_particles,
                               simple_particle_container_type& simple_particles,
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
  /// Old split criterion!
  ///
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
  // template <
  //     split_behavior SplitBehavior = typename split_behaviors::three_splits,
  //     typename Flowmap>
  // auto advect_until_split(Flowmap phi, real_type stepwidth,
  //                         real_type const                 t_end,
  //                         container_type&                 splitted_particles,
  //                         container_type&                 finished_particles,
  //                         simple_particle_container_type& simple_particles,
  //                         std::vector<hierarchy_pair>&    hierarchy_pairs,
  //                         std::mutex&                     hierarchy_mutex,
  //                         std::atomic_uint64_t& uuid_generator) const {
  //   if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
  //     phi.use_caching(false);
  //   }
  //   static constexpr real_type min_tau_step       = 1e-10;
  //   static constexpr real_type max_cond_overshoot = 1e-8;
  //   static constexpr auto      split_cond         =
  //   SplitBehavior::split_cond; static constexpr auto      split_sqr_cond =
  //   split_cond * split_cond; static constexpr auto      split_radii        =
  //   SplitBehavior::radii; static constexpr auto      split_offsets      =
  //   SplitBehavior::offsets; auto const [eigvecs_S, eigvals_S]             =
  //   this->main_axes(); auto const B = eigvecs_S * diag(eigvals_S);  //
  //   current main axes auto const K = *solve(diag(eigvals_S),
  //   transposed(eigvecs_S));
  //
  //   mat_type H, HHt, advected_nabla_phi, assembled_nabla_phi, advected_B,
  //       ghosts_positive_offset, ghosts_negative_offset, prev_ghosts_forward,
  //       prev_ghosts_backward;
  //   auto        min_stepwidth_reached = false;
  //   auto        advected_ellipse      = ellipse_type{*this};
  //   auto        current_radii         = vec_type{};
  //   auto        eig_HHt               = std::pair<mat_type, vec_type>{};
  //   auto        sqr_cond_H            = real_type(1);
  //   auto const& eigvecs_HHt           = eig_HHt.first;
  //   auto const& eigvals_HHt           = eig_HHt.second;
  //
  //   // initialize ghosts
  //   for (std::size_t i = 0; i < num_dimensions(); ++i) {
  //     ghosts_positive_offset.col(i) = x();
  //   }
  //   ghosts_negative_offset = ghosts_positive_offset;
  //
  //   ghosts_positive_offset += B;
  //   ghosts_negative_offset -= B;
  //
  //   // fields for backup
  //   auto t_prev          = t();
  //   auto prev_center     = advected_ellipse.center();
  //   prev_ghosts_forward  = ghosts_positive_offset;
  //   prev_ghosts_backward = ghosts_negative_offset;
  //   auto prev_cond_HHt   = sqr_cond_H;
  //
  //   // repeat as long as particle's ellipse is not wide enough or t_end is
  //   not
  //   // reached or the ellipse gets too small. If the latter happens make it a
  //   // simple massless particle
  //   auto t_advected = t();
  //   while (sqr_cond_H < split_sqr_cond || t_advected < t_end) {
  //     if (!min_stepwidth_reached) {
  //       // backup state before advection
  //       prev_ghosts_forward  = ghosts_positive_offset;
  //       prev_ghosts_backward = ghosts_negative_offset;
  //       prev_center          = advected_ellipse.center();
  //       prev_cond_HHt        = sqr_cond_H;
  //       t_prev               = t_advected;
  //
  //       // increase time
  //       if (t_advected + stepwidth > t_end) {
  //         stepwidth  = t_end - t_advected;
  //         t_advected = t_end;
  //       } else {
  //         t_advected += stepwidth;
  //       }
  //       auto const cur_stepwidth = t_advected - t_prev;
  //
  //       // advect center and ghosts
  //       advected_ellipse.center() =
  //           phi(advected_ellipse.center(), t_advected, cur_stepwidth);
  //       ghosts_positive_offset  = phi(ghosts_positive_offset, t_advected,
  //       cur_stepwidth); ghosts_negative_offset = phi(ghosts_negative_offset,
  //       t_advected, cur_stepwidth);
  //
  //       // make computations
  //       H          = (ghosts_positive_offset - ghosts_negative_offset) *
  //       half; HHt        = H * transposed(H); eig_HHt    =
  //       eigenvectors_sym(HHt); sqr_cond_H = eigvals_HHt(num_dimensions() - 1)
  //       / eigvals_HHt(0);
  //
  //       if (std::isnan(sqr_cond_H)) {
  //         simple_particles.emplace_back(x0(), advected_ellipse.center(),
  //                                       t_advected);
  //       }
  //       advected_nabla_phi  = H * K;
  //       assembled_nabla_phi = advected_nabla_phi * m_nabla_phi;
  //
  //       current_radii        = sqrt(eigvals_HHt);
  //       advected_B           = eigvecs_HHt * diag(current_radii);
  //       advected_ellipse.S() = advected_B * transposed(eigvecs_HHt);
  //     }
  //
  //     // check if particle has reached t_end
  //     if (t_advected == t_end &&
  //         sqr_cond_H <= split_sqr_cond + max_cond_overshoot) {
  //       finished_particles.emplace_back(advected_ellipse, t_advected, x0(),
  //                                       assembled_nabla_phi, m_id);
  //       return;
  //     }
  //
  //     // check if particle's ellipse has reached its splitting width
  //     if ((sqr_cond_H >= split_sqr_cond &&
  //          sqr_cond_H <= split_sqr_cond + max_cond_overshoot) ||
  //         min_stepwidth_reached) {
  //       for (std::size_t i = 0; i < size(split_radii); ++i) {
  //         auto const new_eigvals    = current_radii * split_radii[i];
  //         auto const offset2        = advected_B * split_offsets[i];
  //         auto const offset0        = *solve(assembled_nabla_phi, offset2);
  //         auto       offset_ellipse = ellipse_type{
  //             advected_ellipse.center() + offset2,
  //             eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt)};
  //
  //         splitted_particles.emplace_back(offset_ellipse, t_advected,
  //                                         x0() + offset0,
  //                                         assembled_nabla_phi,
  //                                         uuid_generator);
  //         auto lock = std::lock_guard{hierarchy_mutex};
  //         hierarchy_pairs.emplace_back(splitted_particles.back().m_id, m_id);
  //       }
  //       return;
  //     }
  //     // check if particle's ellipse is wider than its splitting width
  //     if (sqr_cond_H > split_sqr_cond + max_cond_overshoot) {
  //       auto const prev_stepwidth = stepwidth;
  //       stepwidth *= half;
  //       min_stepwidth_reached = stepwidth == prev_stepwidth;
  //       if (stepwidth < min_tau_step) {
  //         min_stepwidth_reached = true;
  //       }
  //       if (!min_stepwidth_reached) {
  //         sqr_cond_H                = prev_cond_HHt;
  //         ghosts_positive_offset            = prev_ghosts_forward;
  //         ghosts_negative_offset           = prev_ghosts_backward;
  //         t_advected                = t_prev;
  //         advected_ellipse.center() = prev_center;
  //       }
  //     }
  //   }
  // }
  //----------------------------------------------------------------------------
  /// New split criterion!
  ///
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
                          real_type const                 t_end,
                          container_type&                 splitted_particles,
                          container_type&                 finished_particles,
                          simple_particle_container_type& simple_particles,
                          std::vector<hierarchy_pair>&    /*hierarchy_pairs*/,
                          std::mutex&                     /*hierarchy_mutex*/,
                          std::atomic_uint64_t& uuid_generator) const {
    if constexpr (is_cacheable<std::decay_t<decltype(phi)>>()) {
      phi.use_caching(false);
    }
    // static constexpr real_type min_tau_step       = 1e-10;
    // static constexpr real_type max_cond_overshoot = 1e-8;
    // static constexpr auto      split_cond         =
    // SplitBehavior::split_cond;
    static constexpr auto split_radii   = SplitBehavior::radii;
    static constexpr auto split_offsets = SplitBehavior::offsets;
    auto const [eigvecs_S, eigvals_S]   = this->main_axes();
    auto const B = eigvecs_S * diag(eigvals_S);  // current main axes
    auto const K = *solve(diag(eigvals_S), transposed(eigvecs_S));

    mat_type H, HHt, D, advected_nabla_phi, assembled_nabla_phi, advected_B,
        ghosts_positive_offset, ghosts_negative_offset, prev_ghosts_forward,
        prev_ghosts_backward;
    auto        advected_ellipse = ellipse_type{*this};
    auto        current_radii    = vec_type{};
    auto        eig_HHt          = std::pair<mat_type, vec_type>{};
    auto        sqr_cond_H       = real_type(1);
    auto        linearity        = real_type(0);
    auto const& eigvecs_HHt      = eig_HHt.first;
    auto const& eigvals_HHt      = eig_HHt.second;

    // initialize ghosts
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      ghosts_positive_offset.col(i) = x();
      ghosts_negative_offset.col(i) = x();
    }
    ghosts_positive_offset += B;
    ghosts_negative_offset -= B;

    // repeat as long as particle's ellipse is not wide enough or t_end is not
    // reached or the ellipse gets too small. If the latter happens make it a
    // simple massless particle
    auto t_advected = t();
    while (t_advected < t_end) {
      if (t_advected + stepwidth > t_end) {
        stepwidth = t_end - t_advected;
      }

      // advect center and ghosts
      advected_ellipse.center() =
          phi(advected_ellipse.center(), t_advected, stepwidth);
      ghosts_positive_offset =
          phi(ghosts_positive_offset, t_advected, stepwidth);
      ghosts_negative_offset =
          phi(ghosts_negative_offset, t_advected, stepwidth);

      // increase time
      if (t_advected + stepwidth + 1e-6 > t_end) {
        t_advected = t_end;
      } else {
        t_advected += stepwidth;
      }

      // make computations
      H = (ghosts_positive_offset - ghosts_negative_offset) * half;
      D = (ghosts_negative_offset + ghosts_positive_offset) * half;
      for (std::size_t i = 0; i < num_dimensions(); ++i) {
        D.col(i) -= advected_ellipse.center();
      }
      linearity = 0;
      for (std::size_t i = 0; i < num_dimensions(); ++i) {
        linearity += dot(D.col(i), D.col(i)) / dot(H.col(i), H.col(i));
      }
      HHt     = H * transposed(H);
      eig_HHt = eigenvectors_sym(HHt);

      if (std::isnan(linearity)) {
        simple_particles.emplace_back(x0(), advected_ellipse.center(),
                                      t_advected);
        return;
      }
      sqr_cond_H = eigvals_HHt(num_dimensions() - 1) / eigvals_HHt(0);

      advected_nabla_phi  = H * K;
      assembled_nabla_phi = advected_nabla_phi * m_nabla_phi;

      current_radii        = sqrt(eigvals_HHt);
      advected_B           = eigvecs_HHt * diag(current_radii);
      advected_ellipse.S() = advected_B * transposed(eigvecs_HHt);

      // check if particle has reached t_end
      if (t_advected == t_end) {
        finished_particles.emplace_back(advected_ellipse, t_advected, x0(),
                                        assembled_nabla_phi, split_depth(),
                                        max_split_depth(), id());
        return;
      }

      // check if particle's ellipse has reached its splitting width
      static auto constexpr linearity_threshold = 1e-3;
      if (split_depth() != max_split_depth() &&
          (linearity >= linearity_threshold || sqr_cond_H > 10)) {
        for (std::size_t i = 0; i < size(split_radii); ++i) {
          auto const new_eigvals    = current_radii * split_radii[i];
          auto const offset2        = advected_B * split_offsets[i];
          auto const offset0        = *solve(assembled_nabla_phi, offset2);
          auto       offset_ellipse = ellipse_type{
              advected_ellipse.center() + offset2,
              eigvecs_HHt * diag(new_eigvals) * transposed(eigvecs_HHt)};

          splitted_particles.emplace_back(
              offset_ellipse, t_advected, x0() + offset0, assembled_nabla_phi,
              split_depth() + 1, max_split_depth(), uuid_generator);
          //auto lock = std::lock_guard{hierarchy_mutex};
          //hierarchy_pairs.emplace_back(splitted_particles.back().m_id, m_id);
        }
        return;
      }
    }
  }
  //----------------------------------------------------------------------------
  auto sampler() const {
    return sampler_type{initial_ellipse(), *this, m_nabla_phi};
  }
  //----------------------------------------------------------------------------
  auto discretize(std::size_t const n, forward_tag const /*tag*/) const {
    return initial_ellipse().discretize(n);
  }
  //----------------------------------------------------------------------------
  auto discretize(std::size_t const n, backward_tag const /*tag*/) const {
    return discretize(n);
  }
};
//------------------------------------------------------------------------------
template <floating_point Real>
auto write_vtp(std::vector<autonomous_particle<Real, 2>> const& particles,
               std::size_t const n, filesystem::path const& path,
               backward_tag const /*tag*/) {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset                    = std::size_t{};
  using header_type              = std::uint64_t;
  using lines_connectivity_int_t = std::int32_t;
  using lines_offset_int_t       = lines_connectivity_int_t;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::to_string(
              vtk::xml::to_type<header_type>())
       << "\">";
  file << "<PolyData>\n";
  for (std::size_t i = 0; i < size(particles); ++i) {
    file << "<Piece"
         << " NumberOfPoints=\"" << n << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"" << n - 1 << "\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<Real>())
         << "\" NumberOfComponents=\"" << 3 << "\"/>";
    auto const num_bytes_points = header_type(sizeof(Real) * 3 * n);
    offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Lines
    file << "<Lines>\n";
    // Lines - connectivity
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<lines_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_lines_connectivity =
        (n - 1) * 2 * sizeof(lines_connectivity_int_t);
    offset += num_bytes_lines_connectivity + sizeof(header_type);
    // Lines - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<lines_offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_lines_offsets =
        sizeof(lines_offset_int_t) * (n - 1) * 2;
    offset += num_bytes_lines_offsets + sizeof(header_type);
    file << "</Lines>\n";
    file << "</Piece>\n";
  }
  file << "</PolyData>\n";
  file << "<AppendedData encoding=\"raw\">_";
  // Writing vertex data to appended data section
  for (auto const& particle : particles) {
    auto const num_bytes_points = header_type(sizeof(Real) * 3 * n);
    using namespace std::ranges;
    auto radial = tatooine::linspace<Real>{0, M_PI * 2, n + 1};
    radial.pop_back();

    auto discretization      = tatooine::line<Real, 3>{};
    auto radian_to_cartesian = [](auto const t) {
      return tatooine::vec{gcem::cos(t), gcem::sin(t), 0};
    };
    auto out_it = std::back_inserter(discretization);
    copy(radial | views::transform(radian_to_cartesian), out_it);
    discretization.set_closed(true);
    for (auto const v : discretization.vertices()) {
      auto v2 = particle.S() * discretization[v].xy() + particle.center();
      discretization[v].x() = v2.x();
      discretization[v].y() = v2.y();
    }

    // Writing points
    file.write(reinterpret_cast<char const*>(&num_bytes_points),
               sizeof(header_type));
    for (auto const v : discretization.vertices()) {
      file.write(reinterpret_cast<char const*>(discretization.at(v).data()),
                 sizeof(Real) * 3);
    }

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<lines_connectivity_int_t>{};
      connectivity_data.reserve((n - 1) * 2);
      for (std::size_t i = 0; i < n - 1; ++i) {
        connectivity_data.push_back(static_cast<lines_connectivity_int_t>(i));
        connectivity_data.push_back(static_cast<lines_connectivity_int_t>(i + 1));
      }

      auto const num_bytes_lines_connectivity =
          header_type((n - 1) * 2 * sizeof(lines_connectivity_int_t));
      file.write(reinterpret_cast<char const*>(&num_bytes_lines_connectivity),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 static_cast<std::streamsize>(num_bytes_lines_connectivity));
    }

    // Writing lines offsets to appended data section
    {
      auto offsets = std::vector<lines_offset_int_t>(n, 2);
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      auto const num_bytes_lines_offsets =
          header_type(sizeof(lines_offset_int_t) * (n - 1) * 2);
      file.write(reinterpret_cast<char const*>(&num_bytes_lines_offsets),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()),
                 static_cast<std::streamsize>(num_bytes_lines_offsets));
    }
  }

  file << "</AppendedData>";
  file << "</VTKFile>";
}
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
//#include <tatooine/reflection.h>
//namespace tatooine::reflection {
//==============================================================================
//template <floating_point Real, std::size_t NumDimensions>
//TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
//    (autonomous_particle<Real, NumDimensions>),
//    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
//    TATOOINE_REFLECTION_INSERT_METHOD(S, S()),
//    TATOOINE_REFLECTION_INSERT_METHOD(x, x()),
//    TATOOINE_REFLECTION_INSERT_METHOD(t, t()),
//    TATOOINE_REFLECTION_INSERT_METHOD(nabla_phi, nabla_phi()))
//==============================================================================
//}  // namespace tatooine::reflection
//==============================================================================
#endif
