#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/huber_loss.h>
#include <tatooine/staggered_flowmap_discretization.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/unstructured_simplicial_grid.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions,
          typename SplitBehavior = autonomous_particle<
              Real, NumDimensions>::split_behaviors::three_splits>
struct autonomous_particle_flowmap_discretization {
  using real_type           = Real;
  using vec_type            = vec<real_type, NumDimensions>;
  using pos_type            = vec_type;
  using particle_type       = autonomous_particle<real_type, NumDimensions>;
  using sampler_type        = typename particle_type::sampler_type;
  using sampler_container_t = std::vector<sampler_type>;
  using mesh_type = unstructured_simplicial_grid<real_type, NumDimensions>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  // std::optional<filesystem::path> m_path;
  std::vector<sampler_type>                                   m_samplers = {};
  mutable std::unique_ptr<pointset<real_type, NumDimensions>> m_centers0 =
      nullptr;
  mutable std::mutex m_centers0_mutex = {};
  mutable std::unique_ptr<pointset<real_type, NumDimensions>> m_centers1 =
      nullptr;
  mutable std::mutex m_centers1_mutex = {};
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  //  explicit autonomous_particle_flowmap_discretization(
  //      filesystem::path const& path)
  //      : m_path{path} {
  //    auto         file              = hdf5::file{*m_path};
  //    auto         particles_on_disk =
  //    file.dataset<particle_type>("finished"); std::size_t const
  //    total_num_particles =
  //        particles_on_disk.dataspace().current_resolution()[0];
  //
  //    auto ps = std::vector<particle_type>(total_num_particles);
  //    particles_on_disk.read(ps);
  //    m_path = std::nullopt;
  //
  //    m_samplers.resize(total_num_particles);
  //#pragma omp parallel for
  //    for (std::size_t i = 0; i < total_num_particles; ++i) {
  //      m_samplers[i] = ps[i].sampler();
  //    }
  //  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t_end,
      arithmetic auto const             tau_step,
      std::vector<particle_type> const& initial_particles,
      std::atomic_uint64_t&             uuid_generator) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t_end, tau_step,
         uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                     tau_step,
      uniform_rectilinear_grid<real_type, NumDimensions> const& g) {
    auto uuid_generator = std::atomic_uint64_t{};
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    auto particles                     = std::vector<particle_type>{};
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
    // auto const small_particle_size =
    //     (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
    //                initial_particle_distribution.dimension(0).spacing()) -
    //      initial_particle_distribution.dimension(0).spacing()) /
    //     2;

    // for (std::size_t i = 0; i < NumDimensions; ++i) {
    //   auto const spacing =
    //   initial_particle_distribution.dimension(i).spacing();
    //   initial_particle_distribution.dimension(i).pop_front();
    //   initial_particle_distribution.dimension(i).front() -= spacing / 2;
    //   initial_particle_distribution.dimension(i).back() -= spacing / 2;
    // }
    // initial_particle_distribution.vertices().iterate_indices(
    //     [&](auto const... is) {
    //       particles.emplace_back(
    //           initial_particle_distribution.vertex_at(is...), t0,
    //           small_particle_size);
    //     });
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step,
         uuid_generator);
  }
  ////----------------------------------------------------------------------------
  // template <typename Flowmap>
  // autonomous_particle_flowmap_discretization(
  //     Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
  //     arithmetic auto const                                tau_step,
  //     uniform_rectilinear_grid<real_type, NumDimensions> const& g,
  //     filesystem::path const&                              path)
  //     : m_path{path} {
  //   static_assert(
  //       std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
  //       "Number of dimensions of flowmap does not match number of
  //       dimensions.");
  //   auto initial_particle_distribution = g.copy_without_properties();
  //   std::vector<particle_type> particles;
  //   for (std::size_t i = 0; i < NumDimensions; ++i) {
  //     auto const spacing =
  //     initial_particle_distribution.dimension(i).spacing();
  //     initial_particle_distribution.dimension(i).pop_front();
  //     initial_particle_distribution.dimension(i).front() -= spacing / 2;
  //     initial_particle_distribution.dimension(i).back() -= spacing / 2;
  //   }
  //   initial_particle_distribution.vertices().iterate_indices(
  //       [&](auto const... is) {
  //         particles.emplace_back(
  //             initial_particle_distribution.vertex_at(is...), t0,
  //             initial_particle_distribution.dimension(0).spacing() / 2);
  //       });
  //   auto const small_particle_size =
  //       (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
  //                  initial_particle_distribution.dimension(0).spacing()) -
  //        initial_particle_distribution.dimension(0).spacing()) /
  //       2;
  //
  //   for (std::size_t i = 0; i < NumDimensions; ++i) {
  //     auto const spacing =
  //     initial_particle_distribution.dimension(i).spacing();
  //     initial_particle_distribution.dimension(i).pop_front();
  //     initial_particle_distribution.dimension(i).front() -= spacing / 2;
  //     initial_particle_distribution.dimension(i).back() -= spacing / 2;
  //   }
  //   initial_particle_distribution.vertices().iterate_indices(
  //       [&](auto const... is) {
  //         particles.emplace_back(
  //             initial_particle_distribution.vertex_at(is...), t0,
  //             initial_particle_distribution.dimension(0).spacing() / 2
  //             // small_particle_size
  //         );
  //       });
  //   fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
  // }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t_end,
      arithmetic auto const tau_step, particle_type const& initial_particle,
      std::atomic_uint64_t& uuid_generator) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, t_end,
         tau_step, uuid_generator);
  }
  //============================================================================
  auto samplers() const -> auto const& { return m_samplers; }
  //----------------------------------------------------------------------------
  auto hierarchy0() const -> auto const& {
    auto l = std::lock_guard{m_centers0_mutex};
    if (m_centers0 == nullptr) {
      m_centers0 = std::make_unique<pointset<real_type, NumDimensions>>();
      for (auto const& sa : m_samplers) {
        m_centers0->insert_vertex(sa.ellipse0().center());
      }
    }
    return m_centers0;
  }
  //----------------------------------------------------------------------------
  auto hierarchy1() const -> auto const& {
    auto l = std::lock_guard{m_centers1_mutex};
    if (m_centers1 == nullptr) {
      m_centers1 = std::make_unique<pointset<real_type, NumDimensions>>();
      for (auto const& sa : m_samplers) {
        m_centers1->insert_vertex(sa.ellipse1().center());
      }
    }
    return m_centers1;
  }
  //----------------------------------------------------------------------------
  auto hierarchy_mutex(forward_tag /*tag*/) const -> auto& {
    return m_centers0_mutex;
  }
  //----------------------------------------------------------------------------
  auto hierarchy_mutex(backward_tag /*tag*/) const -> auto& {
    return m_centers1_mutex;
  }
  //----------------------------------------------------------------------------
  auto hierarchy(forward_tag /*tag*/) const -> auto const& {
    return hierarchy0();
  }
  //----------------------------------------------------------------------------
  auto hierarchy(backward_tag /*tag*/) const -> auto const& {
    return hierarchy1();
  }
  //============================================================================
  auto num_particles() const -> std::size_t {
    // if (m_path) {
    //   auto file              = hdf5::file{*m_path};
    //   auto particles_on_disk = file.dataset<particle_type>("finished");
    //   return particles_on_disk.dataspace().current_resolution()[0];
    // } else {
    return size(m_samplers);
    //}
  }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const t_end, arithmetic auto const tau_step,
            std::atomic_uint64_t& uuid_generator) {
    // if (m_path) {
    //   particle_type::template advect<SplitBehavior>(
    //       std::forward<Flowmap>(flowmap), tau_step, t_end, initial_particles,
    //       *m_path);
    // } else {
    m_samplers.clear();
    auto [advected_particles, simple_particles, edges] =
        particle_type::template advect<SplitBehavior>(
            std::forward<Flowmap>(flowmap), tau_step, t_end, initial_particles,
            uuid_generator);
    m_samplers.reserve(size(advected_particles));
    using namespace std::ranges;
    auto get_sampler = [](auto const& p) { return p.sampler(); };
    copy(advected_particles | views::transform(get_sampler),
         std::back_inserter(m_samplers));
    //}
  }
  //----------------------------------------------------------------------------
  // template <std::size_t... VertexSeq>
  //[[nodiscard]] auto sample(pos_type const&                    p,
  //                          forward_or_backward_tag auto const tag,
  //                          execution_policy::parallel_t [>pol<],
  //                          std::index_sequence<VertexSeq...> [>seq<]) const {
  //  struct data {
  //    real_type                min_dist        =
  //    std::numeric_limits<real_type>::max(); sampler_type const*
  //    nearest_sampler = nullptr; pos_type            p;
  //  };
  //  auto best_per_thread = create_aligned_data_for_parallel<data>();
  //
  //  for_loop(
  //      [&](auto const& sampler) {
  //        auto&      best = *best_per_thread[omp_get_thread_num()];
  //        auto const p1   = sampler.sample(p, tag);
  //        if (auto const cur_dist =
  //                euclidean_length(sampler.opposite_center(tag) - p1);
  //            cur_dist < best.min_dist) {
  //          best.min_dist         = cur_dist;
  //          best.nearest_sampler = &sampler;
  //          best.p                = p1;
  //        }
  //      },
  //      execution_policy::parallel, m_samplers);
  //
  //  auto best = data{};
  //  for (auto const b : best_per_thread) {
  //    auto const& [min_dist, sampler, p] = *b;
  //    if (min_dist < best.min_dist) {
  //      best.min_dist         = min_dist;
  //      best.nearest_sampler = sampler;
  //      best.p                = p;
  //    }
  //  }
  //  return best.p;
  //}
  //----------------------------------------------------------------------------
 public:
  [[nodiscard]] auto sample_nearest_neighbor(
      pos_type const& q, forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto  ps                = pointset<real_type, NumDimensions>{};
    auto& initial_positions = ps.template vertex_property<pos_type>("ps");
    for (auto const& s : m_samplers) {
      auto v               = ps.insert_vertex(s.local_pos(q, tag));
      initial_positions[v] = s.center(opposite(tag));
    }
    auto [v, dist] = ps.nearest_neighbor(pos_type::zeros());
    return ps[v] + initial_positions[v];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_barycentric_coordinate(
      pos_type const& q, forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto local_positions =
        unstructured_simplicial_grid<real_type, NumDimensions, NumDimensions>{};
    auto& initial_positions =
        local_positions.template vertex_property<pos_type>("local_positions");
    for (auto const& s : m_samplers) {
      auto v               = local_positions.insert_vertex(s.local_pos(q, tag));
      initial_positions[v] = s.center(opposite(tag));
    }

    local_positions.build_delaunay_mesh();
    static auto constexpr b = vec<real_type, 3>{0, 0, 1};
    for (auto const s : local_positions.simplices()) {
      auto const barycentric_coordinate =
          local_positions.barycentric_coordinate(s, pos_type::zeros());
      auto is_inside = true;
      for (auto const c : barycentric_coordinate) {
        if (c < 0 || c > 1) {
          is_inside = false;
          break;
        }
      }
      if (is_inside) {
        auto const [v0, v1, v2] = local_positions[s];
        return (local_positions[v0] + initial_positions[v0]) *
                   barycentric_coordinate(0) +
               (local_positions[v1] + initial_positions[v1]) *
                   barycentric_coordinate(1) +
               (local_positions[v2] + initial_positions[v2]) *
                   barycentric_coordinate(2);
      }
    }
    return pos_type::fill(real_type(0) / real_type(0));
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_inverse_distance_local(
      std::size_t num_neighbors, pos_type const& q,
      forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto accumulated_position = pos_type{};
    auto accumulated_weight   = real_type{};

    auto ps = pointset<real_type, NumDimensions>{};
    for (auto const& s : m_samplers) {
      ps.insert_vertex(s.local_pos(q, tag));
    }

    auto [vertices, squared_distances] =
        ps.nearest_neighbors_radius(pos_type::zeros(), 0.01);
    auto squared_distance_it = begin(squared_distances);
    for (auto const v : vertices) {
      auto const x    = m_samplers[v.index()].phi(tag) + ps[v];
      auto const dist = *squared_distance_it;
      // auto const dist = huber_loss(gcem::sqrt(*squared_distance_it));
      if (dist == 0) {
        return x;
      };
      auto const weight = 1 / dist;
      accumulated_position += x * weight;
      accumulated_weight += weight;
      ++squared_distance_it;
    }
    return accumulated_position / accumulated_weight;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_inverse_distance(
      std::size_t num_neighbors, pos_type const& q,
      forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto accumulated_position = pos_type{};
    auto accumulated_weight   = real_type{};

    auto ps = pointset<real_type, NumDimensions>{};
    for (auto const& s : m_samplers) {
      ps.insert_vertex(s.x0(tag));
    }

    auto [vertices, squared_distances] = ps.nearest_neighbors_radius(q, 0.01);
    auto squared_distance_it           = begin(squared_distances);
    for (auto const v : vertices) {
      auto const x    = m_samplers[v.index()](q, tag);
      auto const dist = *squared_distance_it;
      // auto const dist = huber_loss(gcem::sqrt(*squared_distance_it));
      if (dist == 0) {
        return x;
      };
      auto const weight = 1 / dist;
      accumulated_position += x * weight;
      accumulated_weight += weight;
      ++squared_distance_it;
    }
    return accumulated_position / accumulated_weight;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_inverse_distance_without_gradient(
      std::size_t num_neighbors, pos_type const& q,
      forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto accumulated_position = pos_type{};
    auto accumulated_weight   = real_type{};

    auto ps = pointset<real_type, NumDimensions>{};
    for (auto const& s : m_samplers) {
      ps.insert_vertex(s.x0(tag));
    }

    auto [vertices, squared_distances] = ps.nearest_neighbors_radius(q, 0.01);
    auto squared_distance_it           = begin(squared_distances);
    for (auto const v : vertices) {
      auto const x                = m_samplers[v.index()].phi(tag);
      auto const squared_distance = *squared_distance_it + 1e-10;
      // auto const squared_distance =
      // huber_loss(gcem::sqrt(*squared_distance_it));
      if (squared_distance == 0) {
        return x;
      };
      auto const weight = 1 / squared_distance;
      accumulated_position += x * weight;
      accumulated_weight += weight;
      ++squared_distance_it;
    }
    return accumulated_position / accumulated_weight;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_radial_basis_functions(
      pos_type const& q, Real const radius,
      forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto  ps   = pointset<real_type, NumDimensions>{};
    auto& cart = ps.template vertex_property<pos_type>("cartesian");
    ps.vertices().reserve(size(m_samplers));
    for (auto const& s : m_samplers) {
      auto v = ps.insert_vertex(s.x0(tag));
      cart[v] = s.x0(tag);
    }

    auto [vertices, squared_distances] =
        ps.nearest_neighbors_raw(pos_type::zeros(), 20);
    if (vertices.empty()) {
      return pos_type::fill(0.0 / 0.0);
    }

    auto rbf_kernel = thin_plate_spline_from_squared;
    auto const N = vertices.size();

    // construct lower part of symmetric matrix A
    auto A = tensor<real_number>::zeros(N + NumDimensions + 1,
                                        N + NumDimensions + 1);
    auto radial_and_monomial_coefficients =
        tensor<real_number>::zeros(N + NumDimensions + 1, NumDimensions);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = rbf_kernel(squared_euclidean_distance(
            cart[vertices[c]], cart[vertices[r]]));
      }
    }
    // construct polynomial requirements
    for (std::size_t c = 0; c < N; ++c) {
      auto const& p = cart[vertices[c]];
      // constant part
      A(N, c) = 1;

      // linear part
      for (std::size_t i = 0; i < NumDimensions; ++i) {
        A(N + i + 1, c) = p(i);
      }
    }

    for (std::size_t i = 0; i < N; ++i) {
      auto const phi = m_samplers[vertices[i]].phi(tag);
      for (std::size_t j = 0; j < NumDimensions; ++j) {
        radial_and_monomial_coefficients(i, j) = phi(j);
      }
    }
    // do not copy by moving A and radial_and_monomial_coefficients into solver
    radial_and_monomial_coefficients = *solve_symmetric_lapack(
        std::move(A), std::move(radial_and_monomial_coefficients),
        tatooine::lapack::Uplo::Lower);

    auto acc = pos_type{};
    // radial bases
    for (std::size_t i = 0; i < N; ++i) {
      auto const v = vertices[i];
      if (squared_distances[i] == 0) {
        return m_samplers[v].phi(tag);
      }
      for (std::size_t j = 0; j < NumDimensions; ++j) {
        acc(j) += radial_and_monomial_coefficients(i, j) *
                  rbf_kernel(squared_euclidean_distance(q, cart[v]));
      }
    }
    // monomial bases
    for (std::size_t j = 0; j < NumDimensions; ++j) {
      acc(j) += radial_and_monomial_coefficients(N, j);
      for (std::size_t k = 0; k < NumDimensions; ++k) {
        acc(j) += radial_and_monomial_coefficients(N + 1 + k, j) * q(k);
      }
    }
    return acc;
  }
  //----------------------------------------------------------------------------
  template <std::size_t... VertexSeq>
  [[nodiscard]] auto sample_somehow(
      pos_type const& q, forward_or_backward_tag auto const tag,
      execution_policy::sequential_t /*pol*/) const {
    auto  ps                = pointset<real_type, NumDimensions>{};
    auto& initial_positions = ps.template vertex_property<pos_type>("ps");

    for (auto const& s : m_samplers) {
      auto v               = ps.insert_vertex(s.local_pos(tag));
      initial_positions[v] = s.center(opposite(tag));
    }
    auto [vertices, distances] = ps.nearest_neighbors_raw(pos_type::zeros(), 5);
    auto sum                   = real_type{};
    for (auto& d : distances) {
      d = 1 / d;
      sum += d;
    }
    for (auto& d : distances) {
      d /= sum;
    }

    auto p_ret = pos_type{};

    for (std::size_t i = 0; i < vertices.size(); ++i) {
      auto v = typename pointset<real_type, NumDimensions>::vertex_handle{
          std::size_t(vertices[i])};
      p_ret += (ps[v] + initial_positions[v]) * distances[i];
    }
    return p_ret;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample(pos_type const&                     q,
                            forward_or_backward_tag auto const  tag,
                            execution_policy::policy auto const pol) const {
    return sample(q, tag, pol, std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample(pos_type const&                    q,
                            forward_or_backward_tag auto const tag) const {
    return sample(q, tag, execution_policy::sequential);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(
      pos_type const& q, execution_policy::policy auto const pol) const {
    return sample(q, forward, pol);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(pos_type const& q) const {
    return sample(q, forward, execution_policy::sequential);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_type const& q) const {
    return sample(q, backward, execution_policy::sequential);
  }
  auto sample_backward(pos_type const&                     q,
                       execution_policy::policy auto const pol) const {
    return sample(q, backward, pol);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const& q, forward_or_backward_tag auto tag) const {
    return sample(q, tag, execution_policy::sequential);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const& q, forward_or_backward_tag auto tag,
                  execution_policy::policy auto const pol) const {
    return sample(q, tag, pol);
  }
};
//==============================================================================
template <std::size_t NumDimensions,
          typename SplitBehavior = typename autonomous_particle<
              real_number, NumDimensions>::split_behaviors::three_splits>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_number, NumDimensions>;
using autonomous_particle_flowmap_discretization2 =
    AutonomousParticleFlowmapDiscretization<2>;
using autonomous_particle_flowmap_discretization3 =
    AutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions>
using staggered_autonomous_particle_flowmap_discretization =
    staggered_flowmap_discretization<
        autonomous_particle_flowmap_discretization<Real, NumDimensions>>;
//------------------------------------------------------------------------------
template <std::size_t NumDimensions>
using StaggeredAutonomousParticleFlowmapDiscretization =
    staggered_autonomous_particle_flowmap_discretization<real_number,
                                                         NumDimensions>;
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
