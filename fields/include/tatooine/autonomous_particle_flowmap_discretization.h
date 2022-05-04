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
          typename SplitBehavior = typename autonomous_particle<
              Real, NumDimensions>::split_behaviors::three_splits>
struct autonomous_particle_flowmap_discretization {
  using real_type              = Real;
  using vec_type               = vec<real_type, NumDimensions>;
  using pos_type               = vec_type;
  using gradient_type          = mat<real_type, NumDimensions, NumDimensions>;
  using particle_type          = autonomous_particle<real_type, NumDimensions>;
  using sampler_type           = typename particle_type::sampler_type;
  using sampler_container_type = std::vector<sampler_type>;
  using mesh_type = unstructured_simplicial_grid<real_type, NumDimensions>;

  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using flowmap_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<pos_type>;
  using flowmap_gradient_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<
          gradient_type>;

  using cgal_kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using cgal_triangulation_type =
      cgal::delaunay_triangulation_with_info<NumDimensions, cgal_kernel,
                                             vertex_handle>;
  using cgal_point = typename cgal_triangulation_type::Point;

  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  // std::optional<filesystem::path> m_path;
  pointset_type                          m_pointset_forward;
  flowmap_vertex_property_type&          m_flowmaps_forward;
  flowmap_gradient_vertex_property_type& m_flowmap_gradients_forward;
  cgal_triangulation_type                m_triangulation_forward;

  pointset_type                          m_pointset_backward;
  flowmap_vertex_property_type&          m_flowmaps_backward;
  flowmap_gradient_vertex_property_type& m_flowmap_gradients_backward;
  cgal_triangulation_type                m_triangulation_backward;
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  auto pointset(forward_or_backward_tag auto const direction) -> auto& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_pointset_forward;
    } else {
      return m_pointset_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto pointset(forward_or_backward_tag auto const direction) const
      -> auto const& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_pointset_forward;
    } else {
      return m_pointset_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmaps(forward_or_backward_tag auto const direction) -> auto& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_flowmaps_forward;
    } else {
      return m_flowmaps_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmaps(forward_or_backward_tag auto const direction) const
      -> auto const& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_flowmaps_forward;
    } else {
      return m_flowmaps_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmap_gradients(forward_or_backward_tag auto const direction)
      -> auto& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_flowmap_gradients_forward;
    } else {
      return m_flowmap_gradients_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmap_gradients(forward_or_backward_tag auto const direction) const
      -> auto const& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_flowmap_gradients_forward;
    } else {
      return m_flowmap_gradients_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto triangulation(forward_or_backward_tag auto const direction) -> auto& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_triangulation_forward;
    } else {
      return m_triangulation_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto triangulation(forward_or_backward_tag auto const direction) const
      -> auto const& {
    if constexpr (is_same<decltype(direction), forward_tag>) {
      return m_triangulation_forward;
    } else {
      return m_triangulation_backward;
    }
  }
  //============================================================================
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
  //    samplers().resize(total_num_particles);
  //#pragma omp parallel for
  //    for (std::size_t i = 0; i < total_num_particles; ++i) {
  //      samplers()[i] = ps[i].sampler();
  //    }
  //  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t_end,
      arithmetic auto const             tau_step,
      std::vector<particle_type> const& initial_particles,
      std::atomic_uint64_t&             uuid_generator)
      : m_flowmaps_forward{m_pointset_forward
                               .template vertex_property<pos_type>("flowmaps")},
        m_flowmap_gradients_forward{
            m_pointset_forward.template vertex_property<gradient_type>(
                "flowmap_gradients")},

        m_flowmaps_backward{
            m_pointset_backward.template vertex_property<pos_type>("flowmap")},
        m_flowmap_gradients_backward{
            m_pointset_backward.template vertex_property<gradient_type>(
                "flowmap_gradients")} {
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
      uniform_rectilinear_grid<real_type, NumDimensions> const& g)
      : m_flowmaps_forward{m_pointset_forward
                               .template vertex_property<pos_type>("flowmaps")},
        m_flowmap_gradients_forward{
            m_pointset_forward.template vertex_property<gradient_type>(
                "flowmap_gradients")},

        m_flowmaps_backward{
            m_pointset_backward.template vertex_property<pos_type>("flowmap")},
        m_flowmap_gradients_backward{
            m_pointset_backward.template vertex_property<gradient_type>(
                "flowmap_gradients")} {
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
      std::atomic_uint64_t& uuid_generator)
      : m_flowmaps_forward{m_pointset_forward
                               .template vertex_property<pos_type>("flowmaps")},
        m_flowmap_gradients_forward{
            m_pointset_forward.template vertex_property<gradient_type>(
                "flowmap_gradients")},

        m_flowmaps_backward{
            m_pointset_backward.template vertex_property<pos_type>("flowmap")},
        m_flowmap_gradients_backward{
            m_pointset_backward.template vertex_property<gradient_type>(
                "flowmap_gradients")} {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, t_end,
         tau_step, uuid_generator);
  }
  //============================================================================
  auto num_particles() const -> std::size_t {
    // if (m_path) {
    //   auto file              = hdf5::file{*m_path};
    //   auto particles_on_disk = file.dataset<particle_type>("finished");
    //   return particles_on_disk.dataspace().current_resolution()[0];
    // } else {
    return m_pointset_forward.vertices().size();
    //}
  }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const t_end, arithmetic auto const tau_step,
            std::atomic_uint64_t& uuid_generator) {
    fill(std::forward<Flowmap>(flowmap), initial_particles, t_end, tau_step,
         uuid_generator, std::make_index_sequence<NumDimensions>{});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap, std::size_t... Is>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const t_end, arithmetic auto const tau_step,
            std::atomic_uint64_t& uuid_generator,
            std::index_sequence<Is...> /*seq*/) {
    //// if (m_path) {
    ////   particle_type::template advect<SplitBehavior>(
    ////       std::forward<Flowmap>(flowmap), tau_step, t_end,
    ///initial_particles, /       *m_path); / } else {
    // samplers().clear();
    // auto [advected_particles, simple_particles, edges] =
    //     particle_type::template advect<SplitBehavior>(
    //         std::forward<Flowmap>(flowmap), tau_step, t_end,
    //         initial_particles, uuid_generator);
    // samplers().reserve(size(advected_particles));
    // using namespace std::ranges;
    // auto get_sampler = [](auto const& p) { return p.sampler(); };
    // copy(advected_particles | views::transform(get_sampler),
    //      std::back_inserter(samplers()));
    ////}
    auto [advected_particles, simple_particles, edges] =
        particle_type::template advect<SplitBehavior>(
            std::forward<Flowmap>(flowmap), tau_step, t_end, initial_particles,
            uuid_generator);
    using namespace std::ranges;
    auto points_forward = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points_forward.reserve(size(advected_particles));
    auto points_backward = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points_backward.reserve(size(advected_particles));
    for (auto const& p : advected_particles) {
      {
        // forward
        auto v                = m_pointset_forward.insert_vertex(p.x0());
        m_flowmaps_forward[v] = p.x();
        m_flowmap_gradients_forward[v] = *inv(p.nabla_phi());
        points_forward.emplace_back(cgal_point{m_pointset_forward[v](Is)...},
                                    v);
      }
      {
        // backward
        auto v                 = m_pointset_backward.insert_vertex(p.x());
        m_flowmaps_backward[v] = p.x0();
        m_flowmap_gradients_backward[v] = p.nabla_phi();
        points_backward.emplace_back(cgal_point{m_pointset_backward[v](Is)...},
                                     v);
      }
    }

    m_triangulation_forward =
        cgal_triangulation_type{begin(points_forward), end(points_forward)};
    m_triangulation_backward =
        cgal_triangulation_type{begin(points_backward), end(points_backward)};
  }
  //----------------------------------------------------------------------------
  // template <std::size_t... VertexSeq>
  //[[nodiscard]] auto sample(pos_type const&                    p,
  //                          forward_or_backward_tag auto const direction,
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
  //        auto const p1   = sampler.sample(p, direction);
  //        if (auto const cur_dist =
  //                euclidean_length(sampler.opposite_center(direction) - p1);
  //            cur_dist < best.min_dist) {
  //          best.min_dist         = cur_dist;
  //          best.nearest_sampler = &sampler;
  //          best.p                = p1;
  //        }
  //      },
  //      execution_policy::parallel, samplers());
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
  [[nodiscard]] auto sample(
      pos_type const& q, forward_or_backward_tag auto const direction) const {
    return sample(q, direction, std::make_index_sequence<NumDimensions>{});
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  [[nodiscard]] auto sample(pos_type const&                    q,
                            forward_or_backward_tag auto const direction,
                            std::index_sequence<Is...> /*seq*/) const {
    using nnc_per_vertex_type =
        std::vector<std::pair<typename cgal_triangulation_type::Vertex_handle,
                              cgal_kernel::FT>>;

    // coordinates computation
    auto       nnc_per_vertex = nnc_per_vertex_type{};
    auto const result         = CGAL::natural_neighbor_coordinates_2(
                triangulation(direction), cgal_point{q(Is)...},
                std::back_inserter(nnc_per_vertex),
                CGAL::Identity<
            std::pair<typename cgal_triangulation_type::Vertex_handle,
                      cgal_kernel::FT>>{});
    if (!result.third) {
      return pos_type::fill(0.0 / 0.0);
    }
    auto const norm = 1 / result.second;

    auto Z0 = [&] {
      auto sum = pos_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const v        = cgal_handle->info();
        auto const lambda_i = coeff * norm;
        sum += lambda_i * flowmaps(direction)[v];
      }
      return sum;
    }();

    auto xi = [&] {
      auto numerator   = pos_type{};
      auto denominator = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const  v        = cgal_handle->info();
        auto const  lambda_i = coeff * norm;
        auto const& p_i      = pointset(direction)[v];
        auto const& g_i      = flowmap_gradients(direction)[v];
        auto const  xi_i     = [&] {
          if constexpr (tensor_rank<pos_type> == 0) {
            return flowmaps(direction)[v] + dot(g_i, q - p_i);
          } else {
            return flowmaps(direction)[v] + transposed(g_i) * (q - p_i);
          }
        }();
        auto const w = lambda_i / euclidean_distance(q, p_i);
        numerator += w * xi_i;
        denominator += w;
      }
      return numerator / denominator;
    }();

    auto alpha = [&] {
      auto numerator   = pos_type{};
      auto denominator = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const  v        = cgal_handle->info();
        auto const  lambda_i = coeff * norm;
        auto const& p_i      = pointset(direction)[v];
        auto const  w        = lambda_i / squared_euclidean_distance(q, p_i);
        numerator += lambda_i;
        denominator += w;
      }
      return numerator / denominator;
    }();

    auto beta = [&] {
      auto sum = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const  v        = cgal_handle->info();
        auto const  lambda_i = coeff * norm;
        auto const& p_i      = pointset(direction)[v];
        sum += lambda_i * squared_euclidean_distance(q, p_i);
      }
      return sum;
    }();

    return (alpha * Z0 + beta * xi) / (alpha + beta);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const&              q,
                  forward_or_backward_tag auto direction) const {
    return sample(q, direction);
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
