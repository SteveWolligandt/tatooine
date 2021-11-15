#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/unstructured_simplicial_grid.h>
#include <tatooine/staggered_flowmap_discretization.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t NumDimensions>
struct autonomous_particle_flowmap_discretization {
  using real_t              = Real;
  using vec_t               = vec<Real, NumDimensions>;
  using pos_t               = vec_t;
  using particle_t          = autonomous_particle<Real, NumDimensions>;
  using sampler_t           = autonomous_particle_sampler<Real, NumDimensions>;
  using sampler_container_t = std::vector<sampler_t>;
  using mesh_t              = unstructured_simplicial_grid<Real, NumDimensions>;
  using mesh_prop_t =
      typename mesh_t::template vertex_property_t<sampler_t const*>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
 private:
  std::optional<filesystem::path> m_path;
  std::vector<sampler_t>          m_samplers;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers0 = nullptr;
  mutable std::mutex m_centers0_mutex;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers1 = nullptr;
  mutable std::mutex m_centers1_mutex;
  //============================================================================
 public:
  explicit autonomous_particle_flowmap_discretization(
      filesystem::path const& path)
      : m_path{path} {
    auto         file              = hdf5::file{*m_path};
    auto         particles_on_disk = file.dataset<particle_t>("finished");
    size_t const total_num_particles =
        particles_on_disk.dataspace().current_resolution()[0];

    auto ps = std::vector<particle_t>(total_num_particles);
    particles_on_disk.read(ps);
    m_path = std::nullopt;

    m_samplers.resize(total_num_particles);
#pragma omp parallel for
    for (std::size_t i = 0; i < total_num_particles; ++i) {
      m_samplers[i] = ps[i].sampler();
    }
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      std::vector<autonomous_particle<Real, NumDimensions>> const& initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t0 + tau, tau_step);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::vector<autonomous_particle<Real, NumDimensions>> particles;
    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2);
        });
    //auto const small_particle_size =
    //    (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
    //               initial_particle_distribution.dimension(0).spacing()) -
    //     initial_particle_distribution.dimension(0).spacing()) /
    //    2;

    //for (size_t i = 0; i < NumDimensions; ++i) {
    //  auto const spacing = initial_particle_distribution.dimension(i).spacing();
    //  initial_particle_distribution.dimension(i).pop_front();
    //  initial_particle_distribution.dimension(i).front() -= spacing / 2;
    //  initial_particle_distribution.dimension(i).back() -= spacing / 2;
    //}
    //initial_particle_distribution.vertices().iterate_indices(
    //    [&](auto const... is) {
    //      particles.emplace_back(
    //          initial_particle_distribution.vertex_at(is...), t0,
    //          small_particle_size);
    //    });
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      filesystem::path const&                              path)
      : m_path{path} {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::vector<autonomous_particle<Real, NumDimensions>> particles;
    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2);
        });
    auto const small_particle_size =
        (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
                   initial_particle_distribution.dimension(0).spacing()) -
         initial_particle_distribution.dimension(0).spacing()) /
        2;

    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2
              // small_particle_size
          );
        });
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const tau_step,
      std::vector<autonomous_particle<Real, NumDimensions>> const&
          initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, tau, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const                           tau_step,
      autonomous_particle<Real, NumDimensions> const& initial_particle) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, tau,
         tau_step);
  }
  //============================================================================
  auto samplers() const -> auto const& { return m_samplers; }
  //----------------------------------------------------------------------------
  auto hierarchy0() const -> auto const& {
    auto l = std::lock_guard{m_centers0_mutex};
    if (m_centers0 == nullptr) {
      m_centers0 = std::make_unique<pointset<Real, NumDimensions>>();
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
      m_centers1 = std::make_unique<pointset<Real, NumDimensions>>();
      for (auto const& sa : m_samplers) {
        m_centers1->insert_vertex(sa.ellipse1().center());
      }
    }
    return m_centers1;
  }
  //----------------------------------------------------------------------------
  template <typename Tag>
  auto hierarchy_mutex(Tag) const -> auto & {
    if constexpr (is_same<tag::forward_t, Tag>) {
      return m_centers0_mutex;
    } else if constexpr (is_same<tag::backward_t, Tag>) {
      return m_centers1_mutex;
    }
  }
  //----------------------------------------------------------------------------
  template <typename Tag>
  auto hierarchy(Tag) const -> auto const& {
    if constexpr (is_same<tag::forward_t, Tag>) {
      return hierarchy0();
    } else if constexpr (is_same<tag::backward_t, Tag>) {
      return hierarchy1();
    }
  }
  //============================================================================
  auto num_particles() const -> std::size_t {
    if (m_path) {
      auto file              = hdf5::file{*m_path};
      auto particles_on_disk = file.dataset<particle_t>("finished");
      return particles_on_disk.dataspace().current_resolution()[0];
    } else {
      return size(m_samplers);
    }
  }
 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const tau, arithmetic auto const tau_step) {
    if (m_path) {
      autonomous_particle<Real, NumDimensions>::advect_with_3_splits(
          std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles,
          *m_path);
    } else {
      auto particles =
          autonomous_particle<Real, NumDimensions>::advect_with_3_splits(
              std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles);
      m_samplers.clear();
      m_samplers.reserve(size(particles));
      std::transform(
          begin(particles), end(particles),
          std::back_inserter(m_samplers),
          [](auto const& p) { return p.sampler();});
    }
  }
  //----------------------------------------------------------------------------
  template <typename Tag, size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_t const& x, Tag const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    auto         shortest_distance = std::numeric_limits<real_t>::infinity();
    //auto         file              = hdf5::file{*m_path};
    //auto         particles_on_disk = file.dataset<particle_t>("finished");
    sampler_t    nearest_sampler;
    auto const& h = hierarchy(tag);
    std::pair<std::vector<int>, std::vector<Real>> nn;
    {
      auto l = std::lock_guard{hierarchy_mutex(tag)};
      nn     = h->nearest_neighbors_radius_raw(x, 0.1);
    }
    auto const& indices            = nn.first;
    auto const& physical_distances = nn.second;
    static auto m = std::mutex{};
#pragma omp parallel for
    for (size_t j = 0; j < size(indices); ++j) {
      auto const& sampler = m_samplers[indices[j]];
      auto const  local_dist =
          sampler.ellipse(tag).local_squared_euclidean_distance_to_center(x);
      {
        auto       l                 = std::lock_guard{m};
        auto const weighted_distance = physical_distances[j] * local_dist;
        if (weighted_distance < shortest_distance) {
          shortest_distance = weighted_distance;
          nearest_sampler   = sampler;
        }
      }
    }

    if (shortest_distance < std::numeric_limits<real_t>::infinity()) {
      return nearest_sampler.sample(x, tag);
    }
    return pos_t::fill(Real(0) / Real(0));
  }

 public:
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(pos_t const& x) const {
    return sample(x, tag::forward,
                  std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_t const& x) const {
    return sample(x, tag::backward,
                  std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
};
//==============================================================================
template <size_t NumDimensions>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
using autonomous_particle_flowmap_discretization_2 =
    AutonomousParticleFlowmapDiscretization<2>;
using autonomous_particle_flowmap_discretization_3 =
    AutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
template <typename Real, size_t NumDimensions>
using staggered_autonomous_particle_flowmap_discretization =
    staggered_flowmap_discretization<
        autonomous_particle_flowmap_discretization<Real, NumDimensions>>;
//------------------------------------------------------------------------------
template <size_t NumDimensions>
using StaggeredAutonomousParticleFlowmapDiscretization =
    staggered_autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
