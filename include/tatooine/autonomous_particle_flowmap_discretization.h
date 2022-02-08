#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/staggered_flowmap_discretization.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/unstructured_simplicial_grid.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename SplitBehavior = autonomous_particle<
              Real, NumDimensions>::split_behaviors::three_splits>
struct autonomous_particle_flowmap_discretization {
  using real_type              = Real;
  using vec_t               = vec<Real, NumDimensions>;
  using pos_type               = vec_t;
  using particle_type       = autonomous_particle<Real, NumDimensions>;
  using sampler_type        = typename particle_type::sampler_type;
  using sampler_container_t = std::vector<sampler_type>;
  using mesh_type           = unstructured_simplicial_grid<Real, NumDimensions>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  // std::optional<filesystem::path> m_path;
  std::vector<sampler_type>                              m_samplers = {};
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers0 = nullptr;
  mutable std::mutex                                     m_centers0_mutex = {};
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers1 = nullptr;
  mutable std::mutex                                     m_centers1_mutex = {};
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
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const             tau_step,
      std::vector<particle_type> const& initial_particles,
      std::atomic_uint64_t&             uuid_generator) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t0 + tau, tau_step,
         uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    auto uuid_generator = std::atomic_uint64_t{};
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::vector<particle_type> particles;
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
  //     uniform_rectilinear_grid<Real, NumDimensions> const& g,
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
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const             tau_step,
      std::vector<particle_type> const& initial_particles,
      std::atomic_uint64_t&             uuid_generator) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, tau, tau_step,
         uuid_generator);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const tau_step, particle_type const& initial_particle,
      std::atomic_uint64_t& uuid_generator) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, tau,
         tau_step, uuid_generator);
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
            arithmetic auto const tau, arithmetic auto const tau_step,
            std::atomic_uint64_t& uuid_generator) {
    // if (m_path) {
    //   particle_type::template advect<SplitBehavior>(
    //       std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles,
    //       *m_path);
    // } else {
    auto [autonomous_particles, simple_particles, edges] =
        particle_type::template advect<SplitBehavior>(
            std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles,
            uuid_generator);
    m_samplers.clear();
    m_samplers.reserve(size(autonomous_particles));
    using namespace std::ranges;
    auto get_sampler = [](auto const& p) { return p.sampler(); };
    copy(autonomous_particles | views::transform(get_sampler),
         std::back_inserter(m_samplers));
    //}
  }
  //----------------------------------------------------------------------------
  template <std::size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_type const&                       p0,
                            forward_or_backward_tag auto const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    auto nearest_sampler_it = end(m_samplers);
    auto min_dist           = std::numeric_limits<Real>::max();
    for (auto sampler_it = begin(m_samplers); sampler_it != end(m_samplers);
         ++sampler_it) {
      auto const p1 = sampler_it->sample(p0, tag);
      if (auto const cur_dist =
              euclidean_length(sampler_it->opposite_center(tag) - p1);
          cur_dist < min_dist) {
        min_dist           = cur_dist;
        nearest_sampler_it = sampler_it;
      }
    }
    return nearest_sampler_it->sample(p0, tag);
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample(pos_type const&                       x,
                            forward_or_backward_tag auto const tag) const {
    return sample(x, tag, std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(pos_type const& x) const {
    return sample(x, forward);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_type const& x) const { return sample(x, backward); }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const& x, forward_or_backward_tag auto tag) const {
    return sample(x, tag);
  }
};
//==============================================================================
template <std::size_t NumDimensions>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_type, NumDimensions>;
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
    staggered_autonomous_particle_flowmap_discretization<real_type, NumDimensions>;
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
