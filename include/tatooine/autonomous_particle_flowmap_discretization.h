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
  using real_t              = Real;
  using vec_t               = vec<Real, NumDimensions>;
  using pos_t               = vec_t;
  using particle_type       = autonomous_particle<Real, NumDimensions>;
  using sampler_type        = typename particle_type::sampler_type;
  using sampler_container_t = std::vector<sampler_type>;
  using mesh_type           = unstructured_simplicial_grid<Real, NumDimensions>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
 private:
  // std::optional<filesystem::path> m_path;
  std::vector<sampler_type>                              m_samplers;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers0 = nullptr;
  mutable std::mutex                                     m_centers0_mutex;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers1 = nullptr;
  mutable std::mutex                                     m_centers1_mutex;
  //============================================================================
 public:
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
      std::vector<particle_type> const& initial_particles) {
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
              initial_particle_distribution.dimension(0).spacing() / 2);
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
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
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
  //  -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const             tau_step,
      std::vector<particle_type> const& initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, tau, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const tau,
      arithmetic auto const tau_step, particle_type const& initial_particle) {
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
  auto hierarchy_mutex(tag::forward_t /*tag*/) const -> auto& {
    return m_centers0_mutex;
  }
  //----------------------------------------------------------------------------
  auto hierarchy_mutex(tag::backward_t /*tag*/) const -> auto& {
    return m_centers1_mutex;
  }
  //----------------------------------------------------------------------------
  auto hierarchy(tag::forward_t /*tag*/) const -> auto const& {
    return hierarchy0();
  }
  //----------------------------------------------------------------------------
  auto hierarchy(tag::backward_t /*tag*/) const -> auto const& {
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

 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const tau, arithmetic auto const tau_step) {
    // if (m_path) {
    //   particle_type::template advect<SplitBehavior>(
    //       std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles,
    //       *m_path);
    // } else {
    auto [autonomous_particles, simple_particles] =
        particle_type::template advect<SplitBehavior>(
            std::forward<Flowmap>(flowmap), tau_step, tau, initial_particles);
    m_samplers.clear();
    m_samplers.reserve(size(autonomous_particles));
    using namespace std::ranges;
    auto get_sampler = [](auto const& p) { return p.sampler(); };
    copy(autonomous_particles | views::transform(get_sampler),
         std::back_inserter(m_samplers));
    //}
  }
  //----------------------------------------------------------------------------
  template <typename Tag, std::size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_t const& x, Tag const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    sampler_type nearest_sampler;
    auto const&  h  = hierarchy(tag);
    auto         nn = typename pointset<Real, NumDimensions>::vertex_handle{};
    {
      auto l = std::lock_guard{hierarchy_mutex(tag)};
      nn     = h->nearest_neighbor(x);
    }
    return m_samplers[nn.index()].sample(x, tag);
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
template <std::size_t NumDimensions>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
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
    staggered_autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
