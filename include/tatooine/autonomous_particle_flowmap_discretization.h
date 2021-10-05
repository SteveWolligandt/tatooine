#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/unstructured_simplex_grid.h>
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
  using sampler_t           = autonomous_particle_sampler<Real, NumDimensions>;
  using sampler_container_t = std::vector<sampler_t>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
 private:
  sampler_container_t m_samplers;
  //============================================================================
 public:
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::deque<autonomous_particle<Real, NumDimensions>> particles;
    for (size_t i = 0; i < NumDimensions; ++i) {
      initial_particle_distribution.dimension(i).pop_front();
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
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
    //    std::sqrt(2.0 * initial_particle_distribution.dimension(0).spacing()) -
    //    initial_particle_distribution.dimension(0).spacing();
    //
    //for (size_t i = 0; i < NumDimensions; ++i) {
    //  initial_particle_distribution.dimension(i).pop_front();
    //  auto const spacing = initial_particle_distribution.dimension(i).spacing();
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const tau_step,
      std::deque<autonomous_particle<Real, NumDimensions>> const&
          initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t1, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const                           tau_step,
      autonomous_particle<Real, NumDimensions> const& initial_particle) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::deque{initial_particle}, t1,
         tau_step);
  }
  //============================================================================
  auto samplers() const -> auto const& { return m_samplers; }
  //============================================================================
 private:
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const t1, arithmetic auto const tau_step) {
    auto const advected_particles =
        autonomous_particle<Real, NumDimensions>::advect_with_3_splits(
            std::forward<Flowmap>(flowmap), tau_step, t1, initial_particles);
    m_samplers.reserve(size(advected_particles));
    using boost::copy;
    using boost::adaptors::transformed;
    auto constexpr sampler = [](auto const& p) { return p.sampler(); };
    copy(advected_particles | transformed(sampler),
         std::back_inserter(m_samplers));
  }
  //----------------------------------------------------------------------------
  template <typename Tag, size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_t const& x, Tag const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    auto shortest_distance = std::numeric_limits<real_t>::infinity();
    sampler_t const* nearest_sampler   = nullptr;
    for (auto const& sampler : m_samplers) {
      if (auto const dist = sampler.distance(x, tag);
          dist < shortest_distance) {
        shortest_distance = dist;
        nearest_sampler   = &sampler;
      }
    }
    return nearest_sampler->sample(x, tag);
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
using staggered_autonomous_particle_flowmap_discretization_2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization_3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
