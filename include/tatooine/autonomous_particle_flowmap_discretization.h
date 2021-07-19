#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle_flowmap_discretization {
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  //============================================================================
 private:
  std::vector<autonomous_particle_sampler<Real, N>> m_samplers;
  //============================================================================
 public:
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(Flowmap&&             flowmap,
                                             arithmetic auto const t0,
                                             arithmetic auto const t1,
                                             arithmetic auto const tau_step,
                                             uniform_grid<Real, N> const& g) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto  initial_particle_distribution = g.copy_without_properties();
    for (size_t i = 0; i < N; ++i) {
      initial_particle_distribution.dimension(i).pop_front();
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    std::deque<autonomous_particle<Real, N>> particles;
    particles.reserve(initial_particle_distribution.vertices().size());
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing());
        });
    fill(std::forward<Flowmap>(flowmap), particles, t1, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const                            tau_step,
      std::deque<autonomous_particle<Real, N>> const& initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t1, tau_step);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&&                                       flowmap,
            std::deque<autonomous_particle<Real, N>> const& initial_particles,
            arithmetic auto const t1, arithmetic auto const tau_step) {
    auto const advected_particles =
        autonomous_particle<Real, N>::advect_with_3_splits(
            std::forward<Flowmap>(flowmap), tau_step, t1, initial_particles);
    m_samplers.reserve(size(advected_particles));
    boost::copy(
        advected_particles | boost::adaptors::transformed(
                                 [](auto const& p) { return p.sampler(); }),
        std::back_inserter(m_samplers));
  }
  //----------------------------------------------------------------------------
  auto sample_forward(pos_t const& x) const {
    for (auto const& sampler : m_samplers) {
      if (sampler.is_inside0(x)) {
        return sampler.sample_forward(x);
      }
    }
    throw std::runtime_error{"out of domain"};
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_t const& x) const {
    for (auto const& sampler : m_samplers) {
      if (sampler.is_inside0(x)) {
        return sampler.sample_backward(x);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
