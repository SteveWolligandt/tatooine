#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle_flowmap_discretization {
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  template <size_t M, typename... Ts>
  struct grid_type_creator {
    using type = typename grid_type_creator<M - 1, linspace<Real>, Ts...>::type;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  struct grid_type_creator<0, Ts...> {
    using type = grid<Ts...>;
  };
  //----------------------------------------------------------------------------
  using grid_t = typename grid_type_creator<N>::type;
  using grid_vertex_property_t =
      typed_grid_vertex_property_interface<grid_t, pos_t, true>;
  //----------------------------------------------------------------------------
  template <size_t M, template <typename> typename... InterpolationKernels>
  struct grid_sampler_type_creator {
    using type =
        typename grid_sampler_type_creator<M - 1, interpolation::linear,
                                           InterpolationKernels...>::type;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename> typename... InterpolationKernels>
  struct grid_sampler_type_creator<0, InterpolationKernels...> {
    using type =
        tatooine::grid_vertex_property_sampler<grid_vertex_property_t,
                                               InterpolationKernels...>;
  };
  using grid_vertex_property_sampler_t =
      typename grid_sampler_type_creator<N>::type;

  using mesh_t = simplex_mesh<Real, N, N>;
  using mesh_vertex_property_t =
      typename mesh_t::template vertex_property_t<pos_t>;
  using mesh_vertex_property_sampler_t =
      typename mesh_t::template vertex_property_sampler_t<pos_t>;
  //============================================================================
 private:
  Real m_t0;
  Real m_t1;
  Real m_tau;

  grid_t m_initial_particle_setup;
  std::vector<autonomous_particle<Real, N>> m_particles;
  //============================================================================
 private:
  template <typename Flowmap, size_t... Is>
  autonomous_particle_flowmap_discretization(std::index_sequence<Is...> /*seq*/,
                                             Flowmap&&             flowmap,
                                             arithmetic auto const t0,
                                             arithmetic auto const tau,
                                             pos_t const& min, pos_t const& max,
                                             integral auto const... resolution)
      : m_t0{real_t(t0)},
        m_t1{real_t(t0 + tau)},
        m_tau{real_t(tau)},
        m_initial_particle_distribution{linspace<Real>{
            min(Is), max(Is), static_cast<size_t>(resolution)}...} {
    setup_initial_particle_distribution(std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  auto setup_initial_particle_distribution(std::index_sequence<Is...> /*seq*/) {
    (setup_initial_particle_distribution<Is>(), ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t I>
  auto setup_initial_particle_distribution() {
    m_initial_particle_distribution.dimension<I>().pop_front();
    auto const spacing =
        m_initial_particle_distribution.dimension<I>().spacing();
    m_initial_particle_distribution.dimension<I>().front() -= spacing / 2;
    m_initial_particle_distribution.dimension<I>().back() -= spacing / 2;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(Flowmap&&             flowmap,
                                             arithmetic auto const t0,
                                             arithmetic auto const tau,
                                             pos_t const& min, pos_t const& max,
                                             integral auto const... resolution)
      : autonomous_particle_flowmap_discretization{
            std::make_index_sequence<N>{},
            std::forward<Flowmap>(flowmap),
            t0,
            tau,
            min,
            max,
            resolution...} {
    static_assert(
        sizeof...(resolution) == N,
        "Number of resolution components does not match number of dimensions.");
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  auto sample_forward(pos_t const& x) const{

  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
