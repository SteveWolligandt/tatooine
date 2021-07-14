#ifndef TATOOINE_NAIVE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_NAIVE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/particle.h>
#include <tatooine/triangular_mesh.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct naive_flowmap_discretization {
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using grid_t      = typename grid_type_creator<N>::type;
  using grid_prop_t = typed_grid_vertex_property_interface<grid_t, pos_t, true>;
  //============================================================================
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
        tatooine::grid_vertex_property_sampler<grid_prop_t,
                                               InterpolationKernels...>;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using grid_vertex_property_sampler_t =
      typename grid_sampler_type_creator<N>::type;
  using mesh_t = triangular_mesh<Real, 2>;
  using mesh_prop_t =
      typename triangular_mesh<Real, 2>::template vertex_property_t<pos_t>;
  using mesh_sampler_t =
      typename triangular_mesh<Real,
                               2>::template vertex_property_sampler_t<pos_t>;
  //============================================================================
 private:
  Real m_t0;
  Real m_tau;

  grid_t m_forward_grid;
  mesh_t m_backward_mesh;
  //============================================================================
 private:
  template <typename Flowmap, size_t... Is>
  naive_flowmap_discretization(std::index_sequence<Is...> /*seq*/, Flowmap&& flowmap,
                        arithmetic auto const t0, arithmetic auto const tau,
                        pos_t const& min, pos_t const& max,
                        integral auto const... resolution)
      : m_t0{real_t(t0)},
        m_tau{real_t(tau)},
        m_forward_grid{linspace<Real>{min(Is), max(Is),
                                      static_cast<size_t>(resolution)}...} {
    fill(std::forward<Flowmap>(flowmap));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <typename Flowmap>
  naive_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                        arithmetic auto const tau, pos_t const& min,
                        pos_t const& max, integral auto const... resolution)
      : naive_flowmap_discretization{std::make_index_sequence<N>{},
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
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap) -> void {
    auto& discretized_flowmap =
        m_forward_grid.template insert_vertex_property<pos_t>("flowmap");
    m_forward_grid.vertices().iterate_indices([&](auto const... is) {
      discretized_flowmap(is...) =
          flowmap(m_forward_grid.vertex_at(is...), m_t0, m_tau);
    });

    // create backward flowmap
    m_backward_mesh = mesh_t{m_forward_grid};
    auto& backward_base =
        m_backward_mesh.template vertex_property<pos_t>("flowmap");
    for (auto v : m_backward_mesh.vertices()) {
      for (size_t i = 0; i < N; ++i) {
        std::swap(m_backward_mesh[v](i), backward_base[v](i));
      }
    }
    m_backward_mesh.build_delaunay_mesh();
    m_backward_mesh.build_hierarchy();
  }
  //----------------------------------------------------------------------------
  auto forward_mesh() const -> auto const& { return m_forward_grid; }
  auto forward_mesh() -> auto& { return m_forward_grid; }
  auto backward_mesh() const -> auto const& { return m_backward_mesh; }
  auto backward_mesh() -> auto& { return m_backward_mesh; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
