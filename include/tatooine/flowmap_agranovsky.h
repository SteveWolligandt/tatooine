#ifndef TATOOINE_FLOWMAP_AGRANOVSKY_H
#define TATOOINE_FLOWMAP_AGRANOVSKY_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/triangular_mesh.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct flowmap_agranovsky {
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
  using grid_t = typename grid_type_creator<N>::type;
  using prop_t = typed_multidim_property<grid_t, pos_t>;
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
    using type = tatooine::sampler<prop_t, InterpolationKernels...>;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  using sampler_t = typename grid_sampler_type_creator<N>::type;
  //============================================================================
  std::vector<Real> m_t0s;

  std::vector<grid_t>    m_flowmap_forward_grids;
  std::vector<prop_t*>   m_flowmap_forward_bases;
  std::vector<sampler_t> m_flowmap_forward_samplers;

  std::vector<triangular_mesh<Real, 2>> m_flowmap_backward_meshes;
  std::vector<
      typename triangular_mesh<Real, 2>::template vertex_property_t<pos_t>*>
      m_flowmap_backward_bases;
  std::vector<typename triangular_mesh<
      Real, 2>::template vertex_property_sampler_t<pos_t>>
      m_flowmap_backward_samplers;
  //============================================================================
 private:
  template <typename V, size_t... Is>
  flowmap_agranovsky(std::index_sequence<Is...> /*seq*/,
                     vectorfield<V, Real, N> const& v,
                     real_number auto const t0, real_number auto const tau,
                     real_number auto const delta_t, pos_t const& min,
                     pos_t const& max, integral auto const... resolution)
      : m_t0s(static_cast<size_t>(std::ceil(tau / delta_t) + 1)),
        m_flowmap_forward_grids(
            size(m_t0s) - 1,
            grid{linspace<Real>{min(Is), max(Is),
                                static_cast<size_t>(resolution)}...}),
        m_flowmap_backward_meshes(size(m_t0s) - 1) {
    for (size_t i = 0; i < size(m_t0s) - 1; ++i) {
      m_t0s[i] = t0 + delta_t * i;
    }
    m_t0s.back() = t0 + tau;
    fill(v);
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename V>
  flowmap_agranovsky(vectorfield<V, Real, N> const& v,
                     real_number auto const t0, real_number auto const tau,
                     real_number auto const delta_t, pos_t const& min,
                     pos_t const& max, integral auto const... resolution)
      : flowmap_agranovsky{std::make_index_sequence<N>{},
                           v,
                           t0,
                           tau,
                           delta_t,
                           min,
                           max,
                           resolution...} {
    static_assert(
        sizeof...(resolution) == N,
        "number of resolution components does not match number of components");
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto fill(vectorfield<V, Real, N> const& v) -> void {
    // create grids for forward integration
    for (auto& forward_base : m_flowmap_forward_grids) {
      auto& flowmap_prop =
          forward_base.template add_vertex_property<pos_t>("flowmap");
      m_flowmap_forward_bases.push_back(&flowmap_prop);
      m_flowmap_forward_samplers.push_back(
          flowmap_prop.template sampler<interpolation::linear>());
    }
    auto fm = flowmap(v);
    fm.use_caching(false);
    for (size_t i = 0; i < size(m_t0s) - 1; ++i) {
      auto const t0  = m_t0s[i];
      auto const tau = m_t0s[i + 1] - m_t0s[i];
      // create forward flowmap
      auto& forward_grid = m_flowmap_forward_grids[i];
      auto& forward_base = *m_flowmap_forward_bases[i];
      forward_grid.loop_over_vertex_indices([&](auto const... is) {
        forward_base(is...) = fm(forward_grid.vertex_at(is...), t0, tau);
      });

      // create backward flowmap
      auto& backward_mesh = m_flowmap_backward_meshes[i];
      auto& backward_base =
          backward_mesh.template add_vertex_property<pos_t>("backward_base");
      m_flowmap_backward_bases.push_back(&backward_base);
      forward_grid.loop_over_vertex_indices([&](auto const... is) {
        auto v           = backward_mesh.insert_vertex(forward_base(is...));
        backward_base[v] = forward_grid(is...);
      });
      backward_mesh.triangulate_delaunay();
      m_flowmap_backward_samplers.push_back(
          backward_mesh.vertex_property_sampler(backward_base));
    }
  }

 public:
  auto operator()(pos_t const& x, Real const t0, Real const tau) const
      -> pos_t {
    return evaluate(x, t0, tau);
  }
  auto evaluate(pos_t const& x, Real const t0, Real const tau) -> pos_t {
    return pos_t::zeros();
  }
  auto evaluate_full_forward(pos_t x) -> pos_t {
    for (size_t i = 0; i < size(m_t0s) - 1; ++i) {
      x = m_flowmap_forward_samplers[i](x(0), x(1));
    }
    return x;
  }
  auto evaluate_full_backward(pos_t x) -> pos_t {
    for (size_t j = 0; j < size(m_t0s) - 1; ++j) {
      size_t i = size(m_t0s) - j - 1;
      x        = m_flowmap_backward_samplers[i](x(0), x(1));
    }
    return x;
  }
  auto write() const {
    size_t i = 0;
    for (auto& forw_fm : m_flowmap_forward_grids) {
      forw_fm.write_vtk("forward_flowmap" + std::to_string(i++) + ".vtk");
    }
    i = 0;
    for (auto& backw_fm : m_flowmap_backward_meshes) {
      backw_fm.write_vtk("backward_flowmap" + std::to_string(i++) + ".vtk");
    }
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
