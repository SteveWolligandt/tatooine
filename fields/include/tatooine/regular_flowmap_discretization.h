#ifndef TATOOINE_REGULAR_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_REGULAR_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/interpolation.h>
#include <tatooine/particle.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Samples a flow map by advecting particles from a uniform rectilinear grid.
template <typename Real, std::size_t NumDimensions>
struct regular_flowmap_discretization {
  using real_type = Real;
  static auto constexpr num_dimensions() { return NumDimensions; }
  using vec_type = vec<Real, NumDimensions>;
  using pos_type = vec_type;
  template <std::size_t M, typename... Ts>
  struct grid_type_creator {
    using type = typename grid_type_creator<M - 1, linspace<Real>, Ts...>::type;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  struct grid_type_creator<0, Ts...> {
    using type = rectilinear_grid<Ts...>;
  };
  //----------------------------------------------------------------------------
  // using forward_grid_type = typename grid_type_creator<NumDimensions>::type;
  // using forward_grid_vertex_property_type =
  //    detail::rectilinear_grid::typed_vertex_property_interface<
  //        forward_grid_type, pos_type, true>;
  ////----------------------------------------------------------------------------
  // template <std::size_t M, template <typename> typename...
  // InterpolationKernels> struct forward_grid_sampler_type_creator {
  //   using type =
  //       typename forward_grid_sampler_type_creator<M - 1,
  //       interpolation::linear,
  //                                          InterpolationKernels...>::type;
  // };
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ///-
  // template <template <typename> typename... InterpolationKernels>
  // struct forward_grid_sampler_type_creator<0, InterpolationKernels...> {
  //   using type = tatooine::detail::rectilinear_grid::vertex_property_sampler<
  //       forward_grid_vertex_property_type, InterpolationKernels...>;
  // };
  // using forward_grid_vertex_property_sampler_type =
  //     typename forward_grid_sampler_type_creator<NumDimensions>::type;

  using forward_grid_type = pointset<Real, NumDimensions>;
  using forward_grid_vertex_property_type =
      typename forward_grid_type::template typed_vertex_property_type<pos_type>;
  using forward_grid_vertex_property_sampler_type = typename forward_grid_type::
      template natural_neighbor_coordinates_sampler_type<pos_type>;

  using backward_grid_type = pointset<Real, NumDimensions>;
  using backward_grid_vertex_property_type =
      typename backward_grid_type::template typed_vertex_property_type<
          pos_type>;
  using backward_grid_vertex_property_sampler_type =
      typename backward_grid_type::template natural_neighbor_coordinates_sampler_type<
          pos_type>;
  //============================================================================
 private:
  Real m_t0;
  Real m_t1;
  Real m_tau;

  forward_grid_type                          m_forward_grid;
  forward_grid_vertex_property_type*         m_forward_flowmap_discretization;
  std::unique_ptr<forward_grid_vertex_property_sampler_type>  m_forward_sampler;

  backward_grid_type                         m_backward_grid;
  backward_grid_vertex_property_type*        m_backward_flowmap_discretization;
  std::unique_ptr<backward_grid_vertex_property_sampler_type> m_backward_sampler;
  static constexpr auto default_execution_policy = execution_policy::parallel;
  //============================================================================
  template <typename Flowmap, typename ExecutionPolicy, integral Int, std::size_t... Is>
  regular_flowmap_discretization(std::index_sequence<Is...> seq,
                                 Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 ExecutionPolicy execution_policy,
                                 vec<Int, NumDimensions> const& resolution) : regular_flowmap_discretization{seq, std::forward<Flowmap>(flowmap), t0, tau, min, max, execution_policy, resolution(Is)...} {}
  template <typename Flowmap, typename ExecutionPolicy, std::size_t... Is>
  regular_flowmap_discretization(std::index_sequence<Is...> /*seq*/,
                                 Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 ExecutionPolicy /*execution_policy*/,
                                 integral auto const... resolution)
      : m_t0{real_type(t0)},
        m_t1{real_type(t0 + tau)},
        m_tau{real_type(tau)},

        m_forward_grid{},
        m_forward_flowmap_discretization{
            &m_forward_grid.template vertex_property<pos_type>(
                "flowmap_discretization")},
        m_forward_sampler{},

        m_backward_grid{},
        m_backward_flowmap_discretization{
            &m_backward_grid.template vertex_property<pos_type>(
                "flowmap_discretization")},
        m_backward_sampler{} {
    // fill forward pointset from rectilinear grid
    auto grid = rectilinear_grid{linspace<Real>{
        min(Is), max(Is), static_cast<std::size_t>(resolution)}...};
    m_forward_grid.vertices().reserve(grid.vertices().size());
    grid.vertices().iterate_positions(
        [this](auto const& p) { m_forward_grid.insert_vertex(p); });

    // equip positions with flow maps
    m_forward_grid.sample_to_vertex_property(
        [&](auto const& x) mutable {
          if constexpr (requires { flowmap.use_caching(false); }) {
            flowmap.use_caching(false);
          }
          return flowmap(x, m_t0, m_tau);
        },
        "flowmap_discretization"/*, execution_policy*/);

    // create forward sampler
    m_forward_sampler =
        std::make_unique<forward_grid_vertex_property_sampler_type>(
            m_forward_grid.natural_neighbor_coordinates_sampler(
                *m_forward_flowmap_discretization));

    m_backward_grid.vertices().resize(m_forward_grid.vertices().size());
    for (auto const v : m_forward_grid.vertices()) {
      m_backward_grid.vertex_at(v) = m_forward_flowmap_discretization->at(v);
      m_backward_flowmap_discretization->at(v.index()) =
          m_forward_grid.vertex_at(v);
    }
    m_backward_sampler =
        std::make_unique<backward_grid_vertex_property_sampler_type>(
            m_backward_grid.natural_neighbor_coordinates_sampler(
                *m_backward_flowmap_discretization));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <typename Flowmap, typename ExecutionPolicy>
  regular_flowmap_discretization(Flowmap&&             flowmap,
                                 ExecutionPolicy       execution_policy,
                                 arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 integral auto const... resolution)
      : regular_flowmap_discretization{std::make_index_sequence<NumDimensions>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       execution_policy,
                                       resolution...} {
    static_assert(
        sizeof...(resolution) == NumDimensions,
        "Number of resolution components does not match number of dimensions.");
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap, typename ExecutionPolicy, integral Int>
  regular_flowmap_discretization(Flowmap&&             flowmap,
                                 ExecutionPolicy       execution_policy,
                                 arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 vec<Int, NumDimensions> const& resolution)
      : regular_flowmap_discretization{std::make_index_sequence<NumDimensions>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       execution_policy,
                                       resolution} {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  regular_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 integral auto const... resolution)
      : regular_flowmap_discretization{std::make_index_sequence<NumDimensions>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       default_execution_policy,
                                       resolution...} {
    static_assert(
        sizeof...(resolution) == NumDimensions,
        "Number of resolution components does not match number of dimensions.");
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap, integral Int>
  regular_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_type const& min,
                                 pos_type const& max,
                                 vec<Int, NumDimensions> const& resolution)
      : regular_flowmap_discretization{std::make_index_sequence<NumDimensions>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       default_execution_policy,
                                       resolution} {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  /// \{
  auto grid(forward_tag const /*direction*/) const -> auto const& {
    return m_forward_grid;
  }
  //----------------------------------------------------------------------------
  auto grid(forward_tag const /*direction*/) -> auto& { return m_forward_grid; }
  //----------------------------------------------------------------------------
  auto grid(backward_tag const /*direction*/) const -> auto const& {
    return m_backward_grid;
  }
  //----------------------------------------------------------------------------
  auto grid(backward_tag const /*direction*/) -> auto& {
    return m_backward_grid;
  }
  /// \}
  //----------------------------------------------------------------------------
  /// \{
  auto sampler(forward_tag const /*direction*/) const -> auto const& {
    return *m_forward_sampler;
  }
  //----------------------------------------------------------------------------
  auto sampler(forward_tag const /*direction*/) -> auto& {
    return *m_forward_sampler;
  }
  //----------------------------------------------------------------------------
  auto sampler(backward_tag const /*direction*/) const -> auto const& {
    return *m_backward_sampler;
  }
  //----------------------------------------------------------------------------
  auto sampler(backward_tag const /*direction*/) -> auto& {
    return *m_backward_sampler;
  }
  /// \}
  //----------------------------------------------------------------------------
  /// \{
  auto flowmap(forward_tag const /*direction*/) -> auto const& {
    return *m_forward_flowmap_discretization;
  }
  //----------------------------------------------------------------------------
  auto flowmap(backward_tag const /*direction*/) -> auto const& {
    return *m_backward_flowmap_discretization;
  }
  /// \}
  //----------------------------------------------------------------------------
  /// Evaluates flow map in direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t1 - t0)
  auto sample(pos_type const&                    x,
              forward_or_backward_tag auto const direction) const {
    return sampler(direction)(x);
  }
};
//==============================================================================
using regular_flowmap_discretization2 =
    regular_flowmap_discretization<real_number, 2>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
