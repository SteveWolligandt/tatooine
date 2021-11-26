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
template <typename Real, size_t N>
struct regular_flowmap_discretization {
  using real_t = Real;
  static auto constexpr num_dimensions() { return N; }
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  template <size_t M, typename... Ts>
  struct grid_type_creator {
    using type = typename grid_type_creator<M - 1, linspace<Real>, Ts...>::type;
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  struct grid_type_creator<0, Ts...> {
    using type = rectilinear_grid<Ts...>;
  };
  //----------------------------------------------------------------------------
  using forward_grid_t = typename grid_type_creator<N>::type;
  using grid_vertex_property_t =
      typed_rectilinear_grid_vertex_property_interface<forward_grid_t, pos_t, true>;
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
    using type = tatooine::rectilinear_grid_vertex_property_sampler<
        grid_vertex_property_t, InterpolationKernels...>;
  };
  using grid_vertex_property_sampler_t =
      typename grid_sampler_type_creator<N>::type;

  using backward_grid_t = unstructured_simplicial_grid<Real, N, N>;
  using mesh_vertex_property_t =
      typename backward_grid_t::template vertex_property_t<pos_t>;
  using mesh_vertex_property_sampler_t =
      typename backward_grid_t::template vertex_property_sampler_t<pos_t>;
  //============================================================================
 private:
  Real m_t0;
  Real m_t1;
  Real m_tau;

  forward_grid_t                 m_forward_grid;
  grid_vertex_property_t*        m_forward_discretization;
  grid_vertex_property_sampler_t m_forward_sampler;

  backward_grid_t                m_backward_grid;
  mesh_vertex_property_t*        m_backward_discretization;
  mesh_vertex_property_sampler_t m_backward_sampler;
  static constexpr auto default_execution_policy = execution_policy::parallel;
  //============================================================================
 private:
  template <typename Flowmap, typename ExecutionPolicy, size_t... Is>
  regular_flowmap_discretization(std::index_sequence<Is...> /*seq*/,
                                 Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_t const& min,
                                 pos_t const&    max,
                                 ExecutionPolicy execution_policy,
                                 integral auto const... resolution)
      : m_t0{real_t(t0)},
        m_t1{real_t(t0 + tau)},
        m_tau{real_t(tau)},
        m_forward_grid{linspace<Real>{min(Is), max(Is),
                                      static_cast<size_t>(resolution)}...},
        m_forward_discretization{
            &m_forward_grid.template vertex_property<pos_t>(
                "forward_discretization")},
        m_forward_sampler{m_forward_discretization->linear_sampler()},
        m_backward_grid{m_forward_grid},
        m_backward_discretization{
            &m_backward_grid.template vertex_property<pos_t>(
                "backward_discretization")},
        m_backward_sampler{
            m_backward_grid.sampler(*m_backward_discretization)} {
    fill(std::forward<Flowmap>(flowmap), execution_policy);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <typename Flowmap, typename ExecutionPolicy>
  regular_flowmap_discretization(Flowmap&&             flowmap,
                                 ExecutionPolicy       execution_policy,
                                 arithmetic auto const t0,
                                 arithmetic auto const tau, pos_t const& min,
                                 pos_t const& max,
                                 integral auto const... resolution)
      : regular_flowmap_discretization{std::make_index_sequence<N>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       execution_policy,
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
  regular_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                 arithmetic auto const tau, pos_t const& min,
                                 pos_t const& max,
                                 integral auto const... resolution)
      : regular_flowmap_discretization{std::make_index_sequence<N>{},
                                       std::forward<Flowmap>(flowmap),
                                       t0,
                                       tau,
                                       min,
                                       max,
                                       default_execution_policy,
                                       resolution...} {
    static_assert(
        sizeof...(resolution) == N,
        "Number of resolution components does not match number of dimensions.");
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap, typename ExecutionPolicy>
  auto fill(Flowmap&& flowmap, ExecutionPolicy execution_policy) -> void {
    m_forward_grid.vertices().iterate_indices(
        [&](auto const... is) {
          auto flowmap2 = flowmap;
          flowmap2.use_caching(false);
          m_forward_discretization->at(is...) =
              flowmap2(m_forward_grid.vertex_at(is...), m_t0, m_tau);
        },
        execution_policy);

    for (auto v : m_forward_grid.vertices()) {
      m_backward_discretization->at(typename backward_grid_t::vertex_handle{
          v.plain_index()}) = m_forward_discretization->at(v);
    }
    for (auto v : m_backward_grid.vertices()) {
      for (size_t i = 0; i < N; ++i) {
        std::swap(m_backward_discretization->at(v)(i), m_backward_grid[v](i));
      }
    }
    m_backward_grid.build_delaunay_mesh();
    m_backward_grid.build_hierarchy();
  }
  //----------------------------------------------------------------------------
  auto forward_grid() const -> auto const& { return m_forward_grid; }
  auto forward_grid() -> auto& { return m_forward_grid; }
  auto backward_grid() const -> auto const& { return m_backward_grid; }
  auto backward_grid() -> auto& { return m_backward_grid; }
  //----------------------------------------------------------------------------
  auto forward_sampler() const -> auto const& { return *m_forward_sampler; }
  auto forward_sampler() -> auto& { return *m_forward_sampler; }
  auto backward_sampler() const -> auto const& { return *m_backward_sampler; }
  auto backward_sampler() -> auto& { return *m_backward_sampler; }
  //----------------------------------------------------------------------------
  /// evaluates flow map
  auto operator()(pos_t const& x, real_t const t, real_t const tau) const {
    if (t + tau < m_t0 || t + tau > m_t1) {
      throw std::runtime_error{"Flow map out of domain!"};
    }
    if (t == m_t0 && tau > 0) {
      return sample_forward(x, tau);
    } else if (t == m_t1 && tau < 0) {
      return sample_backward(x, tau);
    }
    return sample_forward(find_position_at_t0(x, t), t - m_t0 + tau);
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t1 - t0)
  auto sample_forward(pos_t const& x) const { return m_forward_sampler(x); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Evaluates flow map in forward direction at time t0 with an advection time
  /// of tau.
  /// \param x position
  /// \param tau advection time (needs to be positive)
  /// \returns phi(x, t0, tau)
  auto sample_forward(pos_t const& x, real_t const tau) const {
    assert(tau > 0);
    return interpolation::linear{
        x, m_forward_sampler(x)}((tau - m_t0) / (m_t1 - m_t0));
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in backward direction at time t1 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t0 - t1)
  auto sample_backward(pos_t const& x) const { return m_backward_sampler(x); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Evaluates flow map in backward direction at time t1 with an advection time
  /// of tau.
  /// \param x position
  /// \param tau advection time (needs to be negative)
  /// \returns phi(x, t1, tau)
  auto sample_backward(pos_t const& x, real_t const tau) const {
    assert(tau < 0);
    return interpolation::linear{
        x, m_backward_sampler(x)}((-tau - m_t0) / (m_t1 - m_t0));
  }
  //----------------------------------------------------------------------------
  /// TODO Implement!
  auto find_position_at_t0(pos_t const& /*x*/, real_t const /*t*/) const {
    return pos_t::fill(real_t(0) / real_t(0));
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
