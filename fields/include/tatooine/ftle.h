#ifndef TATOOINE_FTLE_FIELD_H
#define TATOOINE_FTLE_FIELD_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/field.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/differentiated_flowmap.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename FlowmapGradient>
struct ftle_field
    : scalarfield<ftle_field<FlowmapGradient>, typename FlowmapGradient::real_type,
            FlowmapGradient::num_dimensions()> {
  using real_type   = typename FlowmapGradient::real_type;
  using this_type   = ftle_field<FlowmapGradient>;
  using parent_type = scalarfield<this_type, real_type, FlowmapGradient::num_dimensions()>;
  using vec_type = typename FlowmapGradient::vec_type;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  static auto constexpr num_dimensions() {
    return FlowmapGradient::num_dimensions();
  }
  //============================================================================
 private:
  FlowmapGradient m_flowmap_gradient;
  real_type       m_tau;
  //============================================================================
 public:
  template <typename V, typename VReal, size_t N,
            template <typename, size_t> typename ODESolver>
  requires requires(V&& v, typename V::pos_type&& x, typename V::real_type&& t) {
    v(x, t);
  }
  constexpr ftle_field(vectorfield<V, VReal, N> const& v, arithmetic auto tau,
                       ODESolver<VReal, N> const&)
      : m_flowmap_gradient{diff(flowmap<ODESolver>(v))},
        m_tau{static_cast<real_type>(tau)} {}
  //------------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  requires requires(V&& v, typename V::pos_type&& x, typename V::real_type&& t) {
    v(x, t);
  }
  constexpr ftle_field(vectorfield<V, VReal, N> const& v, arithmetic auto tau)
      : m_flowmap_gradient{diff(flowmap(v))}, m_tau{static_cast<real_type>(tau)} {}
  //------------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  requires requires(V&& v, typename V::pos_type&& x, typename V::real_type&& t) {
    { v(x, t) }
    ->std::convertible_to<typename V::tensor_type>;
  }
  constexpr ftle_field(vectorfield<V, VReal, N> const& v, arithmetic auto tau,
                       arithmetic auto eps)
      : m_flowmap_gradient{diff(flowmap(v), eps)},
        m_tau{static_cast<real_type>(tau)} {}
  //------------------------------------------------------------------------------
  template <typename V, typename VReal, size_t N>
  requires requires(V&& v, typename V::pos_type&& x, typename V::real_type&& t) {
    v(x, t);
  }
  constexpr ftle_field(vectorfield<V, VReal, N> const& v, arithmetic auto tau,
                       vec_type const& eps)
      : m_flowmap_gradient{diff(flowmap(v), eps)},
        m_tau{static_cast<real_type>(tau)} {}
  ////------------------------------------------------------------------------------
  // template <typename FlowmapGradient_>
  // constexpr ftle_field(FlowmapGradient_&& flowmap_gradient, arithmetic auto
  // tau)
  //    : m_flowmap_gradient{std::forward<FlowmapGradient_>(flowmap_gradient)},
  //      m_tau{static_cast<real_type>(tau)} {}
  //============================================================================
  auto evaluate(pos_type const& x, real_type t) const -> tensor_type final {
    auto const g       = m_flowmap_gradient(x, t, m_tau);
    auto const eigvals = eigenvalues_sym(transposed(g) * g);
    return gcem::log(gcem::sqrt(eigvals(num_dimensions() - 1))) /
           std::abs(m_tau);
  }
  //----------------------------------------------------------------------------
  auto tau() const { return m_tau; }
  auto tau() -> auto& { return m_tau; }
  void set_tau(real_type tau) { m_tau = tau; }
  //----------------------------------------------------------------------------
  auto flowmap_gradient() const -> const auto& { return m_flowmap_gradient; }
  auto flowmap_gradient() -> auto& { return m_flowmap_gradient; }
};
//==============================================================================
template <typename V, typename Real, size_t N,
          template <typename, size_t> typename ODESolver>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto,
           ODESolver<Real, N>)
    -> ftle_field<
        numerically_differentiated_flowmap<decltype((flowmap<ODESolver>(v)))>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t N>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto)
    -> ftle_field<decltype(diff(flowmap(v)))>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t N>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto, arithmetic auto)
    -> ftle_field<decltype(diff(flowmap(v)))>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t N, typename EpsReal>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto,
           vec<EpsReal, N> const&) -> ftle_field<decltype(diff(flowmap(v)))>;
//==============================================================================
template <typename... Domains, typename Flowmap>
auto ftle(rectilinear_grid<Domains...>& grid, Flowmap&& flowmap,
          arithmetic auto const t0, arithmetic auto const tau,
          execution_policy_tag auto const exec) -> auto& {
  auto const fixed_time_phi = [&flowmap, t0, tau](auto const& x) {
    return flowmap(x, t0, tau);
  };

  auto const& phi =
      grid.sample_to_vertex_property(fixed_time_phi, "[ftle]_phi", exec);
  auto const& nabla_phi =
      grid.sample_to_vertex_property(diff(phi), "[ftle]_nabla_phi", exec);

  auto const ftle_field = [&](integral auto const... is) {
    auto const& nabla_phi_at_pos = nabla_phi(is...);
    auto const  eigvals =
        eigenvalues_sym(transposed(nabla_phi_at_pos) * nabla_phi_at_pos);
    return gcem::log(gcem::sqrt(eigvals(sizeof...(Domains) - 1))) /
           std::abs(tau);
  };
  return grid.sample_to_vertex_property(ftle_field, "ftle", exec);
}
//==============================================================================
template <typename... Domains, typename Flowmap>
auto ftle(rectilinear_grid<Domains...>& grid, Flowmap&& flowmap,
          arithmetic auto const t0, arithmetic auto const tau)
    -> decltype(auto) {
  return ftle(grid, flowmap, t0, tau, execution_policy::sequential);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
