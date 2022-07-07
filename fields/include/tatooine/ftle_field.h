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
  using vec_t    = typename FlowmapGradient::vec_t;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
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
                       vec_t const& eps)
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
    auto const max_eig = max(eigvals);
    return std::log(std::sqrt(max_eig)) / std::abs(m_tau);
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_type const& /*x*/, real_type /*t*/) const
      -> bool final {
    return true;
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
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto, ODESolver<Real, N>)
    -> ftle_field<numerically_differentiated_flowmap<
        decltype((flowmap<ODESolver>(v)))>>;
template <typename V, typename Real, size_t N>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto)
    -> ftle_field<decltype(diff(flowmap(v)))>;
template <typename V, typename Real, size_t N>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto, arithmetic auto)
    -> ftle_field<decltype(diff(flowmap(v)))>;
template <typename V, typename Real, size_t N, typename EpsReal>
ftle_field(vectorfield<V, Real, N> const& v, arithmetic auto,
           vec<EpsReal, N> const&) -> ftle_field<decltype(diff(flowmap(v)))>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
