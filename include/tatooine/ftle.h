#ifndef TATOOINE_FTLE_H
#define TATOOINE_FTLE_H

#include "flowmap.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V, template <typename, size_t> typename Integrator>
struct ftle : field<ftle<V, Integrator>, typename V::real_t, V::num_dimensions()> {
  using real_t       = typename V::real_t;
  using integrator_t = Integrator<real_t, V::num_dimensions()>;
  using this_t       = ftle<V, Integrator>;
  using parent_t     = field<this_t, real_t, V::num_dimensions()>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using flowmap_gradient_t =
      decltype(diff(flowmap{std::declval<V>(), std::declval<integrator_t>(),
                            std::declval<real_t>()}));

  //============================================================================
  private:
  flowmap_gradient_t m_flowmap_gradient;

  //============================================================================
  public:
   template <typename FieldReal, size_t N, typename TauReal>
   ftle(const field<V, FieldReal, N, N>& v, const integrator_t& integrator,
        TauReal tau)
       : m_flowmap_gradient{diff(flowmap{v, integrator, tau})} {}

   tensor_t evaluate(const pos_t& x, real_t t) const {
     auto   g       = m_flowmap_gradient(x, t);
     auto   eigvals = eigenvalues_sym(transpose(g) * g);
     real_t max_eig =
         std::max(std::abs(min(eigvals)), max(eigvals));
     //auto   [eigvals, eigvecs] = eigenvectors(transpose(g) * g);
     //real_t max_eig =
     //    std::max(std::abs(min(real(eigvals))), max(real(eigvals)));
     return std::log(std::sqrt(max_eig)) / std::abs(tau());
   }

   //---------------------------------------------------------------------------
   constexpr bool in_domain(const pos_t& x, real_t t) const {
     return m_flowmap_gradient.in_domain(x, t);
  }

  //----------------------------------------------------------------------------
  auto& flowmap_gradient() { return m_flowmap_gradient; }
  const auto& flowmap_gradient() const { return m_flowmap_gradient; }

  //----------------------------------------------------------------------------
  auto  tau() const { return m_flowmap_gradient.internal_field().tau(); }
  auto& tau() { return m_flowmap_gradient.internal_field().tau(); }
  void set_tau(real_t tau) { m_flowmap_gradient.internal_field().set_tau(tau); }

  //----------------------------------------------------------------------------
  auto  eps() const { return m_flowmap_gradient.internal_field().eps(); }
  auto& eps() { return m_flowmap_gradient.internal_field().eps(); }
  auto  eps(size_t i) const {
    return m_flowmap_gradient.internal_field().eps(i);
  }
  auto& eps(size_t i) { return m_flowmap_gradient.internal_field().eps(i); }
  void set_eps(real_t eps) { m_flowmap_gradient.internal_field().set_eps(eps); }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
