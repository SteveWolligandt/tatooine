#ifndef TATOOINE_FTLE_H
#define TATOOINE_FTLE_H

#include "flowmap.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename V>
struct ftle : field<ftle<V>, typename V::real_t, V::num_dimensions()> {
  using real_t = V::real_t;
  using this_t   = ftle<V>;
  using parent_t = field<this_t, Real, N, N>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  V m_v;
  vec<Real, n> m_eps;

  ftle(const field<V, Real, N, N>& v, Real eps = 1e-6)
      : m_v{v}, m_eps{fill{eps}} {
    pos_t offset;
    mat<Real, n, n> gradient;

    for (size_t i = 0; i < n; i++) {
      offset(i)       = m_eps(i);
      const auto fw   = m_flowmap(x + offset, t0);
      const auto bw   = m_flowmap(x - offset, t0);
      gradient.col(i) = (fw - bw) / (m_eps(i) * 2);
      offset(i) = 0;
    }

    auto   A       = transpose(gradient) * gradient;
    auto   eigvals = eigenvalues_sym(A);
    Real max_eig = std::max(std::abs(min(eigvals)), max(eigvals));
    auto   v       = std::log(std::sqrt(max_eig)) / std::abs(tau());
    if (std::isnan(v) || std::isinf(v)) {
      for (size_t i = 0; i < n; i++) {
        offset(i) = m_eps(i);
        offset(i) = 0;
      }
    }
    return v;
  }

  tensor_t evaluate(const pos_t& x, Real x) const {
  
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    m_field.in_domain(x, t);
  }
};

doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

#endif
