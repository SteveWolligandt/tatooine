#ifndef TATOOINE_FTLE_H
#define TATOOINE_FTLE_H

#include "field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename field_t, typename real_t, size_t N>
struct ftle : field<ftle<field_t, real_t, N>, real_t, N, N> {
  using this_t   = ftle<field_t, real_t, N>;
  using parent_t = field<ftle<field_t, real_t, N>, real_t, N, N>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  field_t m_field;
  vec<real_t, n> m_eps;

  ftle(const field<field_t, real_t, N, N>& f, real_t eps = 1e-6)
      : m_field{f}, m_eps{eps} {
    pos_t offset;
    mat<real_t, n, n> gradient;

    for (size_t i = 0; i < n; i++) {
      offset(i)       = m_eps(i);
      const auto fw   = m_flowmap(x + offset, t0);
      const auto bw   = m_flowmap(x - offset, t0);
      gradient.col(i) = (fw - bw) / (m_eps(i) * 2);
      offset(i) = 0;
    }

    auto   A       = transpose(gradient) * gradient;
    auto   eigvals = eigenvalues_sym(A);
    real_t max_eig = std::max(std::abs(min(eigvals)), max(eigvals));
    auto   v       = std::log(std::sqrt(max_eig)) / std::abs(tau());
    if (std::isnan(v) || std::isinf(v)) {
      for (size_t i = 0; i < n; i++) {
        offset(i) = m_eps(i);
        offset(i) = 0;
      }
    }
    return v;
  }

  tensor_t evaluate(const pos_t& x, real_t x) const {
  
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t) const {
    m_field.in_domain(x, t);
  }
};

doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

#endif
