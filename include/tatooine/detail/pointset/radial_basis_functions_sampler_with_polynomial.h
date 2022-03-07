#ifndef TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_POLYNOMIAL_H
#define TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_POLYNOMIAL_H
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> Kernel>
struct radial_basis_functions_sampler_with_polynomial;
//==============================================================================
template <floating_point Real, typename T,
          invocable<Real> Kernel>
struct radial_basis_functions_sampler_with_polynomial<Real, 2, T, Kernel>
    : field<radial_basis_functions_sampler_with_polynomial<Real, 2,
                                                           T, Kernel>,
            Real, 2, T> {
  using this_type =
      radial_basis_functions_sampler_with_polynomial<Real, 2, T,
                                                     Kernel>;
  using parent_type = field<this_type, Real, 2, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<T>;
  //==========================================================================
  pointset_type const&        m_pointset;
  vertex_property_type const& m_property;
  Kernel                      m_kernel;
  tensor<Real>                m_weights_and_coeffs;
  //==========================================================================
  radial_basis_functions_sampler_with_polynomial(
      pointset_type const& ps, vertex_property_type const& property,
      convertible_to<Kernel> auto&& kernel)
      : m_pointset{ps},
        m_property{property},
        m_kernel{std::forward<decltype(kernel)>(kernel)} {
    auto const N = m_pointset.vertices().size();
    // construct lower part of symmetric matrix A
    auto A = tensor<real_type>::zeros(N + 6, N + 6);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = m_kernel(squared_euclidean_distance(m_pointset.vertex_at(c),
                                                      m_pointset.vertex_at(r)));
      }
    }
    // construct polynomial requirements
    for (std::size_t c = 0; c < N; ++c) {
      auto const& p = m_pointset.vertex_at(c);
      // constant part
      A(N, c)       = 1;
      
      // linear part
      A(N + 1, c)   = p.x();
      A(N + 2, c)   = p.y();

      // quadratic part
      A(N + 3, c)   = p.x() * p.x();
      A(N + 4, c)   = p.x() * p.y();
      A(N + 5, c)   = p.y() * p.y();
    }

    m_weights_and_coeffs = [N] {
      if constexpr (arithmetic<T>) {
        return tensor<T>::zeros(N+6);
      } else if constexpr (static_tensor<T>) {
        return tensor<tensor_value_type<T>>::zeros(N+6, T::num_components());
      }
    }();

    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (arithmetic<T>) {
        m_weights_and_coeffs(i) = m_property[i];
      } else if constexpr (static_tensor<T>) {
        for (std::size_t j = 0; j < T::num_components(); ++j) {
          m_weights_and_coeffs.data()[j] = m_property[i](j);
        }
      }
    }
    // do not copy by moving A and m_weights_and_coeffs into solver
    m_weights_and_coeffs = *solve_symmetric_lapack(std::move(A), std::move(m_weights_and_coeffs),
                                        lapack::Uplo::Lower);
  }
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler_with_polynomial(
      radial_basis_functions_sampler_with_polynomial const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  radial_basis_functions_sampler_with_polynomial(
      radial_basis_functions_sampler_with_polynomial&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler_with_polynomial const&)
      -> radial_basis_functions_sampler_with_polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(radial_basis_functions_sampler_with_polynomial&&) noexcept
      -> radial_basis_functions_sampler_with_polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~radial_basis_functions_sampler_with_polynomial() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto const N   = m_pointset.vertices().size();
    auto acc = T{};
    for (auto const v : m_pointset.vertices()) {
      if constexpr (arithmetic<T>) {
        acc += m_weights_and_coeffs(v.index()) *
                   m_kernel(squared_euclidean_distance(q, m_pointset[v]));
      } else if constexpr (static_tensor<T>) {
        // for (std::size_t j = 0; j < T::num_components(); ++j) {
        //   acc.data()[j] +=
        //       m_weights_and_coeffs(v.index(), j) *
        //       m_kernel(squared_euclidean_distance(q, m_pointset[v]))(j);
        // }
      }
    }
    if constexpr (arithmetic<T>) {
      acc += m_weights_and_coeffs(N) +
             m_weights_and_coeffs(N + 1) * q.x() +
             m_weights_and_coeffs(N + 2) * q.y() +
             m_weights_and_coeffs(N + 3) * q.x() * q.x() +
             m_weights_and_coeffs(N + 4) * q.x() * q.y() +
             m_weights_and_coeffs(N + 5) * q.y() * q.y();
    } else if constexpr (static_tensor<T>) {
    }
    return acc;
  }
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> Kernel>
radial_basis_functions_sampler_with_polynomial(
    tatooine::pointset<Real, NumDimensions> const& ps,
    typed_vector_property<
        typename tatooine::pointset<Real, NumDimensions>::vertex_handle, T>,
    Kernel&& kernel)
    -> radial_basis_functions_sampler_with_polynomial<Real, NumDimensions, T,
                                                      std::decay_t<Kernel>>;
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
