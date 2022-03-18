#ifndef TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_DERIVATIVES_H
#define TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_DERIVATIVES_H
//==============================================================================
#include <tatooine/pointset.h>
#include <tatooine/field.h>
#include <tatooine/concepts.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename F,
          typename DF, invocable<Real>     Kernel>
struct radial_basis_functions_sampler_with_derivatives
    : field<radial_basis_functions_sampler_with_derivatives<Real, NumDimensions,
                                                            F, DF, Kernel>,
            Real, NumDimensions, F> {
  using this_type =
      radial_basis_functions_sampler_with_derivatives<Real, NumDimensions, F,
                                                      DF, Kernel>;
  using parent_type = field<this_type, Real, NumDimensions, F>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  template <typename S>
  using typed_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<S>;
  using f_property_type  = typed_vertex_property_type<F>;
  using df_property_type = typed_vertex_property_type<DF>;
  //==========================================================================
  pointset_type const&    m_pointset;
  f_property_type const&  m_f;
  df_property_type const& m_df;
  Kernel                  m_kernel;
  tensor<Real>            m_coefficients;
  //==========================================================================
  radial_basis_functions_sampler_with_derivatives(
      pointset_type const& ps, f_property_type const& f,
      df_property_type<Df> const& df, convertible_to<Kernel> auto&& kernel)
      : radial_basis_functions_sampler_with_derivatives{
            ps, f, df, std::forward<decltype(kernel)>(kernel),
            execution_policy::sequential} {}
  //----------------------------------------------------------------------------
  radial_basis_functions_sampler_with_derivatives(
      pointset_type const& ps, f_property_type const& f,
      df_property_type<Df> const& df, convertible_to<Kernel> auto&& kernel,
      execution_policy::parallel_t /*pol*/)
      : m_pointset{ps},
        m_f{f},
        m_df{df},
        m_kernel{std::forward<decltype(kernel)>(kernel)},
        m_coefficients{} {
    auto const N = m_pointset.vertices().size();
    // construct lower part of symmetric matrix A
    auto A = tensor<real_type>::zeros(N, N);
#pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = m_kernel(squared_euclidean_distance(m_pointset.vertex_at(c),
                                                      m_pointset.vertex_at(r)));
      }
    }
    m_coefficients = [N] {
      if constexpr (arithmetic<F>) {
        return tensor<F>::zeros(N);
      } else if constexpr (static_tensor<F>) {
        return tensor<tensor_value_type<F>>::zeros(N, F::num_components());
      }
    }();

#pragma omp parallel for
    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (arithmetic<F>) {
        m_coefficients(i) = m_f[i];
      } else if constexpr (static_tensor<F>) {
        for (std::size_t j = 0; j < F::num_components(); ++j) {
          m_coefficients(i, j) = m_f[i].data()[j];
        }
      }
    }
    // do not copy by moving A and m_coefficients into solver
    m_coefficients = *solve_symmetric_lapack(
        std::move(A), std::move(m_coefficients), lapack::Uplo::Lower);
  }
  //----------------------------------------------------------------------------
  radial_basis_functions_sampler_with_derivatives(
      pointset_type const& ps, f_property_type const& f,
      convertible_to<Kernel> auto&& kernel,
      execution_policy::sequential_t /*pol*/)
      : m_pointset{ps},
        m_f{f},
        m_kernel{std::forward<decltype(kernel)>(kernel)} {
    auto const N = m_pointset.vertices().size();
    // construct lower part of symmetric matrix A
    auto A = tensor<real_type>::zeros(N, N);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = m_kernel(squared_euclidean_distance(m_pointset.vertex_at(c),
                                                      m_pointset.vertex_at(r)));
      }
    }
    m_coefficients = [N] {
      if constexpr (arithmetic<F>) {
        return tensor<F>::zeros(N);
      } else if constexpr (static_tensor<F>) {
        return tensor<tensor_value_type<F>>::zeros(N, F::num_components());
      }
    }();

    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (arithmetic<F>) {
        m_coefficients(i) = m_f[i];
      } else if constexpr (static_tensor<F>) {
        for (std::size_t j = 0; j < F::num_components(); ++j) {
          m_coefficients(i, j) = m_f[i].data()[j];
        }
      }
    }
    // do not copy by moving A and m_coefficients into solver
    m_coefficients = *solve_symmetric_lapack(
        std::move(A), std::move(m_coefficients), lapack::Uplo::Lower);
  }
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler_with_derivatives(
      radial_basis_functions_sampler_with_derivatives const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  radial_basis_functions_sampler_with_derivatives(
      radial_basis_functions_sampler_with_derivatives&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler_with_derivatives const&)
      -> radial_basis_functions_sampler_with_derivatives& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(radial_basis_functions_sampler_with_derivatives&&) noexcept
      -> radial_basis_functions_sampler_with_derivatives& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~radial_basis_functions_sampler_with_derivatives() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto acc = F{};
    for (auto const v : m_pointset.vertices()) {
      auto const sqr_dist = squared_euclidean_distance(q, m_pointset[v]);
      if (sqr_dist == 0) {
        return m_f[v];
      }
      if constexpr (arithmetic<F>) {
        acc += m_coefficients(v.index()) * m_kernel(sqr_dist);
      } else if constexpr (static_tensor<F>) {
        for (std::size_t j = 0; j < F::num_components(); ++j) {
          acc.data()[j] += m_coefficients(v.index(), j) * m_kernel(sqr_dist);
        }
      }
    }
    return acc;
  }
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename F,
          invocable<Real> Kernel>
radial_basis_functions_sampler_with_derivatives(
    tatooine::pointset<Real, NumDimensions> const& ps,
    typed_vector_property<
        typename tatooine::pointset<Real, NumDimensions>::vertex_handle, F>,
    Kernel&& kernel)
    -> radial_basis_functions_sampler_with_derivatives<Real, NumDimensions, F,
                                                       std::decay_t<Kernel>>;
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
