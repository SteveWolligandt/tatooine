#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
#ifndef TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_GRADIENTS_H
#define TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_GRADIENTS_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/field.h>
#include <tatooine/pointset.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
namespace rbf_gradient_kernel {
//==============================================================================
static constexpr auto kernel_from_squared(floating_point auto const sqr_r) {
  return sqr_r * sqr_r * gcem::log(sqr_r) / 2;
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared(fixed_size_real_vec<2> auto const& p1, fixed_size_real_vec<2> auto const& p2) {
  auto const x1_m_x2     = p1(0) - p2(0);
  auto const x1_m_x2_sqr = x1_m_x2 * x1_m_x2;
  auto const y1_m_y2     = p1.y() - p2.y();
  auto const y1_m_y2_sqr = y1_m_y2 * y1_m_y2;
  return (gcem::log(y1_m_y2_sqr + x1_m_x2_sqr) * (y1_m_y2_sqr + x1_m_x2_sqr) *
          (y1_m_y2_sqr + x1_m_x2_sqr)) /
         2;
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared_x1(fixed_size_real_vec<2> auto const& p1, fixed_size_real_vec<2> auto const& p2) {
  return ((2 * p1(0) - 2 * p2(0)) * p2(1) * p2(1) +
          (4 * p2(0) - 4 * p1(0)) * p1(1) * p2(1) +
          (2 * p1(0) - 2 * p2(0)) * p1(1) * p1(1) -
          2 * p2(0) * p2(0) * p2(0) + 6 * p1(0) * p2(0) * p2(0) -
          6 * p1(0) * p1(0) * p2(0) + 2 * p1(0) * p1(0) * p1(0)) *
             gcem::log(p2(1) * p2(1) - 2 * p1(1) * p2(1) + p1(1) * p1(1) +
                       p2(0) * p2(0) - 2 * p1(0) * p2(0) +
                       p1(0) * p1(0)) +
         (p1(0) - p2(0)) * p2(1) * p2(1) +
         (2 * p2(0) - 2 * p1(0)) * p1(1) * p2(1) +
         (p1(0) - p2(0)) * p1(1) * p1(1) - p2(0) * p2(0) * p2(0) +
         3 * p1(0) * p2(0) * p2(0) - 3 * p1(0) * p1(0) * p2(0) +
         p1(0) * p1(0) * p1(0);
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared_y1(fixed_size_real_vec<2> auto const& p1, fixed_size_real_vec<2> auto const& p2) {
  return (-2 * p2(1) * p2(1) * p2(1) + 6 * p1(1) * p2(1) * p2(1) +
          (-6 * p1(1) * p1(1) - 2 * p2(0) * p2(0) + 4 * p1(0) * p2(0) -
           2 * p1(0) * p1(0)) *
              p2(1) +
          2 * p1(1) * p1(1) * p1(1) +
          (2 * p2(0) * p2(0) - 4 * p1(0) * p2(0) + 2 * p1(0) * p1(0)) *
              p1(1)) *
             gcem::log(p2(1) * p2(1) - 2 * p1(1) * p2(1) + p1(1) * p1(1) +
                       p2(0) * p2(0) - 2 * p1(0) * p2(0) +
                       p1(0) * p1(0)) -
         p2(1) * p2(1) * p2(1) + 3 * p1(1) * p2(1) * p2(1) +
         (-3 * p1(1) * p1(1) - p2(0) * p2(0) + 2 * p1(0) * p2(0) -
          p1(0) * p1(0)) *
             p2(1) +
         p1(1) * p1(1) * p1(1) +
         (p2(0) * p2(0) - 2 * p1(0) * p2(0) + p1(0) * p1(0)) * p1(1);
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared_x1_x2(fixed_size_real_vec<2> auto const& p1,
                                                fixed_size_real_vec<2> auto const& p2) {
  return (-2 * p2(1) * p2(1) + 4 * p1(1) * p2(1) - 2 * p1(1) * p1(1) -
          6 * p2(0) * p2(0) + 12 * p1(0) * p2(0) - 6 * p1(0) * p1(0)) *
             gcem::log(p2(1) * p2(1) - 2 * p1(1) * p2(1) + p1(1) * p1(1) +
                       p2(0) * p2(0) - 2 * p1(0) * p2(0) +
                       p1(0) * p1(0)) -
         p2(1) * p2(1) + 2 * p1(1) * p2(1) - p1(1) * p1(1) -
         7 * p2(0) * p2(0) + 14 * p1(0) * p2(0) - 7 * p1(0) * p1(0);
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared_y1_x2(fixed_size_real_vec<2> auto const& p1,
                                                fixed_size_real_vec<2> auto const& p2) {
  return ((4 * p1(0) - 4 * p2(0)) * p2(1) +
          (4 * p2(0) - 4 * p1(0)) * p1(1)) *
             gcem::log(p2(1) * p2(1) - 2 * p1(1) * p2(1) + p1(1) * p1(1) +
                       p2(0) * p2(0) - 2 * p1(0) * p2(0) +
                       p1(0) * p1(0)) +
         (6 * p1(0) - 6 * p2(0)) * p2(1) +
         (6 * p2(0) - 6 * p1(0)) * p1(1);
}
//------------------------------------------------------------------------------
static constexpr auto kernel_from_squared_y1_y2(fixed_size_real_vec<2> auto const& p1,
                                                fixed_size_real_vec<2> auto const& p2) {
  return (-6 * p2(1) * p2(1) + 12 * p1(1) * p2(1) - 6 * p1(1) * p1(1) -
          2 * p2(0) * p2(0) + 4 * p1(0) * p2(0) - 2 * p1(0) * p1(0)) *
             gcem::log(p2(1) * p2(1) - 2 * p1(1) * p2(1) + p1(1) * p1(1) +
                       p2(0) * p2(0) - 2 * p1(0) * p2(0) +
                       p1(0) * p1(0)) -
         7 * p2(1) * p2(1) + 14 * p1(1) * p2(1) - 7 * p1(1) * p1(1) -
         p2(0) * p2(0) + 2 * p1(0) * p2(0) - p1(0) * p1(0);
}
//==============================================================================
}  // namespace rbf_gradient_kernel
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename GradientType>
struct radial_basis_functions_sampler_with_gradients;
//==============================================================================
template <floating_point Real, floating_point ValueType>
struct radial_basis_functions_sampler_with_gradients<Real, 2, ValueType,
                                                     vec<ValueType, 2>>
    : field<radial_basis_functions_sampler_with_gradients<Real, 2, ValueType,
                                                          vec<ValueType, 2>>,
            Real, 2, ValueType> {
  using value_type    = ValueType;
  using gradient_type = vec<ValueType, 2>;
  using this_type =
      radial_basis_functions_sampler_with_gradients<Real, 2, value_type,
                                                    gradient_type>;
  using parent_type = field<this_type, Real, 2, value_type>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using vertex_handle = typename pointset_type::vertex_handle;
  template <typename S>
  using typed_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<S>;
  using f_property_type  = typed_vertex_property_type<value_type>;
  using df_property_type = typed_vertex_property_type<gradient_type>;
  //==========================================================================
 private:
  pointset_type const&    m_pointset;
  f_property_type const&  m_f;
  df_property_type const& m_df;
  tensor<Real>            m_coefficients = {};
  //==========================================================================
 public:
  auto pointset() const -> auto const& { return m_pointset; }
  auto coefficients() const -> auto const& { return m_coefficients; }
  auto coefficient(vertex_handle const v) const -> auto const& {
    return m_coefficients(v.index());
  }
  auto coefficient(std::size_t const i) const -> auto const& {
    return m_coefficients(i);
  }
  auto f() const -> auto const& { return m_f; }
  auto f(vertex_handle const v) const -> auto const& { return m_f[v]; }
  auto f(std::size_t const i) const -> auto const& { return m_f[i]; }
  auto df() const -> auto const& { return m_df; }
  auto df(vertex_handle const v) const -> auto const& { return m_df[v]; }
  auto df(std::size_t const i) const -> auto const& { return m_df[i]; }

  //==========================================================================
  radial_basis_functions_sampler_with_gradients(
      radial_basis_functions_sampler_with_gradients const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  radial_basis_functions_sampler_with_gradients(
      radial_basis_functions_sampler_with_gradients&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler_with_gradients const&)
      -> radial_basis_functions_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(radial_basis_functions_sampler_with_gradients&&) noexcept
      -> radial_basis_functions_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~radial_basis_functions_sampler_with_gradients() = default;
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler_with_gradients(pointset_type const&    ps,
                                                f_property_type const&  f,
                                                df_property_type const& df)
      : radial_basis_functions_sampler_with_gradients{
            ps, f, df, execution_policy::sequential} {}
  //----------------------------------------------------------------------------
  //  radial_basis_functions_sampler_with_gradients(pointset_type const& ps,
  //                                                f_property_type const& f,
  //                                                df_property_type<Df> const&
  //                                                df,
  //                                                execution_policy::parallel_t
  //                                                /*pol*/)
  //      : m_pointset{ps}, m_f{f}, m_df{df}, m_coefficients{} {
  //    auto const N = m_pointset.vertices().size();
  //    // construct lower part
  //    of symmetric matrix A auto A = tensor<real_type>::zeros(N, N);
  //#pragma omp parallel for
  //    collapse(2) for (std::size_t c = 0; c < N; ++c) {
  //      for (std::size_t r = c + 1; r < N; ++r) {
  //        A(r, c) = rbf_gradient_kernel::kernel_from_squared(squared_euclidean_distance(
  //            m_pointset.vertex_at(c), m_pointset.vertex_at(r)));
  //      }
  //    }
  //    m_coefficients = [N] {
  //      return tensor<value_type>::zeros(N);
  //    }();
  //
  //#pragma omp parallel for
  //    for (std::size_t i = 0; i < N; ++i) {
  //      m_coefficients(i) = m_f[i];
  //    }
  //    // do not copy by moving
  //    A and m_coefficients into solver m_coefficients =
  //    *solve_symmetric_lapack(
  //        std::move(A), std::move(m_coefficients), lapack::uplo::lower);
  //  }
  //----------------------------------------------------------------------------
  radial_basis_functions_sampler_with_gradients(
      pointset_type const& ps, f_property_type const& f_,
      df_property_type const& df_, execution_policy::sequential_t /*pol*/)
      : m_pointset{ps}, m_f{f_}, m_df{df_} {
    auto const N = pointset().vertices().size();
    // construct lower part of
    // symmetric matrix A
    auto A = tensor<real_type>::zeros(N * 3 + 3, N * 3 + 3);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = rbf_gradient_kernel::kernel_from_squared(
            pointset().vertex_at(c), pointset().vertex_at(r));
      }
    }

    // differentiated to x1
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = N; r < N * 2; ++r) {
        if (c == r - N) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c);
          auto const& p2 = pointset().vertex_at(r - N);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_x1(p1, p2);
        }
      }
    }
    // differentiated to y1
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = N * 2; r < N * 3; ++r) {
        if (c == r - N * 2) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c);
          auto const& p2 = pointset().vertex_at(r - 2 * N);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1(p1, p2);
        }
      }
    }
    //  differentiated to x1 and x2
    for (std::size_t c = N; c < N * 2; ++c) {
      for (std::size_t r = c + 1; r < N * 2; ++r) {
        auto const& p1 = pointset().vertex_at(c - N);
        auto const& p2 = pointset().vertex_at(r - N);
        A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_x1_x2(p1, p2);
      }
    }
    // differentiated to y1 and x2
    for (std::size_t c = N; c < N * 2; ++c) {
      for (std::size_t r = N * 2; r < N * 3; ++r) {
        if (c - N == r - N * 2) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c - N);
          auto const& p2 = pointset().vertex_at(r - N * 2);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1_x2(p1, p2);
        }
      }
    }
    //  differentiated to y1 and y2
    for (std::size_t c = N * 2; c < N * 3; ++c) {
      for (std::size_t r = c + 1; r < N * 3; ++r) {
        auto const& p1 = pointset().vertex_at(c - N * 2);
        auto const& p2 = pointset().vertex_at(r - N * 2);
        A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1_y2(p1, p2);
      }
    }

    //  linear monomial basis constant part
    //  constant part of derivatives is already 0.
    for (std::size_t c = 0; c < N; ++c) {
      A(N * 3, c) = 1;
    }

    //  linear monomial basis linear part x
    for (std::size_t c = 0; c < N; ++c) {
      A(N * 3 + 1, c) = pointset().vertex_at(c).x();
      A(N * 3 + 2, c) = pointset().vertex_at(c).y();

      A(N * 3 + 1, c + N)     = 1;
      A(N * 3 + 2, c + 2 * N) = 1;
    }

    m_coefficients = [N] { return tensor<value_type>::zeros(N * 3 + 3); }();

    for (std::size_t i = 0; i < N; ++i) {
      m_coefficients(i) = f(i);
    }
    for (std::size_t i = N; i < N * 2; ++i) {
      m_coefficients(i) = df(i - N).x();
    }
    for (std::size_t i = N * 2; i < N * 3; ++i) {
      m_coefficients(i) = df(i - N * 2).y();
    }
    // do not copy by moving A
    // and m_coefficients into
    // solver
    m_coefficients = *solve_symmetric_lapack(
        std::move(A), std::move(m_coefficients), lapack::uplo::lower);
  }
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto       acc = value_type{};
    auto const N   = pointset().vertices().size();
    for (auto const v : pointset().vertices()) {
      auto const sqr_dist = squared_euclidean_distance(q, pointset()[v]);
      if (sqr_dist == 0) {
        return f(v);
      }
      acc += coefficient(v) * rbf_gradient_kernel::kernel_from_squared(sqr_dist);
      acc -= coefficient(v + N) * rbf_gradient_kernel::kernel_from_squared_x1(q, pointset()[v]);
      acc -= coefficient(v + N * 2) * rbf_gradient_kernel::kernel_from_squared_y1(q, pointset()[v]);
    }
    acc += coefficient(N * 3);
    acc += coefficient(N * 3 + 1) * q.x();
    acc += coefficient(N * 3 + 2) * q.y();
    return acc;
  }
};
//==============================================================================
template <floating_point Real, floating_point ValueType>
struct radial_basis_functions_sampler_with_gradients<Real, 2, vec<ValueType, 2>,
                                                     mat<ValueType, 2, 2>>
    : field<radial_basis_functions_sampler_with_gradients<Real, 2, vec<ValueType, 2>,
                                                          mat<ValueType, 2, 2>>,
            Real, 2, vec<ValueType, 2>> {
  using value_type    = vec<ValueType, 2>;
  using gradient_type = mat<ValueType, 2, 2>;
  using this_type =
      radial_basis_functions_sampler_with_gradients<Real, 2, value_type,
                                                    gradient_type>;
  using parent_type = field<this_type, Real, 2, value_type>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using vertex_handle = typename pointset_type::vertex_handle;
  template <typename S>
  using typed_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<S>;
  using f_property_type  = typed_vertex_property_type<value_type>;
  using df_property_type = typed_vertex_property_type<gradient_type>;
  //==========================================================================
 private:
  pointset_type const&    m_pointset;
  f_property_type const&  m_f;
  df_property_type const& m_df;
  tensor<Real>            m_coefficients = {};
  //==========================================================================
 public:
  auto pointset() const -> auto const& { return m_pointset; }
  auto coefficients() const -> auto const& { return m_coefficients; }
  auto coefficient(vertex_handle const v, std::size_t const j) const
      -> auto const& {
    return m_coefficients(v.index(), j);
  }
  auto coefficient(std::size_t const i, std::size_t const j) const
      -> auto const& {
    return m_coefficients(i, j);
  }
  auto f() const -> auto const& { return m_f; }
  auto f(vertex_handle const v) const -> auto const& { return m_f[v]; }
  auto f(std::size_t const i) const -> auto const& { return m_f[i]; }
  auto df() const -> auto const& { return m_df; }
  auto df(vertex_handle const v) const -> auto const& { return m_df[v]; }
  auto df(std::size_t const i) const -> auto const& { return m_df[i]; }

  //==========================================================================
  radial_basis_functions_sampler_with_gradients(
      radial_basis_functions_sampler_with_gradients const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  radial_basis_functions_sampler_with_gradients(
      radial_basis_functions_sampler_with_gradients&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler_with_gradients const&)
      -> radial_basis_functions_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(radial_basis_functions_sampler_with_gradients&&) noexcept
      -> radial_basis_functions_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~radial_basis_functions_sampler_with_gradients() = default;
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler_with_gradients(pointset_type const&    ps,
                                                f_property_type const&  f,
                                                df_property_type const& df)
      : radial_basis_functions_sampler_with_gradients{
            ps, f, df, execution_policy::sequential} {}
  //----------------------------------------------------------------------------
  //  radial_basis_functions_sampler_with_gradients(pointset_type const& ps,
  //                                                f_property_type const& f,
  //                                                df_property_type<Df> const&
  //                                                df,
  //                                                execution_policy::parallel_t
  //                                                /*pol*/)
  //      : m_pointset{ps}, m_f{f}, m_df{df}, m_coefficients{} {
  //    auto const N = m_pointset.vertices().size();
  //    // construct lower part
  //    of symmetric matrix A auto A = tensor<real_type>::zeros(N, N);
  //#pragma omp parallel for
  //    collapse(2) for (std::size_t c = 0; c < N; ++c) {
  //      for (std::size_t r = c + 1; r < N; ++r) {
  //        A(r, c) = rbf_gradient_kernel::kernel_from_squared(squared_euclidean_distance(
  //            m_pointset.vertex_at(c), m_pointset.vertex_at(r)));
  //      }
  //    }
  //    m_coefficients = [N] {
  //      return tensor<value_type>::zeros(N);
  //    }();
  //
  //#pragma omp parallel for
  //    for (std::size_t i = 0; i < N; ++i) {
  //      m_coefficients(i) = m_f[i];
  //    }
  //    // do not copy by moving
  //    A and m_coefficients into solver m_coefficients =
  //    *solve_symmetric_lapack(
  //        std::move(A), std::move(m_coefficients), lapack::uplo::lower);
  //  }
  //----------------------------------------------------------------------------
  radial_basis_functions_sampler_with_gradients(
      pointset_type const& ps, f_property_type const& f_,
      df_property_type const& df_, execution_policy::sequential_t /*pol*/)
      : m_pointset{ps}, m_f{f_}, m_df{df_} {
    auto const N = pointset().vertices().size();
    // construct lower part of
    // symmetric matrix A
    auto A = tensor<real_type>::zeros(N * 3 + 3, N * 3 + 3);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c + 1; r < N; ++r) {
        A(r, c) = rbf_gradient_kernel::kernel_from_squared(
            pointset().vertex_at(c), pointset().vertex_at(r));
      }
    }

    // differentiated to x1
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = N; r < N * 2; ++r) {
        if (c == r - N) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c);
          auto const& p2 = pointset().vertex_at(r - N);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_x1(p1, p2);
        }
      }
    }
    // differentiated to y1
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = N * 2; r < N * 3; ++r) {
        if (c == r - N * 2) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c);
          auto const& p2 = pointset().vertex_at(r - 2 * N);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1(p1, p2);
        }
      }
    }
    //  differentiated to x1 and x2
    for (std::size_t c = N; c < N * 2; ++c) {
      for (std::size_t r = c + 1; r < N * 2; ++r) {
        auto const& p1 = pointset().vertex_at(c - N);
        auto const& p2 = pointset().vertex_at(r - N);
        A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_x1_x2(p1, p2);
      }
    }
    // differentiated to y1 and x2
    for (std::size_t c = N; c < N * 2; ++c) {
      for (std::size_t r = N * 2; r < N * 3; ++r) {
        if (c - N == r - N * 2) {
          A(r, c) = 0;
        } else {
          auto const& p1 = pointset().vertex_at(c - N);
          auto const& p2 = pointset().vertex_at(r - N * 2);
          A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1_x2(p1, p2);
        }
      }
    }
    //  differentiated to y1 and y2
    for (std::size_t c = N * 2; c < N * 3; ++c) {
      for (std::size_t r = c + 1; r < N * 3; ++r) {
        auto const& p1 = pointset().vertex_at(c - N * 2);
        auto const& p2 = pointset().vertex_at(r - N * 2);
        A(r, c)        = -rbf_gradient_kernel::kernel_from_squared_y1_y2(p1, p2);
      }
    }

    //  linear monomial basis constant part
    //  constant part of derivatives is already 0.
    for (std::size_t c = 0; c < N; ++c) {
      A(N * 3, c) = 1;
    }

    //  linear monomial basis linear part x
    for (std::size_t c = 0; c < N; ++c) {
      A(N * 3 + 1, c) = pointset().vertex_at(c).x();
      A(N * 3 + 2, c) = pointset().vertex_at(c).y();

      A(N * 3 + 1, c + N)     = 1;
      A(N * 3 + 2, c + 2 * N) = 1;
    }

    m_coefficients = [N] { return tensor<ValueType>::zeros(N * 3 + 3, 2); }();

    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        m_coefficients(i, j) = f(i)(j);
      }
    }
    for (std::size_t i = N; i < N * 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        m_coefficients(i, j) = df(i - N)(j, 0);
      }
    }
    for (std::size_t i = N * 2; i < N * 3; ++i) {
      for (std::size_t j = 0; j < 2; ++j) {
        m_coefficients(i, j) = df(i - N*2)(j, 1);
      }
    }
    // do not copy by moving A
    // and m_coefficients into
    // solver
    m_coefficients = *solve_symmetric_lapack(
        std::move(A), std::move(m_coefficients), lapack::uplo::lower);
  }
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto       acc = value_type{};
    auto const N   = pointset().vertices().size();
    for (std::size_t j = 0; j < 2; ++j) {
    for (auto const v : pointset().vertices()) {
      auto const sqr_dist = squared_euclidean_distance(q, pointset()[v]);
      if (sqr_dist == 0) {
        return f(v);
      }
      acc(j) +=
          coefficient(v, j) * rbf_gradient_kernel::kernel_from_squared(sqr_dist);
      acc(j) -= coefficient(v + N, j) *
                rbf_gradient_kernel::kernel_from_squared_x1(q, pointset()[v]);
      acc(j) -= coefficient(v + N * 2, j) *
                rbf_gradient_kernel::kernel_from_squared_y1(q, pointset()[v]);
    }
    acc(j) += coefficient(N * 3, j);
    acc(j) += coefficient(N * 3 + 1, j) * q.x();
    acc(j) += coefficient(N * 3 + 2, j) * q.y();
    }
    return acc;
  }
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename GradientType>
radial_basis_functions_sampler_with_gradients(
    tatooine::pointset<Real, NumDimensions> const& ps,
    typed_vector_property<
        typename tatooine::pointset<Real, NumDimensions>::vertex_handle,
        ValueType> const&,
    typed_vector_property<
        typename tatooine::pointset<Real, NumDimensions>::vertex_handle,
        GradientType> const&)
    -> radial_basis_functions_sampler_with_gradients<Real, NumDimensions,
                                                     ValueType, GradientType>;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename GradientType>
struct differentiated_radial_basis_functions_sampler_with_gradients;
//==============================================================================
template <floating_point Real, floating_point ValueType>
struct differentiated_radial_basis_functions_sampler_with_gradients<
    Real, 2, ValueType, vec<ValueType, 2>>
    : field<differentiated_radial_basis_functions_sampler_with_gradients<
                Real, 2, ValueType, vec<ValueType, 2>>,
            Real, 2, vec<ValueType, 2>> {
  using this_type =
      differentiated_radial_basis_functions_sampler_with_gradients<
          Real, 2, ValueType, vec<ValueType, 2>>;
  using parent_type = field<this_type, Real, 2, vec<ValueType, 2>>;
  using base_type =
      radial_basis_functions_sampler_with_gradients<Real, 2, ValueType,
                                                    vec<ValueType, 2>>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using vertex_handle = typename pointset_type::vertex_handle;

 private:
  base_type const& m_base;

 public:
  auto base() const -> auto const& { return m_base; }
  auto pointset() const -> auto const& { return base().pointset(); }
  auto coefficients() const -> auto const& { return base().coefficients(); }
  auto coefficient(vertex_handle const v) const -> auto const& {
    return base().coefficient(v.index());
  }
  auto coefficient(std::size_t const i) const -> auto const& {
    return base().coefficient(i);
  }

  explicit differentiated_radial_basis_functions_sampler_with_gradients(
      base_type const& base)
      : m_base{base} {}
  auto f() const -> auto const& { return base().df(); }
  auto f(vertex_handle const v) const -> auto const& { return f()[v]; }
  auto f(std::size_t const i) const -> auto const& { return f()[i]; }
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto       acc = vec<ValueType, 2>{};
    auto const N   = pointset().vertices().size();
    for (auto const v : pointset().vertices()) {
      auto const sqr_dist = squared_euclidean_distance(q, pointset()[v]);
      if (sqr_dist == 0) {
        return f(v);
      }
      acc.x() +=
          coefficient(v) * rbf_gradient_kernel::kernel_from_squared_x1(q, pointset()[v]);
      acc.x() -= coefficient(v + N) *
                 rbf_gradient_kernel::kernel_from_squared_x1_x2(q, pointset()[v]);
      acc.x() -= coefficient(v + N * 2) *
                 rbf_gradient_kernel::kernel_from_squared_y1_x2(q, pointset()[v]);

      acc.y() +=
          coefficient(v) * rbf_gradient_kernel::kernel_from_squared_y1(q, pointset()[v]);
      acc.y() -= coefficient(v + N) *
                 rbf_gradient_kernel::kernel_from_squared_y1_x2(q, pointset()[v]);
      acc.y() -= coefficient(v + N * 2) *
                 rbf_gradient_kernel::kernel_from_squared_y1_y2(q, pointset()[v]);
    }
    acc.x() += coefficient(N * 3 + 1);
    acc.y() += coefficient(N * 3 + 2);
    return acc;
  }
};
//==============================================================================
template <floating_point Real, floating_point ValueType>
struct differentiated_radial_basis_functions_sampler_with_gradients<
    Real, 2, vec<ValueType, 2>, mat<ValueType, 2, 2>>
    : field<differentiated_radial_basis_functions_sampler_with_gradients<
                Real, 2, vec<ValueType, 2>, mat<ValueType, 2, 2>>,
            Real, 2, mat<ValueType, 2, 2>> {
  using this_type =
      differentiated_radial_basis_functions_sampler_with_gradients<
          Real, 2, vec<ValueType, 2>, mat<ValueType, 2, 2>>;
  using parent_type = field<this_type, Real, 2, mat<ValueType, 2, 2>>;
  using base_type =
      radial_basis_functions_sampler_with_gradients<Real, 2, vec<ValueType, 2>,
                                                    mat<ValueType, 2, 2>>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using vertex_handle = typename pointset_type::vertex_handle;

 private:
  base_type const& m_base;

 public:
  auto base() const -> auto const& { return m_base; }
  auto pointset() const -> auto const& { return base().pointset(); }
  auto coefficients() const -> auto const& { return base().coefficients(); }
  auto coefficient(vertex_handle const v, std::size_t const j) const
      -> auto const& {
    return base().coefficient(v.index(), j);
  }
  auto coefficient(std::size_t const i, std::size_t const j) const
      -> auto const& {
    return base().coefficient(i, j);
  }

  explicit differentiated_radial_basis_functions_sampler_with_gradients(
      base_type const& base)
      : m_base{base} {}
  auto f() const -> auto const& { return base().df(); }
  auto f(vertex_handle const v) const -> auto const& { return f()[v]; }
  auto f(std::size_t const i) const -> auto const& { return f()[i]; }
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto       acc = tensor_type{};
    auto const N   = pointset().vertices().size();
    for (auto const v : pointset().vertices()) {
      auto const sqr_dist = squared_euclidean_distance(q, pointset()[v]);
      if (sqr_dist == 0) {
        return f(v);
      }
      for (std::size_t j = 0; j < 2; ++j) {
        acc(j, 0) +=
            coefficient(v, j) *
            rbf_gradient_kernel::kernel_from_squared_x1(q, pointset()[v]);
        acc(j, 0) -=
            coefficient(v + N, j) *
            rbf_gradient_kernel::kernel_from_squared_x1_x2(q, pointset()[v]);
        acc(j, 0) -=
            coefficient(v + N * 2, j) *
            rbf_gradient_kernel::kernel_from_squared_y1_x2(pointset()[v], q);

        acc(j, 1) +=
            coefficient(v, j) *
            rbf_gradient_kernel::kernel_from_squared_y1(q, pointset()[v]);
        acc(j, 1) -=
            coefficient(v + N, j) *
            rbf_gradient_kernel::kernel_from_squared_y1_x2(q, pointset()[v]);
        acc(j, 1) -=
            coefficient(v + N * 2, j) *
            rbf_gradient_kernel::kernel_from_squared_y1_y2(q, pointset()[v]);
      }
    }
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t i = 0; i < 2; ++i) {
        acc(j, i) += coefficient(N * 3 + i + 1, j);
      }
    }
    return acc;
  }
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename GradientType>
auto diff(radial_basis_functions_sampler_with_gradients<
          Real, NumDimensions, ValueType, GradientType> const& f) {
  return differentiated_radial_basis_functions_sampler_with_gradients<
      Real, NumDimensions, ValueType, GradientType>{f};
}
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
#endif
