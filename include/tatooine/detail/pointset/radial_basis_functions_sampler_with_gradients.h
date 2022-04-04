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
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename GradientType>
struct radial_basis_functions_sampler_with_gradients
    : field<radial_basis_functions_sampler_with_gradients<
                Real, NumDimensions, ValueType, GradientType>,
            Real, NumDimensions, ValueType> {
  using this_type =
      radial_basis_functions_sampler_with_gradients<Real, NumDimensions,
                                                    ValueType, GradientType>;
  using parent_type = field<this_type, Real, NumDimensions, ValueType>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  template <typename S>
  using typed_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<S>;
  using f_property_type  = typed_vertex_property_type<ValueType>;
  using df_property_type = typed_vertex_property_type<GradientType>;
  //==========================================================================
  pointset_type const&    m_pointset;
  f_property_type const&  m_f;
  df_property_type const& m_df;
  tensor<Real>            m_coefficients = {};
  //==========================================================================
 private:
  static constexpr auto kernel_from_squared(Real const sqr_r) {
    return sqr_r * sqr_r * gcem::log(sqr_r) / 2;
  }
  static constexpr auto kernel_from_squared_x1(Real const x1, Real const x2,
                                               Real const y1, Real const y2) {
    return (2 * x1 - 2 * x2) * gcem::log(y2 * y2 - 2 * y1 * y2 + y1 * y1 +
                                         x2 * x2 - 2 * x1 * x2 + x1 * x1) -
           3 * x2 + 3 * x1;
  }
  // static constexpr auto kernel_from_squared_x2(Real const x1,
  //                                              Real const x2,
  //                                              Real const y1,
  //                                              Real const y2) {
  //   return (2 * x2 - 2 * x1) * gcem::log(y2 * y2 - 2 * y1 * y2 + y1 * y1 +
  //                                        x2 * x2 - 2 * x1 * x2 + x1 * x1) +
  //          3 * x2 - 3 * x1;
  // }
  static constexpr auto kernel_from_squared_y1(Real const x1, Real const x2,
                                               Real const y1, Real const y2) {
    return (2 * y1 - 2 * y2) * gcem::log(y2 * y2 - 2 * y1 * y2 + y1 * y1 +
                                         x2 * x2 - 2 * x1 * x2 + x1 * x1) -
           3 * y2 + 3 * y1;
  }
  // static constexpr auto kernel_from_squared_y2(Real const x1,
  //                                              Real const x2,
  //                                              Real const y1,
  //                                              Real const y2) {
  //   return (2 * y2 - 2 * y1) * gcem::log(y2 * y2 - 2 * y1 * y2 + y1 * y1 +
  //                                        x2 * x2 - 2 * x1 * x2 + x1 * x1) +
  //          3 * y2 - 3 * y1;
  // }
  static constexpr auto kernel_from_squared_x1_x2(Real const x1, Real const x2,
                                                  Real const y1,
                                                  Real const y2) {
    return -((2 * y2 * y2 - 4 * y1 * y2 + 2 * y1 * y1 + 2 * x2 * x2 -
              4 * x1 * x2 + 2 * x1 * x1) *
                 gcem::log(y2 * y2 - 2 * y1 * y2 + y1 * y1 + x2 * x2 -
                           2 * x1 * x2 + x1 * x1) +
             3 * y2 * y2 - 6 * y1 * y2 + 3 * y1 * y1 + 7 * x2 * x2 -
             14 * x1 * x2 + 7 * x1 * x1) /
           (y2 * y2 - 2 * y1 * y2 + y1 * y1 + x2 * x2 - 2 * x1 * x2 + x1 * x1);
  }
  // static constexpr auto kernel_from_squared_x1_y2(Real const x1,
  //                                                 Real const x2,
  //                                                 Real const y1,
  //                                                 Real const y2) {
  //   return -((4 * x2 - 4 * x1) * y2 + (4 * x1 - 4 * x2) * y1) /
  //          (y2 * y2 - 2 * y1 * y2 + y1 * y1 + x2 * x2 - 2 * x1 * x2 + x1 *
  //          x1);
  // }
  static constexpr auto kernel_from_squared_y1_x2(Real const x1, Real const x2,
                                                  Real const y1,
                                                  Real const y2) {
    return -((4 * x2 - 4 * x1) * y2 + (4 * x1 - 4 * x2) * y1) /
           (y2 * y2 - 2 * y1 * y2 + y1 * y1 + x2 * x2 - 2 * x1 * x2 + x1 * x1);
  }
                                                  Real const x2,
                                                  Real const y1,
                                                  Real const y2) {
                                                    return -((4 * x2 - 4 * x1) *
                                                                 y2 +
                                                             (4 * x1 - 4 * x2) *
                                                                 y1) /
                                                           (y2 * y2 -
                                                            2 * y1 * y2 +
                                                            y1 * y1 + x2 * x2 -
                                                            2 * x1 * x2 +
                                                            x1 * x1);
                                                  }
                                                  static constexpr auto
                                                  kernel_from_squared_y1_y2(
                                                      Real const x1,
                                                      Real const x2,
                                                      Real const y1,
                                                      Real const y2) {
                                                    return -((2 * y2 * y2 -
                                                              4 * y1 * y2 +
                                                              2 * y1 * y1 +
                                                              2 * x2 * x2 -
                                                              4 * x1 * x2 +
                                                              2 * x1 * x1) *
                                                                 gcem::log(
                                                                     y2 * y2 -
                                                                     2 * y1 *
                                                                         y2 +
                                                                     y1 * y1 +
                                                                     x2 * x2 -
                                                                     2 * x1 *
                                                                         x2 +
                                                                     x1 * x1) +
                                                             7 * y2 * y2 -
                                                             14 * y1 * y2 +
                                                             7 * y1 * y1 +
                                                             3 * x2 * x2 -
                                                             6 * x1 * x2 +
                                                             3 * x1 * x1) /
                                                           (y2 * y2 -
                                                            2 * y1 * y2 +
                                                            y1 * y1 + x2 * x2 -
                                                            2 * x1 * x2 +
                                                            x1 * x1);
                                                  }

                                                 public:
                                                  //==========================================================================
                                                  radial_basis_functions_sampler_with_gradients(
                                                      pointset_type const&   ps,
                                                      f_property_type const& f,
                                                      df_property_type const&
                                                          df)
                                                      : radial_basis_functions_sampler_with_gradients{
                                                            ps, f, df,
                                                            execution_policy::
                                                                sequential} {}
                                                  //----------------------------------------------------------------------------
                                                  //  radial_basis_functions_sampler_with_gradients(
                                                  //      pointset_type const&
                                                  //      ps, f_property_type
                                                  //      const& f,
                                                  //      df_property_type<Df>
                                                  //      const& df,
                                                  //      execution_policy::parallel_t
                                                  //      /*pol*/) :
                                                  //      m_pointset{ps},
                                                  //      m_f{f}, m_df{df},
                                                  //      m_coefficients{} {
                                                  //    auto const N =
                                                  //    m_pointset.vertices().size();
                                                  //    // construct lower part
                                                  //    of symmetric matrix A
                                                  //    auto A =
                                                  //    tensor<real_type>::zeros(N,
                                                  //    N);
                                                  //#pragma omp parallel for
                                                  //collapse(2)
                                                  //    for (std::size_t c = 0;
                                                  //    c < N; ++c) {
                                                  //      for (std::size_t r = c
                                                  //      + 1; r < N; ++r) {
                                                  //        A(r, c) =
                                                  //        kernel_from_squared(squared_euclidean_distance(m_pointset.vertex_at(c),
                                                  //                                                      m_pointset.vertex_at(r)));
                                                  //      }
                                                  //    }
                                                  //    m_coefficients = [N] {
                                                  //      if constexpr
                                                  //      (arithmetic<ValueType>)
                                                  //      {
                                                  //        return
                                                  //        tensor<ValueType>::zeros(N);
                                                  //      } else if constexpr
                                                  //      (static_tensor<ValueType>)
                                                  //      {
                                                  //        return
                                                  //        tensor<tensor_value_type<ValueType>>::zeros(N,
                                                  //        ValueType::num_components());
                                                  //      }
                                                  //    }();
                                                  //
                                                  //#pragma omp parallel for
                                                  //    for (std::size_t i = 0;
                                                  //    i < N; ++i) {
                                                  //      if constexpr
                                                  //      (arithmetic<ValueType>)
                                                  //      {
                                                  //        m_coefficients(i) =
                                                  //        m_f[i];
                                                  //      } else if constexpr
                                                  //      (static_tensor<ValueType>)
                                                  //      {
                                                  //        for (std::size_t j =
                                                  //        0; j <
                                                  //        ValueType::num_components();
                                                  //        ++j) {
                                                  //          m_coefficients(i,
                                                  //          j) =
                                                  //          m_f[i].data()[j];
                                                  //        }
                                                  //      }
                                                  //    }
                                                  //    // do not copy by moving
                                                  //    A and m_coefficients
                                                  //    into solver
                                                  //    m_coefficients =
                                                  //    *solve_symmetric_lapack(
                                                  //        std::move(A),
                                                  //        std::move(m_coefficients),
                                                  //        lapack::Uplo::Lower);
                                                  //  }
                                                  //----------------------------------------------------------------------------
                                                  radial_basis_functions_sampler_with_gradients(
                                                      pointset_type const&   ps,
                                                      f_property_type const& f,
                                                      df_property_type<
                                                          Df> const& df,
                                                      execution_policy::
                                                          sequential_t /*pol*/)
                                                      : m_pointset{ps},
                                                        m_f{f},
                                                        m_df{df} {
                                                    auto const N =
                                                        m_pointset.vertices()
                                                            .size();
                                                    // construct lower part of
                                                    // symmetric matrix A
                                                    auto A = tensor<real_type>::
                                                        zeros(N * 3, N * 3);
                                                    for (std::size_t c = 0;
                                                         c < N; ++c) {
                                                      for (std::size_t r =
                                                               c + 1;
                                                           r < N; ++r) {
                                                        A(r, c) = kernel_from_squared(
                                                            squared_euclidean_distance(
                                                                m_pointset
                                                                    .vertex_at(
                                                                        c),
                                                                m_pointset
                                                                    .vertex_at(
                                                                        r)));
                                                      }
                                                    }

                                                    // differentiated to x1
                                                    for (std::size_t c = 0;
                                                         c < N; ++c) {
                                                      for (std::size_t r = N;
                                                           r < N * 2; ++r) {
                                                        auto const& p1 =
                                                            m_pointset
                                                                .vertex_at(c);
                                                        auto const& p2 =
                                                            m_pointset
                                                                .vertex_at(r);
                                                        A(r, c) =
                                                            kernel_from_squared_x1(
                                                                p1.x(), p1.y(),
                                                                p2.x(), p2.y());
                                                      }
                                                    }
                                                    // differentiated to y1
                                                    for (std::size_t c = 0;
                                                         c < N; ++c) {
                                                      for (std::size_t r =
                                                               N * 2;
                                                           r < N * 3; ++r) {
                                                        auto const& p1 =
                                                            m_pointset
                                                                .vertex_at(c);
                                                        auto const& p2 =
                                                            m_pointset
                                                                .vertex_at(r);
                                                        A(r, c) =
                                                            kernel_from_squared_y1(
                                                                p1.x(), p1.y(),
                                                                p2.x(), p2.y());
                                                      }
                                                    }
                                                    //// differentiated to x2
                                                    // for (std::size_t c = 0; c
                                                    // < N; ++c) {
                                                    //   for (std::size_t r = N
                                                    //   ; r < N * 2; ++r) {
                                                    //     auto const& p1 =
                                                    //     m_pointset.vertex_at(c);
                                                    //     auto const& p2 =
                                                    //     m_pointset.vertex_at(r);
                                                    //     A(r, c)        =
                                                    //     -kernel_from_squared_x1_x2(p1.x(),
                                                    //     p1.y(), p2.x(),
                                                    //     p2.y());
                                                    //   }
                                                    // }
                                                    //  differentiated to x1 and
                                                    //  x2
                                                    for (std::size_t c = N;
                                                         c < N * 2; ++c) {
                                                      for (std::size_t r =
                                                               c + 1;
                                                           r < N * 2; ++r) {
                                                        auto const& p1 =
                                                            m_pointset
                                                                .vertex_at(c);
                                                        auto const& p2 =
                                                            m_pointset
                                                                .vertex_at(r);
                                                        A(r, c) =
                                                            -kernel_from_squared_x1_x2(
                                                                p1.x(), p1.y(),
                                                                p2.x(), p2.y());
                                                      }
                                                    }
                                                    // differentiated to y1 and
                                                    // x2
                                                    for (std::size_t c = N;
                                                         c < N * 2; ++c) {
                                                      for (std::size_t r =
                                                               N * 2;
                                                           r < N * 3; ++r) {
                                                        auto const& p1 =
                                                            m_pointset
                                                                .vertex_at(c);
                                                        auto const& p2 =
                                                            m_pointset
                                                                .vertex_at(r);
                                                        A(r, c) =
                                                            kernel_from_squared_y1_x2(
                                                                p1.x(), p1.y(),
                                                                p2.x(), p2.y());
                                                      }
                                                    }
                                                    //// differentiated to y2
                                                    // for (std::size_t c = N *
                                                    // 2; c < N * 3; ++c) {
                                                    //   for (std::size_t r = N
                                                    //   * 2; r < N * 3; ++r) {
                                                    //     auto const& p1 =
                                                    //     m_pointset.vertex_at(c);
                                                    //     auto const& p2 =
                                                    //     m_pointset.vertex_at(r);
                                                    //     A(r, c)        =
                                                    //     kernel_from_squared_y2(p1.x(),
                                                    //     p1.y(), p2.x(),
                                                    //     p2.y());
                                                    //   }
                                                    // }
                                                    //// differentiated to x1
                                                    ///and y2
                                                    // for (std::size_t c = N *
                                                    // 2; c < N * 3; ++c) {
                                                    //   for (std::size_t r = 0;
                                                    //   r < N; ++r) {
                                                    //     auto const& p1 =
                                                    //     m_pointset.vertex_at(c);
                                                    //     auto const& p2 =
                                                    //     m_pointset.vertex_at(r);
                                                    //     A(r, c)        =
                                                    //     kernel_from_squared_x1_y2(p1.x(),
                                                    //     p1.y(), p2.x(),
                                                    //     p2.y());
                                                    //   }
                                                    // }
                                                    //  differentiated to y1 and
                                                    //  y2
                                                    for (std::size_t c = N * 2;
                                                         c < N * 3; ++c) {
                                                      for (std::size_t r =
                                                               c + 1;
                                                           r < N * 3; ++r) {
                                                        auto const& p1 =
                                                            m_pointset
                                                                .vertex_at(c);
                                                        auto const& p2 =
                                                            m_pointset
                                                                .vertex_at(r);
                                                        A(r, c) =
                                                            kernel_from_squared_y1_y2(
                                                                p1.x(), p1.y(),
                                                                p2.x(), p2.y());
                                                      }
                                                    }

                                                    m_coefficients = [N] {
                                                      if constexpr (
                                                          arithmetic<
                                                              ValueType>) {
                                                        return tensor<
                                                            ValueType>::
                                                            zeros(N * 3);
                                                        //} else if constexpr
                                                        //(static_tensor<ValueType>)
                                                        //{
                                                        //  return
                                                        //  tensor<tensor_value_type<ValueType>>::zeros(N*3,
                                                        //  ValueType::num_components());
                                                      }
                                                    }();

                                                    for (std::size_t i = 0;
                                                         i < N; ++i) {
                                                      if constexpr (
                                                          arithmetic<
                                                              ValueType>) {
                                                        m_coefficients(i) =
                                                            m_f[i];
                                                        //} else if constexpr
                                                        //(static_tensor<ValueType>)
                                                        //{
                                                        //  for (std::size_t j =
                                                        //  0; j <
                                                        //  ValueType::num_components();
                                                        //  ++j) {
                                                        //    m_coefficients(i,
                                                        //    j) =
                                                        //    m_f[i].data()[j];
                                                        //  }
                                                      }
                                                    }
                                                    for (std::size_t i = N;
                                                         i < N * 2; ++i) {
                                                      if constexpr (
                                                          arithmetic<
                                                              ValueType>) {
                                                        m_coefficients(i) =
                                                            m_df[i].x();
                                                        //} else if constexpr
                                                        //(static_tensor<ValueType>)
                                                        //{
                                                        //  for (std::size_t j =
                                                        //  0; j <
                                                        //  ValueType::num_components();
                                                        //  ++j) {
                                                        //    m_coefficients(i,
                                                        //    j) =
                                                        //    m_f[i].data()[j];
                                                        //  }
                                                      }
                                                    }
                                                    // do not copy by moving A
                                                    // and m_coefficients into
                                                    // solver
                                                    m_coefficients =
                                                        *solve_symmetric_lapack(
                                                            std::move(A),
                                                            std::move(
                                                                m_coefficients),
                                                            lapack::Uplo::
                                                                Lower);
                                                  }
                                                  //--------------------------------------------------------------------------
                                                  radial_basis_functions_sampler_with_gradients(
                                                      radial_basis_functions_sampler_with_gradients const&) =
                                                      default;
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - -
                                                  radial_basis_functions_sampler_with_gradients(
                                                      radial_basis_functions_sampler_with_gradients&&) noexcept =
                                                      default;
                                                  //--------------------------------------------------------------------------
                                                  auto operator=(
                                                      radial_basis_functions_sampler_with_gradients const&)
                                                      -> radial_basis_functions_sampler_with_gradients& =
                                                      default;
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - -
                                                  auto operator=(
                                                      radial_basis_functions_sampler_with_gradients&&) noexcept
                                                      -> radial_basis_functions_sampler_with_gradients& =
                                                      default;
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - - - - - - -
                                                  // - - - - - - - - -
                                                  ~radial_basis_functions_sampler_with_gradients() =
                                                      default;
                                                  //==========================================================================
                                                  [[nodiscard]] auto evaluate(
                                                      pos_type const& q,
                                                      real_type const /*t*/)
                                                      const -> tensor_type {
                                                    auto acc = ValueType{};
                                                    for (auto const v :
                                                         m_pointset
                                                             .vertices()) {
                                                      auto const sqr_dist =
                                                          squared_euclidean_distance(
                                                              q, m_pointset[v]);
                                                      if (sqr_dist == 0) {
                                                        return m_f[v];
                                                      }
                                                      if constexpr (
                                                          arithmetic<
                                                              ValueType>) {
                                                        acc +=
                                                            m_coefficients(
                                                                v.index()) *
                                                            kernel_from_squared(
                                                                sqr_dist);
                                                      } else if constexpr (
                                                          static_tensor<
                                                              ValueType>) {
                                                        for (
                                                            std::size_t j = 0;
                                                            j <
                                                            ValueType::
                                                                num_components();
                                                            ++j) {
                                                          acc.data()[j] +=
                                                              m_coefficients(
                                                                  v.index(),
                                                                  j) *
                                                              kernel_from_squared(
                                                                  sqr_dist);
                                                        }
                                                      }
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
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
