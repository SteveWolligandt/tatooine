#ifndef TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_POLYNOMIAL_H
#define TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_WITH_POLYNOMIAL_H
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> Kernel>
struct radial_basis_functions_sampler_with_polynomial
    : field<radial_basis_functions_sampler_with_polynomial<Real, NumDimensions,
                                                           T, Kernel>,
            Real, NumDimensions, T> {
  using this_type =
      radial_basis_functions_sampler_with_polynomial<Real, NumDimensions, T,
                                                     Kernel>;
  using parent_type = field<this_type, Real, NumDimensions, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<T>;
  static auto constexpr num_dimensions() { return NumDimensions; }
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
    auto A =
        tensor<real_type>::zeros(N + NumDimensions + 1, N + NumDimensions + 1);
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
      A(N, c) = 1;

      // linear part
      for (std::size_t i = 0; i < NumDimensions; ++i) {
        A(N + i + 1, c) = p(i);
      }
    }

    m_weights_and_coeffs = [N] {
      if constexpr (arithmetic<T>) {
        return tensor<T>::zeros(N + NumDimensions + 1);
      } else if constexpr (static_tensor<T>) {
        return tensor<tensor_value_type<T>>::zeros(N + NumDimensions + 1,
                                                   T::num_components());
      }
    }();

    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (arithmetic<T>) {
        m_weights_and_coeffs(i) = m_property[i];
      } else if constexpr (static_tensor<T>) {
        for (std::size_t j = 0; j < T::num_components(); ++j) {
          m_weights_and_coeffs(i, j) = m_property[i].data()[j];
        }
      }
    }
    // do not copy by moving A and m_weights_and_coeffs into solver
    m_weights_and_coeffs = *solve_symmetric_lapack(
        std::move(A), std::move(m_weights_and_coeffs), lapack::Uplo::Lower);
  }
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler_with_polynomial(
      radial_basis_functions_sampler_with_polynomial const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  radial_basis_functions_sampler_with_polynomial(
      radial_basis_functions_sampler_with_polynomial&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler_with_polynomial const&)
      -> radial_basis_functions_sampler_with_polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  auto operator=(radial_basis_functions_sampler_with_polynomial&&) noexcept
      -> radial_basis_functions_sampler_with_polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  ~radial_basis_functions_sampler_with_polynomial() = default;

  //------------------------------------------------------------------------------
  static auto for_loop(auto&& iteration, std::size_t const degree,
                       std::size_t const dim, std::vector<std::size_t>& status)
      -> void {
    if (dim == degree + 1) {
      return iteration(status);
    } else {
      for (; status[dim] < (dim == 0 ? num_dimensions() : status[dim - 1]);
           ++status[dim]) {
        for_loop(std::forward<decltype(iteration)>(iteration), begin, end,
                 status, dim + 1);
      }
      status[dim] = 0;
    }
  }
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto const N   = m_pointset.vertices().size();
    auto       acc = T{};
    // radial basis functions
    for (auto const v : m_pointset.vertices()) {
      if constexpr (arithmetic<T>) {
        acc += m_weights_and_coeffs(v.index()) *
               m_kernel(squared_euclidean_distance(q, m_pointset[v]));
      } else if constexpr (static_tensor<T>) {
        for (std::size_t j = 0; j < T::num_components(); ++j) {
          acc.data()[j] +=
              m_weights_and_coeffs(v.index(), j) *
              m_kernel(squared_euclidean_distance(q, m_pointset[v]));
        }
      }
    }
    // polynomial part
    if constexpr (arithmetic<T>) {
      acc += m_weights_and_coeffs(N);
      for (std::size_t k = 0; k < NumDimensions; ++k) {
        acc += m_weights_and_coeffs(N + 1 + k) * q(k);
      }
    } else if constexpr (static_tensor<T>) {
      for (std::size_t j = 0; j < T::num_components(); ++j) {
        acc.data()[j] += m_weights_and_coeffs(N, j);
        for (std::size_t k = 0; k < NumDimensions; ++k) {
          acc.data()[j] += m_weights_and_coeffs(N + 1 + k, j) * q(k);
        }
      }
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
        -> radial_basis_functions_sampler_with_polynomial<
            Real, NumDimensions, T, std::decay_t<Kernel>>;
    //==============================================================================
    }  // namespace tatooine::detail::pointset
    //==============================================================================
#endif
