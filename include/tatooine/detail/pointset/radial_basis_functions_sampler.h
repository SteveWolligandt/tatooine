#ifndef TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_H
#define TATOOINE_DETAIL_POINTSET_RADIAL_BASIS_FUNCTIONS_SAMPLER_H
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T>
struct radial_basis_functions_sampler
    : field<radial_basis_functions_sampler<Real, NumDimensions, T>, Real,
            NumDimensions, T> {
  static_assert(flann_available(),
                "Inverse Distance Weighting Sampler needs FLANN!");
  using this_type   = radial_basis_functions_sampler<Real, NumDimensions, T>;
  using parent_type = field<this_type, Real, NumDimensions, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type    = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<T>;
  //==========================================================================
  pointset_type const&        m_pointset;
  vertex_property_type const& m_property;
  Real                        m_epsilon;
  //==========================================================================
  radial_basis_functions_sampler(pointset_type const&        ps,
                                 vertex_property_type const& property,
                                 Real const                  epsilon)
      : m_pointset{ps}, m_property{property}, m_epsilon{epsilon} {}
  //--------------------------------------------------------------------------
  radial_basis_functions_sampler(radial_basis_functions_sampler const&) =
      default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  radial_basis_functions_sampler(radial_basis_functions_sampler&&) noexcept =
      default;
  //--------------------------------------------------------------------------
  auto operator=(radial_basis_functions_sampler const&)
      -> radial_basis_functions_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(radial_basis_functions_sampler&&) noexcept
      -> radial_basis_functions_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~radial_basis_functions_sampler() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& q, real_type const /*t*/) const
      -> tensor_type {
    auto const N = m_pointset.vertices().size();
    // construct lower part of symmetric matrix A
    auto       A = tensor<real_type>::zeros(N, N);
    for (std::size_t c = 0; c < N; ++c) {
      for (std::size_t r = c; r < N; ++r) {
        auto const sqr_dist =
            c == r ? 0
                   : squared_euclidean_distance(m_pointset.vertex_at(c),
                                                m_pointset.vertex_at(r));
        A(r, c) = std::exp(-(m_epsilon * m_epsilon * sqr_dist));
      }
    }
    auto B = [N] {
      if constexpr (arithmetic<T>) {
        return tensor<T>::zeros(N);
      } else if constexpr (static_tensor<T>) {
        return tensor<tensor_value_type<T>>::zeros(N, T::num_components());
      }
    }();

    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (arithmetic<T>) {
        B(i) = m_property[i];
      } else if constexpr (static_tensor<T>) {
        for (std::size_t j = 0; j < T::num_components(); ++j) {
          B.data()[j] = m_property[i](j);
        }
      }
    }
    // do not copy by moving A and B into solver
    B = *solve_symmetric_lapack(std::move(A), std::move(B), lapack::Uplo::Lower);
    auto acc = T{};
    for (std::size_t i = 0; i < N; ++i){
      if constexpr (arithmetic<T>) {
        acc += B(i) * m_property[i];
      } else if constexpr (static_tensor<T>) {
        for (std::size_t j = 0; j < T::num_components(); ++j) {
          acc.data()[j] += B(i, j) * m_property[i](j);
        }
      }
    }
    return acc;
  }
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
