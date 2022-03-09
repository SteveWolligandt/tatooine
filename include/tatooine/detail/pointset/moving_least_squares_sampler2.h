#ifndef TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLER2_H
#define TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLER2_H
//==============================================================================
#include <tatooine/detail/pointset/moving_least_squares_samplerN.h>
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
/// Moving Least Squares Sampler of scattered data in 2 Dimensions.
/// \see <em>An As-Short-As-Possible Introduction to the Least Squares,
/// Weighted Least Squares and Moving Least Squares Methods for Scattered Data
/// Approximation and Interpolation</em> \cite nealen2004LeastSquaresIntro.
template <floating_point Real, typename T>
struct moving_least_squares_sampler<Real, 2, T>
    : field<moving_least_squares_sampler<Real, 2, T>, Real, 2, T> {
  static_assert(flann_available(), "Moving Least Squares Sampler needs FLANN!");
  using this_type   = moving_least_squares_sampler<Real, 2, T>;
  using parent_type = field<this_type, Real, 2, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, 2>;
  using property_type =
      typename pointset_type::template typed_vertex_property_t<T>;
  using vertex_handle = typename pointset_type::vertex_handle;
  //==========================================================================
  pointset_type const& m_pointset;
  property_type const& m_property;
  Real                 m_radius;
  //==========================================================================
  moving_least_squares_sampler(pointset_type const& ps,
                               property_type const& property, Real const radius)
      : m_pointset{ps}, m_property{property}, m_radius{radius} {}
  //--------------------------------------------------------------------------
  moving_least_squares_sampler(moving_least_squares_sampler const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  moving_least_squares_sampler(moving_least_squares_sampler&&) noexcept =
      default;
  //--------------------------------------------------------------------------
  auto operator                        =(moving_least_squares_sampler const&)
      -> moving_least_squares_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(moving_least_squares_sampler&&) noexcept
      -> moving_least_squares_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~moving_least_squares_sampler() = default;
  //--------------------------------------------------------------------------
 private:
  [[nodiscard]] auto evaluate_0_neighbors(pos_type const& q) const {
    if constexpr (is_arithmetic<tensor_type>) {
      return Real(0) / Real(0);
    } else {
      return tensor_type::fill(Real(0) / Real(0));
    }
  }
  //------------------------------------------------------------------------------
  [[nodiscard]] auto evaluate_1_neighbors(
      std::vector<int> const& indices) const {
    return m_property[vertex_handle{indices[0]}];
  }
  //------------------------------------------------------------------------------
  [[nodiscard]] auto evaluate_2_neighbors(
      std::vector<int> const&  indices,
      std::vector<Real> const& distances) const {
    auto const& p0 = m_property[vertex_handle{indices[0]}];
    auto const& p1 = m_property[vertex_handle{indices[1]}];

    auto d0 = distances[0];
    auto d1 = distances[1];

    auto const d_norm = 1 / (d0 + d1);
    d0 *= d_norm;
    d1 *= d_norm;

    return p0 * d1 + p1 * d0;
  }
  //------------------------------------------------------------------------------
  [[nodiscard]] auto evaluate_more_than_2_neighbors(
      std::vector<int> const& indices, std::vector<Real> const& distances,
      pos_type const& q) const {
    auto const num_neighbors = size(indices);
    auto const w             = construct_weights(num_neighbors, distances);
    auto const F             = construct_F(num_neighbors, indices);
    auto const B             = construct_B(num_neighbors, indices, q);
    auto const BtW           = transposed(B) * diag(w);
    auto const C             = *solve(BtW * B, BtW * F);

    if constexpr (tensor_num_components<T> == 1) {
      return C(0);
    } else {
      auto ret = T{};
      for (std::size_t i = 0; i < tensor_num_components<T>; ++i) {
        ret(i) = C(0, i);
      }
      return ret;
    }
  }
  //------------------------------------------------------------------------------
  auto construct_B(std::size_t const       num_neighbors,
                   std::vector<int> const& indices, pos_type const& q) const {
    auto B               = allocate_B(num_neighbors);
    auto local_positions = std::vector<pos_type>(num_neighbors);
    std::ranges::copy(indices | std::views::transform([&](auto const i) {
                        return m_pointset.vertex_at(i) - q;
                      }),
                      begin(local_positions));
    if (num_neighbors >= 3) {
      construct_linear_part_of_B(local_positions, q, B);
    }
    if (num_neighbors >= 6) {
      construct_quadratic_part_of_B(local_positions, q, B);
    }
    if (num_neighbors >= 10) {
      construct_cubic_part_of_B(local_positions, q, B);
    }
    return B;
  }
  //------------------------------------------------------------------------------
  auto allocate_B(std::size_t const num_neighbors) const {
    if (num_neighbors >= 10) {
      return tensor<Real>::ones(num_neighbors, 10);
    }
    if (num_neighbors >= 6) {
      return tensor<Real>::ones(num_neighbors, 6);
    }
    if (num_neighbors >= 3) {
      return tensor<Real>::ones(num_neighbors, 3);
    }
    return tensor<Real>::ones(1, 1);
  }
  //------------------------------------------------------------------------------
  auto construct_linear_part_of_B(std::vector<pos_type> const& local_positions,
                                  pos_type const& q, auto& B) const {
    auto i = std::size_t{};
    for (auto const& x : local_positions) {
      B(i, 1) = x.x();
      B(i, 2) = x.y();
      ++i;
    }
  }
  //------------------------------------------------------------------------------
  auto construct_quadratic_part_of_B(
      std::vector<pos_type> const& local_positions, pos_type const& q,
      auto& B) const {
    auto i = std::size_t{};
    for (auto const& x : local_positions) {
      B(i, 3) = x.x() * x.x();
      B(i, 4) = x.x() * x.y();
      B(i, 5) = x.y() * x.y();
      ++i;
    }
  }
  //------------------------------------------------------------------------------
  auto construct_cubic_part_of_B(std::vector<pos_type> const& local_positions,
                                 pos_type const& q, auto& B) const {
    auto i = std::size_t{};
    for (auto const& x : local_positions) {
      B(i, 6) = x.x() * x.x() * x.x();
      B(i, 7) = x.x() * x.x() * x.y();
      B(i, 8) = x.x() * x.y() * x.y();
      B(i, 9) = x.y() * x.y() * x.y();
      ++i;
    }
  }
  //----------------------------------------------------------------------------
  auto construct_weights(std::size_t const        num_neighbors,
                         std::vector<Real> const& distances) const {
    auto w                  = tensor<Real>::zeros(num_neighbors);
    auto weighting_function = [&](auto const d) {
      auto const s =  1 - d / m_radius;//normalized_distance
      return 1 - 6 * s * s + 8 * s * s * s - 3 * s * s * s * s;
    };
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      w(i) = weighting_function(distances[i]);
    }
    return w;
  }
  //----------------------------------------------------------------------------
  /// Represents function values f(x_i)
  auto construct_F(std::size_t const       num_neighbors,
                   std::vector<int> const& indices) const {
    auto F = tensor_num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, tensor_num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      if constexpr (tensor_num_components<T> == 1) {
        F(i) = m_property[vertex_handle{indices[i]}];
      } else {
        for (std::size_t j = 0; j < tensor_num_components<T>; ++j) {
          F(i, j) = m_property[vertex_handle{indices[i]}](j);
        }
      }
    }
    return F;
  }
  //==========================================================================
 public:
  [[nodiscard]] auto evaluate(pos_type const& q, Real const /*t*/) const
      -> tensor_type {
    auto [indices, distances] =
        m_pointset.nearest_neighbors_radius_raw(q, m_radius);
    switch (size(indices)) {
      case 0:
        return evaluate_0_neighbors(q);
      case 1:
        return m_property[vertex_handle{indices[0]}];
      case 2:
        return evaluate_2_neighbors(indices, distances);
      default:
        return evaluate_more_than_2_neighbors(indices, distances, q);
    }
  }
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
