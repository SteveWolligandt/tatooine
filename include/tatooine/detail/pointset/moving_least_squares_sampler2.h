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
template <typename Real, typename T>
struct moving_least_squares_sampler<Real, 2, T>
    : field<moving_least_squares_sampler<Real, 2, T>, Real, 2, T> {
  static_assert(flann_available(), "Moving Least Squares Sampler needs FLANN!");
  using this_type   = moving_least_squares_sampler<Real, 2, T>;
  using parent_type = field<this_type, Real, 2, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_t        = tatooine::pointset<Real, 2>;
  using vertex_property_t = typename pointset_t::template typed_vertex_property_t<T>;
  using vertex_handle     = typename pointset_t::vertex_handle;
  //==========================================================================
  pointset_t const&        m_pointset;
  vertex_property_t const& m_property;
  Real                     m_radius = 1;
  //==========================================================================
  moving_least_squares_sampler(pointset_t const&        ps,
                               vertex_property_t const& property,
                               Real const               radius = 1)
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
  [[nodiscard]] auto evaluate_1_neighbors(std::vector<int> const& indices) const {
      return m_property[vertex_handle{indices[0]}];
  }
  //------------------------------------------------------------------------------
  [[nodiscard]] auto evaluate_2_neighbors(std::vector<int> const& indices, std::vector<Real> const& distances) const {
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
      std::vector<int> const& indices,
      std::vector<Real>& distances, pos_type const& q) const {
    auto const num_neighbors = size(indices);
    for (auto& d : distances) {
      d /= m_radius;
      d = 1 - d;
    }
    auto const w   = construct_w(num_neighbors, distances);
    auto const F   = construct_F(num_neighbors, indices);
    auto const B   = construct_B(num_neighbors, indices, q);
    auto const BtW = transposed(B) * diag(w);

    if constexpr (num_components<T> == 1) {
      return solve(BtW * B, BtW * F)(0);
    } else {
      T    ret{};
      auto C = solve(BtW * B, BtW * F);
      for (std::size_t i = 0; i < num_components<T>; ++i) {
        ret(i) = C(0, i);
      }
      return ret;
    }
  }
  //------------------------------------------------------------------------------
  auto construct_linear_part_of_B(std::size_t const       num_neighbors,
                                  std::vector<int> const& indices,
                                  pos_type const& q, auto& B) const {
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
    }
  }
  //------------------------------------------------------------------------------
  auto construct_quadratic_part_of_B(std::size_t const       num_neighbors,
                                     std::vector<int> const& indices,
                                     pos_type const& q, auto& B) const {
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 3) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).x() - q.x());
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 4) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y());
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 5) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y());
    }
  }
  //------------------------------------------------------------------------------
  auto construct_cubic_part_of_B(std::size_t const       num_neighbors,
                                 std::vector<int> const& indices,
                                 pos_type const& q, auto& B) const {
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 6) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).x() - q.x());
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 7) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y());
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 8) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y());
    }
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      B(i, 9) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                (m_pointset.vertex_at(indices[i]).y() - q.y());
    }
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
  auto construct_B(std::size_t const       num_neighbors,
                   std::vector<int> const& indices, pos_type const& q) const {
    auto B = allocate_B(num_neighbors);
    if (num_neighbors >= 3) {
      construct_linear_part_of_B(num_neighbors, indices, q, B);
    }
    if (num_neighbors >= 6) {
      construct_quadratic_part_of_B(num_neighbors, indices, q, B);
    }
    // cubic terms of polynomial
    if (num_neighbors >= 10) {
      construct_cubic_part_of_B(num_neighbors, indices, q, B);
    }
    return B;
  }
  //----------------------------------------------------------------------------
  auto construct_w(std::size_t const        num_neighbors,
                   std::vector<Real> const& distances) const {
    auto w = tensor<Real>::zeros(num_neighbors);
    // build w
    auto weighting_function = [&](auto const d) {
      return 1 / d - 1 / m_radius;
      // return std::exp(-d * d);
    };
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      w(i) = weighting_function(distances[i]);
    }
    return w;
  }
  //----------------------------------------------------------------------------
  auto construct_F(std::size_t const       num_neighbors,
                   std::vector<int> const& indices) const {
    auto F = num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      if constexpr (num_components<T> == 1) {
        F(i) = m_property[vertex_handle{indices[i]}];
      } else {
        for (std::size_t j = 0; j < num_components<T>; ++j) {
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
