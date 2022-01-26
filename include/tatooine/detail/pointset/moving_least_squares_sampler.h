#ifndef TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLER_H
#define TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLER_H
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
  using this_t   = moving_least_squares_sampler<Real, 2, T>;
  using parent_t = field<this_t, Real, 2, T>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using pointset_t        = tatooine::pointset<Real, 2>;
  using vertex_property_t = typename pointset_t::template vertex_property_t<T>;
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
  auto operator=(moving_least_squares_sampler const&)
      -> moving_least_squares_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(moving_least_squares_sampler&&) noexcept
      -> moving_least_squares_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~moving_least_squares_sampler() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_t const& q, Real const /*t*/) const
      -> tensor_t {
    auto        nn      = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
    auto const& indices = nn.first;
    auto&       distances = nn.second;
    for (auto& d : distances) {
      d /= m_radius;
      d = 1 - d;
    }
    auto const num_neighbors = size(indices);

    if (num_neighbors == 0) {
      if constexpr (is_arithmetic<tensor_t>) {
        return Real(0) / Real(0);
      } else {
        return tensor_t::fill(Real(0) / Real(0));
      }
    }
    if (num_neighbors == 1) {
      return m_property[vertex_handle{indices[0]}];
    }

    auto w = tensor<Real>::zeros(num_neighbors);
    auto F = num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    auto B = [&] {
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
    }();

    // build w
    auto weighting_function = [&](auto const d) {
      return 1 / d - 1 / m_radius;
      // return std::exp(-d * d);
    };
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      w(i) = weighting_function(distances[i]);
    }
    // build F
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      if constexpr (num_components<T> == 1) {
        F(i) = m_property[vertex_handle{indices[i]}];
      } else {
        for (std::size_t j = 0; j < num_components<T>; ++j) {
          F(i, j) = m_property[vertex_handle{indices[i]}](j);
        }
      }
    }
    // build B
    // linear terms of polynomial
    if (num_neighbors >= 3) {
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
      }
    }
    // quadratic terms of polynomial
    if (num_neighbors >= 6) {
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
    // cubic terms of polynomial
    if (num_neighbors >= 10) {
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
};
//============================================================================
/// Moving Least Squares Sampler of scattered data in 3 Dimensions.
/// \see <em>An As-Short-As-Possible Introduction to the Least Squares,
/// Weighted Least Squares and Moving Least Squares Methods for Scattered Data
/// Approximation and Interpolation</em> \cite nealen2004LeastSquaresIntro.
template <typename Real, typename T>
struct moving_least_squares_sampler<Real, 3, T>
    : field<moving_least_squares_sampler<Real, 3, T>, Real, 3, T> {
  static_assert(flann_available(), "Moving Least Squares Sampler needs FLANN!");
  using this_t   = moving_least_squares_sampler<Real, 3, T>;
  using parent_t = field<this_t, Real, 3, T>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using pointset_t        = tatooine::pointset<Real, 3>;
  using vertex_handle     = typename pointset_t::vertex_handle;
  using vertex_property_t = typename pointset_t::template vertex_property_t<T>;
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
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_t const& q, real_t const /*t*/) const
      -> tensor_t {
    auto const  nn      = m_pointset.nearest_neighbors_radius_raw(q, m_radius);
    auto const& indices = nn.first;
    auto const& distances     = nn.second;
    auto const  num_neighbors = size(indices);
    if (num_neighbors == 0) {
      return T{Real(0) / Real(0)};
    }
    if (num_neighbors == 1) {
      return m_property[vertex_handle{indices[0]}];
    }

    auto w = tensor<Real>::zeros(num_neighbors);
    auto F = num_components<T> > 1
                 ? tensor<Real>::zeros(num_neighbors, num_components<T>)
                 : tensor<Real>::zeros(num_neighbors);
    auto B = [&] {
      if (num_neighbors >= 20) {
        return tensor<Real>::ones(num_neighbors, 20);
      } else if (num_neighbors >= 10) {
        return tensor<Real>::ones(num_neighbors, 10);
      } else if (num_neighbors >= 4) {
        return tensor<Real>::ones(num_neighbors, 4);
      }
      return tensor<Real>::ones(1, 1);
    }();
    // build w
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      // if (distances[i] == 0) {
      //  return m_property[vertex_handle{indices[i]}];
      //}
      // w(i) = 1 / distances[i] - 1 / m_radius;
      w(i) = std::exp(-(m_radius - distances[i]) * (m_radius - distances[i]));
    }
    // build f
    for (std::size_t i = 0; i < num_neighbors; ++i) {
      if constexpr (num_components<T> == 1) {
        F(i, 0) = m_property[vertex_handle{indices[i]}];
      } else {
        for (std::size_t j = 0; j < num_components<T>; ++j) {
          F(i, j) = m_property[vertex_handle{indices[i]}](j);
        }
      }
    }
    // build B
    // linear terms of polynomial
    if (num_neighbors >= 4) {
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 1) = m_pointset.vertex_at(indices[i]).x() - q.x();
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 2) = m_pointset.vertex_at(indices[i]).y() - q.y();
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 3) = m_pointset.vertex_at(indices[i]).z() - q.z();
      }
    }
    // quadratic terms of polynomial
    if (num_neighbors >= 10) {
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 4) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                  (m_pointset.vertex_at(indices[i]).x() - q.x());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 5) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                  (m_pointset.vertex_at(indices[i]).y() - q.y());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 6) = (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                  (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 7) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                  (m_pointset.vertex_at(indices[i]).y() - q.y());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 8) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                  (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 9) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                  (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
    }
    // cubic terms of polynomial
    if (num_neighbors >= 20) {
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 10) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).x() - q.x());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 11) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 12) = (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 13) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 14) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 15) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 16) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 17) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 18) = (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
      for (std::size_t i = 0; i < num_neighbors; ++i) {
        B(i, 19) = (m_pointset.vertex_at(indices[i]).x() - q.x()) *
                   (m_pointset.vertex_at(indices[i]).y() - q.y()) *
                   (m_pointset.vertex_at(indices[i]).z() - q.z());
      }
    }
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
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
