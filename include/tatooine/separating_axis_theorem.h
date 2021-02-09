#ifndef TATOOINE_SEPARATING_AXIS_THEOREM_H
#define TATOOINE_SEPARATING_AXIS_THEOREM_H
//==============================================================================
#include <tatooine/vec.h>

#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Return true if n is a separating axis of polygon0 and polygon1.
template <typename Real>
auto is_separating_axis(vec<Real, 2> const&              n,
                        std::vector<vec<Real, 2>> const& polygon0,
                        std::vector<vec<Real, 2>> const& polygon1) {
  Real min0, min1, max0, max1;
  min0 = min1 = std::numeric_limits<Real>::infinity();
  max0 = max1 = -std::numeric_limits<Real>::infinity();

  for (auto const& v : polygon0) {
    auto const projection = dot(v, n);
    min0                  = std::min(min0, projection);
    max0                  = std::max(max0, projection);
  }
  for (auto const& v : polygon1) {
    auto const projection = dot(v, n);
    min1                  = std::min(min1, projection);
    max1                  = std::max(max1, projection);
  }
  return !(max0 >= min1 && max1 >= min0);
}
//------------------------------------------------------------------------------
/// from
/// https://hackmd.io/@US4ofdv7Sq2GRdxti381_A/ryFmIZrsl?type=view#:~:text=Separating%20axis%20theorem%20(SAT)&text=In%20simple%20terms%2C%20the%20SAT,objects'%20projections%20do%20not%20overlap.
template <typename Real>
auto has_separating_axis(std::vector<vec<Real, 2>> const& polygon0,
                         std::vector<vec<Real, 2>> const& polygon1) -> bool {
  using vec_t    = vec<Real, 2>;
  using vec_list = std::vector<vec_t>;
  vec_list normals;
  normals.reserve(size(polygon0) + size(polygon1));

  vec_t e;
  for (size_t i = 0; i < size(polygon0) - 1; ++i) {
    e = polygon0[i + 1] - polygon0[i];
    normals.push_back(vec_t{-e(1), e(0)});
  }
  for (size_t i = 0; i < size(polygon1) - 1; ++i) {
    e = polygon1[i + 1] - polygon1[i];
    normals.push_back(vec_t{-e(1), e(0)});
  }
  e = polygon0.front() - polygon0.back();
  normals.push_back(vec_t{-e(1), e(0)});
  e = polygon1.front() - polygon1.back();
  normals.push_back(vec_t{-e(1), e(0)});

  for (auto const& n : normals) {
    if (is_separating_axis(n, polygon0, polygon1)) {
      return true;
    }
  }
  return false;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
