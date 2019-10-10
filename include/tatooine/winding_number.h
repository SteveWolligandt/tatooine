#ifndef TATOOINE_WINDING_NUMBER_H_
#define TATOOINE_WINDING_NUMBER_H_

#include "edgeset.h"

//==============================================================================
namespace tatooine {
//==============================================================================

/// tests if a point is Left|On|Right of an infinite line defined by p0 and p1
/// of pointset ps1.
/// \return > 0 left of the line through P0 and P1
///         = 0 on the line
///         < 0 right of the line
template <typename Real0, typename Real1>
inline auto side(const pointset<Real0, 2>& ps0, const pointset<Real1, 2>& ps1,
                 typename pointset<Real0, 2>::vertex cp,
                 typename pointset<Real1, 2>::vertex p0,
                 typename pointset<Real1, 2>::vertex p1) {
  return (ps1[p1](0) - ps1[p0](0)) * (ps0[cp](1) - ps1[p0](1)) -
         (ps0[cp](0) - ps1[p0](0)) * (ps1[p1](1) - ps1[p0](1));
}
//------------------------------------------------------------------------------
template <typename Real>
inline auto side(const pointset<Real, 2>&           ps,
                 typename pointset<Real, 2>::vertex cp,
                 typename pointset<Real, 2>::vertex p0,
                 typename pointset<Real, 2>::vertex p1) {
  return side(ps, ps, cp, p0, p1);
}
//------------------------------------------------------------------------------
template <typename Real0, typename Real1>
inline auto side(const pointset<Real0, 2>& ps0, const edgeset<Real1, 2>& es1,
                 typename pointset<Real0, 2>::vertex p,
                 typename edgeset<Real1, 2>::Edge    e) {
  return side(ps0, es1, p, es1[e][0], es1[e][1]);
}
//------------------------------------------------------------------------------
template <typename Real>
inline auto side(const edgeset<Real, 2>&            es,
                 typename pointset<Real, 2>::vertex p,
                 typename edgeset<Real, 2>::Edge    e) {
  return side(es, es, p, e);
}

//==============================================================================
/// winding number for a point inside or outside a polygon
/// \return the winding number
template <typename Real0, typename Real1>
inline int winding_number(
    const pointset<Real0, 2>& ps0, typename pointset<Real0, 2>::vertex p,
    const pointset<Real1, 2>&                               ps1,
    const std::vector<typename pointset<Real1, 2>::vertex>& polygon) {
  int wn = 0;  // the  winding number counter

  // loop through all edges of the polygon
  for (size_t i = 0; i < polygon.size(); ++i) {
    // edge from polygon[i] to  polygon[i_next]
    size_t i_next = i + 1 == polygon.size() ? 0 : i + 1;
    if (ps1[polygon[i]](1) <= ps0[p](1)) {      // y <= p.y
      if (ps1[polygon[i_next]](1) > ps0[p](1))  // an upward crossing
        if (side(ps0, ps1, p, polygon[i], polygon[i_next]) >
            0)   // p left of  edge
          ++wn;  // have  a valid up intersect
    } else if (ps1[polygon[i_next]](1) <= ps0[p](1)) {  // a downward crossing
      if (side(ps0, ps1, p, polygon[i], polygon[i_next]) <
          0)   // P right of edge
        --wn;  // have  a valid down intersect
    }
  }
  return wn;
}

//------------------------------------------------------------------------------
template <typename Real>
inline int winding_number(
    const pointset<Real, 2>& ps, typename pointset<Real, 2>::vertex p,
    const std::vector<typename pointset<Real, 2>::vertex>& polygon) {
  return winding_number(ps, p, ps, polygon);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
