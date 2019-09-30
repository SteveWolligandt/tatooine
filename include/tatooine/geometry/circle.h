#ifndef TATOOINE_GEOMETRY_CIRCLE_H
#define TATOOINE_GEOMETRY_CIRCLE_H

#include "../tensor.h"
#include "primitive.h"

//==============================================================================
namespace tatooine::geometry {
//==============================================================================

template <typename Real>
struct circle : primitive<Real, 2> {
  using this_t   = circle<Real>;
  using parent_t = primitive<Real, 2>;
  using typename parent_t::pos_t;

  //============================================================================
 private:
  Real  m_radius;
  pos_t m_center;

  //============================================================================
 public:
  circle(Real radius, pos_t&& center)
      : m_radius{radius}, m_center{std::move(center)} {}
  circle(Real radius, const pos_t& center)
      : m_radius{radius}, m_center{center} {}

  //----------------------------------------------------------------------------
  circle(const circle&) = default;
  circle(circle&&) = default;
  circle& operator=(const circle&) = default;
  circle& operator=(circle&&) = default;

  //============================================================================
  bool is_inside(const pos_t& x) const override {
    return distance(center, x) <= radius;
  }

  //----------------------------------------------------------------------------
  auto  radius() const { return m_radius; }
  auto& radius() { return m_radius; }

  //----------------------------------------------------------------------------
  const auto& center() const { return m_center; }
  auto&       center() { return m_center; }
};

//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================

#endif
