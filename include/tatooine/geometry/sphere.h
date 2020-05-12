#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include "../tensor.h"
#include "primitive.h"
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <typename Real, size_t N>
struct sphere : primitive<Real, N> {
  using this_t   = sphere<Real, N>;
  using parent_t = primitive<Real, N>;
  using typename parent_t::pos_t;
  //============================================================================
 private:
  Real  m_radius;
  pos_t m_center;
  //============================================================================
 public:
  sphere(Real radius, pos_t&& center)
      : m_radius{radius}, m_center{std::move(center)} {}
  sphere(Real radius, const pos_t& center)
      : m_radius{radius}, m_center{center} {}
  //----------------------------------------------------------------------------
  sphere(const sphere&) = default;
  sphere(sphere&&) = default;
  sphere& operator=(const sphere&) = default;
  sphere& operator=(sphere&&) = default;
  //============================================================================
  bool is_inside(const pos_t& x) const override {
    return distance(center, x) <= m_radius;
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

