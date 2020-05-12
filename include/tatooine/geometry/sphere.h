#ifndef TATOOINE_GEOMETRY_SPHERE_H
#define TATOOINE_GEOMETRY_SPHERE_H
//==============================================================================
#include "../tensor.h"
#include "primitive.h"
#include "sphere_ray_intersection.h"
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
  sphere() : m_radius{1}, m_center{pos_t::zeros()} {}
  explicit sphere(Real radius) : m_radius{radius}, m_center{pos_t::zeros()} {}
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
  std::optional<intersection<Real, N>> check_intersection(
      const ray<Real, N>& r, const Real min_t = 0) const override {
    return tatooine::geometry::check_intersection(*this, r, min_t);
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

