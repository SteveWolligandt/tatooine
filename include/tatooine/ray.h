#ifndef TATOOINE_RAY_H
#define TATOOINE_RAY_H
//==============================================================================
#include "tensor.h"
//==============================================================================
namespace tatooine {
//==============================================================================
  template <typename Real, size_t N>
struct ray{
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  //============================================================================
 private:
   pos_t m_origin;
   vec_t m_direction;
  //============================================================================
 public:
  ray(const ray&)     = default;
  ray(ray&&) noexcept = default;
  //----------------------------------------------------------------------------
  ray& operator=(const ray&) = default;
  ray& operator=(ray&&) noexcept = default;
  //----------------------------------------------------------------------------
  ray(const pos_t& origin, const vec_t& direction)
      : m_origin{origin}, m_direction{direction} {}
  ray(pos_t&& origin, const vec_t& direction)
      : m_origin{std::move(origin)}, m_direction{direction} {}
  ray(const pos_t& origin, vec_t&& direction)
      : m_origin{origin}, m_direction{std::move(direction)} {}
  ray(pos_t&& origin, vec_t&& direction)
      : m_origin{std::move(origin)}, m_direction{std::move(direction)} {}
  //============================================================================
  auto&       origin() { return m_origin; }
  const auto& origin() const { return m_origin; }
  //----------------------------------------------------------------------------
  auto&       direction() const { return m_direction; }
  const auto& direction() const { return m_direction; }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto operator=(Real t) const { return at(t); }
  [[nodiscard]] auto at(Real t) const {return m_origin + m_direction * t;}
  //----------------------------------------------------------------------------
  void normalize() { m_direction = normalize(m_direction); }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
