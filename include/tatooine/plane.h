#ifndef TATOOINE_PLANE_H
#define TATOOINE_PLANE_H
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t N>
struct plane {
  //----------------------------------------------------------------------------
 private:
  vec<T, N> m_origin;
  vec<T, N> m_normal;
  //----------------------------------------------------------------------------
 public:
  plane(vec<T, N> const& origin, vec<T, N> const& normal)
      : m_origin{origin}, m_normal{normal} {}
  plane(vec<T, N>&& origin, vec<T, N> const& normal)
      : m_origin{std::move(origin)}, m_normal{normal} {}
  plane(vec<T, N> const& origin, vec<T, N>&& normal)
      : m_origin{origin}, m_normal{std::move(normal)} {}
  plane(vec<T, N>&& origin, vec<T, N>&& normal)
      : m_origin{std::move(origin)}, m_normal{std::move(normal)} {}
  //----------------------------------------------------------------------------
  auto normal() -> auto& { return m_normal; }
  auto normal() const -> auto const& { return m_normal; }
  //----------------------------------------------------------------------------
  auto origin() -> auto& { return m_origin; }
  auto origin() const -> auto const& { return m_origin; }
};
template <size_t N>
using Plane   = plane<real_t, N>;
using plane2  = Plane<2>;
using plane3  = Plane<3>;
using plane4  = Plane<4>;
using plane5  = Plane<5>;
using plane6  = Plane<6>;
using plane7  = Plane<7>;
using plane8  = Plane<8>;
using plane9  = Plane<9>;
using plane10 = Plane<10>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
