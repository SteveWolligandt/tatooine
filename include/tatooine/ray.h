#ifndef TATOOINE_RAY_H
#define TATOOINE_RAY_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct ray {
  static_assert(is_floating_point<Real>);
  template <typename OtherReal, size_t OtherN>
  friend struct ray;
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  //============================================================================
 private:
  pos_t m_origin;
  vec_t m_direction;
  //============================================================================
 public:
  ray(ray const&)     = default;
  ray(ray&&) noexcept = default;
  //----------------------------------------------------------------------------
  template <floating_point OtherReal>
  ray(ray<OtherReal, N> const& other)
      : m_origin{other.m_origin}, m_direction{other.m_direction} {}
  //----------------------------------------------------------------------------
  ray& operator=(ray const&) = default;
  ray& operator=(ray&&) noexcept = default;
  //----------------------------------------------------------------------------
  ray(pos_t const& origin, vec_t const& direction)
      : m_origin{origin}, m_direction{direction} {}
  ray(pos_t&& origin, vec_t const& direction)
      : m_origin{std::move(origin)}, m_direction{direction} {}
  ray(pos_t const& origin, vec_t&& direction)
      : m_origin{origin}, m_direction{std::move(direction)} {}
  ray(pos_t&& origin, vec_t&& direction)
      : m_origin{std::move(origin)}, m_direction{std::move(direction)} {}
  //============================================================================
  auto origin() -> auto& {
    return m_origin;
  }
  auto origin() const -> auto const& {
    return m_origin;
  }
  auto origin(size_t i) -> auto& {
    return m_origin(i);
  }
  auto origin(size_t i) const {
    return m_origin(i);
  }
  //----------------------------------------------------------------------------
  auto direction() -> auto& {
    return m_direction;
  }
  auto direction() const -> auto const& {
    return m_direction;
  }
  auto direction(size_t i) -> auto& {
    return m_direction(i);
  }
  auto direction(size_t i) const {
    return m_direction(i);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto operator()(Real t) const {
    return at(t);
  }
  [[nodiscard]] auto at(Real t) const {
    return m_origin + m_direction * t;
  }
  //----------------------------------------------------------------------------
  void normalize() {
    m_direction = tatooine::normalize(m_direction);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
