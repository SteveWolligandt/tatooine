#ifndef __RAY_H__
#define __RAY_H__

#include <glm/glm.hpp>

//==============================================================================
namespace yavin {
//==============================================================================

template <typename T>
class ray {
 public:
  ray(T x, T y, T z, T dir_x, T dir_y, T dir_z)
      : m_origin{x, y, z}, m_dir{dir_x, dir_y, dir_z} {}
  ray(const glm::vec3& _origin, const glm::vec3& _dir)
      : m_origin(_origin), m_dir(_dir) {}
  ray(glm::vec3&& _origin, glm::vec3&& _dir)
      : m_origin(std::move(_origin)), m_dir(std::move(_dir)) {}

  const auto& x() const { return m_origin; }
  const auto& x(size_t i) const { return m_origin[i]; }

  const auto& dir() const { return m_dir; }
  const auto& dir(size_t i) const { return m_dir[i]; }

  auto operator()(T length) const { return m_origin + length * m_dir; }

 private:
  glm::tvec3<T> m_origin;
  glm::tvec3<T> m_dir;
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
