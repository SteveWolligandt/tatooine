#ifndef __YAVIN_COLLISION_H__
#define __YAVIN_COLLISION_H__

#include <glm/glm.hpp>
#include "sceneobject.h"
//==============================================================================
namespace yavin {
//==============================================================================

template <typename T>
class collision {
 public:
  collision(scene_object<T> _o, const glm::tvec3<T>& _x)
      : m_scene_object(&_o), m_x(_x) {}
  collision(scene_object<T> _o, glm::tvec3<T>&& _x)
      : m_scene_object(&_o), m_x(std::move(_x)) {}
  collision(scene_object<T>* _o, const glm::tvec3<T>& _x)
      : m_scene_object(_o), m_x(_x) {}
  collision(scene_object<T>* _o, glm::tvec3<T>&& _x)
      : m_scene_object(_o), m_x(std::move(_x)) {}

  collision(const collision<T>& other)
      : m_scene_object(other.m_scene_object), m_x(other.m_x) {}
  collision(collision<T>&& other)
      : m_scene_object(other.m_scene_object), m_x(std::move(other.m_x)) {}

  auto& operator=(const collision<T>& other) {
    m_scene_object = other.m_scene_object;
    m_x            = other.m_x;
    return *this;
  }

  auto& operator=(collision<T>&& other) {
    m_scene_object = other.m_scene_object;
    m_x            = std::move(other.m_x);
    return *this;
  }

  auto&       x() { return m_x; }
  const auto& x() const { return m_x; }
  const auto& x(size_t i) { return m_x[i]; }
  auto&       object() { return *m_scene_object; }
  const auto& object() const { return *m_scene_object; }

 private:
  scene_object<T>* m_scene_object;
  glm::tvec3<T>    m_x;
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
