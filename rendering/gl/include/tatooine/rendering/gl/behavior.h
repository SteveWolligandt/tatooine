#ifndef __YAVIN_BEHAVIOUR_H__
#define __YAVIN_BEHAVIOUR_H__

#include <optional>
#include "camera.h"
#include "ray.h"

//==============================================================================
namespace yavin {
//==============================================================================

template <typename T>
class scene_object;
template <typename T>
class collision;

template <typename T>
class behavior {
 public:
  behavior(scene_object<T>* o) : m_scene_object(o) {}
  behavior(const behavior<T>& other)
      : m_scene_object(other.m_scene_object) {}
  behavior(behavior<T>&& other) : m_scene_object(other.m_scene_object) {}

  virtual void on_mouse_down(T /*x*/, T /*y*/) {}
  virtual void on_mouse_up(T /*x*/, T /*y*/) {}
  virtual void on_mouse_moved(T /*x*/, T /*y*/) {}
  virtual void on_collision(const collision<T>& /*c*/) {}
  virtual void on_render(const camera& /*cam*/) {}

  virtual std::optional<collision<T>> check_collision(
      const ray<T>& /*r*/) {
    return {};
  }

  virtual behavior<T>& operator=(const behavior& other) {
    m_scene_object = other.m_scene_object;
    return *this;
  }

  virtual behavior<T>& operator=(behavior&& other) {
    m_scene_object = other.m_scene_object;
    return *this;
  }

  auto&       object() { return *m_scene_object; }
  const auto& object() const { return *m_scene_object; }

 protected:
  scene_object<T>* m_scene_object;
};

//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
