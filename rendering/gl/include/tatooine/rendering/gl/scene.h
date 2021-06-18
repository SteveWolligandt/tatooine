#ifndef __SCENE_H__
#define __SCENE_H__

#include <algorithm>
#include <iostream>
#include <vector>
#include "camera.h"
#include "collision.h"
#include "ray.h"
#include "sceneobject.h"

//==============================================================================
namespace yavin {
//==============================================================================

template <typename T = double>
class scene : public std::vector<scene_object<T>> {
 public:
  std::vector<std::unique_ptr<shader>> m_shader_pool;

  template <typename Shader, typename... Args>
  auto& add_shader(Args&&... args) {
    m_shader_pool.emplace_back(new Shader(std::forward<Args>(args)...));
    return *dynamic_cast<Shader*>(m_shader_pool.back().get());
  }

  std::optional<collision<T>> cast_ray(const ray<T>& r) const {
    T                             shortest_dist = 1e10;
    std::optional<collision<T>> closest_collision;
    for (const auto& obj : *this)
      if (auto c = obj.check_collision(r); c)
        if (T dist = glm::distance(c->x(), r.x()); shortest_dist > dist) {
          shortest_dist     = dist;
          closest_collision = c;
        }

    if (closest_collision)
      closest_collision->object().on_collision(*closest_collision);

    return closest_collision;
  }

  void push_back(const scene_object<T>& obj) {
    std::vector<scene_object<T>>::push_back(obj);
    obj.m_appropriate_scenes.push_back(this);
  }

  void push_back(scene_object<T>&& obj) {
    std::vector<scene_object<T>>::push_back(std::move(obj));
    obj.m_appropriate_scenes.push_back(this);
  }

  void push_back_dispatcher(std::function<void()>&& dispatcher) {
    m_dispatch_functions.push_back(std::move(dispatcher));
  }

  template <class... Args>
  auto& emplace_back(Args&&... args) {
    std::vector<scene_object<T>>::emplace_back(std::forward<Args>(args)...);
    this->back().m_appropriate_scenes.push_back(this);
    return this->back();
  }

  std::optional<collision<T>> cast_ray(const ray<T>& r, T x, T y) const {
    auto c = cast_ray(r);
    if (c) c->object().on_mouse_down(x, y);
    return c;
  }

  void on_mouse_down(T x, T y) {
    for (auto& obj : *this) obj.on_mouse_down(x, y);
  }

  void on_mouse_up(T x, T y) {
    for (auto& obj : *this) obj.on_mouse_up(x, y);
  }

  void on_mouse_moved(T x, T y) {
    for (auto& obj : *this) obj.on_mouse_moved(x, y);
  }

  void on_collision(const collision<T>& c) {
    for (auto& obj : *this) obj.on_collision(c);
  }

  void on_render(const camera& cam) {
    gl::viewport(cam.viewport());
    for (auto& obj : *this) obj.on_render(cam);
  }

  void dispatch() {
    for (auto& dispatcher : m_dispatch_functions) dispatcher();
    m_dispatch_functions.clear();
  }

 private:
  std::vector<std::function<void()>> m_dispatch_functions;
};

//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
