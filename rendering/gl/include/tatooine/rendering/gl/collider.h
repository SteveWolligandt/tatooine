#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include <glm/glm.hpp>
#include "behavior.h"
#include "collision.h"

//==============================================================================
namespace yavin {
//==============================================================================

template <typename T>
class collider : public behavior<T> {
 public:
  collider(scene_object<T>* o) : behavior<T>(o) {}
  collider(const collider& other) : behavior<T>(other) {}
  collider(collider&& other) : behavior<T>(other) {}

  virtual collider<T>& operator=(const collider& other) {
    behavior<T>::operator=(other);
    return *this;
  }

  virtual collider<T>& operator=(collider&& other) {
    behavior<T>::operator=(other);
    return *this;
  }
};

template <typename T>
class aabb_collider : public collider<T> {
 public:
  aabb_collider(scene_object<T>* o, T center_x, T center_y, T center_z,
                T size_x, T size_y, T size_z)
      : collider<T>(o),
        m_center{center_x, center_y, center_z},
        m_size{size_x, size_y, size_z} {}
  aabb_collider(const aabb_collider& other)
      : collider<T>(other), m_center(other.m_center), m_size(other.m_size) {}
  aabb_collider(aabb_collider&& other)
      : collider<T>(other),
        m_center(std::move(other.m_center)),
        m_size(std::move(other.m_size)) {}

  virtual aabb_collider<T>& operator=(const aabb_collider& other) {
    collider<T>::operator=(other);
    m_center               = other.m_center;
    m_size                 = other.m_size;
    return *this;
  }

  virtual aabb_collider<T>& operator=(aabb_collider&& other) {
    collider<T>::operator=(other);
    m_center               = std::move(other.m_center);
    m_size                 = std::move(other.m_size);
    return *this;
  }

  const auto& size() const { return m_size; }
  const auto& size(size_t i) const { return m_size[i]; }

  auto&       center() { return m_center; }
  const auto& center() const { return m_center; }
  const auto& center(size_t i) const { return m_center[i]; }
  auto        transformed_center() {
    return this->object().transformation_matrix() * glm::vec4(m_center, 1);
  }

  T left() const { return transformed_center()[0] - m_size[0] * 0.5; }
  T right() const { return transformed_center()[0] + m_size[0] * 0.5; }

  T front() const { return transformed_center()[1] - m_size[1] * 0.5; }
  T back() const { return transformed_center()[1] + m_size[1] * 0.5; }

  T bottom() const { return transformed_center()[2] - m_size[2] * 0.5; }
  T top() const { return transformed_center()[2] + m_size[2] * 0.5; }

  virtual std::optional<collision<T>> check_collision(
      const ray<T>& r) override {
    T    shortest_dist = 1e10;
    auto tr_center     = transformed_center();
    T    le            = tr_center[0] - m_size[0] * 0.5;
    T    ri            = tr_center[0] + m_size[0] * 0.5;
    T    bo            = tr_center[1] - m_size[1] * 0.5;
    T    to            = tr_center[1] + m_size[1] * 0.5;
    T    fr            = tr_center[2] - m_size[2] * 0.5;
    T    ba            = tr_center[2] + m_size[2] * 0.5;

    T             left_dist = (le - r.x(0)) / r.dir(0);
    glm::tvec3<T> x_left    = r(left_dist);
    bool          left_hit  = (left_dist >= 0) && (x_left[1] >= bo) &&
                    (x_left[1] <= to) && (x_left[2] >= fr) && (x_left[2] <= ba);

    T             right_dist = (ri - r.x(0)) / r.dir(0);
    glm::tvec3<T> x_right    = r(right_dist);
    bool          right_hit  = (right_dist >= 0) && (x_right[1] >= bo) &&
                     (x_right[1] <= to) && (x_right[2] >= fr) &&
                     (x_right[2] <= ba);

    T             bottom_dist = (bo - r.x(1)) / r.dir(1);
    glm::tvec3<T> x_bottom    = r(bottom_dist);
    bool          bottom_hit  = (bottom_dist >= 0) && (x_bottom[0] >= le) &&
                      (x_bottom[0] <= ri) && (x_bottom[2] >= fr) &&
                      (x_bottom[2] <= ba);

    T             top_dist = (to - r.x(1)) / r.dir(1);
    glm::tvec3<T> x_top    = r(top_dist);
    bool top_hit = (top_dist >= 0) && (x_top[0] >= le) && (x_top[0] <= ri) &&
                   (x_top[2] >= fr) && (x_top[2] <= ba);

    T             front_dist = (fr - r.x(2)) / r.dir(2);
    glm::tvec3<T> x_front    = r(front_dist);
    bool          front_hit  = (front_dist >= 0) && (x_front[0] >= le) &&
                     (x_front[0] <= ri) && (x_front[1] >= bo) &&
                     (x_front[1] <= to);

    T             back_dist = (ba - r.x(2)) / r.dir(2);
    glm::tvec3<T> x_back    = r(back_dist);
    bool          back_hit  = (back_dist >= 0) && (x_back[0] >= le) &&
                    (x_back[0] <= ri) && (x_back[1] >= bo) && (x_back[1] <= to);

    if (left_hit || right_hit || bottom_hit || top_hit || front_hit ||
        back_hit) {
      if (left_hit && shortest_dist > left_dist) shortest_dist = left_dist;
      if (right_hit && shortest_dist > right_dist) shortest_dist = right_dist;
      if (bottom_hit && shortest_dist > bottom_dist)
        shortest_dist = bottom_dist;
      if (top_hit && shortest_dist > top_dist) shortest_dist = top_dist;
      if (front_hit && shortest_dist > front_dist) shortest_dist = front_dist;
      if (back_hit && shortest_dist > back_dist) shortest_dist = back_dist;
      return collision<T>(&this->object(), r(shortest_dist));
    }
    return {};
  }

 private:
  glm::tvec3<T> m_center;
  glm::tvec3<T> m_size;
};

template <typename collider>
struct is_collider {
  static constexpr bool value = false;
};

template <typename T>
struct is_collider<collider<T>> {
  static constexpr bool value = true;
};

template <typename T>
struct is_collider<aabb_collider<T>> {
  static constexpr bool value = true;
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
