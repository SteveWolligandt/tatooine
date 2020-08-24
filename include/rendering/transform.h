#ifndef YAVIN_TRANSFORM_H
#define YAVIN_TRANSFORM_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <iostream>
#include "vec.h"
#include "mat.h"

//==============================================================================
namespace yavin {
//==============================================================================
class transform {
 protected:
  mat4  m_matrix;

 public:
  transform() : m_matrix{mat4::eye()} {}
  transform(const transform& other)
      : m_matrix(other.m_matrix) {}
  transform(transform&& other)
      : m_matrix(std::move(other.m_matrix)) {}
  virtual ~transform() {}

  mat4& matrix() { return m_matrix; }

  const mat4& matrix() const { return m_matrix; }

  void look_at(const vec3& eye, const vec3& center,
               const vec3& up = {0, 1, 0}) {
    m_matrix = *inverse(look_at_matrix(eye, center, up));
  }

  void translate(const vec3& t) {
    m_matrix = translation_matrix(t) * m_matrix;
  }
  void translate(const float x, const float y, const float z) {
    m_matrix = translation_matrix(x,y,z) * m_matrix;
  }
  vec3 translation() {
    return {m_matrix(0, 3), m_matrix(1, 3), m_matrix(2, 3)};
  }

  void scale(const vec3& s) {
    m_matrix = scale_matrix(s) * m_matrix;
  }
  void scale(const float x, const float y, const float z) {
    m_matrix = scale_matrix(x, y, z) * m_matrix;
  }
  void scale(const float s) {
    m_matrix = scale_matrix(s) * m_matrix ;
  }

  void rotate(const float angle, const vec3& axis) {
    m_matrix = rotation_matrix(angle, axis) * m_matrix;
  }
  void rotate(const float angle, const float axis_x, const float axis_y,
              const float axis_z) {
    m_matrix = rotation_matrix(angle, axis_x, axis_y, axis_z) * m_matrix;
  }
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
