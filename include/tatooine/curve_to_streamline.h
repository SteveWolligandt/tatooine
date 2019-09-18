#ifndef TATOOINE_CURVE_TO_STREAMLINE_H
#define TATOOINE_CURVE_TO_STREAMLINE_H

#include <tatooine/field.h>

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real>
struct curve_to_streamline {
  //============================================================================
  using vec2  = vec<Real, 2>;
  using vec3  = vec<Real, 3>;
  using mat32 = mat<Real, 3, 2>;

  //============================================================================
  std::vector<mat32> bases;
  std::vector<vec2>  grad_alpha;
  std::vector<vec3>  tangents;

  //============================================================================
  template <typename V, typename Line>
  auto operator()(const field<V, Real, 3, 3>& v, Real t0, Line line,
                  Real initial_stepsize, Real delta, size_t num_its) {
    bases.resize(line.size());
    grad_alpha.resize(line.size());
    tangents.resize(line.size());

    Real cur_stepsize = initial_stepsize;
    for (unsigned int i = 0; i < num_its; i++) {
      calc_tangents(line);
      calc_plane_basis();
      redistribute_points(line);
      calc_gradient_alpha(v, t0, line);
      step(line, cur_stepsize);
      cur_stepsize *= delta;
    }
    return line;
  }

  //---------------------------------------------------------------------------
  template <typename Line>
  void step(Line& line, Real stepsize) {
    for (size_t i = 0; i < line.size(); i++) {
      auto refinement_dir = bases[i] * grad_alpha[i] * stepsize;
      line.vertex_at(i) -= refinement_dir;
    }
  }

  //----------------------------------------------------------------------------
  template <typename Line>
  void calc_tangents(const Line& line) {
    // for (unsigned int i = 0; i < line.size(); i++) {
    //  tangents[i] = normalize(line.tangent(i));
    //}
    for (unsigned int i = 0; i < line.size(); i++) {
      if (i == 0) {
        tangents[i] = line.vertex_at(i + 1) - line.vertex_at(i);
      } else if (i == line.size() - 1) {
        tangents[i] = line.vertex_at(i) - line.vertex_at(i - 1);
      } else {
        tangents[i] = line.vertex_at(i + 1) - line.vertex_at(i - 1);
      }
    }
  }

  //------------------------------------------------------------------------------
  void calc_plane_basis() {
    for (size_t i = 0; i < tangents.size(); i++) {
      const auto& t = tangents[i];
      vec3        aux_vec{0, 1, 0};
      auto        tn = normalize(t);
      if (approx_equal(aux_vec, tn, 1e-10)) { aux_vec = {1, 0, 0}; }
      if (approx_equal(aux_vec, tn, 1e-10)) { aux_vec = {0, 0, 1}; }
      while (approx_equal(aux_vec, tn, 1e-10)) {
        aux_vec = normalize(vec3::randu());
      }
      bases[i].col(0) = normalize(cross(aux_vec, t));
      bases[i].col(1) = normalize(cross(t, bases[i].col(0)));
    }
  }

  //------------------------------------------------------------------------------
  template <typename Line>
  void redistribute_points(Line& line) const {
    auto         redistributed_points = line.vertices();
    const size_t start_idx            = (line.is_closed() ? 0 : 1);
    const size_t end_idx = (line.is_closed() ? line.size() : (line.size() - 1));

    for (size_t i = start_idx; i < end_idx; ++i) {
      const auto center = (line.vertex_at(i + 1) + line.vertex_at(i - 1)) * 0.5;
      const auto& t     = tangents[i];
      auto correction   = t * (dot(center - line.vertex_at(i), t) / dot(t, t));
      redistributed_points[i] += correction;
    }
    line.vertices() = std::move(redistributed_points);
  }

  //------------------------------------------------------------------------------
  template <typename V, typename Line>
  void calc_gradient_alpha(const V& v, Real t0, const Line& line) {
    for (size_t i = 0; i < line.size(); ++i) {
      const auto& x = line.vertex_at(i);

      const Real offset = 1e-6;

      const auto v_x_pos = normalize(v(x + bases[i].col(0) * offset, t0));
      const auto v_x_neg = normalize(v(x - bases[i].col(0) * offset, t0));
      const auto v_y_pos = normalize(v(x + bases[i].col(1) * offset, t0));
      const auto v_y_neg = normalize(v(x - bases[i].col(1) * offset, t0));

      const auto        tn          = normalize(tangents[i]);
      auto              alpha_x_pos = min_angle(tn, v_x_pos);
      auto              alpha_x_neg = min_angle(tn, v_x_neg);
      auto              alpha_y_pos = min_angle(tn, v_y_pos);
      auto              alpha_y_neg = min_angle(tn, v_y_neg);

      if (std::isnan(alpha_x_pos)) { alpha_x_pos = 0; }
      if (std::isnan(alpha_x_neg)) { alpha_x_neg = 0; }
      if (std::isnan(alpha_y_pos)) { alpha_y_pos = 0; }
      if (std::isnan(alpha_y_neg)) { alpha_y_neg = 0; }

      grad_alpha[i] = {(alpha_x_pos - alpha_x_neg) / (offset * 2),
                       (alpha_y_pos - alpha_y_neg) / (offset * 2)};
    }
  }
};

curve_to_streamline()->curve_to_streamline<double>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
