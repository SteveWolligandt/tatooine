#ifndef TATOOINE_STEADIFICATION_RENDERERS_H
#define TATOOINE_STEADIFICATION_RENDERERS_H

#include <tatooine/streamsurface.h>

//==============================================================================
namespace tatooine::steadification {
//==============================================================================
struct streamsurface_renderer
    : rendering::gl::indexeddata<vec2f, vec2f, float, float,
                         float> {
  //============================================================================
  using parent_t = rendering::gl::indexeddata<vec2f, vec2f, float,
                                      float, float>;
  using typename parent_t::ibo_data_vec;
  using typename parent_t::vbo_data_vec;
  //============================================================================
  template <typename Real>
  streamsurface_renderer(const simple_tri_mesh<Real, 2>& mesh, const Real t0)
      : indexeddata{to_vbo_data(mesh, t0), to_ibo_data(mesh)} {}
  //----------------------------------------------------------------------------
  streamsurface_renderer(const streamsurface_renderer& other) = default;
  //----------------------------------------------------------------------------
  streamsurface_renderer(streamsurface_renderer&& other) = default;
  //============================================================================
  template <typename Real>
  static vbo_data_vec to_vbo_data(const simple_tri_mesh<Real, 2>& mesh,
                                  const Real t0) {
    using namespace boost;
    vbo_data_vec vbo_data;
    vbo_data.reserve(mesh.num_vertices());

    const auto& uv_prop = mesh.template vertex_property<vec<Real, 2>>("uv");
    const auto& vf_prop = mesh.template vertex_property<vec<Real, 2>>("v");
    const auto& curvature_prop =
        mesh.template vertex_property<Real>("curvature");

    boost::transform(
        mesh.vertices(), std::back_inserter(vbo_data), [&](auto v) {
          return vbo_data_t{
              vec2f{float(mesh[v](0)), float(mesh[v](1))},
              vec2f{float(vf_prop[v](0)), float(vf_prop[v](1))},
              float(uv_prop[v](1)), t0, float(curvature_prop[v])};
        });

    return vbo_data;
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  static ibo_data_vec to_ibo_data(const simple_tri_mesh<Real, 2>& mesh) {
    using namespace boost;
    using namespace adaptors;
    ibo_data_vec ibo_data;
    auto         is_triangle = [&mesh](auto f) { return mesh[f].size() == 3; };
    auto         get_face = [&mesh](auto f) -> const auto& { return mesh[f]; };

    for (const auto& vs :
         mesh.faces() | filtered(is_triangle) | transformed(get_face)) {
      copy(vs | transformed([](auto h) { return h.i; }),
           std::back_inserter(ibo_data));
    }
    return ibo_data;
  }
  //----------------------------------------------------------------------------
  void draw() const;
};
template <size_t N>
struct line_renderer : rendering::gl::indexeddata<vec<float, N>> {
  //============================================================================
  using parent_t = rendering::gl::indexeddata<vec<float, N>>;
  using typename parent_t::ibo_data_vec;
  using typename parent_t::vbo_data_vec;
  //============================================================================
  template <typename Real>
  line_renderer(const std::vector<line<Real, N>>& lines)
      : rendering::gl::indexeddata<vec<float, N>>{to_vbo_data(lines), to_ibo_data(lines)} {}
  //----------------------------------------------------------------------------
  line_renderer(const line_renderer& other) = default;
  //----------------------------------------------------------------------------
  line_renderer(line_renderer&& other) = default;
  //============================================================================
  template <typename Real, size_t... Is>
  static vbo_data_vec to_vbo_data(const std::vector<line<Real, N>>& lines,
                                  std::index_sequence<Is...> /*seq*/) {
    using namespace boost;
    vbo_data_vec vbo_data;

    for (const auto& line : lines) {
      for (const auto& v : line.vertices()) {
        vbo_data.push_back(
            typename parent_t::vbo_data_t{{static_cast<float>(v(Is))...}});
      }
    }
    return vbo_data;
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  static vbo_data_vec to_vbo_data(const std::vector<line<Real, N>>& lines) {
    return to_vbo_data(lines, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  static ibo_data_vec to_ibo_data(const std::vector<line<Real, N>>& lines) {
    ibo_data_vec ibo_data;
    size_t       j = 0;
    for (const auto& line : lines) {
      for (size_t i = 0; i < line.num_vertices() - 1; ++i, ++j) {
        ibo_data.push_back(j);
        ibo_data.push_back(j + 1);
      }
      ++j;
    }
    return ibo_data;
  }
  //----------------------------------------------------------------------------
  void draw() const { this->draw_lines(); }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
