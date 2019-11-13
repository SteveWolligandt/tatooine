#ifndef RENDERERS_H
#define RENDERERS_H

#include <tatooine/streamsurface.h>

#include <yavin>

//==============================================================================
namespace tatooine {
//==============================================================================
struct streamsurface_renderer
    : yavin::indexeddata<yavin::vec2, yavin::vec2, yavin::vec2> {
  //============================================================================
  using parent_t = yavin::indexeddata<yavin::vec2, yavin::vec2, yavin::vec2>;
  using vbo_data_vec = parent_t::vbo_data_vec;
  using ibo_data_vec = parent_t::ibo_data_vec;
  //============================================================================
  template <typename Real>
  streamsurface_renderer(const mesh<Real, 3>& mesh)
      : indexeddata{mesh_to_vbo_data(mesh), mesh_to_ibo_data(mesh)} {}
  //----------------------------------------------------------------------------
  streamsurface_renderer(const streamsurface_renderer& other) = default;
  //----------------------------------------------------------------------------
  streamsurface_renderer(streamsurface_renderer&& other) = default;
  //============================================================================
  template <typename Real>
  static vbo_data_vec mesh_to_vbo_data(const mesh<Real, 3>& mesh) {
    using namespace boost;
    vbo_data_vec vbo_data;
    vbo_data.reserve(mesh.num_vertices());
    const auto& uv_prop =
        mesh.template vertex_property<vec<Real, 2>>("uv");
    const auto& vf_prop =
        mesh.template vertex_property<vec<Real, 2>>("v");
    boost::transform(
        mesh.vertices(), std::back_inserter(vbo_data), [&](auto v) {
          return vbo_data_t{
              yavin::vec2{(float)mesh[v][0], (float)mesh[v][1]},
              yavin::vec2{(float)vf_prop[v][0], (float)vf_prop[v][1]},
              yavin::vec2{(float)uv_prop[v][0], (float)uv_prop[v][1]}};
        });

    return vbo_data;
  }
  //----------------------------------------------------------------------------
  template <typename Real>
  static ibo_data_vec mesh_to_ibo_data(const mesh<Real, 3>& mesh) {
    using namespace boost;
    using namespace adaptors;
    ibo_data_vec ibo_data;
    auto         is_triangle = [&mesh](auto f) { return mesh[f].size() == 3; };
    auto         is_quad     = [&mesh](auto f) { return mesh[f].size() == 4; };
    auto         triangle_filter = filtered(is_triangle);
    auto         quad_filter     = filtered(is_quad);
    auto         to_vertices =
        transformed([&mesh](auto f) -> const auto& { return mesh[f]; });
    auto to_indices   = transformed([](auto h) { return h.i; });
    auto quad_to_tris = [](const auto& quad) {
      return std::array{quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]};
    };
    auto triangles = mesh.faces() | triangle_filter | to_vertices;
    auto quads     = mesh.faces() | quad_filter | to_vertices;
    for (const auto& vs : triangles)
      copy(vs | to_indices, std::back_inserter(ibo_data));
    for (const auto& vs : quads)
      copy(quad_to_tris(vs) | to_indices, std::back_inserter(ibo_data));
    return ibo_data;
  }
  //----------------------------------------------------------------------------
  void draw() const;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
