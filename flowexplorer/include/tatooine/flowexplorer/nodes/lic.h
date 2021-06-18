#ifndef TATOOINE_FLOWEXPLORER_NODES_LIC_H
#define TATOOINE_FLOWEXPLORER_NODES_LIC_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/gpu/texture_shader.h>
#include <tatooine/rendering/matrices.h>
#include <tatooine/flowexplorer/nodes/vectorfield_to_gpu.h>
#include <tatooine/rendering/gl/texture.h>
#include <tatooine/rendering/gl/indexdata.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct lic : renderable<lic> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = vectorfield_to_gpu;
  using bb_t          = flowexplorer::nodes::axis_aligned_bounding_box<2>;

  //----------------------------------------------------------------------------
  // attributes
  //----------------------------------------------------------------------------
 private:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // pins
  vectorfield_t* m_v  = nullptr;
  bb_t*          m_bb = nullptr;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // internal
  bool                                 m_calculating          = false;
  bool                                 m_needs_another_update = false;
  std::unique_ptr<gpu::texture_shader> m_shader;
  std::unique_ptr<rendering::gl::tex2rgba<float>>                 m_lic_tex;
  rendering::gl::indexeddata<vec<float, 2>, vec<float, 2>, float> m_quad;
  std::mutex                                                      m_mutex;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // user data
  vec<int, 2> m_lic_res;
  int         m_num_samples;
  double      m_stepsize;
  float       m_alpha;
  std::string m_seed_str;

 public:
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  lic(flowexplorer::scene& s);

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  auto init() -> void;
  auto write_png() -> void;
  //----------------------------------------------------------------------------
  auto setup_pins() -> void;
  //----------------------------------------------------------------------------
  auto setup_quad() -> void;
  //----------------------------------------------------------------------------
  auto render(mat4f const& projection_matrix, mat4f const& view_matrix)
      -> void override;
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override;
  //----------------------------------------------------------------------------
  auto calculate_lic() -> void;
  //----------------------------------------------------------------------------
  auto update_shader(mat4f const& projection_matrix, mat4f const& view_matrix)
      -> void;
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::input_pin&  this_pin,
                        ui::output_pin& other_pin) override;
  //----------------------------------------------------------------------------
  void on_pin_disconnected(ui::input_pin& this_pin) override;
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override;
  //----------------------------------------------------------------------------
  // setters / getters
  //----------------------------------------------------------------------------
  auto lic_res() -> auto& { return m_lic_res; }
  auto lic_res() const -> auto const& { return m_lic_res; }
  //----------------------------------------------------------------------------
  auto num_samples() -> auto& { return m_num_samples; }
  auto num_samples() const { return m_num_samples; }
  //----------------------------------------------------------------------------
  auto stepsize() -> auto& { return m_stepsize; }
  auto stepsize() const { return m_stepsize; }
  //----------------------------------------------------------------------------
  auto alpha() -> auto& { return m_alpha; }
  auto alpha() const { return m_alpha; }
  //----------------------------------------------------------------------------
  auto seed() -> auto& { return m_seed_str; }
  auto seed() const -> auto const& { return m_seed_str; }
  //----------------------------------------------------------------------------
  auto update(std::chrono::duration<double> const& /*dt*/) -> void override;
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    auto const changed = renderable<lic>::draw_properties();

    if (m_lic_tex) {
      if (ImGui::Button("write png")) {
        write_png();
      }
    }

    return changed;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::lic,
    TATOOINE_REFLECTION_INSERT_METHOD(resolution, lic_res()),
    TATOOINE_REFLECTION_INSERT_GETTER(num_samples),
    TATOOINE_REFLECTION_INSERT_GETTER(stepsize),
    TATOOINE_REFLECTION_INSERT_GETTER(alpha),
    TATOOINE_REFLECTION_INSERT_GETTER(seed))
#endif
