#ifndef TATOOINE_FLOWEXPLORER_NODES_LIC_H
#define TATOOINE_FLOWEXPLORER_NODES_LIC_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/boundingbox.h>
#include <tatooine/gpu/texture_shader.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/matrices.h>
#include <yavin>
#include <tatooine/rendering/yavin_interop.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct lic : renderable<lic> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::vectorfield<double, 2>;
  using bb_t          = flowexplorer::nodes::boundingbox<2>;

 private:
  //----------------------------------------------------------------------------
  vectorfield_t const*                    m_v           = nullptr;
  bb_t*                                   m_boundingbox = nullptr;
  bool                                    m_calculating = false;
  std::unique_ptr<gpu::texture_shader>    m_shader;
  std::unique_ptr<yavin::tex2rgba<float>> m_lic_tex;
  yavin::indexeddata<vec<float, 2>, vec<float, 2>, float> m_quad;
  vec<int, 2>                                             m_lic_res;
  vec<int, 2> m_vectorfield_sample_res;
  double      m_t;
  int         m_num_samples;
  double      m_stepsize;
  float       m_alpha;

 public:
  //----------------------------------------------------------------------------
  lic(flowexplorer::scene& s);
  //----------------------------------------------------------------------------
  lic(lic const& other);
  //============================================================================
  auto init() -> void;
  //----------------------------------------------------------------------------
  auto setup_pins() -> void;
  //----------------------------------------------------------------------------
  auto setup_quad() -> void;
  //----------------------------------------------------------------------------
  auto render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) -> void override;
  //----------------------------------------------------------------------------
  auto update(const std::chrono::duration<double>&) -> void override {}
  //----------------------------------------------------------------------------
  auto calculate_lic() -> void;
  //----------------------------------------------------------------------------
  auto update_shader(mat<float, 4, 4> const& projection_matrix,
                     mat<float, 4, 4> const& view_matrix) -> void;
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override ;
  //----------------------------------------------------------------------------
  void on_pin_disconnected(ui::pin& this_pin) override ;
  //----------------------------------------------------------------------------
  auto           is_transparent() const -> bool override;
  //============================================================================
  auto lic_res() -> auto& {
    return m_lic_res;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto lic_res() const -> auto const& {
    return m_lic_res;
  }
  //----------------------------------------------------------------------------
  auto vectorfield_sample_res() -> auto& {
    return m_vectorfield_sample_res;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vectorfield_sample_res() const -> auto const& {
    return m_vectorfield_sample_res;
  }
  //----------------------------------------------------------------------------
  auto t() -> auto& {
    return m_t;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto t() const {
    return m_t;
  }
  //----------------------------------------------------------------------------
  auto num_samples() -> auto& {
    return m_num_samples;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto num_samples() const {
    return m_num_samples;
  }
  //----------------------------------------------------------------------------
  auto stepsize() -> auto& {
    return m_stepsize;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto stepsize() const {
    return m_stepsize;
  }
  //----------------------------------------------------------------------------
  auto alpha() -> auto& {
    return m_alpha;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto alpha() const {
    return m_alpha;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::lic,
              TATOOINE_REFLECTION_INSERT_METHOD(resolution, lic_res()),
              TATOOINE_REFLECTION_INSERT_GETTER(vectorfield_sample_res),
              TATOOINE_REFLECTION_INSERT_GETTER(t),
              TATOOINE_REFLECTION_INSERT_GETTER(num_samples),
              TATOOINE_REFLECTION_INSERT_GETTER(stepsize),
              TATOOINE_REFLECTION_INSERT_GETTER(alpha))
#endif
