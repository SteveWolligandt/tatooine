#ifndef TATOOINE_FLOWEXPLORER_NODES_VECTORFIELD_TO_GPU_H
#define TATOOINE_FLOWEXPLORER_NODES_VECTORFIELD_TO_GPU_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/yavin_interop.h>
#include <tatooine/field.h>
#include <tatooine/gpu/upload.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct vectorfield_to_gpu : ui::node<vectorfield_to_gpu>, yavin::tex2rg32f {
  using vectorfield_t = parent::vectorfield<double, 2>;
  using bb_t          = flowexplorer::nodes::axis_aligned_bounding_box<2>;
  using tex_t         = yavin::tex2rg32f;
  //----------------------------------------------------------------------------
  vectorfield_t* m_v  = nullptr;
  bb_t*          m_bb = nullptr;
  vec<int, 2> m_res;
  real_t m_t;
  //----------------------------------------------------------------------------
  auto bounding_box() const { return m_bb; }
  //----------------------------------------------------------------------------
  auto resolution() const -> auto const& { return m_res; }
  auto resolution() -> auto& { return m_res; }
  //----------------------------------------------------------------------------
  auto time() const { return m_t; }
  auto time() -> auto& { return m_t; }
  //----------------------------------------------------------------------------
  vectorfield_to_gpu(flowexplorer::scene& s)
      : ui::node<vectorfield_to_gpu>{"Vectorfield to GPU", s, typeid(vectorfield_to_gpu)},
        m_res{100, 100} {
    insert_input_pin<vectorfield_t>("2D Vector Field");
    insert_input_pin<bb_t>("2D Bounding Box");
  }
  virtual ~vectorfield_to_gpu() = default;
  //----------------------------------------------------------------------------
  auto upload() {
    *dynamic_cast<tex_t*>(this) = gpu::upload_tex<float>(
        sample_to_vector(
            *m_v,
            uniform_grid<real_t, 2>{
                linspace{m_bb->min(0), m_bb->max(0), static_cast<size_t>(m_res->at(0))},
                linspace{m_bb->min(1), m_bb->max(1), static_cast<size_t>(m_res->at(1))}},
            m_t),
        m_res(0), m_res(1));
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::input_pin&  this_pin,
                        ui::output_pin& other_pin) override {
    if (std::find(begin(this_pin.types()), end(this_pin.types()),
                  &typeid(bb_t)) != end(this_pin.types())) {
      m_bb = dynamic_cast<bb_t*>(&other_pin.node());
    } else if (std::find(begin(this_pin.types()), end(this_pin.types()),
                         &typeid(vectorfield_t)) != end(this_pin.types())) {
      m_v = dynamic_cast<vectorfield_t*>(&other_pin.node());
    }
    if (m_bb != nullptr && m_v != nullptr) {
      upload();
    }
  }
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void override {
    if (m_v != nullptr && m_bb != nullptr) {
      upload();
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::vectorfield_to_gpu,
    TATOOINE_REFLECTION_INSERT_GETTER(resolution),
    TATOOINE_REFLECTION_INSERT_GETTER(time));
#endif
