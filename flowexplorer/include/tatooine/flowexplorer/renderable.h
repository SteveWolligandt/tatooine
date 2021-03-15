#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <chrono>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/ray.h>
#include <tatooine/mat.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
namespace base {
struct renderable : ui::base::node {
  bool m_picked = false;
  //============================================================================
  renderable(std::string const& title, flowexplorer::scene& s);
  template <typename T>
  renderable(std::string const& title, flowexplorer::scene& s, T& ref)
      : ui::base::node{title, s, ref} {}
  //renderable(renderable const& w)                        = default;
  //renderable(renderable&& w) noexcept                    = default;
  //auto operator=(renderable const& w) -> renderable&     = default;
  //auto operator=(renderable&& w) noexcept -> renderable& = default;
  virtual ~renderable()                                  = default;

  virtual auto update(std::chrono::duration<double> const& /*dt*/) -> void {}
  virtual auto render(mat<float, 4, 4> const& projection_matrix,
                      mat<float, 4, 4> const& view_matrix) -> void = 0;
  virtual auto is_transparent() const -> bool { return false; }
  virtual auto on_mouse_drag(int /*offset_x*/, int /*offset_y*/) -> bool {
    return false;
  }
  auto         on_mouse_clicked() -> void { m_picked = true; }
  auto         on_mouse_released() -> void { m_picked = false; }
  auto         is_picked() -> bool { return m_picked; }
  virtual auto check_intersection(ray<float, 3> const& /*r*/) const -> bool {
    return false;
  }
};
}  // namespace base
template <typename Child>
struct renderable : base::renderable, ui::node_serializer<Child> {
  using base::renderable::renderable;
  using serializer_t = ui::node_serializer<Child>; 
  //============================================================================
  auto serialize() const -> toml::table override{
    return serializer_t::serialize(*dynamic_cast<Child const*>(this));
  }
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialization) -> void override {
    return serializer_t::deserialize(*dynamic_cast<Child*>(this),
                                               serialization);
  }
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    return serializer_t::draw_properties(*dynamic_cast<Child*>(this));
  }
  //----------------------------------------------------------------------------
  auto type_name() const -> std::string_view override {
    return serializer_t::type_name();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
