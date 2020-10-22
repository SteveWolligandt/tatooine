#ifndef TATOOINE_FLOWEXPLORER_RENDERABLE_H
#define TATOOINE_FLOWEXPLORER_RENDERABLE_H
//==============================================================================
#include <chrono>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/mat.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
namespace base {
struct renderable : ui::base::node {
  renderable(std::string const& title, flowexplorer::scene& s);
  renderable(renderable const& w)                        = default;
  renderable(renderable&& w) noexcept                    = default;
  auto operator=(renderable const& w) -> renderable&     = default;
  auto operator=(renderable&& w) noexcept -> renderable& = default;
  virtual ~renderable()                                  = default;

  virtual void        update(const std::chrono::duration<double>& dt) {}
  virtual void        render(const mat<float, 4, 4>& projection_matrix,
                             const mat<float, 4, 4>& view_matrix) = 0;
  virtual bool        is_transparent() const                      = 0;
};
}  // namespace base
template <typename Child>
struct renderable : base::renderable, ui::node_serializer<Child> {
  using base::renderable::renderable;
  //============================================================================
  auto serialize() const -> toml::table override{
    return ui::node_serializer<Child>::serialize();
  }
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialization) -> void override {
    return ui::node_serializer<Child>::deserialize(serialization);
  }
  //----------------------------------------------------------------------------
  auto draw_ui() -> void override {
    return ui::node_serializer<Child>::draw_ui();
  }
  //----------------------------------------------------------------------------
  auto type_name() const -> std::string_view override {
    return ui::node_serializer<Child>::type_name();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
