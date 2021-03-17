#ifndef TATOOINE_FLOWEXPLORER_TOGGLEABLE_H
#define TATOOINE_FLOWEXPLORER_TOGGLEABLE_H
//==============================================================================
namespace tatooine::flowexplorer{
//==============================================================================
struct toggleable {
 private:
  bool m_active;

 public:
  explicit constexpr toggleable(bool active = true) : m_active{active} {}
  constexpr toggleable(toggleable const &) = default;
  constexpr toggleable(toggleable &&)      = default;
  constexpr auto operator=(toggleable const &) -> toggleable& = default;
  constexpr auto operator=(toggleable &&) -> toggleable& = default;
  ~toggleable() = default;

  constexpr virtual auto set_active(bool active = true) -> void {
    m_active = active;
  }
  constexpr auto activate() -> void { set_active(true); }
  constexpr auto deactivate() -> void { set_active(false); }
  constexpr auto toggle() -> void { set_active(!is_active()); }
  constexpr auto is_active() const -> bool const & { return m_active; }
  constexpr auto is_active() -> bool & { return m_active; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
