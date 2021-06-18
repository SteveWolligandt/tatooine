#ifndef YAVIN_SHADERSOURCE_H
#define YAVIN_SHADERSOURCE_H
//==============================================================================
#include <string>
//==============================================================================
namespace yavin {
//==============================================================================
struct shadersource {
 private:
  std::string m_source;

 public:
  explicit shadersource(std::string const& src) : m_source{src} {}
  explicit shadersource(std::string_view const& src) : m_source{src} {}
  explicit shadersource(char const* src) : m_source{src} {}
  //----------------------------------------------------------------------------
  shadersource()                                           = default;
  shadersource(shadersource const&)                        = default;
  shadersource(shadersource&&) noexcept                    = default;
  //----------------------------------------------------------------------------
  auto operator=(shadersource const&) -> shadersource&     = default;
  auto operator=(shadersource&&) noexcept -> shadersource& = default;
  //----------------------------------------------------------------------------
  ~shadersource()                                          = default;
  //============================================================================
  auto string() const -> auto const& { return m_source; }
  auto string() -> auto& { return m_source; }
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
