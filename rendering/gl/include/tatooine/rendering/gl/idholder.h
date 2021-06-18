#ifndef YAVIN_ID_HOLDER_H
#define YAVIN_ID_HOLDER_H
//==============================================================================
#include <utility>
#include "glincludes.h"
//==============================================================================
namespace yavin {
//==============================================================================
template <typename ID>
struct id_holder_default_param;

//==============================================================================
template <>
struct id_holder_default_param<GLuint> {
  constexpr static GLuint value = 0;
};

//==============================================================================
template <>
struct id_holder_default_param<GLint> {
  constexpr static GLuint value = -1;
};

//==============================================================================
template <typename ID>
static constexpr auto id_holder_default_param_v =
    id_holder_default_param<ID>::value;

//==============================================================================
template <typename ID>
struct id_holder {
  static constexpr auto default_val = id_holder_default_param_v<ID>;

 private:
  //----------------------------------------------------------------------------
  ID m_id;

 public:
  //----------------------------------------------------------------------------
  id_holder() : m_id{default_val} {}
  //----------------------------------------------------------------------------
  explicit id_holder(ID _id) : m_id{_id} {}
  //----------------------------------------------------------------------------
  id_holder(id_holder const& other) = delete;
  id_holder(id_holder&& other) noexcept
      : m_id{std::exchange(other.m_id, default_val)} {}
  //----------------------------------------------------------------------------
  auto operator=(id_holder const& other) -> id_holder& = delete;
  auto operator=(id_holder&& other) noexcept -> id_holder& {
    swap(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  ~id_holder() = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto id() const { return m_id; }
  //----------------------------------------------------------------------------
 protected:
  void set_id(ID id) { m_id = id; }
  //----------------------------------------------------------------------------
  auto id_ptr() { return &m_id; }
  auto id_ref() -> auto& { return m_id; }
  //----------------------------------------------------------------------------
 public:
  void swap(id_holder& other) { std::swap(m_id, other.m_id); }
};
//==============================================================================
template <typename ID>
void swap(id_holder<ID>& i0, id_holder<ID>& i1) {
  i0.swap(i1);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
