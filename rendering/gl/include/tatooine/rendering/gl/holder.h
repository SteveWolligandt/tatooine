#ifndef YAVIN_HOLDER_H
#define YAVIN_HOLDER_H
//==============================================================================
#include <memory>
//==============================================================================
namespace yavin {
//==============================================================================
struct base_holder {
  virtual ~base_holder() = default;
};

template <typename T>
struct holder : base_holder {
  using held_type = T;
  //----------------------------------------------------------------------------
 protected:
  T m_held_object;
  //----------------------------------------------------------------------------
 public:
  template <typename _T>
  holder(_T&& obj) : m_held_object{std::forward<_T>(obj)} {}
  virtual ~holder() = default;
  //----------------------------------------------------------------------------
  const auto& get() const { return m_held_object; }
  auto&       get() { return m_held_object; }
};

// copy when having rvalue
template <typename T>
holder(T &&)->holder<T>;

// keep reference when having lvalue
template <typename T>
holder(const T&)->holder<const T&>;

template <typename T>
auto hold(T&& t) {
  return std::unique_ptr<base_holder>{new holder{std::forward<T>(t)}};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
