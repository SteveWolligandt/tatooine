#include<tatooine/reflection.h>
#include<array>
#include<iostream>
//==============================================================================
using namespace tatooine;
//==============================================================================
class S {
  public:
  std::array<int, 5> m_a;
  float              m_b;
  double             m_c;

 public:
  S() = default;
  S(std::array<int, 5>const& a, float const b, double const c):m_a{a}, m_b{b}, m_c{c} {}

  [[nodiscard]] auto a() const -> auto const& { return m_a; }
  [[nodiscard]] auto a(size_t const i) const { return m_a[i]; }
  [[nodiscard]] auto b() const { return m_b; }
  [[nodiscard]] auto c() const { return m_c; }

  auto set_a(std::array<int, 5> const& a) { m_a = a; }
  auto set_a(size_t const i, int const a) { m_a[i] = a; }
  auto set_b(float const b) { m_b = b; }
  auto set_c(float const c) { m_c = c; }
};
//==============================================================================
TATOOINE_MAKE_REFLECTABLE(S,
                          TATOOINE_REFLECTION_INSERT_METHOD(a, m_a));
//==============================================================================
auto main() -> int {
  S s;
  std::cout << "number of fields: " << reflection::field_count<S>() << '\n';
}
