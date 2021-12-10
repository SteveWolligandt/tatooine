#include <tatooine/demangling.h>
#include <tatooine/handle.h>
#include <tatooine/pointset.h>
#include <tatooine/iterator_facade.h>
#include <tatooine/vec.h>

#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct my_range {
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    std::size_t m_vh{};

    iterator() = default;  // ITERATORS NEED TO BE DEFAULT-CONSTRUCTIBLE!!!
    iterator(std::size_t const vh) : m_vh{vh} {}
    iterator(iterator const&)     = default;
    iterator(iterator&&) noexcept = default;
    auto operator=(iterator const&) -> iterator& = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;
    ~iterator()                                      = default;

    constexpr auto               increment() { ++m_vh; }
    constexpr auto               decrement() { --m_vh; }
    [[nodiscard]] auto           dereference() const { return m_vh; }
    [[nodiscard]] constexpr bool equal_to(iterator other) const {
      return m_vh == other.m_vh;
    }
    [[nodiscard]] auto at_end() const { return m_vh == 100; }
  };
  auto                  begin() { return iterator{std::size_t{0}}; }
  static constexpr auto end() { return typename iterator::sentinel_type{}; }
};
//==============================================================================
/// With this you can use rvalues with `operator|`
template <>
inline constexpr const bool std::ranges::enable_borrowed_range<my_range> = true;
//==============================================================================
auto main() -> int {
  auto v = std::vector<std::size_t>{};
  auto r = my_range{};
  for (auto i : r) {
    std::cout << i << '\n';
  }
  auto square = [](auto const x) { return x * x; };

  using namespace std::ranges;

  // use borrowed range
  copy(my_range{} | views::transform(square), std::back_inserter(v));
  copy(r | views::transform(square),
       std::back_inserter(v));

  for (auto i : v) {
    std::cout << i << '\n';
  }

  using it_traits = std::iterator_traits<my_range::iterator>;
  std::cout << "iterator_category: "
            << type_name<it_traits::iterator_category>() << '\n';
  std::cout << "reference:         " << type_name<it_traits::reference>()
            << '\n';
  std::cout << "pointer:           " << type_name<it_traits::pointer>() << '\n';
  std::cout << "value_type:        " << type_name<it_traits::value_type>()
            << '\n';
  std::cout << "difference_type:   " << type_name<it_traits::difference_type>()
            << '\n';
}
