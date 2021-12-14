#include <tatooine/demangling.h>
#include <tatooine/iterator_facade.h>
#include <tatooine/vec.h>

#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>
//==============================================================================
using namespace tatooine;
//==============================================================================
// Range that uses iterators that inherit from iterator_facade.
struct my_range {
  // Iterator implementation.
  // It uses a sentinel as end.
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    std::size_t m_vh{};

    // CTORS
    constexpr iterator() =
        default;  // ITERATORS NEED TO BE DEFAULT-CONSTRUCTIBLE!!!
    constexpr explicit iterator(std::size_t const vh) : m_vh{vh} {}
    constexpr iterator(iterator const&)     = default;
    constexpr iterator(iterator&&) noexcept = default;
    // Assign ops
    constexpr auto operator=(iterator const&) -> iterator& = default;
    constexpr auto operator=(iterator&&) noexcept -> iterator& = default;
    // DTOR
    ~iterator() = default;

    constexpr auto increment() { ++m_vh; }
    constexpr auto               decrement() { --m_vh; }
    [[nodiscard]] constexpr auto dereference() const { return m_vh; }
    [[nodiscard]] constexpr auto equal(iterator other) const {
      return m_vh == other.m_vh;
    }
    [[nodiscard]] constexpr auto at_end() const { return m_vh == 10; }
  };
  static constexpr auto begin() { return iterator{std::size_t{0}}; }
  static constexpr auto end() { return typename iterator::sentinel_type{}; }
};
//==============================================================================
// With this you can use rvalues with `operator|`
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
  copy(r | views::transform(square), std::back_inserter(v));

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
// OUTPUT:
// 0
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// 8
// 9
// 0
// 1
// 4
// 9
// 16
// 25
// 36
// 49
// 64
// 81
// 0
// 1
// 4
// 9
// 16
// 25
// 36
// 49
// 64
// 81
// iterator_category: std::bidirectional_iterator_tag
// reference:         unsigned long
// pointer:           tatooine::arrow_proxy<unsigned long>
// value_type:        unsigned long
// difference_type:   long
