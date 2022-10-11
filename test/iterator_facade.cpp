#include <tatooine/iterator_facade.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
struct forward_iterator_range_without_sentinel {
  class iterator : public iterator_facade<iterator> {
    long m_state;

   public:
    explicit iterator(long p_state) : m_state{p_state} {}

    explicit iterator()                              = default;
    iterator(iterator const&)                        = default;  // MANDATORY
    iterator(iterator&&) noexcept                    = default;
    auto operator=(iterator const&) -> iterator&     = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;

    constexpr auto dereference() const { return m_state; }  // MANDATORY
    constexpr auto increment() { ++m_state; }               // MANDATORY
    constexpr auto equal(iterator const& other) const {     // MANDATORY
      return m_state == other.m_state;
    }
  };

  static auto begin() { return iterator{0}; }
  static auto end() { return iterator{4}; }
};
//==============================================================================
struct forward_iterator_range_with_sentinel {
  class iterator : public iterator_facade<iterator> {
    long m_state;

   public:
    using sentinel_type = iterator_sentinel;  // MANDATORY FOR SENTINEL SUPPORT
    explicit iterator(long p_state) : m_state{p_state} {}

    explicit iterator()                              = default;  // MANDATORY
    iterator(iterator const&)                        = default;
    iterator(iterator&&) noexcept                    = default;
    auto operator=(iterator const&) -> iterator&     = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;

    constexpr auto dereference() const { return m_state; }  // MANDATORY
    constexpr auto increment() { ++m_state; }               // MANDATORY
    constexpr auto equal(iterator const& other) const {     // MANDATORY
      return m_state == other.m_state;
    }
    auto at_end() const {  // MANDATORY FOR SENTINEL SUPPORT
      return m_state > 3;
    }
  };

  static auto begin() { return iterator{0}; }
  static auto end() { return iterator::sentinel_type{}; }
};
//==============================================================================
struct bidirectional_iterator_range_without_sentinel {
  class iterator : public iterator_facade<iterator> {
    long m_state;

   public:
    explicit iterator(long p_state) : m_state{p_state} {}

    explicit iterator()                              = default;
    iterator(iterator const&)                        = default;  // MANDATORY
    iterator(iterator&&) noexcept                    = default;
    auto operator=(iterator const&) -> iterator&     = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;

    constexpr auto dereference() const { return m_state; }  // MANDATORY
    constexpr auto increment() { ++m_state; }               // MANDATORY
    constexpr auto decrement() {  // MANDATORY FOR BEING BIDIRECTIONAL
      --m_state;
    }
    constexpr auto equal(iterator const& other) const {  // MANDATORY
      return m_state == other.m_state;
    }
  };

  static auto begin() { return iterator{0}; }
  static auto end() { return iterator{4}; }
};
//==============================================================================
struct bidirectional_iterator_range_with_sentinel {
  class iterator : public iterator_facade<iterator> {
    long m_state;

   public:
    using sentinel_type = iterator_sentinel;  // MANDATORY FOR SENTINEL SUPPORT
    explicit iterator(long p_state) : m_state{p_state} {}

    explicit iterator()                              = default;  // MANDATORY
    iterator(iterator const&)                        = default;
    iterator(iterator&&) noexcept                    = default;
    auto operator=(iterator const&) -> iterator&     = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;

    constexpr auto dereference() const { return m_state; }  // MANDATORY
    constexpr auto increment() { ++m_state; }               // MANDATORY
    constexpr auto decrement() {  // MANDATORY FOR BEING BIDIRECTIONAL
      --m_state;
    }
    constexpr auto equal(iterator const& other) const {  // MANDATORY
      return m_state == other.m_state;
    }
    auto at_end() const {  // MANDATORY FOR SENTINEL SUPPORT
      return m_state > 3;
    }
  };

  static auto begin() { return iterator{0}; }
  static auto end() { return iterator::sentinel_type{}; }
};
//==============================================================================
struct bidirectional_iterator_range_with_distance_to_and_advance {
  static constexpr std::size_t size = 4;
  class iterator : public iterator_facade<iterator> {
    long m_state;

   public:
    using sentinel_type = iterator_sentinel;  // MANDATORY FOR SENTINEL SUPPORT
    explicit iterator(long p_state) : m_state{p_state} {}

    explicit iterator()                              = default;  // MANDATORY
    iterator(iterator const&)                        = default;
    iterator(iterator&&) noexcept                    = default;
    auto operator=(iterator const&) -> iterator&     = default;
    auto operator=(iterator&&) noexcept -> iterator& = default;

    constexpr auto dereference() const { return m_state; }  // MANDATORY
    constexpr auto increment() { ++m_state; }               // MANDATORY
    constexpr auto decrement() {  // MANDATORY FOR BEING BIDIRECTIONAL
      --m_state;
    }
    constexpr auto equal(iterator const& other) const {  // MANDATORY
      return m_state == other.m_state;
    }
    auto at_end() const {  // MANDATORY FOR SENTINEL SUPPORT
      return m_state >= static_cast<long>(size);
    }
    auto distance_to(iterator const& other) const
        -> std::ptrdiff_t {  // MANDATORY FOR RANDOM ACCESS
      return other.m_state - m_state;
    }
    constexpr auto distance_to(sentinel_type const /*sentinel*/) const
        -> std::ptrdiff_t {  // MANDATORY FOR RANDOM ACCESS WITH SENTINEL
      return size - m_state;
    }
    constexpr auto advance(
        std::ptrdiff_t const off) {  // MANDATORY FOR RANDOM ACCESS

      m_state += off;
    }
  };

  static auto begin() { return iterator{0}; }
  static auto end() { return iterator::sentinel_type{}; }
};
//==============================================================================
TEMPLATE_TEST_CASE("iterator_facade_forward", "[iterator_facade][forward]",
                   forward_iterator_range_without_sentinel,
                   forward_iterator_range_with_sentinel,
                   bidirectional_iterator_range_without_sentinel,
                   bidirectional_iterator_range_with_sentinel) {
  using range_type = TestType;
  auto const range = range_type{};
  using Catch::Matchers::Equals;
  SECTION("incrementing") {
    auto it = range.begin();
    REQUIRE(*it == 0);
    ++it;
    REQUIRE(*it == 1);
  }
  SECTION("copy range") {
    STATIC_REQUIRE(std::ranges::range<range_type>);
    auto const expected_copied_elems = std::vector<int>{0, 1, 2, 3};
    auto       actual_copied_elems   = std::vector<int>{};
    std::ranges::copy(range, std::back_inserter(actual_copied_elems));
    REQUIRE_THAT(actual_copied_elems, Equals(expected_copied_elems));
  }
}
//==============================================================================
TEMPLATE_TEST_CASE("iterator_facade_backward", "[iterator_facade][backward]",
                   bidirectional_iterator_range_without_sentinel,
                   bidirectional_iterator_range_with_sentinel) {
  using range_type = TestType;
  auto const range = range_type{};
  SECTION("decrementing") {
    auto it = ++range.begin();
    REQUIRE(*it == 1);
    --it;
    REQUIRE(*it == 0);
  }
}
//==============================================================================
TEMPLATE_TEST_CASE("iterator_facade_random_access",
                   "[iterator_facade][random_access]",
                   bidirectional_iterator_range_with_distance_to_and_advance) {
  using range_type = TestType;
  auto const range = range_type{};
  auto       it    = begin(range);

  auto vec = std::vector<long>{};
  std::ranges::copy(range, back_inserter(vec));

  REQUIRE(it[0] == 0);
  REQUIRE(it[1] == 1);
  REQUIRE(it[2] == 2);
  REQUIRE(it[3] == 3);
  REQUIRE(*it == 0);
  it += 2;
  REQUIRE(*it == 2);
  it -= 1;
  REQUIRE(*it == 1);
  advance(it);
  REQUIRE(*it == 2);
  advance(it, -2);
  REQUIRE(*it == 0);

  REQUIRE(next(begin(range)) - begin(range) == next(begin(vec)) - begin(vec));
  REQUIRE(next(begin(range), 2) - begin(range) == next(begin(vec), 2) - begin(vec));
  //REQUIRE(end(range) - begin(range) == end(vec) - begin(vec));
  //REQUIRE(begin(range) - end(range) == begin(vec) - end(vec));
  SECTION("distance") {
    auto range_it = begin(range);
    auto vec_it   = begin(vec);

    REQUIRE(distance(range_it, next(range_it)) ==
            distance(vec_it, next(vec_it)));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
