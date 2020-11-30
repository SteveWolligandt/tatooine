#ifndef TATOOINE_MULTIDIM_ARRAY_H
#define TATOOINE_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_ordering.h>
#include <tatooine/linspace.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_size.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>
#include <tatooine/make_array.h>

#include <array>
#include <png++/png.hpp>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Indexing, typename MemLoc, size_t... Resolution>
class static_multidim_array
    : public static_multidim_size<Indexing, Resolution...> {
  //============================================================================
  // assertions
  //============================================================================
  static_assert(
      std::is_same<MemLoc, tag::heap>::value ||
          std::is_same<MemLoc, tag::stack>::value,
      "MemLoc must either be tatooine::tag::heap or tatooine::tag::stack");
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type = T;
  using this_t     = static_multidim_array<T, Indexing, MemLoc, Resolution...>;
  using parent_t   = static_multidim_size<Indexing, Resolution...>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_dimensions;
  using parent_t::num_components;
  using parent_t::plain_index;
  using parent_t::size;
  using container_t =
      std::conditional_t<std::is_same<MemLoc, tag::stack>::value,
                         std::array<T, num_components()>, std::vector<T>>;

  //============================================================================
  // static methods
  //============================================================================
 private:
  static constexpr auto init_data(T const init = T{}) {
    if constexpr (std::is_same<MemLoc, tag::stack>::value) {
      return make_array<num_components()>(init);
    } else {
      return std::vector(num_components(), init);
    }
  }
  //============================================================================
  // members
  //============================================================================
 private:
  container_t m_data;
  //============================================================================
  // factories
  //============================================================================
 public:
  static constexpr auto zeros() { return this_t{tag::zeros}; }
  //------------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::ones}; }
  //------------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  requires real_number<T>
  static auto randu(T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  requires real_number<T>
  static auto randn(T mean = 0, T stddev = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal{mean, stddev, std::forward<RandEng>(eng)}};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr static_multidim_array(static_multidim_array const& other) = default;
  constexpr static_multidim_array(static_multidim_array&& other) noexcept =
      default;
  //----------------------------------------------------------------------------
  constexpr auto operator       =(static_multidim_array const& other)
      -> static_multidim_array& = default;
  constexpr auto operator       =(static_multidim_array&& other) noexcept
      -> static_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~static_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  explicit constexpr static_multidim_array(
      static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...> const& other)
      : m_data(init_data()) {
    for (auto is : static_multidim(Resolution...)) { at(is) = other(is); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  constexpr auto operator=(
      static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...> const& other)
      -> static_multidim_array& {
    for (auto is : tatooine::static_multidim{Resolution...}) {
      at(is) = other(is);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim_array(convertible_to<T> auto&&... ts)
      : m_data{static_cast<T>(ts)...} {
    static_assert(sizeof...(ts) == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array() : m_data(init_data(T{})) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit constexpr static_multidim_array(tag::fill<S> const& f)
      : m_data(init_data(static_cast<T>(f.value))) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires real_number<T>
  explicit constexpr static_multidim_array(tag::zeros_t /*z*/)
      : m_data(init_data(0)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires real_number<T>
  explicit constexpr static_multidim_array(tag::ones_t /*o*/)
      : m_data(init_data(1)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit static_multidim_array(std::vector<T> const& data)
      : m_data(begin(data), end(data)) {
    assert(data.size() == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      std::array<T, num_components()> const& data)
      : m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires std::is_same_v<MemLoc, tag::stack>
  explicit constexpr static_multidim_array(std::array<T, num_components()>&& data)
      : m_data(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires std::is_same_v<MemLoc, tag::heap>
  explicit constexpr static_multidim_array(std::vector<T>&& data)
      : m_data(std::move(data)) {
    assert(num_components() == data.size());
  }
  //----------------------------------------------------------------------------
  template <typename RandomReal, typename Engine>
  requires real_number<T>
  explicit constexpr static_multidim_array(
      random_uniform<RandomReal, Engine>& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  requires real_number<T>
  explicit constexpr static_multidim_array(
      random_uniform<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  requires real_number<T>
  explicit constexpr static_multidim_array(
      random_normal<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  requires real_number<T>
  explicit constexpr static_multidim_array(
      random_normal<RandomReal, Engine>& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  [[nodiscard]] constexpr auto at(integral auto const... is) const -> const
      auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  constexpr auto at(IndexRange const& index_range) const -> auto const& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    m_data[plain_index(index_range)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  constexpr auto at(IndexRange const& index_range) -> auto& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    return m_data[plain_index(index_range)];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator()(integral auto const... is) const
      ->  auto const& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  template <range IndexRange>
  constexpr auto operator()(IndexRange const& index_range) const -> const
      auto& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    return m_data[plain_index(index_range)];
  }
  //----------------------------------------------------------------------------
  template <range IndexRange>
  constexpr auto operator()(IndexRange const& index_range) -> auto& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    m_data[plain_index(index_range)];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator[](size_t i) -> auto& {
    assert(i < num_components());
    return m_data[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto operator[](size_t i) const -> auto const& {
    assert(i < num_components());
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data() -> auto& { return m_data; }
  [[nodiscard]] constexpr auto data() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data_ptr() -> T* { return m_data.data(); }
  [[nodiscard]] constexpr auto data_ptr() const -> T const* {
    return m_data.data();
  }
  //============================================================================
  template <typename F>
  constexpr void unary_operation(F&& f) {
    for (auto i : indices()) { at(i) = f(at(i)); }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing,
            typename OtherMemLoc>
  constexpr void binary_operation(
      F&& f, static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                         Resolution...> const& other) {
    for (auto const& i : indices()) { at(i) = f(at(i), other(i)); }
  }
};

//==============================================================================
template <typename T, typename Indexing = x_fastest>
class dynamic_multidim_array : public dynamic_multidim_size<Indexing> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type = T;
  using this_t     = dynamic_multidim_array<T, Indexing>;
  using parent_t   = dynamic_multidim_size<Indexing>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_dimensions;
  using parent_t::num_components;
  using parent_t::plain_index;
  using parent_t::size;
  using container_t = std::vector<T>;
  //============================================================================
  // members
  //============================================================================
  container_t m_data;
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto... size) {
    return this_t{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  static auto zeros(std::vector<UInt> const& size) {
    return this_t{tag::zeros, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  static auto zeros(std::array<UInt, N> const& size) {
    return this_t{tag::zeros, size};
  }
  //------------------------------------------------------------------------------
  static auto ones(integral auto... size) {
    return this_t{tag::ones, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  static auto ones(std::vector<UInt> const& size) {
    return this_t{tag::ones, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  static auto ones(std::array<UInt, N> const& size) {
    return this_t{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(T min, T max, std::initializer_list<UInt>&& size,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ///-
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(std::initializer_list<UInt>&& size, T min = 0, T
  // max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::vector<UInt> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(std::vector<UInt> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::array<UInt, N> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(std::array<UInt, N> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::vector<UInt> const&          size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::array<UInt, N> const&        size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   integral auto... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::vector<UInt> const&     size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::array<UInt, N> const&   size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   integral auto... size) {
    return this_t{std::move(rand),
                  std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::vector<UInt> const&         size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::array<UInt, N> const&       size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   integral auto... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::vector<UInt> const&    size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::array<UInt, N> const&  size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand, integral auto... size) {
    return this_t{std::move(rand),
                  std::vector{static_cast<size_t>(size)...}};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  dynamic_multidim_array()                                        = default;
  dynamic_multidim_array(dynamic_multidim_array const& other)     = default;
  dynamic_multidim_array(dynamic_multidim_array&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dynamic_multidim_array const& other)
      -> dynamic_multidim_array& = default;
  auto operator=(dynamic_multidim_array&& other) noexcept
      -> dynamic_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  explicit constexpr dynamic_multidim_array(
      dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      : parent_t{other} {
    auto it = begin(other.data());
    for (auto& v : m_data) { v = static_cast<T>(*(it++)); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  auto operator=(dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      -> dynamic_multidim_array& {
    if (parent_t::operator!=(other)) { resize(other.size()); }
    parent_t::operator=(other);
    for (auto i : indices()) { at(i) = other(i); }
    return *this;
  }
  //============================================================================
  explicit dynamic_multidim_array(integral auto... size)
      : parent_t{size...}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit dynamic_multidim_array(tag::fill<S> const& f,
                                  integral auto... size)
      : parent_t{size...}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::zeros_t const& /*z*/,
                                  integral auto... size)
      : parent_t{size...}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::ones_t const& /*o*/,
                                  integral auto... size)
      : parent_t{size...}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<T> const& data,
                                  integral auto... size)
      : parent_t{size...}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<T>&& data,
                                  integral auto... size)
      : parent_t{size...}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  explicit dynamic_multidim_array(std::vector<UInt> const& size)
      : parent_t{size}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, unsigned_integral UInt>
  dynamic_multidim_array(tag::fill<S> const&      f,
                         std::vector<UInt> const& size)
      : parent_t{size}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(tag::zeros_t const& /*z*/,
                         std::vector<UInt> const& size)
      : parent_t{size}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(tag::ones_t const& /*o*/,
                         std::vector<UInt> const& size)
      : parent_t{size}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T> const&    data,
                         std::vector<UInt> const& size)
      : parent_t{size}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&&         data,
                         std::vector<UInt> const& size)
      : parent_t{size}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <size_t N, unsigned_integral UInt>
  explicit dynamic_multidim_array(std::array<UInt, N> const& size)
      : parent_t{size}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename S, unsigned_integral UInt>
  dynamic_multidim_array(tag::fill<S> const&        f,
                         std::array<UInt, N> const& size)
      : parent_t{size}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(tag::zeros_t const& /*z*/,
                         std::array<UInt, N> const& size)
      : parent_t{size}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(tag::ones_t const& /*o*/,
                         std::array<UInt, N> const& size)
      : parent_t{size}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T> const&      data,
                         std::array<UInt, N> const& size)
      : parent_t{size}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&&           data,
                         std::array<UInt, N> const& size)
      : parent_t{size}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, real_number RandomReal, typename Engine>
 requires real_number<T>
  dynamic_multidim_array(random_uniform<RandomReal, Engine> const& rand,
                         std::vector<UInt> const&                  size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, real_number RandomReal, typename Engine>
  requires real_number<T>
  dynamic_multidim_array(random_uniform<RandomReal, Engine>&& rand,
                         std::vector<UInt> const&             size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, real_number RandomReal,
            typename Engine>
  requires real_number<T>
  dynamic_multidim_array(random_normal<RandomReal, Engine> const& rand,
                         std::array<UInt, N> const&               size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, real_number RandomReal,
            typename Engine>
  requires real_number<T>
  dynamic_multidim_array(random_normal<RandomReal, Engine>&& rand,
                         std::array<UInt, N> const&          size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //============================================================================
  // methods
  //============================================================================
  auto at(integral auto const... is) -> auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  auto at(IndexRange const& index_range) -> auto& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    assert(in_range(index_range));
    return m_data[plain_index(index_range)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  auto at(IndexRange const& index_range) const -> auto const& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    assert(in_range(index_range));
    return m_data[plain_index(index_range)];
  }
  //------------------------------------------------------------------------------
  auto operator()(integral auto const... is) -> auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
#ifndef NDEBUG
  if (!in_range(is...)) {
    std::cerr << "will now crash because indices [ ";
    ((std::cerr << is << ' '), ...);
    std::cerr << "] are not in range\n";
  }
#endif
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  auto operator()(IndexRange const& index_range) -> auto& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    assert(in_range(index_range));
    return at(index_range);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range IndexRange>
  auto operator()(IndexRange const& index_range) const -> auto const& {
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(index_range.size() == num_dimensions());
    assert(in_range(index_range));
    return at(index_range);
  }
  //----------------------------------------------------------------------------
  auto operator[](size_t i) const -> auto const& { return m_data[i]; }
  auto operator[](size_t i) -> auto& { return m_data[i]; }
  //----------------------------------------------------------------------------
  void resize(integral auto... size) {
    parent_t::resize(size...);
    m_data.resize(num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  void resize(std::vector<UInt> const& res, T const value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_components(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  void resize(std::vector<UInt>&& res, T const value = T{}) {
    parent_t::resize(std::move(res));
    m_data.resize(num_components(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  void resize(std::array<Int, N> const& res, T const value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_components(), value);
  }
  //----------------------------------------------------------------------------
  constexpr auto data() -> auto& { return m_data; }
  constexpr auto data() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  constexpr auto data_ptr() -> T* { return m_data.data(); }
  constexpr auto data_ptr() const -> T const* { return m_data.data(); }
  //============================================================================
  template <typename F>
  void unary_operation(F&& f) {
    for (auto is : indices()) { at(is) = f(at(is)); }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing>
  constexpr void binary_operation(
      F&& f, dynamic_multidim_array<OtherT, OtherIndexing> const& other) {
    assert(parent_t::operator==(other));
    for (auto const& is : indices()) { at(is) = f(at(is), other(is)); }
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename T, typename Indexing>
dynamic_multidim_array(dynamic_multidim_array<T, Indexing> const&)
    -> dynamic_multidim_array<T, Indexing>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename Indexing>
dynamic_multidim_array(dynamic_multidim_array<T, Indexing> &&)
    -> dynamic_multidim_array<T, Indexing>;
//----------------------------------------------------------------------------
template <typename T, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, T const& initial)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, std::vector<T> const&)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, std::vector<T> &&)
    -> dynamic_multidim_array<T, x_fastest>;
//----------------------------------------------------------------------------
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, T const& initial)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, std::vector<T> const&)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, std::vector<T> &&)
    -> dynamic_multidim_array<T, x_fastest>;

//==============================================================================
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename FReal,
          size_t... Resolution>
auto interpolate(
    static_multidim_array<T0, Indexing0, MemLoc0, Resolution...> const& arr0,
    static_multidim_array<T1, Indexing1, MemLoc1, Resolution...> const& arr1,
    FReal factor) {
  static_multidim_array<promote_t<T0, T1>, IndexingOut, MemLocOut,
                        Resolution...>
      interpolated{arr0};

  if constexpr (sizeof...(Resolution) == 2) {
#ifndef NDEBUG
#pragma omp parallel for collapse(2)
#endif
    for (size_t iy = 0; iy < interpolated.size(1); ++iy) {
      for (size_t ix = 0; ix < interpolated.size(0); ++ix) {
        interpolated(ix, iy) =
            interpolated.data(ix, iy) * (1 - factor) + arr1(ix, iy) * factor;
      }
    }
  } else if constexpr (sizeof...(Resolution) == 3) {
#ifndef NDEBUG
#pragma omp parallel for collapse(3)
#endif
    for (size_t iz = 0; iz < interpolated.size(2); ++iz) {
      for (size_t iy = 0; iy < interpolated.size(1); ++iy) {
        for (size_t ix = 0; ix < interpolated.size(0); ++ix) {
          interpolated(ix, iy, iz) = interpolated(ix, iy, iz) * (1 - factor) +
                                     arr1(ix, iy, iz) * factor;
        }
      }
    }
  } else {
    for (size_t is : interpolated.indices()) {
      interpolated(is) = interpolated(is) * (1 - factor) + arr1(is) * factor;
    }
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename LinReal, typename TReal,
          size_t... Resolution>
auto interpolate(
    static_multidim_array<T0, Indexing0, MemLoc0, Resolution...> const& arr0,
    static_multidim_array<T1, Indexing1, MemLoc1, Resolution...> const& arr1,
    linspace<LinReal> const& ts, TReal t) {
  return interpolate<MemLocOut, IndexingOut>(
      arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
//==============================================================================
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename FReal>
auto interpolate(dynamic_multidim_array<T0, Indexing0> const& arr0,
                 dynamic_multidim_array<T1, Indexing1> const& arr1,
                 FReal                                        factor) {
  if (factor == 0) { return arr0; }
  if (factor == 1) { return arr1; }
  assert(arr0.dyn_size() == arr1.dyn_size());
  dynamic_multidim_array<promote_t<T0, T1>, IndexingOut> interpolated{arr0};

  for (auto is : interpolated.indices()) {
    interpolated(is) = interpolated(is) * (1 - factor) + arr1(is) * factor;
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename LinReal,
          typename TReal>
auto interpolate(dynamic_multidim_array<T0, Indexing0> const& arr0,
                 dynamic_multidim_array<T1, Indexing1> const& arr1,
                 linspace<LinReal> const& ts, TReal t) {
  return interpolate<IndexingOut>(arr0, arr1,
                                  (t - ts.front()) / (ts.back() - ts.front()));
}
//#include "vtk_legacy.h"
// template <typename T, typename Indexing, typename MemLoc, size_t...
// Resolution> void write_vtk(
//    static_multidim_array<T, Indexing, MemLoc, Resolution...> const& arr,
//    std::string const& filepath, vec<double, 3> const& origin,
//    vec<double, 3> const& spacing,
//    std::string const&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    auto const res = arr.size();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_components());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
////------------------------------------------------------------------------------
// template <typename T, typename Indexing>
// void write_vtk(dynamic_multidim_array<T, Indexing> const& arr,
//               std::string const& filepath, vec<double, 3> const& origin,
//               vec<double, 3> const& spacing,
//               std::string const&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    auto const res = arr.size();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_components());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
//
template <floating_point Real>
void write_png(dynamic_multidim_array<Real> const& arr,
               std::string const&                  filepath) {
  if (arr.num_dimensions() != 2) {
    throw std::runtime_error{
        "multidim array needs 2 dimensions for writing as png."};
  }

  png::image<png::rgb_pixel> image(arr.size(0), arr.size(1));
  for (unsigned int y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      unsigned int idx = x + arr.size(0) * y;

      image[image.get_height() - 1 - y][x].red =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
      image[image.get_height() - 1 - y][x].green =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
      image[image.get_height() - 1 - y][x].blue =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
    }
  }
  image.write(filepath);
}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Real>
// void write_png(dynamic_multidim_array<vec<Real, 2> const>& arr,
//               std::string const&                          filepath) {
//  if (arr.num_dimensions() != 2) {
//    throw std::runtime_error{
//        "multidim array needs 2 dimensions for writing as png."};
//  }
//
//  png::image<png::rgb_pixel> image(dimension(0).size(), dimension(1).size());
//  for (unsigned int y = 0; y < image.get_height(); ++y) {
//    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
//      unsigned int idx = x + dimension(0).size() * y;
//
//      image[image.get_height() - 1 - y][x].red =
//          std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 0])) * 255;
//      image[image.get_height() - 1 - y][x].green =
//          std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 1])) * 255;
//      image[image.get_height() - 1 - y][x].blue =
//          std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 2])) * 255;
//      image[image.get_height() - 1 - y][x].alpha =
//          std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 3])) * 255;
//    }
//  }
//  image.write(filepath);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
