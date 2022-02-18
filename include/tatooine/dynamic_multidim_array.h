#ifndef TATOOINE_DYNAMIC_MULTIDIM_ARRAY_H
#define TATOOINE_DYNAMIC_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_order.h>
#include <tatooine/linspace.h>
#include <tatooine/make_array.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_size.h>
#include <tatooine/png.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename IndexOrder = x_fastest>
class dynamic_multidim_array : public dynamic_multidim_size<IndexOrder> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type = T;
  using this_type     = dynamic_multidim_array<T, IndexOrder>;
  using parent_type   = dynamic_multidim_size<IndexOrder>;
  using parent_type::in_range;
  using parent_type::indices;
  using parent_type::num_components;
  using parent_type::num_dimensions;
  using parent_type::plain_index;
  using parent_type::size;
  using container_t = std::vector<T>;
  //============================================================================
  // members
  //============================================================================
  container_t m_data;
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto const... size) {
    return this_type{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  static auto zeros(std::vector<Int> const& size) {
    return this_type{tag::zeros, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral Int>
  static auto zeros(std::array<Int, N> const& size) {
    return this_type{tag::zeros, size};
  }
  //------------------------------------------------------------------------------
  static auto ones(integral auto const... size) {
    return this_type{tag::ones, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  static auto ones(std::vector<Int> const& size) {
    return this_type{tag::ones, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral Int>
  static auto ones(std::array<Int, N> const& size) {
    return this_type{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(T min, T max, std::initializer_list<UInt>&& size,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_type{random::uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ///-
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(std::initializer_list<UInt>&& size, T min = 0, T
  // max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_type{random::uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::vector<UInt> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(std::vector<UInt> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt,
            typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::array<UInt, N> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt,
            typename RandEng = std::mt19937_64>
  static auto randu(std::array<UInt, N> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand,
                   std::vector<UInt> const&           size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand,
                   std::array<UInt, N> const&         size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng, integral... Size>
  static auto rand(random::uniform<T, RandEng> const& rand,
                   Size const... size) {
    return this_type{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   std::vector<UInt> const&      size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   std::array<UInt, N> const&    size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng, integral... Size>
  static auto rand(random::uniform<T, RandEng>&& rand, Size const... size) {
    return this_type{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand,
                   std::vector<UInt> const&          size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand,
                   std::array<UInt, N> const&        size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng, integral... Size>
  static auto rand(random::normal<T, RandEng> const& rand, Size const... size) {
    return this_type{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand,
                   std::vector<UInt> const&     size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand,
                   std::array<UInt, N> const&   size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng, integral... Size>
  static auto rand(random::normal<T, RandEng>&& rand, Size... size) {
    return this_type{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  dynamic_multidim_array()                                        = default;
  dynamic_multidim_array(dynamic_multidim_array const& other)     = default;
  dynamic_multidim_array(dynamic_multidim_array&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                  =(dynamic_multidim_array const& other)
      -> dynamic_multidim_array& = default;
  auto operator                  =(dynamic_multidim_array&& other) noexcept
      -> dynamic_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  explicit constexpr dynamic_multidim_array(
      dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      : parent_type{other} {
    auto it = begin(other.data());
    for (auto& v : m_data) {
      v = static_cast<T>(*(it++));
    }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  auto operator=(dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      -> dynamic_multidim_array& {
    if (parent_type::operator!=(other)) {
      resize(other.size());
    }
    parent_type::operator=(other);
    for (auto i : indices()) {
      at(i) = other(i);
    }
    return *this;
  }
  //============================================================================
  explicit dynamic_multidim_array(integral auto const... size)
      : parent_type{size...}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, integral... Size>
  explicit dynamic_multidim_array(tag::fill<S> const& f, Size const... size)
      : parent_type{size...}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::zeros_t const& /*z*/,
                                  integral auto const... size)
      : parent_type{size...}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::ones_t const& /*o*/,
                                  integral auto const... size)
      : parent_type{size...}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<T> const& data,
                                  integral auto const... size)
      : parent_type{size...}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<T>&& data,
                                  integral auto const... size)
      : parent_type{size...}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  explicit dynamic_multidim_array(std::vector<UInt> const& size)
      : parent_type{size}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, unsigned_integral UInt>
  dynamic_multidim_array(tag::fill<S> const& f, std::vector<UInt> const& size)
      : parent_type{size}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(tag::zeros_t const& /*z*/,
                         std::vector<UInt> const& size)
      : parent_type{size}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(tag::ones_t const& /*o*/,
                         std::vector<UInt> const& size)
      : parent_type{size}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T> const&    data,
                         std::vector<UInt> const& size)
      : parent_type{size}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&& data, std::vector<UInt> const& size)
      : parent_type{size}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <size_t N, unsigned_integral UInt>
  explicit dynamic_multidim_array(std::array<UInt, N> const& size)
      : parent_type{size}, m_data(num_components(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename S, unsigned_integral UInt>
  dynamic_multidim_array(tag::fill<S> const& f, std::array<UInt, N> const& size)
      : parent_type{size}, m_data(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(tag::zeros_t const& /*z*/,
                         std::array<UInt, N> const& size)
      : parent_type{size}, m_data(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(tag::ones_t const& /*o*/,
                         std::array<UInt, N> const& size)
      : parent_type{size}, m_data(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T> const&      data,
                         std::array<UInt, N> const& size)
      : parent_type{size}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&& data, std::array<UInt, N> const& size)
      : parent_type{size}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, std::size_t N, arithmetic RandomReal,
            typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::uniform<RandomReal, Engine> const& rand,
      std::array<UInt, N> const&                 size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, std::size_t N, arithmetic RandomReal,
            typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::uniform<RandomReal, Engine>&& rand,
      std::array<UInt, N> const&            size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, arithmetic RandomReal, typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::uniform<RandomReal, Engine> const& rand,
      std::vector<UInt> const&                   size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, arithmetic RandomReal, typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::uniform<RandomReal, Engine>&& rand, std::vector<UInt> const& size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, arithmetic RandomReal,
            typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::normal<RandomReal, Engine> const& rand,
      std::array<UInt, N> const&                size)
      : dynamic_multidim_array{size} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, arithmetic RandomReal,
            typename Engine>
  requires arithmetic<T> dynamic_multidim_array(
      random::normal<RandomReal, Engine>&& rand,
      std::array<UInt, N> const&           size)
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
  auto at(integral_range auto const& indices) -> auto& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data[plain_index(indices)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral_range auto const& indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data[plain_index(indices)];
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
  auto operator()(integral_range auto const& indices) -> auto& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral_range auto const& indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return at(indices);
  }
  //----------------------------------------------------------------------------
  auto operator[](size_t i) const -> auto const& { return m_data[i]; }
  auto operator[](size_t i) -> auto& { return m_data[i]; }
  //----------------------------------------------------------------------------
  void resize(integral auto const... size) {
    parent_type::resize(size...);
    m_data.resize(num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(integral_range auto const& res, T const value = T{}) {
    parent_type::resize(res);
    m_data.resize(num_components(), value);
  }
  //----------------------------------------------------------------------------
  constexpr auto data() -> auto& { return m_data; }
  constexpr auto data() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data(size_t const i) -> auto& {
    return m_data[i];
  }
  [[nodiscard]] constexpr auto data(size_t const i) const -> auto const& {
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  constexpr auto data_ptr() -> T* { return m_data.data(); }
  constexpr auto data_ptr() const -> T const* { return m_data.data(); }
  //============================================================================
  template <typename F>
  void unary_operation(F&& f) {
    for (auto is : indices()) {
      at(is) = f(at(is));
    }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing>
  constexpr void binary_operation(
      F&& f, dynamic_multidim_array<OtherT, OtherIndexing> const& other) {
    assert(parent_type::operator==(other));
    for (auto const& is : indices()) {
      at(is) = f(at(is), other(is));
    }
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename T, typename IndexOrder>
dynamic_multidim_array(dynamic_multidim_array<T, IndexOrder> const&)
    -> dynamic_multidim_array<T, IndexOrder>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename IndexOrder>
dynamic_multidim_array(dynamic_multidim_array<T, IndexOrder> &&)
    -> dynamic_multidim_array<T, IndexOrder>;
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
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename FReal>
auto interpolate(dynamic_multidim_array<T0, Indexing0> const& arr0,
                 dynamic_multidim_array<T1, Indexing1> const& arr1,
                 FReal                                        factor) {
  if (factor == 0) {
    return arr0;
  }
  if (factor == 1) {
    return arr1;
  }
  assert(arr0.dyn_size() == arr1.dyn_size());
  dynamic_multidim_array<common_type<T0, T1>, IndexingOut> interpolated{arr0};

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
////------------------------------------------------------------------------------
// template <typename T, typename IndexOrder>
// void write_vtk(dynamic_multidim_array<T, IndexOrder> const& arr,
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
#if TATOOINE_PNG_AVAILABLE
template <arithmetic Real>
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
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
