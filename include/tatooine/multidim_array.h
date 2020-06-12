#ifndef TATOOINE_MULTIDIM_ARRAY_H
#define TATOOINE_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_ordering.h>
#include <tatooine/linspace.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_resolution.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>

#include <array>
#include <png++/png.hpp>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename T, typename Indexing, typename MemLoc, size_t... Resolution>
class static_multidim_array
    : public static_multidim_resolution<Indexing, Resolution...> {
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
  using this_t   = static_multidim_array<T, Indexing, MemLoc, Resolution...>;
  using parent_t = static_multidim_resolution<Indexing, Resolution...>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_dimensions;
  using parent_t::num_elements;
  using parent_t::plain_index;
  using parent_t::size;
  using container_t =
      std::conditional_t<std::is_same<MemLoc, tag::stack>::value,
                         std::array<T, num_elements()>, std::vector<T>>;

  //============================================================================
  // static methods
  //============================================================================
 private:
  static constexpr auto init_data(T init = T{}) {
    if constexpr (std::is_same<MemLoc, tag::stack>::value) {
      return make_array<T, num_elements()>(init);
    } else {
      return std::vector(num_elements(), init);
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
  template <typename RandEng = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T> = true>
  static auto randu(T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T> = true>
  static auto randn(T mean = 0, T stddev = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal{mean, stddev, std::forward<RandEng>(eng)}};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr static_multidim_array(const static_multidim_array& other) = default;
  constexpr static_multidim_array(static_multidim_array&& other) noexcept =
      default;
  //----------------------------------------------------------------------------
  constexpr auto operator       =(const static_multidim_array& other)
      -> static_multidim_array& = default;
  constexpr auto operator       =(static_multidim_array&& other) noexcept
      -> static_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~static_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  explicit constexpr static_multidim_array(
      const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...>& other)
      : m_data(init_data()) {
    for (auto is : static_multidim(Resolution...)) { at(is) = other(is); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  constexpr auto operator=(
      const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...>& other)
      -> static_multidim_array& {
    for (auto is : tatooine::static_multidim{Resolution...}) {
      at(is) = other(is);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim_array(convertible_to<T> auto&&... ts)
      : m_data{static_cast<T>(ts)...} {
    static_assert(sizeof...(ts) == num_elements());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array() : m_data(init_data(T{})) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit constexpr static_multidim_array(const tag::fill<S>& f)
      : m_data(init_data(static_cast<T>(f.value))) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _T = T, enable_if_arithmetic<_T> = true>
  explicit constexpr static_multidim_array(tag::zeros_t /*z*/)
      : m_data(init_data(0)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _T = T, enable_if_arithmetic<_T> = true>
  explicit constexpr static_multidim_array(tag::ones_t /*o*/)
      : m_data(init_data(1)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit static_multidim_array(const std::vector<T>& data)
      : m_data(begin(data), end(data)) {
    assert(data.size() == num_elements());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      const std::array<T, num_elements()>& data)
      : m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <
      typename _MemLoc                                                 = MemLoc,
      std::enable_if_t<std::is_same<_MemLoc, tag::stack>::value, bool> = true>
  explicit constexpr static_multidim_array(std::array<T, num_elements()>&& data)
      : m_data(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <
      typename _MemLoc                                                = MemLoc,
      std::enable_if_t<std::is_same<_MemLoc, tag::heap>::value, bool> = true>
  explicit constexpr static_multidim_array(std::vector<T>&& data)
      : m_data(std::move(data)) {
    assert(num_elements() == data.size());
  }
  ////----------------------------------------------------------------------------
  template <typename RandomReal, typename Engine, typename _T = T,
            enable_if_arithmetic<_T> = true>
  explicit constexpr static_multidim_array(
      random_uniform<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _T = T,
            enable_if_arithmetic<_T> = true>
  explicit constexpr static_multidim_array(
      random_normal<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  [[nodiscard]] constexpr auto at(integral auto... is) const -> const auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  constexpr auto at(integral auto... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto at(const std::array<UInt, num_dimensions()>& is) const -> const
      auto& {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto at(const std::array<UInt, num_dimensions()>& is) -> auto& {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto at(const std::vector<UInt>& is) const -> const auto& {
    assert(is.size() == num_dimensions());
    m_data[plain_index(is)];
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto at(const std::vector<UInt>& is) -> auto& {
    assert(is.size() == num_dimensions());
    return m_data[plain_index(is)];
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto... is) const -> const auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto operator()(const std::array<UInt, num_dimensions()>& is) const
      -> const auto& {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  constexpr auto operator()(const std::array<UInt, num_dimensions()>& is)
      -> auto& {
    return at(is);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator[](size_t i) -> auto& {
    return m_data[i];
  }
  [[nodiscard]] constexpr auto operator[](size_t i) const -> const auto& {
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data() -> auto& { return m_data; }
  [[nodiscard]] constexpr auto data() const -> const auto& { return m_data; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data_ptr() -> T* { return m_data.data(); }
  [[nodiscard]] constexpr auto data_ptr() const -> const T* {
    return m_data.data();
  }
  //============================================================================
  template <typename F>
  constexpr void unary_operation(F&& f) {
    for (auto is : indices()) { at(is) = f(at(is)); }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing,
            typename OtherMemLoc>
  constexpr void binary_operation(
      F&& f, const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                         Resolution...>& other) {
    for (const auto& is : indices()) { at(is) = f(at(is), other(is)); }
  }
};

//==============================================================================
template <typename T, typename Indexing = x_fastest>
class dynamic_multidim_array : public dynamic_multidim_resolution<Indexing> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using this_t   = dynamic_multidim_array<T, Indexing>;
  using parent_t = dynamic_multidim_resolution<Indexing>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_dimensions;
  using parent_t::num_elements;
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
  static auto zeros(integral auto... resolution) {
    return this_t{tag::zeros, resolution...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  static auto zeros(const std::vector<UInt>& resolution) {
    return this_t{tag::zeros, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  static auto zeros(const std::array<UInt, N>& resolution) {
    return this_t{tag::zeros, resolution};
  }
  //------------------------------------------------------------------------------
  static auto ones(integral auto... resolution) {
    return this_t{tag::ones, resolution...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  static auto ones(const std::vector<UInt>& resolution) {
    return this_t{tag::ones, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  static auto ones(const std::array<UInt, N>& resolution) {
    return this_t{tag::ones, resolution};
  }
  //------------------------------------------------------------------------------
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(T min, T max, std::initializer_list<UInt>&& resolution,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(resolution))};
  //}
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ///-
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(std::initializer_list<UInt>&& resolution, T min = 0, T
  // max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(resolution))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>

  static auto randu(T min, T max, const std::vector<UInt>& resolution,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(const std::vector<UInt>& resolution, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, const std::array<UInt, N>& resolution,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(const std::array<UInt, N>& resolution, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        resolution};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(const random_uniform<T, RandEng>& rand,
                   const std::vector<UInt>&          resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(const random_uniform<T, RandEng>& rand,
                   const std::array<UInt, N>&        resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(const random_uniform<T, RandEng>& rand,
                   integral auto... resolution) {
    return this_t{rand, std::vector{static_cast<size_t>(resolution)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   const std::vector<UInt>&     resolution) {
    return this_t{std::move(rand), resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   const std::array<UInt, N>&   resolution) {
    return this_t{std::move(rand), resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   integral auto... resolution) {
    return this_t{std::move(rand),
                  std::vector{static_cast<size_t>(resolution)...}};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(const random_normal<T, RandEng>& rand,
                   const std::vector<UInt>&         resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(const random_normal<T, RandEng>& rand,
                   const std::array<UInt, N>&       resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(const random_normal<T, RandEng>& rand,
                   integral auto... resolution) {
    return this_t{rand, std::vector{static_cast<size_t>(resolution)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   const std::vector<UInt>&    resolution) {
    return this_t{std::move(rand), resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   const std::array<UInt, N>&  resolution) {
    return this_t{std::move(rand), resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand, integral auto... resolution) {
    return this_t{std::move(rand),
                  std::vector{static_cast<size_t>(resolution)...}};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  dynamic_multidim_array()                                        = default;
  dynamic_multidim_array(const dynamic_multidim_array& other)     = default;
  dynamic_multidim_array(dynamic_multidim_array&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                  =(const dynamic_multidim_array& other)
      -> dynamic_multidim_array& = default;
  auto operator                  =(dynamic_multidim_array&& other) noexcept
      -> dynamic_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  explicit constexpr dynamic_multidim_array(
      const dynamic_multidim_array<OtherT, OtherIndexing>& other)
      : parent_t{other} {
    auto it = begin(other.data());
    for (auto& v : m_data) { v = static_cast<T>(*(it++)); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  auto operator=(const dynamic_multidim_array<OtherT, OtherIndexing>& other)
      -> dynamic_multidim_array& {
    if (parent_t::operator!=(other)) { resize(other.size()); }
    parent_t::operator=(other);
    for (auto is : indices()) { at(is) = other(is); }
    return *this;
  }
  //============================================================================
  explicit dynamic_multidim_array(integral auto... resolution)
      : parent_t{resolution...}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit dynamic_multidim_array(const tag::fill<S>& f,
                                  integral auto... resolution)
      : parent_t{resolution...}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(const tag::zeros_t& /*z*/,
                                  integral auto... resolution)
      : parent_t{resolution...}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(const tag::ones_t& /*o*/,
                                  integral auto... resolution)
      : parent_t{resolution...}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(const std::vector<T>& data,
                                  integral auto... resolution)
      : parent_t{resolution...}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<T>&& data,
                                  integral auto... resolution)
      : parent_t{resolution...}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  explicit dynamic_multidim_array(const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, unsigned_integral UInt>
  dynamic_multidim_array(const tag::fill<S>&      f,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(const tag::zeros_t& /*z*/,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(const tag::ones_t& /*o*/,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(const std::vector<T>&    data,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&&         data,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <size_t N, unsigned_integral UInt>
  explicit dynamic_multidim_array(const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename S, unsigned_integral UInt>
  dynamic_multidim_array(const tag::fill<S>&        f,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(const tag::zeros_t& /*z*/,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(const tag::ones_t& /*o*/,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(const std::vector<T>&      data,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, unsigned_integral UInt>
  dynamic_multidim_array(std::vector<T>&&           data,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, real_number RandomReal, typename Engine,
            typename _T = T, enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(const random_uniform<RandomReal, Engine>& rand,
                         const std::vector<UInt>&                  resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, real_number RandomReal, typename Engine,
            typename _T = T,
            enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(random_uniform<RandomReal, Engine>&& rand,
                         const std::vector<UInt>&             resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, real_number RandomReal,
            typename Engine, typename _T = T, enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(const random_normal<RandomReal, Engine>& rand,
                         const std::array<UInt, N>&               resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N, real_number RandomReal,
            typename Engine, typename _T = T, enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(random_normal<RandomReal, Engine>&& rand,
                         const std::array<UInt, N>&          resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //============================================================================
  // methods
  //============================================================================
  auto at(integral auto... is) -> auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto... is) const -> const auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  auto at(const std::vector<UInt>& is) -> auto& {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_index(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  auto at(const std::vector<UInt>& is) const -> const auto& {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_index(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  auto at(const std::array<UInt, N>& is) -> auto& {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_index(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  auto at(const std::array<UInt, N>& is) const -> const auto& {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_index(is)];
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto... is) -> auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto... is) const -> const auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  auto operator()(const std::vector<UInt>& is) -> auto& {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  auto operator()(const std::vector<UInt>& is) const -> const auto& {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  auto operator()(const std::array<UInt, N>& is) -> auto& {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  auto operator()(const std::array<UInt, N>& is) const -> const auto& {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  //----------------------------------------------------------------------------
  auto operator[](size_t i) -> auto& { return m_data[i]; }
  auto operator[](size_t i) const -> const auto& { return m_data[i]; }
  //----------------------------------------------------------------------------
  void resize(integral auto... resolution) {
    parent_t::resize(resolution...);
    m_data.resize(num_elements());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  void resize(const std::vector<UInt>& res, const T value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  void resize(std::vector<UInt>&& res, const T value = T{}) {
    parent_t::resize(std::move(res));
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt, size_t N>
  void resize(const std::array<UInt, N>& res, const T value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_elements(), value);
  }
  //----------------------------------------------------------------------------
  constexpr auto data() -> auto& { return m_data; }
  constexpr auto data() const -> const auto& { return m_data; }
  //----------------------------------------------------------------------------
  constexpr auto data_ptr() -> T* { return m_data.data(); }
  constexpr auto data_ptr() const -> const T* { return m_data.data(); }
  //============================================================================
  template <typename F>
  void unary_operation(F&& f) {
    for (auto is : indices()) { at(is) = f(at(is)); }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing>
  constexpr void binary_operation(
      F&& f, const dynamic_multidim_array<OtherT, OtherIndexing>& other) {
    assert(parent_t::operator==(other));
    for (const auto& is : indices()) { at(is) = f(at(is), other(is)); }
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename T, typename Indexing>
dynamic_multidim_array(const dynamic_multidim_array<T, Indexing>&)
    -> dynamic_multidim_array<T, Indexing>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename Indexing>
dynamic_multidim_array(dynamic_multidim_array<T, Indexing> &&)
    -> dynamic_multidim_array<T, Indexing>;
//----------------------------------------------------------------------------
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const T& initial)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const std::vector<T>&)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, std::vector<T> &&)
    -> dynamic_multidim_array<T, x_fastest>;
//----------------------------------------------------------------------------
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const T& initial)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const std::vector<T>&)
    -> dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, std::vector<T> &&)
    -> dynamic_multidim_array<T, x_fastest>;

//==============================================================================
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename FReal,
          size_t... Resolution>
auto interpolate(
    const static_multidim_array<T0, Indexing0, MemLoc0, Resolution...>& arr0,
    const static_multidim_array<T1, Indexing1, MemLoc1, Resolution...>& arr1,
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
    const static_multidim_array<T0, Indexing0, MemLoc0, Resolution...>& arr0,
    const static_multidim_array<T1, Indexing1, MemLoc1, Resolution...>& arr1,
    const linspace<LinReal>& ts, TReal t) {
  return interpolate<MemLocOut, IndexingOut>(
      arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
//==============================================================================
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename FReal>
auto interpolate(const dynamic_multidim_array<T0, Indexing0>& arr0,
                 const dynamic_multidim_array<T1, Indexing1>& arr1,
                 FReal                                        factor) {
  if (factor == 0) { return arr0; }
  if (factor == 1) { return arr1; }
  assert(arr0.dyn_resolution() == arr1.dyn_resolution());
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
auto interpolate(const dynamic_multidim_array<T0, Indexing0>& arr0,
                 const dynamic_multidim_array<T1, Indexing1>& arr1,
                 const linspace<LinReal>& ts, TReal t) {
  return interpolate<IndexingOut>(arr0, arr1,
                                  (t - ts.front()) / (ts.back() - ts.front()));
}
//#include "vtk_legacy.h"
// template <typename T, typename Indexing, typename MemLoc, size_t...
// Resolution> void write_vtk(
//    const static_multidim_array<T, Indexing, MemLoc, Resolution...>& arr,
//    const std::string& filepath, const vec<double, 3>& origin,
//    const vec<double, 3>& spacing,
//    const std::string&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    const auto res = arr.size();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_elements());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
////------------------------------------------------------------------------------
// template <typename T, typename Indexing>
// void write_vtk(const dynamic_multidim_array<T, Indexing>& arr,
//               const std::string& filepath, const vec<double, 3>& origin,
//               const vec<double, 3>& spacing,
//               const std::string&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    const auto res = arr.size();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_elements());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
//
template <floating_point Real>
void write_png(const dynamic_multidim_array<Real>& arr,
               const std::string&                  filepath) {
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
// void write_png(const dynamic_multidim_array<vec<Real, 2>>& arr,
//               const std::string&                          filepath) {
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
