#ifndef TATOOINE_MULTIDIM_ARRAY_H
#define TATOOINE_MULTIDIM_ARRAY_H
//==============================================================================
#include <array>
#include <vector>

#include "linspace.h"
#include "multidim.h"
#include "multidim_resolution.h"
#include "random.h"
//==============================================================================
namespace tatooine {
//==============================================================================
struct heap {};
struct stack {};

template <typename Real>
struct fill {
  Real value;
};
#if has_cxx17_support()
template <typename Real>
fill(Real)->fill<Real>;
#endif

struct zeros_t {};
static constexpr zeros_t zeros;

struct ones_t {};
static constexpr ones_t ones;

template <typename T, typename Indexing, typename MemLoc, size_t... Resolution>
class static_multidim_array
    : public static_multidim_resolution<Indexing, Resolution...> {
  //============================================================================
  // assertions
  //============================================================================
  static_assert(std::is_same_v<MemLoc, heap> || std::is_same_v<MemLoc, stack>,
                "MemLoc must either be tatooine::heap or tatooine::stack");
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
  using parent_t::plain_idx;
  using parent_t::resolution;
  using parent_t::size;
  using container_t =
      std::conditional_t<std::is_same_v<MemLoc, stack>,
                         std::array<T, num_elements()>, std::vector<T>>;

  //============================================================================
  // static methods
  //============================================================================
 private:
  static constexpr auto init_data(T init = T{}) {
    if constexpr (std::is_same_v<MemLoc, stack>) {
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
  static constexpr auto zeros() { return this_t{tatooine::zeros}; }
  //------------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tatooine::ones}; }
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
  constexpr static_multidim_array(static_multidim_array&& other)      = default;
  constexpr static_multidim_array& operator                           =(
      const static_multidim_array& other) = default;
  constexpr static_multidim_array& operator=(static_multidim_array&& other) =
      default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  constexpr static_multidim_array(
      const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...>& other)
      : m_data(init_data()) {
    for (auto is : static_multidim(Resolution...)) { at(is) = other(is); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  constexpr static_multidim_array& operator=(
      const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                  Resolution...>& other) {
    for (auto is : tatooine::static_multidim{Resolution...}) {
      at(is) = other(is);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... Ts,
            std::enable_if_t<sizeof...(Ts) == num_elements(), bool> = true>
  constexpr static_multidim_array(Ts&&... ts) : m_data{static_cast<T>(ts)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array() : m_data(init_data(T{})) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  constexpr static_multidim_array(const fill<S>& f)
      : m_data(init_data(static_cast<T>(f.value))) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _T = T, enable_if_arithmetic<_T> = true>
  constexpr static_multidim_array(zeros_t /*z*/) : m_data(init_data(0)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _T = T, enable_if_arithmetic<_T> = true>
  constexpr static_multidim_array(ones_t /*o*/) : m_data(init_data(1)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static_multidim_array(const std::vector<T>& data)
      : m_data(begin(data), end(data)) {
    assert(data.size() == num_elements());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array(const std::array<T, num_elements()>& data)
      : m_data(begin(data), end(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _MemLoc                                       = MemLoc,
            std::enable_if_t<std::is_same_v<_MemLoc, stack>, bool> = true>
  constexpr static_multidim_array(std::array<T, num_elements()>&& data)
      : m_data(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _MemLoc                                      = MemLoc,
            std::enable_if_t<std::is_same_v<_MemLoc, heap>, bool> = true>
  constexpr static_multidim_array(std::vector<T>&& data)
      : m_data(std::move(data)) {
    assert(num_elements() == data.size());
  }
  //----------------------------------------------------------------------------
  template <typename RandomReal, typename Engine, typename _T = T,
            enable_if_arithmetic<_T> = true>
  constexpr static_multidim_array(random_uniform<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _T = T,
            enable_if_arithmetic<_T> = true>
  constexpr static_multidim_array(random_normal<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr const auto& at(Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto& at(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  constexpr const auto& at(const std::array<UInt, num_dimensions()>& is) const {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  constexpr auto& at(const std::array<UInt, num_dimensions()>& is) {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr const auto& operator()(Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto& operator()(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  constexpr const auto& operator()(
      const std::array<UInt, num_dimensions()>& is) const {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  constexpr auto& operator()(const std::array<UInt, num_dimensions()>& is) {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_data[i]; }
  constexpr const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  constexpr auto&       data() { return m_data; }
  constexpr const auto& data() const { return m_data; }
  //----------------------------------------------------------------------------
  constexpr T*       data_ptr() { return m_data.data(); }
  constexpr const T* data_ptr() const { return m_data.data(); }
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
template <typename T, typename Indexing>
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
  using parent_t::plain_idx;
  using parent_t::resolution;
  using parent_t::size;
  using container_t = std::vector<T>;
  //============================================================================
  // members
  //============================================================================
  container_t m_data;
  //============================================================================
  // factories
  //============================================================================
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  static auto zeros(Resolution... resolution) {
    return this_t{tatooine::zeros, resolution...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  static auto zeros(const std::vector<UInt>& resolution) {
    return this_t{tatooine::zeros, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  static auto zeros(const std::array<UInt, N>& resolution) {
    return this_t{tatooine::zeros, resolution};
  }
  //------------------------------------------------------------------------------
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  static auto ones(Resolution... resolution) {
    return this_t{tatooine::ones, resolution...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  static auto ones(const std::vector<UInt>& resolution) {
    return this_t{tatooine::ones, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  static auto ones(const std::array<UInt, N>& resolution) {
    return this_t{tatooine::ones, resolution};
  }
  //------------------------------------------------------------------------------
  //template <typename UInt, typename RandEng = std::mt19937_64,
  //          enable_if_unsigned_integral<UInt> = true>
  //static auto randu(T min, T max, std::initializer_list<UInt>&& resolution,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(resolution))};
  //}
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //template <typename UInt, typename RandEng = std::mt19937_64,
  //          enable_if_unsigned_integral<UInt> = true>
  //static auto randu(std::initializer_list<UInt>&& resolution, T min = 0, T max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(resolution))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, typename RandEng = std::mt19937_64,
            enable_if_unsigned_integral<UInt> = true>
  static auto randu(T min, T max, const std::vector<UInt>& resolution,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
                  resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, typename RandEng = std::mt19937_64,
            enable_if_unsigned_integral<UInt> = true>
  static auto randu(const std::vector<UInt>& resolution, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
                  resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, typename RandEng = std::mt19937_64,
            enable_if_unsigned_integral<UInt> = true>
  static auto randu(T min, T max, const std::array<UInt, N>& resolution,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
                  resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, typename RandEng = std::mt19937_64,
            enable_if_unsigned_integral<UInt> = true>
  static auto randu(const std::array<UInt, N>& resolution, T min = 0,
                    T         max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
                  resolution};
  }
  //----------------------------------------------------------------------------
  template <typename UInt, typename RandEng,
            enable_if_unsigned_integral<UInt> = true>
  static auto rand(random_uniform<T, RandEng>&  rand,
                   const std::vector<UInt>& resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, typename RandEng,
            enable_if_unsigned_integral<UInt> = true>
  static auto rand(random_uniform<T, RandEng>&  rand,
                   const std::array<UInt, N>& resolution) {
    return this_t{rand, resolution};
  }
  //----------------------------------------------------------------------------
  template <typename UInt, typename RandEng,
            enable_if_unsigned_integral<UInt> = true>
  static auto rand(random_normal<T, RandEng>&  rand,
                   const std::vector<UInt>& resolution) {
    return this_t{rand, resolution};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, typename RandEng,
            enable_if_unsigned_integral<UInt> = true>
  static auto rand(random_normal<T, RandEng>&  rand,
                   const std::array<UInt, N>& resolution) {
    return this_t{rand, resolution};
  }
  //============================================================================
  // ctors
  //============================================================================
 public:
  dynamic_multidim_array()                                    = default;
  dynamic_multidim_array(const dynamic_multidim_array& other) = default;
  dynamic_multidim_array(dynamic_multidim_array&& other)      = default;
  dynamic_multidim_array& operator=(const dynamic_multidim_array& other) =
      default;
  dynamic_multidim_array& operator=(dynamic_multidim_array&& other) = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  constexpr dynamic_multidim_array(
      const dynamic_multidim_array<OtherT, OtherIndexing>& other)
      : parent_t{other} {
    auto it = begin(other.data());
    for (auto& v : m_data) { v = static_cast<T>(*(it++)); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  dynamic_multidim_array& operator=(
      const dynamic_multidim_array<OtherT, OtherIndexing>& other) {
    if (parent_t::operator!=(other)) { resize(other.resolution()); }
    parent_t::operator=(other);
    for (auto is : indices()) { at(is) = other(is); }
    return *this;
  }
  //============================================================================
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(Resolution... resolution)
      : parent_t{resolution...}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, typename S, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(const fill<S>& f, Resolution... resolution)
      : parent_t{resolution...}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(const zeros_t& /*z*/, Resolution... resolution)
      : parent_t{resolution...}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(const ones_t& /*o*/, Resolution... resolution)
      : parent_t{resolution...}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(const std::vector<T>&    data,
                         Resolution...resolution)
      : parent_t{resolution...}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  dynamic_multidim_array(std::vector<T>&&         data,
                         Resolution...resolution)
      : parent_t{resolution...}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const fill<S>& f, const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const zeros_t& /*z*/,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const ones_t& /*o*/, const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<T>&    data,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(std::vector<T>&&         data,
                         const std::vector<UInt>& resolution)
      : parent_t{resolution}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), T{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename S, typename UInt,
            enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const fill<S>&             f,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const zeros_t& /*z*/,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const ones_t& /*o*/,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(num_elements(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<T>&      data,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_array(std::vector<T>&&           data,
                         const std::array<UInt, N>& resolution)
      : parent_t{resolution}, m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <typename UInt, typename RandomReal, typename Engine,
            typename _T = T, enable_if_unsigned_integral<UInt> = true,
            enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(random_uniform<RandomReal, Engine>&& rand,
                                   const std::vector<UInt>& resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, typename RandomReal, typename Engine,
            typename _T = T, enable_if_unsigned_integral<UInt> = true,
            enable_if_arithmetic<_T> = true>
  dynamic_multidim_array(random_normal<RandomReal, Engine>&& rand,
                         const std::array<UInt, N>&          resolution)
      : dynamic_multidim_array{resolution} {
    this->unary_operation(
        [&](const auto& /*c*/) { return static_cast<T>(rand.get()); });
  }
  //============================================================================
  // methods
  //============================================================================
  template <typename... Is, enable_if_integral<Is...> = true>
  auto& at(Is... is) {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& at(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  auto& at(const std::vector<UInt>& is) {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  const auto& at(const std::vector<UInt>& is) const {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  auto& at(const std::array<UInt, N>& is) {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  const auto& at(const std::array<UInt, N>& is) const {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto& operator()(Is... is) {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& operator()(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  auto& operator()(const std::vector<UInt>& is) {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  const auto& operator()(const std::vector<UInt>& is) const {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  auto& operator()(const std::array<UInt, N>& is) {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  const auto& operator()(const std::array<UInt, N>& is) const {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  //----------------------------------------------------------------------------
  auto&       operator[](size_t i) { return m_data[i]; }
  const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  template <typename... Resolution>
  void resize(Resolution... resolution) {
    parent_t::resize(resolution...);
    m_data.resize(num_elements());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  void resize(const std::vector<UInt>& res, const T value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  void resize(std::vector<UInt>&& res, const T value = T{}) {
    parent_t::resize(std::move(res));
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  void resize(const std::array<UInt, N>& res, const T value = T{}) {
    parent_t::resize(res);
    m_data.resize(num_elements(), value);
  }
  //----------------------------------------------------------------------------
  constexpr auto&       data() { return m_data; }
  constexpr const auto& data() const { return m_data; }
  //----------------------------------------------------------------------------
  constexpr T*       data_ptr() { return m_data.data(); }
  constexpr const T* data_ptr() const { return m_data.data(); }
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
#if has_cxx17_support()
template <typename T, typename Indexing>
dynamic_multidim_array(const dynamic_multidim_array<T, Indexing>&)
    ->dynamic_multidim_array<T, Indexing>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename Indexing>
dynamic_multidim_array(dynamic_multidim_array<T, Indexing> &&)
    ->dynamic_multidim_array<T, Indexing>;
//----------------------------------------------------------------------------
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
//----------------------------------------------------------------------------
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
#endif

//==============================================================================
template <typename MemLocOut = stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename FReal,
          size_t... Resolution>
constexpr auto interpolate(
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
              interpolated.data(ix, iy) * (1 - factor) +
              arr1(ix, iy) * factor;
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
      interpolated(is) =
          interpolated(is) * (1 - factor) + arr1(is) * factor;
    }
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename MemLocOut = stack, typename IndexingOut = x_fastest,
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
template <typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename FReal>
auto interpolate(const dynamic_multidim_array<T0, Indexing0>& arr0,
                 const dynamic_multidim_array<T1, Indexing1>& arr1,
                 FReal                                        factor) {
  assert(arr0.dyn_resolution() == arr1.dyn_resolution());
  dynamic_multidim_array<promote_t<T0, T1>, IndexingOut> interpolated{arr0};

  for (size_t is : interpolated.indices()) {
    interpolated(is) = interpolated.data(is) * (1 - factor) + arr1(is) * factor;
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename IndexingOut = x_fastest, typename T0, typename T1, typename Indexing0,
          typename Indexing1, typename LinReal, typename TReal>
auto interpolate(const dynamic_multidim_array<T0, Indexing0>& arr0,
                 const dynamic_multidim_array<T1, Indexing1>& arr1,
                 const linspace<LinReal>& ts, TReal t) {
  return interpolate<IndexingOut>(
      arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
//#include "vtk_legacy.h"
//==============================================================================
namespace tatooine {
//==============================================================================
//template <typename T, typename Indexing, typename MemLoc, size_t... Resolution>
//void write_vtk(
//    const static_multidim_array<T, Indexing, MemLoc, Resolution...>& arr,
//    const std::string& filepath, const vec<double, 3>& origin,
//    const vec<double, 3>& spacing,
//    const std::string&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    const auto res = arr.resolution();
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
//template <typename T, typename Indexing>
//void write_vtk(const dynamic_multidim_array<T, Indexing>& arr,
//               const std::string& filepath, const vec<double, 3>& origin,
//               const vec<double, 3>& spacing,
//               const std::string&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    const auto res = arr.resolution();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_elements());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
