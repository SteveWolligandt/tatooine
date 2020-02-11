#ifndef TATOOINE_MULTIDIM_RESOLUTION_H
#define TATOOINE_MULTIDIM_RESOLUTION_H

#include <array>
#include <cassert>
#include <numeric>
#include <sstream>

#include "functional.h"
#include "index_ordering.h"
#include "multidim.h"
#include "template_helper.h"
#include "type_traits.h"
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Indexing, size_t... Resolution>
struct static_multidim_resolution {
  static constexpr size_t num_dimensions() { return sizeof...(Resolution); }
#if has_cxx17_support()
  //----------------------------------------------------------------------------
  static constexpr size_t num_elements() { return (Resolution * ...); }
#else
  //----------------------------------------------------------------------------
  static constexpr size_t num_elements() {
    constexpr auto res = size();
    size_t         acc = 1;
    for (auto r : res) { acc *= r; }
    return acc;
  }
#endif
  //----------------------------------------------------------------------------
  static constexpr auto size() {
    return std::array<size_t, num_dimensions()>{Resolution...};
  }
  //----------------------------------------------------------------------------
  static constexpr auto size(size_t i) { return size()[i]; }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  static constexpr bool in_range(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
#if has_cxx17_support()
    return ((is >= 0) && ...) &&
           ((static_cast<size_t>(is) < Resolution) && ...);
#else
    const std::array<size_t, num_elements()> js{static_cast<size_t>(is)...};
    for (size_t i = 0; i < num_elements(); ++i) {
      if (js[i] < 0 || js[i] >= size(i)) { return false; }
    }
    return true;
#endif
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  static constexpr auto in_range(const std::array<size_t, N>& is) {
    return invoke_unpacked([](auto... is) { return in_range(is...); },
                           unpack(is));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto in_range(const std::vector<size_t>& is) {
    for (size_t i = 0; i < is.size(); ++i) {
      if (is[i] < 0 || is[i] >= size(i)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
//#ifndef NDEBUG
//  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
//  static auto plain_index(Is... is) {
//    if (!in_range(std::forward<Is>(is)...)) {
//      std::stringstream ss;
//      ss << "is out of bounds: [ ";
//      for (auto i :
//           std::array<size_t, sizeof...(Is)>{static_cast<size_t>(is)...}) {
//        ss << std::to_string(i) + " ";
//      }
//      ss << "]\n";
//      throw std::runtime_error{ss.str()};
//    }
//#else
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  static constexpr auto plain_index(Is... is) {
//#endif
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return Indexing::plain_index(
        std::array<size_t, num_dimensions()>{Resolution...}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  static constexpr auto plain_index(const std::array<size_t, N>& is) {
    return invoke_unpacked([](auto... is) { return plain_index(is...); },
                           unpack(is));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Is, enable_if_integral<Is> = true>
  static auto plain_index(const std::vector<Is>& is) {
//#ifndef NDEBUG
//    if (!in_range(std::forward<Is>(is)...)) {
//      std::stringstream ss;
//      ss << "is out of bounds: [ ";
//      for (auto i :
//           std::array<size_t, sizeof...(Is)>{static_cast<size_t>(is)...}) {
//        ss << std::to_string(i) + " ";
//      }
//      ss << "]\n";
//      throw std::runtime_error{ss.str()};
//    }
//#endif
    assert(is.size() == num_dimensions() &&
           "number of indices does not match number of dimensions");
    return Indexing::plain_index(
        std::array<size_t, num_dimensions()>{Resolution...}, is);
  }
  //----------------------------------------------------------------------------
  static constexpr auto indices() { return static_multidim{Resolution...}; }
};
//==============================================================================
template <typename Indexing>
class dynamic_multidim_resolution {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  std::vector<size_t> m_size;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  dynamic_multidim_resolution() = default;
  dynamic_multidim_resolution(const dynamic_multidim_resolution& other) =
      default;
  dynamic_multidim_resolution(dynamic_multidim_resolution&& other) = default;
  dynamic_multidim_resolution& operator                            =(
      const dynamic_multidim_resolution& other) = default;
  dynamic_multidim_resolution& operator=(dynamic_multidim_resolution&& other) =
      default;
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  dynamic_multidim_resolution(
      const dynamic_multidim_resolution<OtherIndexing>& other)
      : m_size{other.size()} {}
  template <typename OtherIndexing>
  dynamic_multidim_resolution(
      dynamic_multidim_resolution<OtherIndexing>&& other)
      : m_size{std::move(other.m_size)} {}
  template <typename OtherIndexing>
  dynamic_multidim_resolution& operator=(
      const dynamic_multidim_resolution& other) {
    m_size = other.m_size;
    return *this;
  }
  template <typename OtherIndexing>
  dynamic_multidim_resolution& operator=(dynamic_multidim_resolution&& other) {
    m_size = std::move(other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... Resolution,
            enable_if_integral<std::decay_t<Resolution>...> = true>
  dynamic_multidim_resolution(Resolution... resolution)
      : m_size{static_cast<size_t>(resolution)...} {}
  //----------------------------------------------------------------------------
  dynamic_multidim_resolution(const std::vector<size_t>& resolution)
      : m_size(resolution) {}
  //----------------------------------------------------------------------------
  dynamic_multidim_resolution(std::vector<size_t>&& resolution)
      : m_size(std::move(resolution)) {}
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_resolution(const std::vector<UInt>& resolution)
      : m_size(begin(resolution), end(resolution)) {}
  //----------------------------------------------------------------------------
  template <typename UInt, size_t N, enable_if_unsigned_integral<UInt> = true>
  dynamic_multidim_resolution(const std::array<UInt, N>& resolution)
      : m_size(begin(resolution), end(resolution)) {}

  //----------------------------------------------------------------------------
  // comparisons
  //----------------------------------------------------------------------------
 public:
  template <typename OtherIndexing>
  bool operator==(
      const dynamic_multidim_resolution<OtherIndexing>& other) const {
    if (num_dimensions() != other.num_dimensions()) { return false; }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] != other.size(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  bool operator!=(
      const dynamic_multidim_resolution<OtherIndexing>& other) const {
    if (num_dimensions() == other.num_dimensions()) { return false; }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] == other.size(i)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  size_t num_dimensions() const { return m_size.size(); }
  //----------------------------------------------------------------------------
  const auto& size() const { return m_size; }
  /// \return size of dimensions i
  auto size(const size_t i) const { return m_size[i]; }
  //----------------------------------------------------------------------------
  size_t num_elements() const {
    return std::accumulate(begin(m_size), end(m_size), size_t(1),
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
  template <typename... Resolution,
            enable_if_integral<std::decay_t<Resolution>...> = true>
  void resize(Resolution... resolution) {
    m_size = {static_cast<size_t>(resolution)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  void resize(const std::array<size_t, N>& resolution) {
    m_size = std::vector(begin(resolution), end(resolution));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t>&& resolution) {
    m_size = std::move(resolution);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(const std::vector<size_t>& resolution) {
    m_size = resolution;
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr auto in_range(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    constexpr size_t            N = sizeof...(is);
    const std::array<size_t, N> arr_is{static_cast<size_t>(is)...};
    for (size_t i = 0; i < N; ++i) {
      if (arr_is[i] < 0 || arr_is[i] >= size(i)) { return false; }
    }
    return true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  constexpr auto in_range(const std::array<size_t, N>& is) const {
    assert(N == num_dimensions());
    return invoke_unpacked([this](auto... is) { return in_range(is...); },
                           unpack(is));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto in_range(const std::vector<size_t>& is) const {
    for (size_t i = 0; i < is.size(); ++i) {
      if (is[i] < 0 || is[i] >= size(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr auto plain_index(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return Indexing::plain_index(m_size, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto plain_index(const std::vector<size_t>& is) const {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return Indexing::plain_index(m_size, is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  constexpr auto plain_index(const std::array<size_t, N>& is) const {
    assert(N == num_dimensions());
    return invoke_unpacked(
        [&](auto&&... is) { return Indexing::plain_index(m_size, is...); },
        unpack(is));
  }
  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t gi) const {
    return Indexing::multi_index(m_size, gi);
  }
  //----------------------------------------------------------------------------
  constexpr auto indices() const { return dynamic_multidim{m_size}; }
};

//==============================================================================
// deduction guides
//==============================================================================
#if has_cxx17_support()
dynamic_multidim_resolution()->dynamic_multidim_resolution<x_fastest>;

template <typename Indexing>
dynamic_multidim_resolution(const dynamic_multidim_resolution<Indexing>&)
    ->dynamic_multidim_resolution<Indexing>;
template <typename Indexing>
dynamic_multidim_resolution(dynamic_multidim_resolution<Indexing> &&)
    ->dynamic_multidim_resolution<Indexing>;
template <typename... Resolution>
dynamic_multidim_resolution(Resolution...)
    ->dynamic_multidim_resolution<x_fastest>;
#endif

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
