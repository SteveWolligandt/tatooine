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
  static constexpr auto num_dimensions() { return sizeof...(Resolution); }
  static constexpr auto num_components() { return (Resolution * ...); }
  //----------------------------------------------------------------------------
  static constexpr auto size() {
    return std::array<size_t, num_dimensions()>{Resolution...};
  }
  //----------------------------------------------------------------------------
  static constexpr auto size(size_t i) { return size()[i]; }
  //----------------------------------------------------------------------------
  static constexpr auto in_range(integral auto... is) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return ((is >= 0) && ...) &&
           ((static_cast<size_t>(is) < Resolution) && ...);
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
//  static auto plain_index(integral auto... is) {
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
  static constexpr auto plain_index(integral auto... is) {
//#endif
    static_assert(sizeof...(is) == num_dimensions(),
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
  template <integral Is>
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
  //----------------------------------------------------------------------------
  dynamic_multidim_resolution(const dynamic_multidim_resolution& other)
    = default;
  dynamic_multidim_resolution(dynamic_multidim_resolution&& other) noexcept
    = default;
  //----------------------------------------------------------------------------
  auto operator=(const dynamic_multidim_resolution& other)
      -> dynamic_multidim_resolution& = default;
  auto operator=(dynamic_multidim_resolution&& other) noexcept
      -> dynamic_multidim_resolution& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_resolution() = default;
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  explicit dynamic_multidim_resolution(
      const dynamic_multidim_resolution<OtherIndexing>& other)
      : m_size{other.size()} {}

  template <typename OtherIndexing>
  explicit dynamic_multidim_resolution(
      dynamic_multidim_resolution<OtherIndexing>&& other)
      : m_size{std::move(other.m_size)} {}

  template <typename OtherIndexing>
  auto operator=(const dynamic_multidim_resolution& other)
      -> dynamic_multidim_resolution& {
    m_size = other.m_size;
    return *this;
  }
  template <typename OtherIndexing>
  auto operator=(dynamic_multidim_resolution&& other)
      -> dynamic_multidim_resolution& {
    m_size = std::move(other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_resolution(integral auto... resolution)
      : m_size{static_cast<size_t>(resolution)...} {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_resolution(std::vector<size_t> resolution)
      : m_size(std::move(resolution)) {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_resolution(std::vector<size_t>&& resolution)
      : m_size(std::move(resolution)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt>
  explicit dynamic_multidim_resolution(const std::vector<UInt>& resolution)
      : m_size(begin(resolution), end(resolution)) {}
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, size_t N>
  explicit dynamic_multidim_resolution(const std::array<UInt, N>& resolution)
      : m_size(begin(resolution), end(resolution)) {}

  //----------------------------------------------------------------------------
  // comparisons
  //----------------------------------------------------------------------------
 public:
  template <typename OtherIndexing>
  auto operator==(
      const dynamic_multidim_resolution<OtherIndexing>& other) const {
    if (num_dimensions() != other.num_dimensions()) { return false; }
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] != other.size(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator!=(
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
  auto num_dimensions() const { return m_size.size(); }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto size() const -> const auto& { return m_size; }
  /// \return size of dimensions i
  auto size(const size_t i) const { return m_size[i]; }
  //----------------------------------------------------------------------------
  auto num_components() const {
    return std::accumulate(begin(m_size), end(m_size), size_t(1),
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
  void resize(integral auto... resolution) {
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
  constexpr auto in_range(integral auto... is) const {
    assert(sizeof...(is) == num_dimensions());
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
  constexpr auto plain_index(integral auto... is) const {
    assert(sizeof...(is) == num_dimensions());
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

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
