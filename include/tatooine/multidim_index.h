#ifndef TATOOINE_MULTIDIMENSION_H
#define TATOOINE_MULTIDIMENSION_H

#include <array>
#include <numeric>
#include <sstream>

#include "functional.h"
#include "template_helper.h"
#include "type_traits.h"
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
/// converts multi-dimensional index to a one dimensional index where first
/// dimensions grows fastest
struct x_fastest {
  template <typename ResIt, typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(ResIt res_it, Is... is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    map(
        [&](size_t i) {
          idx += i * multiplier;
          multiplier *= *res_it;
          ++res_it;
        },
        is...);
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename... Is,
            enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::array<size_t, N>& resolution,
                                    Is... is) {
    return plain_idx(begin(resolution), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::vector<size_t>& resolution,
                                    Is... is) {
    return plain_idx(begin(resolution), is...);
  }
  //----------------------------------------------------------------------------
  static auto multi_index(const std::vector<size_t>& resolution,
                                    size_t                     plain_idx) {
    std::vector<size_t> is(resolution.size());
    size_t              multiplier =
        std::accumulate(begin(resolution), std::prev(end(resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto res_it = std::prev(end(resolution), 2);
    for (size_t j = 0; j < resolution.size(); ++j, --res_it) {
      size_t i = resolution.size() - 1 - j;
      is[i]    = plain_idx / multiplier;
      plain_idx -= is[i] * multiplier;
      if (res_it >= begin(resolution)) { multiplier /= *res_it; }
    }
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  constexpr static auto multi_index(const std::array<size_t, N>& resolution,
                                    size_t                       plain_idx) {
    auto   is = make_array<size_t, N>();
    size_t multiplier =
        std::accumulate(begin(resolution), std::prev(end(resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto res_it = std::prev(end(resolution), 2);
    for (size_t j = 0; j < N; ++j, --res_it) {
      size_t i = N - 1 - j;
      is[i]    = plain_idx / multiplier;
      plain_idx -= is[i] * multiplier;
      if (res_it >= begin(resolution)) { multiplier /= *res_it; }
    }
    return is;
  }
};
//==============================================================================
/// converts multi-dimensional index to a one dimensional index where first
/// dimensions grows slowest
struct x_slowest {
 private:
  template <typename Resolution, typename... Is,
            enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t internal_plain_idx(const Resolution& resolution,
                                             Is... p_is) {
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < is.size(); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      multiplier *= resolution[is.size() - 1 - i];
    }
    return idx;
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::vector<size_t>& resolution,
                                    Is... is) {
    assert(sizeof...(Is) == num_dims());
    return internal_plain_idx(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::array<size_t, N>& resolution,
                                    Is... is) {
    static_assert(sizeof...(Is) == N);
    return internal_plain_idx(resolution, is...);
  }

  //----------------------------------------------------------------------------
  static auto multi_index(const std::vector<size_t>& resolution,
                          size_t                     /*plain_idx*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::vector<size_t>&, size_t) not "
        "implemented"};
    std::vector<size_t> is(resolution.size());
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  static auto multi_index(const std::array<size_t, N>& /*resolution*/,
                          size_t                       /*plain_idx*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::array<size_t, N>&, size_t) not "
        "implemented"};
    auto is = make_array<size_t, N>();
    return is;
  }
};
//==============================================================================
/// converts multi-dimensional index to a one dimensional index using a
/// space-filling curve algorithm
struct hilbert_curve {
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::vector<size_t>& /*resolution*/,
                                    Is... /*is*/) {
    throw std::runtime_error{
        "hilbert_curve::plain_idx(const std::vector<size_t>&, Is... is) not "
        "implemented"};
    return 0;
  }
  template <size_t N, typename... Is,
            enable_if_integral<std::decay_t<Is>...> = true>
  constexpr static size_t plain_idx(const std::array<size_t, N>& /*resolution*/,
                                    Is... /*is*/) {
    throw std::runtime_error{
        "hilbert_curve::plain_idx(const std::array<size_t, N>&, Is... is) not "
        "implemented"};
    return 0;
  }
  //----------------------------------------------------------------------------
  static auto multi_index(const std::vector<size_t>& resolution,
                          size_t                     /*plain_idx*/) {
    std::vector<size_t> is(resolution.size());
    throw std::runtime_error{
        "hilbert_curve::multi_index(const std::vector<size_t>&, size_t) not "
        "implemented"};
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  static auto multi_index(const std::array<size_t, N>& /*resolution*/,
                          size_t                       /*plain_idx*/) {
    auto is = make_array<size_t, N>();
    throw std::runtime_error{
        "hilbert_curve::multi_index(const std::array<size_t, N>&, size_t) not "
        "implemented"};
    return is;
  }
};
//==============================================================================
template <typename Indexing, size_t... Resolution>
struct static_multidimension {
  static constexpr size_t num_dimensions() { return sizeof...(Resolution); }
#if has_cxx17_support()
  static constexpr size_t num_elements() { return (Resolution * ...); }
#else
  static constexpr size_t num_elements() {
    constexpr auto res = resolution();
    return std::accumulate(begin(res), end(res), size_t(1),
                           std::multiplies<size_t>{});
  }
#endif
  static constexpr auto resolution() { return std::array<size_t, num_dimensions()>{Resolution...}; }
  static constexpr auto size(size_t i) { return resolution()[i]; }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  static constexpr bool in_range(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
#if has_cxx17_support()
    return ((is >= 0) && ...) &&
           ((static_cast<size_t>(is) < Resolution) && ...);
#else
    const std::array<size_t, N> is{static_cast<size_t>(is)...};
    for (size_t i = 0; i < N; ++i) {
      if (is[i] < 0 || is[i] >= resolution(i)) { return false; }
    }
    return true;
#endif
  }
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  template <size_t N>
  static constexpr auto in_range(const std::array<size_t, N>& is) {
    return invoke_unpacked([](auto... is) { return in_range(is...); },
                           unpack(is));
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<std::decay_t<Is>...> = true>
  static constexpr auto plain_idx(Is... is) {
# ifndef NDEBUG
    if (!in_range(std::forward<Is>(is)...)) {
      std::stringstream ss;
      ss << "is out of bounds: [ ";
      for (auto i : std::array<size_t, sizeof...(Is)>{
               static_cast<size_t>(is)...}) {
        ss << std::to_string(i) + " ";
      }
      ss << "]\n";
      throw std::runtime_error{ss.str()};
    }
# endif
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return Indexing::plain_idx(
        std::array<size_t, num_dimensions()>{static_cast<size_t>(is)...});
  }
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  template <size_t N>
  static constexpr auto plain_idx(const std::array<size_t, N>& is) {
    return invoke_unpacked([](auto... is) { return plain_idx(is...); },
                           unpack(is));
  }
  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t gi) {
    return Indexing::multi_index(
        std::array<size_t, num_dimensions()>{Resolution...}, gi);
  }
};
//==============================================================================
template <typename Indexing>
class dynamic_multidim_index {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  std::vector<size_t> m_resolution;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  dynamic_multidim_index()                                    = default;
  dynamic_multidim_index(const dynamic_multidim_index& other) = default;
  dynamic_multidim_index(dynamic_multidim_index&& other)      = default;
  dynamic_multidim_index& operator=(const dynamic_multidim_index& other) =
      default;
  dynamic_multidim_index& operator=(dynamic_multidim_index&& other) = default;
  //----------------------------------------------------------------------------
  template <typename... Resolution,
            enable_if_integral<std::decay_t<Resolution>...> = true>
  dynamic_multidim_index(Resolution... resolution)
      : m_resolution{static_cast<size_t>(resolution)...} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  size_t num_dimensions() const { return m_resolution.size(); }
  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution; }
  /// \return size of dimensions i
  auto size(const size_t i) const { return m_resolution[i]; }
  //----------------------------------------------------------------------------
  size_t num_elements() const {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t(1),
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
  template <typename... Resolution,
            enable_if_integral<std::decay_t<Resolution>...>    = true>
  void resize(Resolution... resolution) {
    m_resolution = {static_cast<size_t>(resolution)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  void resize(const std::array<size_t, N>& resolution) {
    m_resolution = std::vector(begin(resolution), end(resolution));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t>&& resolution) {
    m_resolution = std::move(resolution);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(const std::vector<size_t>& resolution) {
    m_resolution = resolution;
  }
  //----------------------------------------------------------------------------
  template <typename... Is,
            enable_if_integral<std::decay_t<Is>...> = true>
  constexpr auto in_range(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    constexpr size_t N = sizeof...(is);
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
  //----------------------------------------------------------------------------
  template <typename... Is,
            enable_if_integral<std::decay_t<Is>...> = true>
  constexpr auto plain_idx(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return Indexing::plain_idx(m_resolution, is...);
  }
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  template <size_t N>
  constexpr auto plain_idx(const std::array<size_t, N>& is) const {
    assert(N == num_dimensions());
    return invoke_unpacked([this](auto... is) { return plain_idx(is...); },
                           unpack(is));
  }
  //----------------------------------------------------------------------------
  constexpr auto multi_index(size_t gi) const {
    return Indexing::multi_index(m_resolution, gi);
  }
};

//==============================================================================
// deduction guides
//==============================================================================
#if has_cxx17_support()
dynamic_multidim_index()->dynamic_multidim_index<x_fastest>;

template <typename Indexing>
dynamic_multidim_index(const dynamic_multidim_index<Indexing>&)
    ->dynamic_multidim_index<Indexing>;
template <typename Indexing>
dynamic_multidim_index(dynamic_multidim_index<Indexing> &&)
    ->dynamic_multidim_index<Indexing>;
template <typename... Resolution>
dynamic_multidim_index(Resolution...)->dynamic_multidim_index<x_fastest>;
#endif

//==============================================================================
template <size_t n>
struct multi_index_iterator;

//==============================================================================
template <size_t n>
struct multi_index {
  //----------------------------------------------------------------------------
  constexpr multi_index(std::array<std::pair<size_t, size_t>, n> ranges)
      : m_ranges{ranges} {}

  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr multi_index(const std::pair<Ts, Ts>&... ranges)
      : m_ranges{std::make_pair(static_cast<size_t>(ranges.first),
                                static_cast<size_t>(ranges.second))...} {}

  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr multi_index(Ts const (&... ranges)[2])
      : m_ranges{std::make_pair(static_cast<size_t>(ranges[0]),
                                static_cast<size_t>(ranges[1]))...} {}

  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_ranges[i]; }
  constexpr const auto& operator[](size_t i) const { return m_ranges[i]; }

  //----------------------------------------------------------------------------
  constexpr auto&       ranges() { return m_ranges; }
  constexpr const auto& ranges() const { return m_ranges; }

  //----------------------------------------------------------------------------
  constexpr auto begin() { return begin(std::make_index_sequence<n>{}); }
  constexpr auto end() { return end(std::make_index_sequence<n>{}); }

 private:
  //============================================================================
  std::array<std::pair<size_t, size_t>, n> m_ranges;

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto begin(std::index_sequence<Is...> /*is*/) {
    return multi_index_iterator<n>{*this,
                                   std::array<size_t, n>{((void)Is, 0)...}};
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto end(std::index_sequence<Is...> /*is*/) {
    std::array<size_t, n> a{((void)Is, 0)...};
    a.back() = m_ranges.back().second + 1;
    return multi_index_iterator<n>{*this, std::move(a)};
  }
};

//==============================================================================
template <size_t n>
struct multi_index_iterator {
  //----------------------------------------------------------------------------
  const multi_index<n>  m_cont;
  std::array<size_t, n> m_status;

  //----------------------------------------------------------------------------
  constexpr multi_index_iterator(const multi_index<n>&        c,
                                 const std::array<size_t, n>& status)
      : m_cont{c}, m_status{status} {}

  //----------------------------------------------------------------------------
  constexpr multi_index_iterator(const multi_index_iterator& other)
      : m_cont{other.m_cont}, m_status{other.m_status} {}

  //----------------------------------------------------------------------------
  constexpr void operator++() {
    ++m_status.front();
    auto range_it  = begin(m_cont.ranges());
    auto status_it = begin(m_status);
    for (; range_it != prev(end(m_cont.ranges())); ++status_it, ++range_it) {
      if (range_it->second < *status_it) {
        *status_it = 0;
        ++(*(status_it + 1));
      }
    }
  }

  //----------------------------------------------------------------------------
  constexpr auto operator==(const multi_index_iterator& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (m_status[i] != other.m_status[i]) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  constexpr auto operator!=(const multi_index_iterator& other) const {
    return !operator==(other);
  }

  //----------------------------------------------------------------------------
  constexpr auto operator*() const { return m_status; }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if has_cxx17_support()
template <typename... Ts>
multi_index(const std::pair<Ts, Ts>&... ranges)->multi_index<sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
multi_index(Ts const (&... ranges)[2])->multi_index<sizeof...(Ts)>;
#endif

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
