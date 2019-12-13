#ifndef TATOOINE_MULTIDIM_H
#define TATOOINE_MULTIDIM_H

#include <cassert>

#include "type_traits.h"
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t N>
struct static_multidim_iterator;
//==============================================================================
template <size_t N>
struct static_multidim {
  //----------------------------------------------------------------------------
  // static methods
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t num_dimensions() { return N; }

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<std::pair<size_t, size_t>, N> m_ranges;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  constexpr static_multidim(std::array<std::pair<size_t, size_t>, N> ranges)
      : m_ranges{ranges} {}
  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr static_multidim(const std::pair<Ts, Ts>&... ranges)
      : m_ranges{std::make_pair(static_cast<size_t>(ranges.first),
                                static_cast<size_t>(ranges.second))...} {}
  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr static_multidim(Ts const (&... ranges)[2])
      : m_ranges{std::make_pair(static_cast<size_t>(ranges[0]),
                                static_cast<size_t>(ranges[1]))...} {}
  //----------------------------------------------------------------------------
  template <typename... Res, enable_if_integral<Res...> = true>
  constexpr static_multidim(Res... res)
      : m_ranges{std::make_pair(static_cast<size_t>(0),
                                static_cast<size_t>(res))...} {}
  //----------------------------------------------------------------------------
  constexpr static_multidim(const std::array<size_t, N>& res)
      : m_ranges(make_array<std::pair<size_t, size_t>, N>()) {
    for (size_t i = 0; i < N; ++i) { m_ranges[i].second = res[i]; }
  }
  //----------------------------------------------------------------------------
  static_multidim(const std::vector<size_t>& res)
      : m_ranges(make_array<std::pair<size_t, size_t>, N>()) {
    assert(res.size() == N);
    for (size_t i = 0; i < N; ++i) { m_ranges[i].second = res[i]; }
  }
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_ranges[i]; }
  constexpr const auto& operator[](size_t i) const { return m_ranges[i]; }
  //----------------------------------------------------------------------------
  constexpr auto&       ranges() { return m_ranges; }
  constexpr const auto& ranges() const { return m_ranges; }
  //----------------------------------------------------------------------------
  constexpr auto begin() {
    return static_multidim_iterator<N>{*this, make_array<size_t, N>()};
  }
  //----------------------------------------------------------------------------
  constexpr auto end() {
    auto a   = make_array<size_t, N>();
    a.back() = m_ranges.back().second;
    return static_multidim_iterator<N>{*this, std::move(a)};
  }
};

//==============================================================================
template <size_t N>
struct static_multidim_iterator {
  //----------------------------------------------------------------------------
  const static_multidim<N>  m_cont;
  std::array<size_t, N> m_status;

  //----------------------------------------------------------------------------
  constexpr static_multidim_iterator(const static_multidim<N>&        c,
                                 const std::array<size_t, N>& status)
      : m_cont{c}, m_status{status} {}

  //----------------------------------------------------------------------------
  constexpr static_multidim_iterator(const static_multidim_iterator& other)
      : m_cont{other.m_cont}, m_status{other.m_status} {}

  //----------------------------------------------------------------------------
  constexpr void operator++() {
    ++m_status.front();
    auto range_it  = begin(m_cont.ranges());
    auto status_it = begin(m_status);
    for (; range_it != prev(end(m_cont.ranges())); ++status_it, ++range_it) {
      if (range_it->second <= *status_it) {
        *status_it = 0;
        ++(*(status_it + 1));
      }
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto operator==(const static_multidim_iterator& other) const {
    for (size_t i = 0; i < N; ++i) {
      if (m_status[i] != other.m_status[i]) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator!=(const static_multidim_iterator& other) const {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator*() const { return m_status; }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if has_cxx17_support()
template <typename... Ts>
static_multidim(const std::pair<Ts, Ts>&... ranges)->static_multidim<sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static_multidim(Ts const (&... ranges)[2])->static_multidim<sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Res>
static_multidim(Res... res)->static_multidim<sizeof...(Res)>;
#endif
//==============================================================================
struct dynamic_multidim {
 private:
  std::vector<std::pair<size_t, size_t>> m_ranges;
  struct iterator {
    //----------------------------------------------------------------------------
    const dynamic_multidim* m_cont;
    std::vector<size_t>     m_status;

    //----------------------------------------------------------------------------
    iterator(const dynamic_multidim&    c,
                              const std::vector<size_t>& status)
        : m_cont{&c}, m_status{status} {}
    //----------------------------------------------------------------------------
    iterator(const dynamic_multidim& c,
                              std::vector<size_t>&&   status)
        : m_cont{&c}, m_status{std::move(status)} {}
    //----------------------------------------------------------------------------
    iterator(const iterator& other) = default;
    //----------------------------------------------------------------------------
    iterator(iterator&& other) = default;

    //----------------------------------------------------------------------------
    void operator++() {
      ++m_status.front();
      auto range_it  = m_cont->ranges().begin();
      auto status_it = m_status.begin();
      for (; range_it != prev(m_cont->ranges().end()); ++status_it, ++range_it) {
        if (range_it->second <= *status_it) {
          *status_it = 0;
          ++(*(status_it + 1));
        }
      }
    }
    //----------------------------------------------------------------------------
    constexpr auto operator==(const iterator& other) const {
      if (m_cont != other.m_cont) { return false; }
      for (size_t i = 0; i < m_cont->num_dimensions(); ++i) {
        if (m_status[i] != other.m_status[i]) { return false; }
      }
      return true;
    }
    //----------------------------------------------------------------------------
    constexpr auto operator!=(const iterator& other) const {
      return !operator==(other);
    }
    //----------------------------------------------------------------------------
    const auto& operator*() const { return m_status; }
  };

 public:
  //----------------------------------------------------------------------------
  dynamic_multidim(const std::vector<std::pair<size_t, size_t>>& ranges)
      : m_ranges(ranges) {}
  //----------------------------------------------------------------------------
  dynamic_multidim(std::vector<std::pair<size_t, size_t>>&& ranges)
      : m_ranges(std::move(ranges)) {}
  //----------------------------------------------------------------------------
  dynamic_multidim(const std::vector<size_t>& res)
      : m_ranges(res.size(), std::make_pair<size_t, size_t>(0, 0)) {
    for (size_t i = 0; i < res.size(); ++i) { m_ranges[i].second = res[i]; }
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  dynamic_multidim(const std::array<std::pair<size_t, size_t>, N>& ranges)
      : m_ranges(ranges.begin(), ranges.end()) {}
  //----------------------------------------------------------------------------
  template <size_t N>
  constexpr dynamic_multidim(const std::array<size_t, N>& res)
      : m_ranges(N, std::make_pair<size_t, size_t>(0, 0)) {
    for (size_t i = 0; i < N; ++i) { m_ranges[i].second = res[i]; }
  }
  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr dynamic_multidim(const std::pair<Ts, Ts>&... ranges)
      : m_ranges{std::make_pair(static_cast<size_t>(ranges.first),
                                static_cast<size_t>(ranges.second))...} {}

  //----------------------------------------------------------------------------
  template <typename... Ts, enable_if_integral<Ts...> = true>
  constexpr dynamic_multidim(Ts const (&... ranges)[2])
      : m_ranges{std::make_pair(static_cast<size_t>(ranges[0]),
                                static_cast<size_t>(ranges[1]))...} {}

  //----------------------------------------------------------------------------
  template <typename... Res, enable_if_integral<Res...> = true>
  constexpr dynamic_multidim(Res... res)
      : m_ranges{std::make_pair(static_cast<size_t>(0),
                                static_cast<size_t>(res))...} {}


  //----------------------------------------------------------------------------
  auto&       operator[](size_t i) { return m_ranges[i]; }
  const auto& operator[](size_t i) const { return m_ranges[i]; }

  //----------------------------------------------------------------------------
  std::vector<std::pair<size_t, size_t>>& ranges() {
    return m_ranges;
  }
  //----------------------------------------------------------------------------
  const std::vector<std::pair<size_t, size_t>>& ranges() const {
    return m_ranges;
  }
  //----------------------------------------------------------------------------
  auto begin() {
    return iterator{*this, std::vector<size_t>(m_ranges.size(), 0)};
  }
  //----------------------------------------------------------------------------
  auto end() {
    std::vector<size_t>v(m_ranges.size());
    v.back() = m_ranges.back().second;
    return iterator{*this, std::move(v)};
  }
  //----------------------------------------------------------------------------
  size_t num_dimensions() const { return m_ranges.size(); }
};


//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
