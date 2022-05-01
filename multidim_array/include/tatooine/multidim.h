#ifndef TATOOINE_MULTIDIM_H
#define TATOOINE_MULTIDIM_H
//==============================================================================
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

#include <cassert>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t N>
struct static_multidim_iterator;
//==============================================================================
template <std::size_t N>
struct static_multidim {
  //----------------------------------------------------------------------------
  // static methods
  //----------------------------------------------------------------------------
 public:
  static constexpr auto num_dimensions() { return N; }

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<std::pair<std::size_t, std::size_t>, N> m_ranges;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim(
      std::array<std::pair<std::size_t, std::size_t>, N> ranges)
      : m_ranges{ranges} {}
  //----------------------------------------------------------------------------
  template <integral... Ts>
  explicit constexpr static_multidim(std::pair<Ts, Ts> const&... ranges)
      : m_ranges{std::make_pair(static_cast<std::size_t>(ranges.first),
                                static_cast<std::size_t>(ranges.second))...} {}
  //----------------------------------------------------------------------------
  template <integral... Ts>
  constexpr static_multidim(Ts const (&... ranges)[2])  // NOLINT
      : m_ranges{std::make_pair(static_cast<std::size_t>(ranges[0]),
                                static_cast<std::size_t>(ranges[1]))...} {}
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim(integral auto const... res)
      : m_ranges{std::make_pair(static_cast<std::size_t>(0),
                                static_cast<std::size_t>(res))...} {}
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim(std::array<std::size_t, N> const& res)
      : m_ranges(make_array<std::pair<std::size_t, std::size_t>, N>()) {
    for (std::size_t i = 0; i < N; ++i) {
      m_ranges[i].second = res[i];
    }
  }
  //----------------------------------------------------------------------------
  explicit static_multidim(std::vector<std::size_t> const& res)
      : m_ranges(make_array<std::pair<std::size_t, std::size_t>, N>()) {
    assert(res.size() == N);
    for (std::size_t i = 0; i < N; ++i) {
      m_ranges[i].second = res[i];
    }
  }
  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto operator[](std::size_t i) -> auto& { return m_ranges[i]; }
  constexpr auto operator[](std::size_t i) const -> auto const& {
    return m_ranges[i];
  }
  //----------------------------------------------------------------------------
  constexpr auto ranges() -> auto& { return m_ranges; }
  constexpr auto ranges() const -> auto const& { return m_ranges; }
  //----------------------------------------------------------------------------
  constexpr auto begin() {
    return static_multidim_iterator<N>{*this, make_array<std::size_t, N>()};
  }
  //----------------------------------------------------------------------------
  constexpr auto end() {
    auto a   = make_array<std::size_t, N>();
    a.back() = m_ranges.back().second;
    return static_multidim_iterator<N>{*this, std::move(a)};
  }
};

//==============================================================================
template <std::size_t N>
struct static_multidim_iterator {
  //----------------------------------------------------------------------------
  static_multidim<N> const   m_cont;
  std::array<std::size_t, N> m_status;

  //----------------------------------------------------------------------------
  constexpr static_multidim_iterator(static_multidim<N> const&         c,
                                     std::array<std::size_t, N> const& status)
      : m_cont{c}, m_status{status} {}

  //----------------------------------------------------------------------------
  constexpr static_multidim_iterator(static_multidim_iterator const& other) =
      default;
  constexpr static_multidim_iterator(
      static_multidim_iterator&& other) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator          =(static_multidim_iterator const& other)
      -> static_multidim_iterator& = default;
  constexpr auto operator          =(static_multidim_iterator&& other) noexcept
      -> static_multidim_iterator& = default;
  //----------------------------------------------------------------------------
  ~static_multidim_iterator() = default;

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
  constexpr auto operator==(static_multidim_iterator const& other) const {
    for (std::size_t i = 0; i < N; ++i) {
      if (m_status[i] != other.m_status[i]) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator!=(static_multidim_iterator const& other) const {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator*() const { return m_status; }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
static_multidim(std::pair<Ts, Ts> const&... ranges)
    -> static_multidim<sizeof...(Ts)>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static_multidim(Ts const (&... ranges)[2])  // NOLINT
    ->static_multidim<sizeof...(Ts)>;       // NOLINT
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Res>
static_multidim(Res... res) -> static_multidim<sizeof...(Res)>;
//==============================================================================
struct dynamic_multidim {
 private:
  std::vector<std::pair<std::size_t, std::size_t>> m_ranges;
  struct iterator {
    //--------------------------------------------------------------------------
    dynamic_multidim const*  m_cont;
    std::vector<std::size_t> m_status;
    //--------------------------------------------------------------------------
    iterator(dynamic_multidim const& c, std::vector<std::size_t> status)
        : m_cont{&c}, m_status{std::move(status)} {}
    //--------------------------------------------------------------------------
    iterator(iterator const& other) = default;
    iterator(iterator&& other)      = default;
    //--------------------------------------------------------------------------
    auto operator=(iterator const& other) -> iterator& = default;
    auto operator=(iterator&& other) -> iterator& = default;
    //--------------------------------------------------------------------------
    ~iterator() = default;
    //--------------------------------------------------------------------------
    void operator++() {
      ++m_status.front();
      auto range_it  = m_cont->ranges().begin();
      auto status_it = m_status.begin();
      for (; range_it != prev(m_cont->ranges().end());
           ++status_it, ++range_it) {
        if (range_it->second <= *status_it) {
          *status_it = 0;
          ++(*(status_it + 1));
        }
      }
    }
    //--------------------------------------------------------------------------
    auto operator==(iterator const& other) const {
      if (m_cont != other.m_cont) {
        return false;
      }
      for (std::size_t i = 0; i < m_cont->num_dimensions(); ++i) {
        if (m_status[i] != other.m_status[i]) {
          return false;
        }
      }
      return true;
    }
    //--------------------------------------------------------------------------
    auto operator!=(iterator const& other) const { return !operator==(other); }
    //--------------------------------------------------------------------------
    auto operator*() const -> auto const& { return m_status; }
  };

 public:
  //----------------------------------------------------------------------------
  explicit dynamic_multidim(
      std::vector<std::pair<std::size_t, std::size_t>> ranges)
      : m_ranges(std::move(ranges)) {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim(
      std::vector<std::pair<std::size_t, std::size_t>>&& ranges)
      : m_ranges(std::move(ranges)) {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim(std::vector<std::size_t> const& res)
      : m_ranges(res.size(), std::make_pair<std::size_t, std::size_t>(0, 0)) {
    for (std::size_t i = 0; i < res.size(); ++i) {
      m_ranges[i].second = res[i];
    }
  }
  //----------------------------------------------------------------------------
  template <std::size_t N>
  explicit dynamic_multidim(
      std::array<std::pair<std::size_t, std::size_t>, N> const& ranges)
      : m_ranges(ranges.begin(), ranges.end()) {}
  //----------------------------------------------------------------------------
  template <std::size_t N>
  explicit constexpr dynamic_multidim(std::array<std::size_t, N> const& res)
      : m_ranges(N, std::make_pair<std::size_t, std::size_t>(0, 0)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_ranges[i].second = res[i];
    }
  }
  //----------------------------------------------------------------------------
  template <integral... Ts>
  explicit constexpr dynamic_multidim(std::pair<Ts, Ts> const&... ranges)
      : m_ranges{std::make_pair(static_cast<std::size_t>(ranges.first),
                                static_cast<std::size_t>(ranges.second))...} {}

  //----------------------------------------------------------------------------
  template <integral... Ts>
  constexpr dynamic_multidim(Ts const (&... ranges)[2])  // NOLINT
      : m_ranges{std::make_pair(static_cast<std::size_t>(ranges[0]),
                                static_cast<std::size_t>(ranges[1]))...} {}

  //----------------------------------------------------------------------------
  explicit constexpr dynamic_multidim(integral auto const... res)
      : m_ranges{std::make_pair(static_cast<std::size_t>(0),
                                static_cast<std::size_t>(res))...} {}

  //----------------------------------------------------------------------------
  auto operator[](std::size_t i) -> auto& { return m_ranges[i]; }
  auto operator[](std::size_t i) const -> auto const& { return m_ranges[i]; }

  //----------------------------------------------------------------------------
  [[nodiscard]] auto ranges()
      -> std::vector<std::pair<std::size_t, std::size_t>>& {
    return m_ranges;
  }
  [[nodiscard]] auto ranges() const
      -> std::vector<std::pair<std::size_t, std::size_t>> const& {
    return m_ranges;
  }
  //----------------------------------------------------------------------------
  auto begin() {
    return iterator{*this, std::vector<std::size_t>(m_ranges.size(), 0)};
  }
  //----------------------------------------------------------------------------
  auto end() {
    std::vector<std::size_t> v(m_ranges.size());
    v.back() = m_ranges.back().second;
    return iterator{*this, std::move(v)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto num_dimensions() const -> std::size_t {
    return m_ranges.size();
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
