#ifndef TATOOINE_VERTEX_H
#define TATOOINE_VERTEX_H
//==============================================================================
#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include "linspace.h"
#include "tensor.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class grid;
template <typename Real, size_t N>
struct grid_vertex_iterator;
//==============================================================================
template <typename Real, size_t N>
struct grid_vertex {
  using grid_t          = grid<Real, N>;
  using linspace_iter_t = typename linspace<Real>::iterator;
  using iterator        = grid_vertex_iterator<Real, N>;
  //==========================================================================
  std::array<linspace_iter_t, N> iterators;
  //==========================================================================
  grid_vertex(const grid_vertex& other) : iterators{other.iterators} {}
  //--------------------------------------------------------------------------
  grid_vertex(grid_vertex&& other) : iterators{std::move(other.iterators)} {}
  //--------------------------------------------------------------------------
 private:
  template <typename... Is,  size_t... Js>
  grid_vertex(const grid_t& g, std::index_sequence<Js...>, Is... is)
   : iterators{linspace_iter_t{&g.dimension(Js), std::size_t(is)}...} {}
  //--------------------------------------------------------------------------
 public:
  template <typename... Is, enable_if_integral<Is...> = true>
  explicit grid_vertex(const grid_t& g, Is... is)
    : grid_vertex{g, std::make_index_sequence<N>{}, is...} {
    static_assert(sizeof...(Is) == N);
  }
  //--------------------------------------------------------------------------
  template <typename... Its>
  explicit grid_vertex(linspace_iter_t head_it, const Its&... tail_it)
      : iterators{head_it, tail_it...} {
    static_assert(sizeof...(Its) == N - 1,
                  "number of linspace iterators does not match N");
  }
  //--------------------------------------------------------------------------
  auto operator=(const grid_vertex<Real, N>& other) -> grid_vertex& {
    if (this == &other) { return *this; }
    for (size_t i = 0; i < N; ++i) { iterators[i] = other.iterators[i]; }
    return *this;
  }
  //--------------------------------------------------------------------------
  auto operator=(grid_vertex<Real, N>&& other) -> grid_vertex& {
    for (size_t i = 0; i < N; ++i) { iterators[i] = other.iterators[i]; }
    return *this;
  }
  //--------------------------------------------------------------------------
  ~grid_vertex() = default;
  //--------------------------------------------------------------------------
  auto operator++() -> auto& {
    ++iterators.front();
    for (size_t i = 0; i < N - 1; ++i) {
      if (iterators[i] == iterators[i].end()) {
        iterators[i].to_begin();
        ++iterators[i + 1];
      }
    }
    return *this;
  }
  //--------------------------------------------------------------------------
  auto operator--() -> auto& {
    for (size_t i = 0; i < N; ++i)
      if (iterators[i] == iterators[i].begin()) {
        iterators[i].to_end();
        --iterators[i];
      } else {
        --iterators[i];
        break;
      }
    return *this;
  }
  auto operator[](size_t i)       -> auto&       { return iterators[i]; }
  auto operator[](size_t i) const -> const auto& { return iterators[i]; }
  //--------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto position(std::index_sequence<Is...> /*is*/) const {
    return vec<Real, N>{(*iterators[Is])...};
  }
  //--------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto to_array(std::index_sequence<Is...> /*is*/) const {
    return std::array<Real, N>{*iterators[Is]...};
  }
  //--------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto indices(std::index_sequence<Is...> /*is*/) const {
    return vec<size_t, N>{iterators[Is].i()...};
  }
  //--------------------------------------------------------------------------
 public:
  constexpr auto position() const {
    return position(std::make_index_sequence<N>());
  }
  //--------------------------------------------------------------------------
  constexpr auto to_array() const {
    return to_array(std::make_index_sequence<N>());
  }
  //--------------------------------------------------------------------------
  constexpr auto indices() const {
    return indices(std::make_index_sequence<N>());
  }
  //--------------------------------------------------------------------------
  constexpr auto global_index() const {
    size_t idx = 0;
    size_t factor = 1;
    for (const auto& it : iterators) {
      idx += factor * it.i();
      factor *= it.linspace().size();
    }
    return idx;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator==(const grid_vertex& other) const -> bool {
    for (size_t i = 0; i < N; ++i) {
      if (iterators[i] != other.iterators[i]) { return false; }
    }
    return true;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<(const grid_vertex& other) const -> bool {
    for (size_t i = 0; i < N; ++i) {
      if (iterators[i].i() < other.iterators[i].i()) return true;
      if (iterators[i].i() > other.iterators[i].i()) return false;
    }
    return false;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>(const grid_vertex& other) const -> bool {
    return !operator<(other);
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<=(const grid_vertex& other) const -> bool {
    return operator==(other) || operator<(other);
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>=(const grid_vertex& other) const -> bool {
    return operator==(other) || operator>(other);
  }
  //--------------------------------------------------------------------------
  constexpr auto operator!=(const grid_vertex& other) const -> bool {
    return !operator==(other);
  }
  //--------------------------------------------------------------------------
  auto to_string() const {
    std::string str;
    str += "[ ";
    for (const auto& i : iterators) str += std::to_string(i.i()) + ' ';
    str += "]";
    return str;
  }
};
//==============================================================================
template <typename Real, size_t N>
struct grid_vertex_iterator {
  using difference_type   = size_t;
  using value_type        = grid_vertex<Real, N>;
  using pointer           = value_type*;
  using reference         = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;

  grid_vertex<Real, N> v;

  auto operator*() -> auto& { return v; }
  auto operator++() -> auto& {
    ++v;
    return *this;
  }
  auto operator--() -> auto& {
    --v;
    return *this;
  }
  auto operator==(const grid_vertex_iterator& other) { return v == other.v; }
  auto operator!=(const grid_vertex_iterator& other) { return v != other.v; }
};

//==============================================================================

template <typename Real, size_t N>
inline auto operator<<(std::ostream& out, const grid_vertex<Real, N>& v)
    -> auto& {
  out << "[ ";
  for (const auto& i : v.iterators) out << i.i() << ' ';
  out << "]";
  return out;
}

template <typename Real, size_t N>
auto next(grid_vertex_iterator<Real, N> it, size_t num) {
  for (size_t i = 0; i < num; ++i) ++it;
  return it;
}

template <typename Real, size_t N>
auto prev(grid_vertex_iterator<Real, N> it, size_t num) {
  for (size_t i = 0; i < num; ++i) --it;
  return it;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
