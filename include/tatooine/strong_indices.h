#ifndef TATOOINE_STRONG_INDICES_H
#define TATOOINE_STRONG_INDICES_H

//==============================================================================
namespace tatooine {
//==============================================================================

struct index {
  static constexpr size_t invalid_idx = std::numeric_limits<size_t>::max();

  //==========================================================================
  size_t i;

  //==========================================================================
  constexpr index() : i{invalid_idx} {}
  constexpr index(size_t _i) : i{_i} {}
  constexpr index(const index&) = default;
  constexpr index(index&&)      = default;
  constexpr index& operator=(const index&) = default;
  constexpr index& operator=(index&&) = default;

  //--------------------------------------------------------------------------
  auto& operator++() {
    ++this->i;
    return *this;
  }
  auto& operator--() {
    --this->i;
    return *this;
  }
  auto& operator=(size_t i) {
    this->i = i;
    return *this;
  }
};

//==============================================================================
struct vertex : index {
  constexpr vertex() = default;
  constexpr vertex(size_t i) : index{i} {}
  constexpr vertex(const vertex&) = default;
  constexpr vertex(vertex&&)      = default;
  constexpr vertex& operator=(const vertex&) = default;
  constexpr vertex& operator=(vertex&&) = default;

  constexpr bool operator==(vertex other) const { return this->i == other.i; }
  constexpr bool operator!=(vertex other) const { return this->i != other.i; }
  constexpr bool operator<(vertex other) const { return this->i < other.i; }
  static constexpr auto invalid() { return vertex{index::invalid_idx}; }
};

//==============================================================================
struct edge : index {
  constexpr edge() = default;
  constexpr edge(size_t i) : index{i} {}
  constexpr edge(const edge&) = default;
  constexpr edge(edge&&)      = default;
  constexpr edge& operator=(const edge&) = default;
  constexpr edge& operator=(edge&&) = default;

  constexpr bool operator==(const edge& other) const {
    return this->i == other.i;
  }
  constexpr bool operator!=(const edge& other) const {
    return this->i != other.i;
  }
  constexpr bool operator<(const edge& other) const {
    return this->i < other.i;
  }
  static constexpr auto invalid() { return edge{index::invalid_idx}; }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
