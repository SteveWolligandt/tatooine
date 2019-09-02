#ifndef TATOOINE_CHUNKED_DATA_H
#define TATOOINE_CHUNKED_DATA_H

#include <array>
#include <boost/range/algorithm.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "functional.h"
#include "multidimension.h"
#include "utility.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t... Sizes>
struct chunk : static_multidimension<Sizes...> {
  using parent_t                 = static_multidimension<Sizes...>;
  using data_t                   = std::vector<T>;
  using iterator                 = typename data_t::iterator;
  using const_iterator           = typename data_t::const_iterator;
  static constexpr auto num_data = parent_t::num_data;
  data_t                m_data;

  chunk() : m_data(num_data) {}

  chunk(const chunk&) = default;
  chunk(chunk&&)      = default;
  chunk& operator=(const chunk&) = default;
  chunk& operator=(chunk&&) = default;

  //----------------------------------------------------------------------------
  template <typename... Indices>
  T& operator()(Indices&&... indices) {
    return at(std::forward<Indices>(indices)...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices>
  T& at(Indices&&... indices) {
    static_assert((std::is_integral_v<std::decay_t<Indices>> && ...),
                  "chunk::operator() only takes integral types");
    auto gi = this->global_idx(std::forward<Indices>(indices)...);
    return m_data[gi];
  }

  //----------------------------------------------------------------------------
  template <typename... Indices>
  const T& operator()(Indices&&... indices) const {
    return at(std::forward<Indices>(indices)...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices>
  const T& at(Indices&&... indices) const {
    static_assert((std::is_integral_v<std::decay_t<Indices>> && ...),
                  "chunk::operator() only takes integral types");
    return m_data[this->global_idx(std::forward<Indices>(indices)...)];
  }

  //----------------------------------------------------------------------------
  auto begin() { return begin(m_data); }
  auto begin() const { return begin(m_data); }

  //----------------------------------------------------------------------------
  auto end() { return end(m_data); }
  auto end() const { return end(m_data); }

  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T>...>
  void randu(T min = 0, T max = 1,
             RandomEngine&& random_engine = RandomEngine{
                 std::random_device{}()}) {
    auto distribution = [min, max] {
      if constexpr (std::is_integral_v<T>) {
        return std::uniform_int_distribution<T>{min, max};
      } else if (std::is_floating_point_v<T>) {
        return std::uniform_real_distribution<T>{min, max};
      }
    }();
    boost::generate(m_data, [&random_engine, &distribution] {
      return distribution(random_engine);
    });
  }
};

//------------------------------------------------------------------------------
template <typename T, size_t N, size_t R, size_t... Is>
auto create_chunk(std::index_sequence<Is...> /*is*/) {
  return chunk<T, ((void)Is, R)...>{};
}

//------------------------------------------------------------------------------
template <typename T, size_t N, size_t R>
auto create_chunk() {
  return create_chunk<T, N, R>(std::make_index_sequence<N>{});
}

//------------------------------------------------------------------------------
template <typename S, size_t N, size_t R>
struct create_chunk_type {
  using T = decltype(create_chunk<S, N, R>());
};

//==============================================================================
template <typename T, size_t N, size_t ChunkRes = 8>
struct chunked_data {
  //============================================================================
  using this_t    = chunked_data<T, N, ChunkRes>;
  using chunk_t   = typename create_chunk_type<T, N, ChunkRes>::T;
  using indices_t = std::make_index_sequence<N>;

  //============================================================================
  static constexpr auto chunk_res = chunk_t::sizes();

  //============================================================================
  dynamic_multidimension<N>             m_data_structure;
  dynamic_multidimension<N>             m_chunk_structure;
  std::vector<std::unique_ptr<chunk_t>> m_chunks;

 private:
  //============================================================================
  template <typename S, size_t... Is>
  chunked_data(const std::array<S, N>& data_sizes)
      : m_data_structure{data_sizes},
        m_chunk_structure{static_cast<size_t>(
            std::ceil(static_cast<double>(data_sizes(Is)) / ChunkRes))...},
        m_chunks{std::accumulate(begin(this->sizes()),
                                 end(this->sizes()), size_t(1),
                                 std::multiplies<size_t>{})} {}

 public:
  //----------------------------------------------------------------------------
  template <typename... Sizes,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Sizes>> && ...)>,
            typename = std::enable_if_t<sizeof...(Sizes) == N>>
  chunked_data(Sizes&&... sizes)
      : m_data_structure{static_cast<size_t>(sizes)...},
        m_chunk_structure{static_cast<size_t>(
            std::ceil(static_cast<double>(sizes) / ChunkRes))...},
        m_chunks{m_chunk_structure.num_data()} {}

  //----------------------------------------------------------------------------
  template <typename Container, typename... Sizes,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Sizes>> && ...)>,
            typename = std::enable_if_t<sizeof...(Sizes) == N>>
  chunked_data(Container&& data, Sizes&&... sizes)
      : m_data_structure{static_cast<size_t>(sizes)...},
        m_chunk_structure{static_cast<size_t>(
            std::ceil(static_cast<double>(sizes) / ChunkRes))...},
        m_chunks{m_chunk_structure.num_data()} {
    assert(data.size() == (static_cast<size_t>(sizes) * ...));
    size_t i = 0;
    for (const auto& d : data) { (*this)[i++] = d; }
  }

  //----------------------------------------------------------------------------
 private:
  template <typename S, size_t... Is>
  chunked_data(const std::array<S, N>& sizes,
               std::index_sequence<Is...> /*is*/)
      : chunked_data{sizes[Is]...} {}

 public:
  template <typename S, enable_if_integral<S>...>
  chunked_data(const std::array<S, N>& sizes)
      : chunked_data{sizes, indices_t{}} {}

  //----------------------------------------------------------------------------
  chunked_data(const chunked_data& other)
      : m_chunk_structure{other.m_chunk_structure},
        m_data_structure{other.m_data_structure},
        m_chunks(other.m_chunks.size()) {
    copy_chunks(other);
  }

  //----------------------------------------------------------------------------
  chunked_data& operator=(const chunked_data& other) {
    m_chunk_structure = other.m_chunk_structure;
    m_data_structure  = other.m_data_structure;
    copy_chunks(other);
    return *this;
  }

  //----------------------------------------------------------------------------
  chunked_data(chunked_data&& other) = default;
  chunked_data& operator=(chunked_data&& other) = default;

  //----------------------------------------------------------------------------
  template <typename Container>
  chunked_data& operator=(const Container& container) {
    assert(container.size() == m_data_structure.num_data());
    size_t i = 0;
    for (const auto& d : container) { (*this)[i++] = d; }
    return *this;
  }

  //----------------------------------------------------------------------------
  void copy_chunks(const chunked_data& other) {
    auto chunk_it = begin(m_chunks);
    for (const auto& chunk : other.m_chunks) {
      if (chunk) { *(chunk_it++) = std::make_unique<chunk_t>(*chunk); }
    }
  }

  //----------------------------------------------------------------------------
  size_t num_data() const { return m_data_structure.num_data(); }

  //----------------------------------------------------------------------------
  const auto& size() const { return m_data_structure.size(); }
  auto        size(size_t i) const { return m_data_structure.size(i); }

  //----------------------------------------------------------------------------
  T& operator[](size_t global_idx) {
    assert(global_idx < m_data_structure.num_data());
    return at(m_data_structure.multi_index(global_idx));
  }
  const T& operator[](size_t global_idx) const {
    assert(global_idx < m_data_structure.num_data());
    return at(m_data_structure.multi_index(global_idx));
  }

  //----------------------------------------------------------------------------
  template <typename... Indices,
            enable_if_integral<std::decay_t<Indices>...>...>
  T& operator()(Indices... indices) {
    return at(indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices,
            enable_if_integral<std::decay_t<Indices>...>...,
            std::enable_if_t<sizeof...(Indices) == N>...>
  T& at(Indices... indices) {
    assert(m_data_structure.in_range(indices...));
    size_t gi = m_chunk_structure.global_idx((indices / ChunkRes)...);
    if (!m_chunks[gi]) { m_chunks[gi] = std::make_unique<chunk_t>(); }
    return m_chunks[gi]->at((indices % ChunkRes)...);
  }

  //----------------------------------------------------------------------------
  template <typename Tensor, typename S, enable_if_integral<S>...>
  decltype(auto) at(const base_tensor<Tensor, S, N>& indices) {
    return invoke_unpacked(
        [this](const auto... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, typename S, enable_if_integral<S>...>
  decltype(auto) operator()(const base_tensor<Tensor, S, N>& indices) {
    return at(indices);
  }

  //----------------------------------------------------------------------------
  template <typename S, enable_if_integral<S>...>
  decltype(auto) operator()(const std::array<S, N>& indices) {
    return at(indices);
  }

  //----------------------------------------------------------------------------
  template <typename S, enable_if_integral<S>...>
  decltype(auto) at(const std::array<S, N>& indices) {
    return invoke_unpacked(
        [this](const auto... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }

  //----------------------------------------------------------------------------
  template <typename... Indices,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Indices>> && ...)>>
  T operator()(Indices&&... indices) const {
    return at(std::forward<Indices>(indices)...);
  }
  template <typename... Indices,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Indices>> && ...)>>
  T at(Indices&&... indices) const {
    assert(m_data_structure.in_range(std::forward<Indices>(indices)...));
    size_t gi = m_chunk_structure.global_idx((indices / ChunkRes)...);
    if (!m_chunks[gi]) { return T{}; }
    return m_chunks[gi]->at((indices % ChunkRes)...);
  }

  //----------------------------------------------------------------------------
  template <typename S, enable_if_integral<S>...>
  T operator()(const std::array<S, N>& indices) const {
    return at(indices);
  }

  //----------------------------------------------------------------------------
  template <typename S, enable_if_integral<S>...>
  T at(const std::array<S, N>& indices) const {
    return invoke_unpacked(
        [this](const auto... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }

  //----------------------------------------------------------------------------
  template <typename... Sizes, size_t... Is,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Sizes>> && ...)>,
            typename = std::enable_if_t<sizeof...(Sizes) == N>>
  void resize(Sizes&&... sizes) {
    return resize(indices_t{}, std::forward<Sizes>(sizes)...);
  }

  //----------------------------------------------------------------------------
  template <typename... Sizes, size_t... Is,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Sizes>> && ...)>,
            typename = std::enable_if_t<sizeof...(Sizes) == N>>
  void resize(std::index_sequence<Is...> /*is*/, Sizes&&... sizes) {
    dynamic_multidimension<N> new_chunk_structure{static_cast<size_t>(
        std::ceil(static_cast<double>(sizes) / ChunkRes))...};
    std::vector<std::unique_ptr<chunk_t>> new_chunks(
        new_chunk_structure.num_data());

    for (auto mi : tatooine::multi_index(
             {size_t(0),
              std::min<size_t>(sizes, m_chunk_structure.size(Is)) -
                  1}...)) {
      new_chunks[new_chunk_structure.global_idx(mi)] =
          std::move(m_chunks[m_chunk_structure.global_idx(mi)]);
    }
    m_chunk_structure = std::move(new_chunk_structure);
    m_data_structure.resize(static_cast<size_t>(sizes)...);
    m_chunks = std::move(new_chunks);
  }

  //----------------------------------------------------------------------------
  auto unchunk() {
    std::vector<T> data(size());
    for (size_t i = 0; i < size(); ++i) { data[i] = (*this)[i]; }
    return data;
  }

  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T>...>
  void randu(T min = 0, T max = 1,
             RandomEngine&& random_engine = RandomEngine{
                 std::random_device{}()}) {
    for (auto& chunk : m_chunks) {
      if (!chunk) { chunk = std::make_unique<chunk_t>(); }
      chunk->randu(min, max, std::forward<RandomEngine>(random_engine));
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
