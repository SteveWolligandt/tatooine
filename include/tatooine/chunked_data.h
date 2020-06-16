#ifndef TATOOINE_CHUNKED_DATA_H
#define TATOOINE_CHUNKED_DATA_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/functional.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

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
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t N, size_t R, size_t... Is>
auto create_chunk(std::index_sequence<Is...> /*is*/) {
  return static_multidim_array<T, x_fastest, tag::heap, ((void)Is, R)...>{};
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
template <typename T, size_t N, size_t ChunkRes = 128>
struct chunked_data {
  //============================================================================
  using this_t    = chunked_data<T, N, ChunkRes>;
  using chunk_t   = typename create_chunk_type<T, N, ChunkRes>::T;
  using indices_t = std::make_index_sequence<N>;
  //----------------------------------------------------------------------------
  static constexpr auto chunk_res = chunk_t::size();
  //----------------------------------------------------------------------------
  dynamic_multidim_resolution<x_fastest> m_data_structure;
  dynamic_multidim_resolution<x_fastest> m_chunk_structure;
  std::vector<std::unique_ptr<chunk_t>>  m_chunks;

 private:
  //============================================================================
  template <typename S, size_t... Is>
  chunked_data(std::array<S, N> const& data_sizes)
      : m_data_structure{data_sizes},
        m_chunk_structure{static_cast<size_t>(
            std::ceil(static_cast<double>(data_sizes(Is)) / ChunkRes))...},
        m_chunks{std::accumulate(begin(this->sizes()), end(this->sizes()),
                                 size_t(1), std::multiplies<size_t>{})} {}

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
        m_chunks{m_chunk_structure.num_elements()} {}

  //----------------------------------------------------------------------------
  template <typename Container, typename... Sizes,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Sizes>> && ...)>,
            typename = std::enable_if_t<sizeof...(Sizes) == N>>
  chunked_data(Container&& data, Sizes&&... sizes)
      : m_data_structure{static_cast<size_t>(sizes)...},
        m_chunk_structure{static_cast<size_t>(
            std::ceil(static_cast<double>(sizes) / ChunkRes))...},
        m_chunks{m_chunk_structure.num_elements()} {
    assert(data.size() == (static_cast<size_t>(sizes) * ...));
    size_t i = 0;
    for (auto const& d : data) { (*this)[i++] = d; }
  }

  //----------------------------------------------------------------------------
 private:
  template <typename S, size_t... Is>
  chunked_data(std::array<S, N> const& sizes, std::index_sequence<Is...> /*is*/)
      : chunked_data{sizes[Is]...} {}

 public:
  template <integral S>
  chunked_data(std::array<S, N> const& sizes)
      : chunked_data{sizes, indices_t{}} {}

  //----------------------------------------------------------------------------
  chunked_data(chunked_data const& other)
      : m_data_structure{other.m_data_structure},
        m_chunk_structure{other.m_chunk_structure},
        m_chunks(other.m_chunks.size()) {
    copy_chunks(other);
  }

  //----------------------------------------------------------------------------
  chunked_data& operator=(chunked_data const& other) {
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
  auto operator=(Container const& container) -> auto& {
    // assert(container.size() == m_data_structure.num_elements());
    size_t i = 0;
    for (auto const& d : container) { (*this)[i++] = d; }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto copy_chunks(chunked_data const& other) -> void {
    auto chunk_it = begin(m_chunks);
    for (auto const& chunk : other.m_chunks) {
      if (chunk) { *(chunk_it++) = std::make_unique<chunk_t>(*chunk); }
    }
  }
  //----------------------------------------------------------------------------
  auto num_elements() const { return m_data_structure.num_elements(); }
  //----------------------------------------------------------------------------
  auto size() const -> auto const& { return m_data_structure.size(); }
  auto size(size_t i) const { return m_data_structure.size(i); }
  //----------------------------------------------------------------------------
  auto operator[](size_t global_idx) -> T& {
    assert(global_idx < m_data_structure.num_elements());
    return at(m_data_structure.multi_index(global_idx));
  }
  auto operator[](size_t global_idx) const -> T const& {
    assert(global_idx < m_data_structure.num_elements());
    return at(m_data_structure.multi_index(global_idx));
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto... indices) -> T& { return at(indices...); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Indices, std::enable_if_t<sizeof...(Indices) == N>...>
  auto at(Indices... indices) -> T& {
    assert(m_data_structure.in_range(indices...));
    size_t gi = m_chunk_structure.plain_index((indices / ChunkRes)...);
    if (!m_chunks[gi]) { m_chunks[gi] = std::make_unique<chunk_t>(); }
    return m_chunks[gi]->at((indices % ChunkRes)...);
  }
  //----------------------------------------------------------------------------
  auto at(integral auto... indices) const -> T const& {
    assert(m_data_structure.in_range(indices...));
    size_t gi = m_chunk_structure.plain_index((indices / ChunkRes)...);
    if (!m_chunks[gi]) {
      static constexpr T t{};
      return t;
    }
    return m_chunks[gi]->at((indices % ChunkRes)...);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto... indices) const -> T const& {
    return at(indices...);
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, integral S>
  auto at(base_tensor<Tensor, S, N> const& indices) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, integral S>
  auto operator()(base_tensor<Tensor, S, N> const& indices) -> T& {
    return at(indices);
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto operator()(std::array<S, N> const& indices) -> T& {
    return at(indices);
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::array<S, N> const& indices) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& indices) -> T& {
    assert(N == indices.size());
    assert(m_data_structure.in_range(indices));
    auto global_is = indices;
    for (auto& i : global_is) { i /= ChunkRes; }
    size_t gi = m_chunk_structure.plain_index(global_is);
    if (!m_chunks[gi]) { m_chunks[gi] = std::make_unique<chunk_t>(); }
    auto local_is = indices;
    for (auto& i : local_is) { i %= ChunkRes; }
    return m_chunks[gi]->at(local_is);
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& indices) const {
    assert(N == indices.size());
    assert(m_data_structure.in_range(indices));
    auto global_is = indices;
    for (auto& i : global_is) { i /= ChunkRes; }
    size_t gi = m_chunk_structure.plain_index(global_is);
    if (!m_chunks[gi]) { return T{}; }
    auto local_is = indices;
    for (auto& i : local_is) { i %= ChunkRes; }
    return m_chunks[gi]->at(local_is);
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto operator()(std::array<S, N> const& indices) const -> T const& {
    return at(indices);
  }

  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::array<S, N> const& indices) const -> T const& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }

  //----------------------------------------------------------------------------
  template <integral... Sizes,
            std::enable_if_t<sizeof...(Sizes) == N, bool> = true>
  auto resize(Sizes... sizes) -> void {
    return resize(indices_t{}, std::forward<Sizes>(sizes)...);
  }
  //----------------------------------------------------------------------------
  template <integral... Sizes, size_t... Is,
            std::enable_if_t<sizeof...(Sizes) == N, bool> = true>
  auto resize(std::index_sequence<Is...> /*is*/, Sizes&&... sizes) -> void {
    dynamic_multidim_resolution<x_fastest> new_chunk_structure{
        static_cast<size_t>(
            std::ceil(static_cast<double>(sizes) / ChunkRes))...};
    std::vector<std::unique_ptr<chunk_t>> new_chunks(
        new_chunk_structure.num_elements());

    for_loop(
        [this, &new_chunk_structure, &new_chunks](auto... is) {
          new_chunks[new_chunk_structure.plain_index(is...)] =
              std::move(m_chunks[m_chunk_structure.plain_index(is...)]);
        },
        m_chunk_structure.size(Is)...);
    m_chunk_structure = std::move(new_chunk_structure);
    m_data_structure.resize(static_cast<size_t>(sizes)...);
    m_chunks = std::move(new_chunks);
  }
  //----------------------------------------------------------------------------
  auto unchunk() const {
    std::vector<T> data(num_elements());
    for (size_t i = 0; i < num_elements(); ++i) { data[i] = (*this)[i]; }
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename _T = T, enable_if_tensor<_T> = true>
  auto unchunk_plain() const {
    using real_t          = typename T::real_t;
    constexpr auto      n = T::num_components();
    std::vector<real_t> data;
    data.reserve(num_elements() * n);
    for (size_t i = 0; i < num_elements(); ++i) {
      for (size_t j = 0; j < n; ++j) { data.push_back((*this)[i][j]); }
    }
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T>...>
  auto randu(T min = 0, T max = 1,
             RandomEngine&& random_engine = RandomEngine{
                 std::random_device{}()}) -> void {
    for (auto& chunk : m_chunks) {
      if (!chunk) { chunk = std::make_unique<chunk_t>(); }
      chunk->randu(min, max, std::forward<RandomEngine>(random_engine));
    }
  }

  template <typename _T = T, enable_if_arithmetic<_T>...>
  auto min_value() const {
    T min = std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) { min = std::min(min, *chunk->min_element()); }
    }
    return min;
  }

  template <typename _T = T, enable_if_arithmetic<_T>...>
  auto max_value() const {
    T max = -std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) { max = std::max(max, *chunk->max_element()); }
    }
    return max;
  }

  template <typename _T = T, enable_if_arithmetic<_T>...>
  auto minmax_value() const {
    std::pair minmax{std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max()};
    auto& [min, max] = minmax;
    for (auto const& chunk : m_chunks) {
      if (chunk) {
        auto [chunkmin, chunkmax] = chunk->minmax_element();
        if (!std::isinf(*chunkmin) && !std::isnan(*chunkmin)) {
          min = std::min(min, *chunkmin);
        }
        if (!std::isinf(*chunkmax) && !std::isnan(*chunkmax)) {
          max = std::max(max, *chunkmax);
        }
      }
    }
    return minmax;
  }

  template <typename _T = T, enable_if_arithmetic<_T>...>
  auto normalize() -> void {
    auto [min, max] = minmax_value();
    auto normalizer = 1.0 / (max - min);

    for (auto const& chunk : m_chunks) {
      if (chunk) {
        for (auto& val : *chunk) { val = (val - min) * normalizer; }
      }
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
