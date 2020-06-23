#ifndef TATOOINE_CHUNKED_MULTIDIM_ARRAY_H
#define TATOOINE_CHUNKED_MULTIDIM_ARRAY_H
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
template <typename T, typename Indexing = x_fastest>
struct chunked_multidim_array {
  //============================================================================
  using value_type = T;
  using this_t     = chunked_multidim_array<T, Indexing>;
  using chunk_t    = dynamic_multidim_array<T, Indexing>;
  //----------------------------------------------------------------------------
  dynamic_multidim_resolution<Indexing> m_data_structure;
  std::vector<size_t>                   m_internal_chunk_res;
  dynamic_multidim_resolution<Indexing> m_chunk_structure;
  std::vector<std::unique_ptr<chunk_t>> m_chunks;
  //============================================================================
  chunked_multidim_array(chunked_multidim_array const& other)
      : m_data_structure{other.m_data_structure},
        m_internal_chunk_res{other.m_internal_chunk_res},
        m_chunk_structure{other.m_chunk_structure},
        m_chunks(other.m_chunks.size()) {
    copy_chunks(other);
  }
  //----------------------------------------------------------------------------
  chunked_multidim_array& operator=(chunked_multidim_array const& other) {
    m_chunk_structure    = other.m_chunk_structure;
    m_internal_chunk_res = other.m_internal_chunk_res;
    m_data_structure     = other.m_data_structure;
    copy_chunks(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  chunked_multidim_array(chunked_multidim_array&& other) = default;
  chunked_multidim_array& operator=(chunked_multidim_array&& other) = default;
  //----------------------------------------------------------------------------
  template <typename Container>
  auto operator=(Container const& container) -> auto& {
    assert(container.size() == m_data_structure.num_components());
    size_t i = 0;
    for (auto const& d : container) { (*this)[i++] = d; }
    return *this;
  }
  //----------------------------------------------------------------------------
  chunked_multidim_array(std::vector<size_t> const& resolution,
                         std::vector<size_t> const& chunk_res) {
    resize(resolution, chunk_res);
  }
  //----------------------------------------------------------------------------
  template <range Range>
  chunked_multidim_array(Range&& data, std::vector<size_t> const& resolution,
                         std::vector<size_t> const& chunk_res) {
    resize(resolution, chunk_res);
    size_t i = 0;
    for (auto const& d : data) { (*this)[i++] = d; }
  }
  //----------------------------------------------------------------------------
  template <can_read<this_t> Reader>
  chunked_multidim_array(Reader&&                   reader,
                         std::vector<size_t> const& chunk_res) {
    m_internal_chunk_res = chunk_res;
    reader.read(*this);
  }
  //==============================================================================
  void resize(std::vector<size_t> resolution) {
    // apply full resolution
    m_data_structure.resize(resolution);

    // transform to chunk resolution and apply
    auto res_it  = begin(resolution);
    auto size_it = begin(m_internal_chunk_res);
    for (; res_it < end(resolution); ++res_it, ++size_it) {
      *res_it = static_cast<size_t>(std::ceil(static_cast<double>(*res_it) /
                                              static_cast<double>(*size_it)));
    }
    m_chunk_structure.resize(resolution);
    m_chunks.resize(m_chunk_structure.num_components());
  }
  //----------------------------------------------------------------------------
  void resize(std::vector<size_t> const& resolution,
              std::vector<size_t> const& chunk_res) {
    m_internal_chunk_res = chunk_res;
    resize(resolution);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral auto const... sizes) -> void {
    return resize(std::vector{static_cast<size_t>(sizes)...});
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_data_structure.num_dimensions(); }
  //----------------------------------------------------------------------------
  auto internal_chunk_resolution() const { return m_internal_chunk_res; }
  //----------------------------------------------------------------------------
  auto chunk_resolution() const { return m_chunk_structure.size(); }
  //----------------------------------------------------------------------------
  auto resolution() const { return m_data_structure.size(); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto plain_internal_chunk_index_from_global_indices(
      std::index_sequence<Is...>, integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return m_chunk_structure.plain_index(
        (indices % m_internal_chunk_res[Is])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto plain_internal_chunk_index_from_global_indices(
      integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return plain_internal_chunk_index_from_global_indices(
        std::make_index_sequence<sizeof...(indices)>{}, indices...);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto plain_chunk_index_from_global_indices(
      std::index_sequence<Is...>, integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return m_chunk_structure.plain_index(
        (indices / m_internal_chunk_res[Is])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto plain_chunk_index_from_global_indices(
      integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return plain_chunk_index_from_global_indices(
        std::make_index_sequence<sizeof...(indices)>{}, indices...);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto internal_chunk_indices_from_global_indices(
      std::index_sequence<Is...>, integral auto const... indices) const {
    return std::array{
        static_cast<size_t>(indices % m_internal_chunk_res[Is])...};
  }
  //----------------------------------------------------------------------------
 public:
  template <size_t... Is>
  auto internal_chunk_indices_from_global_indices(
      integral auto const... indices) const {
    return internal_chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(indices)>{}, indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  auto internal_chunk_indices_from_global_indices(
      std::vector<Int> indices) const {
    assert(size(indices) == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      indices[i] %= m_internal_chunk_res[i];
    }
    return indices;
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  auto chunk_indices_from_global_indices(std::vector<Int> indices) const {
    assert(size(indices) == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      indices[i] /= m_internal_chunk_res[i];
    }
    return indices;
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  auto global_indices_from_chunk_indices(std::vector<Int> indices) const {
    assert(indices.size() == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      indices[i] *= m_internal_chunk_res[i];
    }
    return indices;
  }
  //----------------------------------------------------------------------------
  auto plain_chunk_index_from_chunk_indices(
      integral auto const... chunk_indices) const {
    assert(sizeof...(chunk_indices) == num_dimensions());
    return m_chunk_structure.plain_index(chunk_indices...);
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  auto plain_chunk_index_from_chunk_indices(
      std::vector<Int> const& chunk_indices) const {
    assert(chunk_indices.size() == num_dimensions());
    return m_chunk_structure.plain_index(chunk_indices);
  }
  //----------------------------------------------------------------------------
  auto chunk_at(integral auto const... chunk_indices) -> auto& {
    assert(sizeof...(chunk_indices) == num_dimensions());
    return m_chunks[plain_chunk_index_from_chunk_indices(chunk_indices...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto chunk_at(size_t i) -> auto& { return m_chunks[i]; }
  //----------------------------------------------------------------------------
  auto chunk_at_is_null(integral auto const... chunk_indices) {
    assert(sizeof...(chunk_indices) == num_dimensions());
    return m_chunks[plain_chunk_index_from_chunk_indices(chunk_indices...)] ==
           nullptr;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto chunk_at_is_null(size_t i) { return m_chunks[i] == nullptr; }
  //----------------------------------------------------------------------------
  auto create_chunk_at(integral auto const... chunk_indices) -> auto& {
    assert(sizeof...(chunk_indices) == num_dimensions());
    auto const i = plain_chunk_index_from_chunk_indices(chunk_indices...);
    m_chunks[i]  = std::make_unique<chunk_t>(m_internal_chunk_res);
    return m_chunks[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto create_chunk_at(size_t i) -> auto& {
    m_chunks[i] = std::make_unique<chunk_t>(m_internal_chunk_res);
    return m_chunks[i];
  }
  //----------------------------------------------------------------------------
  auto destroy_chunk_at(integral auto const... chunk_indices) {
    assert(sizeof...(chunk_indices) == num_dimensions());
    m_chunks[plain_chunk_index_from_chunk_indices(chunk_indices...)].reset();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto destroy_chunk_at(size_t i) { m_chunks[i].reset(); }
  //----------------------------------------------------------------------------
  auto copy_chunks(chunked_multidim_array const& other) -> void {
    auto chunk_it = begin(m_chunks);
    for (auto const& chunk : other.m_chunks) {
      if (chunk) { *(chunk_it++) = std::make_unique<chunk_t>(*chunk); }
    }
  }
  //----------------------------------------------------------------------------
  auto num_components() const { return m_data_structure.num_components(); }
  //----------------------------------------------------------------------------
  auto size() const -> auto const& { return m_data_structure.size(); }
  auto size(size_t i) const { return m_data_structure.size(i); }
  //----------------------------------------------------------------------------
  auto operator[](size_t plain_index) -> T& {
    assert(plain_index < m_data_structure.num_components());
    return at(m_data_structure.multi_index(plain_index));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t plain_index) const -> T const& {
    assert(plain_index < m_data_structure.num_components());
    return at(m_data_structure.multi_index(plain_index));
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto... indices) -> T& { return at(indices...); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto const... indices) -> T& {
    assert(sizeof...(indices) == num_dimensions());
    assert(m_data_structure.in_range(indices...));
    size_t plain_index = plain_chunk_index_from_global_indices(indices...);
    if (chunk_at_is_null(plain_index)) { create_chunk_at(plain_index); }
    auto const internal_indices =
        internal_chunk_indices_from_global_indices(indices...);
    return m_chunks[plain_index]->at(internal_indices);
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == num_dimensions());
    assert(m_data_structure.in_range(indices...));
    size_t plain_index = plain_chunk_index_from_global_indices(indices...);
    if (!m_chunks[plain_index]) {
      static constexpr T t{};
      return t;
    }
    return (
        *m_chunks[plain_index])[plain_internal_chunk_index_from_global_indices(
        indices...)];
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == num_dimensions());
    return at(indices...);
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, size_t N, integral S>
  auto at(base_tensor<Tensor, S, N> const& indices) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  //----------------------------------------------------------------------------
  template <integral S, size_t N>
  auto at(std::array<S, N> const& indices) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& indices) -> T& {
    assert(num_dimensions() == indices.size());
    assert(m_data_structure.in_range(indices));
    size_t const plain_index = m_chunk_structure.plain_index(
        chunk_indices_from_global_indices(indices));
    if (!m_chunks[plain_index]) { create_chunk_at(plain_index); }
    return m_chunks[plain_index]->at(
        internal_chunk_indices_from_global_indices(indices));
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& indices) const -> T const& {
    assert(num_dimensions() == indices.size());
    assert(m_data_structure.in_range(indices));
    auto global_is = indices;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      global_is[i] /= indices[i];
    }
    size_t plain_index = m_chunk_structure.plain_index(global_is);
    if (!m_chunks[plain_index]) {
      static const T t{};
      return t;
    }
    auto local_is = indices;
    for (size_t i = 0; i < num_dimensions(); ++i) { local_is[i] %= indices[i]; }
    return m_chunks[plain_index]->at(local_is);
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, size_t N, integral S>
  auto operator()(base_tensor<Tensor, S, N> const& indices) -> T& {
    return at(indices);
  }
  //----------------------------------------------------------------------------
  template <integral S, size_t N>
  auto operator()(std::array<S, N> const& indices) -> T& {
    return at(indices);
  }
  //----------------------------------------------------------------------------
  template <integral S, size_t N>
  auto operator()(std::array<S, N> const& indices) const -> T const& {
    return at(indices);
  }
  //----------------------------------------------------------------------------
  template <integral S, size_t N>
  auto at(std::array<S, N> const& indices) const -> T const& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(indices));
  }
  ////----------------------------------------------------------------------------
  // auto unchunk() const {
  //  std::vector<T> data(num_components());
  //  for (size_t i = 0; i < num_components(); ++i) { data[i] = (*this)[i]; }
  //  return data;
  //}
  ////----------------------------------------------------------------------------
  // template <typename _T = T, enable_if_tensor<_T> = true>
  // auto unchunk_plain() const {
  //  using real_t          = typename T::real_t;
  //  constexpr auto      n = T::num_components();
  //  std::vector<real_t> data;
  //  data.reserve(num_components() * n);
  //  for (size_t i = 0; i < num_components(); ++i) {
  //    for (size_t j = 0; j < n; ++j) { data.push_back((*this)[i][j]); }
  //  }
  //  return data;
  //}
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
