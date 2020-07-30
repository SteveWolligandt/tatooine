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
  using this_t            = chunked_multidim_array<T, Indexing>;
  using chunk_t           = dynamic_multidim_array<T, Indexing>;
  using chunk_ptr_t       = std::unique_ptr<chunk_t>;
  using chunk_ptr_field_t = std::vector<chunk_ptr_t>;
  //----------------------------------------------------------------------------
 private:
  dynamic_multidim_size<Indexing> m_data_structure;
  std::vector<size_t>             m_internal_chunk_size;
  dynamic_multidim_size<Indexing> m_chunk_structure;

 protected:
  mutable chunk_ptr_field_t m_chunks;
  //============================================================================
 public:
  chunked_multidim_array(chunked_multidim_array const& other)
      : m_data_structure{other.m_data_structure},
        m_internal_chunk_size{other.m_internal_chunk_size},
        m_chunk_structure{other.m_chunk_structure},
        m_chunks(other.m_chunks.size()) {
    copy_chunks(other);
  }
  //----------------------------------------------------------------------------
  chunked_multidim_array& operator=(chunked_multidim_array const& other) {
    m_chunk_structure    = other.m_chunk_structure;
    m_internal_chunk_size = other.m_internal_chunk_size;
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
  chunked_multidim_array(std::vector<size_t> const& size,
                         std::vector<size_t> const& chunk_size) {
    resize(size, chunk_size);
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  chunked_multidim_array(std::array<size_t, N> const& size,
                         std::vector<size_t> const&   chunk_size) {
    m_internal_chunk_size = chunk_size;
    resize(std::vector<size_t>(begin(size), end(size)));
  }
  //----------------------------------------------------------------------------
  template <range Range>
  chunked_multidim_array(Range&& data, std::vector<size_t> const& size,
                         std::vector<size_t> const& chunk_size) {
    resize(size, chunk_size);
    size_t i = 0;
    for (auto const& d : data) { (*this)[i++] = d; }
  }
  //----------------------------------------------------------------------------
  template <can_read<this_t> Reader>
  chunked_multidim_array(Reader&&                   reader,
                         std::vector<size_t> const& chunk_size) {
    m_internal_chunk_size = chunk_size;
    reader.read(*this);
  }
  //==============================================================================
  void resize(std::vector<size_t> size) {
    // apply full size
    m_data_structure.resize(size);

    // transform to chunk size and apply
    auto size_it       = begin(size);
    auto chunk_size_it = begin(m_internal_chunk_size);
    for (; size_it < end(size); ++size_it, ++chunk_size_it) {
      *size_it = static_cast<size_t>(std::ceil(
          static_cast<double>(*size_it) / static_cast<double>(*chunk_size_it)));
    }
    m_chunk_structure.resize(size);
    m_chunks.resize(m_chunk_structure.num_components());
  }
  //----------------------------------------------------------------------------
  void resize(std::vector<size_t> const& size,
              std::vector<size_t> const& chunk_size) {
    m_internal_chunk_size = chunk_size;
    resize(size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral auto const... sizes) -> void {
    return resize(std::vector{static_cast<size_t>(sizes)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, integral Int, size_t N>
  auto resize(base_tensor<Tensor, Int, N> const& v) -> void {
    assert(N == num_dimensions());
    std::vector<size_t> s(num_dimensions());
    for (size_t i = 0; i < N; ++i) {
      s[i] = v(i);
    }
    return resize(s);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  auto resize(std::array<Int, N> const& v) {
    assert(N == num_dimensions());
    return resize(std::vector<size_t>(begin(v), end(v)));
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto plain_internal_chunk_index_from_global_indices(
      size_t plain_chunk_index, std::index_sequence<Is...>,
      integral auto const... indices) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(indices) == m_chunks[plain_chunk_index]->num_dimensions());

    return m_chunks[plain_chunk_index]->plain_index(
        (indices % m_internal_chunk_size[Is])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto plain_internal_chunk_index_from_global_indices(
      size_t plain_chunk_index, integral auto const... indices) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(indices) == m_chunks[plain_chunk_index]->num_dimensions());
    return plain_internal_chunk_index_from_global_indices(
        plain_chunk_index, std::make_index_sequence<sizeof...(indices)>{},
        indices...);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto plain_chunk_index_from_global_indices(
      std::index_sequence<Is...>, integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return m_chunk_structure.plain_index(
        (indices / m_internal_chunk_size[Is])...);
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
        static_cast<size_t>(indices % m_internal_chunk_size[Is])...};
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
  //template <integral Int>
  //auto internal_chunk_indices_from_global_indices(
  //    std::vector<Int> indices) const {
  //  assert(size(indices) == num_dimensions());
  //  for (size_t i = 0; i < num_dimensions(); ++i) {
  //    indices[i] %= m_internal_chunk_size[i];
  //  }
  //  return indices;
  //}
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq>
  auto chunk_indices_from_global_indices(std::index_sequence<Seq...>,
                                         integral auto const... indices) const {
    return std::vector{
        (static_cast<size_t>(indices) / m_internal_chunk_size[Seq])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto chunk_indices_from_global_indices(integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    return chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(indices)>{}, indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  auto chunk_indices_from_global_indices(std::vector<Int> indices) const {
    assert(size(indices) == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      indices[i] /= m_internal_chunk_size[i];
    }
    return indices;
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  auto global_indices_from_chunk_indices(std::vector<Int> indices) const {
    assert(indices.size() == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      indices[i] *= m_internal_chunk_size[i];
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
  auto chunk_at(size_t const chunk_index0, integral auto const... chunk_indices)
      -> auto& {
    if constexpr (sizeof...(chunk_indices) == 0) {
      return m_chunks[chunk_index0];
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      return m_chunks[plain_chunk_index_from_chunk_indices(chunk_index0,
                                                           chunk_indices...)];
    }
  }
  //----------------------------------------------------------------------------
  auto chunk_at(size_t const chunk_index0, integral auto const... chunk_indices)
     const -> auto const& {
    if constexpr (sizeof...(chunk_indices) == 0) {
      return m_chunks[chunk_index0];
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      return m_chunks[plain_chunk_index_from_chunk_indices(chunk_index0,
                                                           chunk_indices...)];
    }
  }
  //----------------------------------------------------------------------------
  auto chunk_at_is_null(integral auto const chunk_index0,
                        integral auto const... chunk_indices) const {
    if constexpr (sizeof...(chunk_indices) == 0) {
      return m_chunks[chunk_index0] == nullptr;
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      return m_chunks[plain_chunk_index_from_chunk_indices(
                 chunk_index0, chunk_indices...)] == nullptr;
    }
  }
  //----------------------------------------------------------------------------
  auto create_all_chunks() {
    for (auto& chunk : m_chunks) {
      chunk = std::make_unique<chunk_t>(m_internal_chunk_size);
    }
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(size_t const chunk_index0,
                       integral auto const... chunk_indices) const -> const
      auto& {
    if constexpr (sizeof...(chunk_indices) == 0) {
      m_chunks[chunk_index0] = std::make_unique<chunk_t>(m_internal_chunk_size);
      return m_chunks[chunk_index0];
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      auto const i =
          plain_chunk_index_from_chunk_indices(chunk_index0, chunk_indices...);
      m_chunks[i] = std::make_unique<chunk_t>(m_internal_chunk_size);
      return m_chunks[i];
    }
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(size_t const chunk_index0,
                       integral auto const... chunk_indices) -> auto& {
    if constexpr (sizeof...(chunk_indices) == 0) {
      m_chunks[chunk_index0] = std::make_unique<chunk_t>(m_internal_chunk_size);
      return m_chunks[chunk_index0];
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      auto const i =
          plain_chunk_index_from_chunk_indices(chunk_index0, chunk_indices...);
      m_chunks[i] = std::make_unique<chunk_t>(m_internal_chunk_size);
      return m_chunks[i];
    }
  }
  //----------------------------------------------------------------------------
  auto destroy_chunk_at(size_t const chunk_index0,
                        integral auto const... chunk_indices) const {
    if constexpr (sizeof...(chunk_indices) == 0) {
      m_chunks[chunk_index0].reset();
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      m_chunks[plain_chunk_index_from_chunk_indices(chunk_indices...)].reset();
    }
  }
  //----------------------------------------------------------------------------
  auto copy_chunks(chunked_multidim_array const& other) -> void {
    auto chunk_it = begin(m_chunks);
    for (auto const& chunk : other.m_chunks) {
      if (chunk) { *(chunk_it++) = std::make_unique<chunk_t>(*chunk); }
    }
  }
  //----------------------------------------------------------------------------
  auto clear() {
    for (auto& chunk : m_chunks) {
      if (chunk != nullptr) { chunk.reset(); }
    }
  }
  //----------------------------------------------------------------------------
  auto in_range(integral auto const... is) const {
    return m_data_structure.in_range(is...);
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_data_structure.num_dimensions(); }
  auto num_components() const { return m_data_structure.num_components(); }
  auto num_chunks() const { return m_chunks.size(); }
  //----------------------------------------------------------------------------
  auto size() const -> auto const& { return m_data_structure.size(); }
  auto size(size_t i) const { return m_data_structure.size(i); }
  //----------------------------------------------------------------------------
  auto chunk_size() const { return m_chunk_structure.size(); }
  auto chunk_size(size_t i) const { return m_chunk_structure.size(i); }
  //----------------------------------------------------------------------------
  auto internal_chunk_size() const {
    return m_internal_chunk_size;
  }
  auto internal_chunk_size(size_t const i) const {
    return m_internal_chunk_size[i];
  }
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
  auto at(integral auto const... indices) -> T& {
    assert(sizeof...(indices) == num_dimensions());
    assert(in_range(indices...));
    size_t const plain_index =
        plain_chunk_index_from_global_indices(indices...);
    if (chunk_at_is_null(plain_index)) { create_chunk_at(plain_index); }
    size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, indices...);
    return (*m_chunks[plain_index])[plain_internal_index];
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == num_dimensions());
    assert(in_range(indices...));
    size_t const plain_index = plain_chunk_index_from_global_indices(indices...);
    if (chunk_at_is_null(plain_index)) {
      static constexpr T t{};
      return t;
    }
    size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, indices...);
    return (*m_chunks[plain_index])[plain_internal_index];
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
    assert(in_range(indices));
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
    assert(in_range(indices));
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
  auto operator()(integral auto const... indices) -> T& {
    assert(sizeof...(indices) == num_dimensions());
    return at(indices...);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == num_dimensions());
    return at(indices...);
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
