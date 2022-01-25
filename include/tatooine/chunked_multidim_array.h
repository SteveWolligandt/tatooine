#ifndef TATOOINE_CHUNKED_MULTIDIM_ARRAY_H
#define TATOOINE_CHUNKED_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/concepts.h>
#include <tatooine/dynamic_multidim_array.h>
#include <tatooine/functional.h>
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
template <typename T, typename GlobalIndexOrder = x_fastest,
          typename LocalIndexOrder = GlobalIndexOrder>
struct chunked_multidim_array : dynamic_multidim_size<GlobalIndexOrder> {
  //============================================================================
  using value_type = T;
  using parent_t   = dynamic_multidim_size<GlobalIndexOrder>;
  using this_t  = chunked_multidim_array<T, GlobalIndexOrder, LocalIndexOrder>;
  using chunk_t = dynamic_multidim_array<T, LocalIndexOrder>;
  using chunk_ptr_t          = std::unique_ptr<chunk_t>;
  using chunk_ptr_field_t    = std::vector<chunk_ptr_t>;
  using global_index_order_t = GlobalIndexOrder;
  using local_index_order_t  = LocalIndexOrder;
  //----------------------------------------------------------------------------
  using parent_t::in_range;
  using parent_t::multi_index;
  using parent_t::num_components;
  using parent_t::num_dimensions;
  using parent_t::size;
  //----------------------------------------------------------------------------
 private:
  std::vector<std::size_t>                    m_internal_chunk_size;
  dynamic_multidim_size<LocalIndexOrder> m_chunk_structure;

 protected:
  mutable chunk_ptr_field_t m_chunks;
  //============================================================================
 public:
  chunked_multidim_array(chunked_multidim_array const& other)
      : parent_t{other},
        m_internal_chunk_size{other.m_internal_chunk_size},
        m_chunk_structure{other.m_chunk_structure},
        m_chunks(other.m_chunks.size()) {
    copy_chunks(other);
  }
  //----------------------------------------------------------------------------
  template <range SizeRange, range ChunkSizeRange>
      requires(is_integral<typename std::decay_t<SizeRange>::value_type>) &&
      (is_integral<typename std::decay_t<ChunkSizeRange>::value_type>)
          chunked_multidim_array(SizeRange&&      size,
                                 ChunkSizeRange&& chunk_size) {
    resize(std::forward<SizeRange>(size),
           std::forward<ChunkSizeRange>(chunk_size));
  }
  //----------------------------------------------------------------------------
  template <range Range>
  chunked_multidim_array(Range&& data, std::vector<std::size_t> const& size,
                         std::vector<std::size_t> const& chunk_size) {
    resize(size, chunk_size);
    std::size_t i = 0;
    for (auto const& d : data) {
      (*this)[i++] = d;
    }
  }
  //----------------------------------------------------------------------------
  template <can_read<this_t> Reader>
  chunked_multidim_array(Reader&&                   reader,
                         std::vector<std::size_t> const& chunk_size) {
    m_internal_chunk_size = chunk_size;
    reader.read(*this);
  }
  //==============================================================================
  template <range SizeRange>
  auto resize(SizeRange&& size) -> void requires(
      is_integral<typename std::decay_t<SizeRange>::value_type>) {
    // apply full size
    parent_t::resize(size);

    // transform to chunk size and apply
    auto size_it       = begin(size);
    auto chunk_size_it = begin(m_internal_chunk_size);
    for (; size_it < end(size); ++size_it, ++chunk_size_it) {
      *size_it = static_cast<std::size_t>(std::ceil(
          static_cast<double>(*size_it) / static_cast<double>(*chunk_size_it)));
    }
    m_chunk_structure.resize(size);
    m_chunks.resize(m_chunk_structure.num_components());
  }
  //----------------------------------------------------------------------------
  template <range SizeRange, range ChunkSizeRange>
  auto resize(SizeRange&& size, ChunkSizeRange&& chunk_size)
  requires (is_integral<typename std::decay_t<SizeRange>::value_type>) &&
           (is_integral<typename std::decay_t<ChunkSizeRange>::value_type>) {
    m_internal_chunk_size.resize(chunk_size.size());
    std::copy(begin(chunk_size), end(chunk_size), begin(m_internal_chunk_size));
    std::size_t i = 0;
    for (auto& s : m_internal_chunk_size) {
      s = std::min<std::size_t>(s, size[i++]);
    }
    resize(std::forward<SizeRange>(size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral auto const... sizes) -> void {
    return resize(std::vector{static_cast<std::size_t>(sizes)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, integral Int, std::size_t N>
  auto resize(base_tensor<Tensor, Int, N> const& v) -> void {
    assert(N == num_dimensions());
    std::vector<std::size_t> s(num_dimensions());
    for (std::size_t i = 0; i < N; ++i) {
      s[i] = v(i);
    }
    return resize(s);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, std::size_t N>
  auto resize(std::array<Int, N> const& v) {
    assert(N == num_dimensions());
    return resize(std::vector<std::size_t>(begin(v), end(v)));
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto plain_internal_chunk_index_from_global_indices(
      std::size_t plain_chunk_index, std::index_sequence<Seq...>,
      integral auto const... is) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(is) == m_chunks[plain_chunk_index]->num_dimensions());
    return m_chunks[plain_chunk_index]->plain_index(
        (is % m_internal_chunk_size[Seq])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto plain_internal_chunk_index_from_global_indices(
      std::size_t plain_chunk_index, integral auto const... is) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(is) == m_chunks[plain_chunk_index]->num_dimensions());
    return plain_internal_chunk_index_from_global_indices(
        plain_chunk_index, std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto plain_chunk_index_from_global_indices(
      std::index_sequence<Seq...> /*seq*/, integral auto const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return m_chunk_structure.plain_index((is / m_internal_chunk_size[Seq])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto plain_chunk_index_from_global_indices(integral auto const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return plain_chunk_index_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto internal_chunk_indices_from_global_indices(
      std::index_sequence<Seq...>, integral auto const... is) const {
    return std::array{static_cast<std::size_t>(is % m_internal_chunk_size[Seq])...};
  }
  //----------------------------------------------------------------------------
 public:
  auto internal_chunk_indices_from_global_indices(
      integral auto const... is) const {
    return internal_chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // template <integral Int>
  // auto internal_chunk_indices_from_global_indices(
  //    std::vector<Int> is) const {
  //  assert(size(is) == num_dimensions());
  //  for (std::size_t i = 0; i < num_dimensions(); ++i) {
  //    is[i] %= m_internal_chunk_size[i];
  //  }
  //  return is;
  //}
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto chunk_indices_from_global_indices(std::index_sequence<Seq...>,
                                         integral auto const... is) const {
    return std::vector{
        (static_cast<std::size_t>(is) / m_internal_chunk_size[Seq])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto chunk_indices_from_global_indices(integral auto const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  auto chunk_indices_from_global_indices(std::vector<Int> is) const {
    assert(size(is) == num_dimensions());
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      is[i] /= m_internal_chunk_size[i];
    }
    return is;
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  auto global_indices_from_chunk_indices(std::vector<Int> is) const {
    assert(is.size() == num_dimensions());
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      is[i] *= m_internal_chunk_size[i];
    }
    return is;
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
  auto chunk_at(integral auto const chunk_index0,
                integral auto const... chunk_indices) const -> auto const& {
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
  auto create_all_chunks() const -> void {
    for (std::size_t i = 0; i < num_chunks(); ++i) {
      create_chunk_at(i);
    }
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(std::size_t const               plain_chunk_index,
                       std::vector<std::size_t> const& multi_indices) const
      -> auto const& {
    assert(multi_indices.size() == num_dimensions());
    std::vector<std::size_t> chunk_size = m_internal_chunk_size;
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (multi_indices[i] == m_chunk_structure.size(i) - 1) {
        chunk_size[i] = this->size(i) % chunk_size[i];
        if (chunk_size[i] == 0) {
          chunk_size[i] = m_internal_chunk_size[i];
        }
      }
    }
    m_chunks[plain_chunk_index] = std::make_unique<chunk_t>(chunk_size);
    return m_chunks[plain_chunk_index];
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(std::size_t const plain_chunk_index) const -> auto const& {
    assert(plain_chunk_index < num_chunks());
    return create_chunk_at(plain_chunk_index,
                           m_chunk_structure.multi_index(plain_chunk_index));
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(integral auto const chunk_index0,
                       integral auto const chunk_index1,
                       integral auto const... chunk_indices) const
      -> auto const& {
    assert(sizeof...(chunk_indices) + 2 == num_dimensions());
    return create_chunk_at(
        plain_chunk_index_from_chunk_indices(chunk_index0, chunk_index1,
                                             chunk_indices...),
        std::vector<std::size_t>{chunk_index0, chunk_index1, chunk_indices...});
  }
  //----------------------------------------------------------------------------
  auto destroy_chunk_at(integral auto const chunk_index0,
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
      if (chunk) {
        *(chunk_it++) = std::make_unique<chunk_t>(*chunk);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto clear() {
    for (auto& chunk : m_chunks) {
      if (chunk != nullptr) {
        chunk.reset();
      }
    }
  }
  //----------------------------------------------------------------------------
  auto num_chunks() const { return m_chunks.size(); }
  //----------------------------------------------------------------------------
  auto chunk_size() const { return m_chunk_structure.size(); }
  auto chunk_size(std::size_t i) const { return m_chunk_structure.size(i); }
  //----------------------------------------------------------------------------
  auto internal_chunk_size() const { return m_internal_chunk_size; }
  auto internal_chunk_size(std::size_t const i) const {
    return m_internal_chunk_size[i];
  }
  //----------------------------------------------------------------------------
  auto operator[](std::size_t plain_index) -> T& {
    assert(plain_index < num_components());
    return at(multi_index(plain_index));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](std::size_t plain_index) const -> T const& {
    assert(plain_index < num_components());
    return at(multi_index(plain_index));
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) -> T& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    std::size_t const plain_index = plain_chunk_index_from_global_indices(is...);
    if (chunk_at_is_null(plain_index)) {
      create_chunk_at(plain_index);
    }
    std::size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, is...);
    return (*m_chunks[plain_index])[plain_internal_index];
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) const -> T const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    std::size_t const plain_index = plain_chunk_index_from_global_indices(is...);
    if (chunk_at_is_null(plain_index)) {
      static constexpr T t{};
      return t;
    }
    std::size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, is...);
    return (*m_chunks[plain_index])[plain_internal_index];
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, std::size_t N, integral S>
  auto at(base_tensor<Tensor, S, N> const& is) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  //----------------------------------------------------------------------------
  template <integral S, std::size_t N>
  auto at(std::array<S, N> const& is) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& is) -> T& {
    assert(num_dimensions() == is.size());
    assert(in_range(is));
    std::size_t const plain_chunk_index =
        m_chunk_structure.plain_index(chunk_indices_from_global_indices(is));
    if (!m_chunks[plain_chunk_index]) {
      create_chunk_at(plain_chunk_index);
    }
    return m_chunks[plain_chunk_index]->at(
        internal_chunk_indices_from_global_indices(is));
  }
  //----------------------------------------------------------------------------
  template <integral S>
  auto at(std::vector<S> const& is) const -> T const& {
    assert(num_dimensions() == is.size());
    assert(in_range(is));
    auto global_is = is;
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      global_is[i] /= is[i];
    }
    std::size_t const plain_chunk_index = m_chunk_structure.plain_index(global_is);
    if (!m_chunks[plain_chunk_index]) {
      static const T t{};
      return t;
    }
    auto local_is = is;
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      local_is[i] %= is[i];
    }
    return m_chunks[plain_chunk_index]->at(local_is);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) -> T& {
    assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> T const& {
    assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename Tensor, std::size_t N, integral S>
  auto operator()(base_tensor<Tensor, S, N> const& is) -> T& {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <integral S, std::size_t N>
  auto operator()(std::array<S, N> const& is) -> T& {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <integral S, std::size_t N>
  auto operator()(std::array<S, N> const& is) const -> T const& {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <integral S, std::size_t N>
  auto at(std::array<S, N> const& is) const -> T const& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  ////----------------------------------------------------------------------------
  // auto unchunk() const {
  //  std::vector<T> data(num_components());
  //  for (std::size_t i = 0; i < num_components(); ++i) { data[i] = (*this)[i]; }
  //  return data;
  //}
  ////----------------------------------------------------------------------------
  //
  // auto unchunk_plain() const requires is_arithmetic<T>{
  //  using real_t          = typename T::real_t;
  //  constexpr auto      n = T::num_components();
  //  std::vector<real_t> data;
  //  data.reserve(num_components() * n);
  //  for (std::size_t i = 0; i < num_components(); ++i) {
  //    for (std::size_t j = 0; j < n; ++j) { data.push_back((*this)[i][j]); }
  //  }
  //  return data;
  //}
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  requires is_arithmetic<T> auto randu(
      T min = 0, T max = 1,
      RandomEngine&& random_engine = RandomEngine{std::random_device{}()})
      -> void {
    for (auto& chunk : m_chunks) {
      if (!chunk) {
        chunk = std::make_unique<chunk_t>();
      }
      chunk->randu(min, max, std::forward<RandomEngine>(random_engine));
    }
  }

  auto min_value() const requires is_arithmetic<T> {
    T min = std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) {
        min = std::min(min, *chunk->min_element());
      }
    }
    return min;
  }

  auto max_value() const requires is_arithmetic<T> {
    T max = -std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) {
        max = std::max(max, *chunk->max_element());
      }
    }
    return max;
  }

  auto minmax_value() const requires is_arithmetic<T> {
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

  auto normalize() -> void requires is_arithmetic<T> {
    auto [min, max] = minmax_value();
    auto normalizer = 1.0 / (max - min);

    for (auto const& chunk : m_chunks) {
      if (chunk) {
        for (auto& val : *chunk) {
          val = (val - min) * normalizer;
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  /// Iterates over indices so that coherent chunk indices are being processed
  /// together.
  template <typename Iteration>
  auto iterate_over_indices(Iteration&& iteration) const {
    std::vector<std::pair<std::size_t, std::size_t>> outer_ranges(num_dimensions());
    std::vector<std::pair<std::size_t, std::size_t>> inner_ranges(num_dimensions());

    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      outer_ranges[i] = {0, m_chunk_structure.size(i)};
    }

    // outer loop iterates over chunks
    for_loop(
        [&](auto const& outer_indices) {
          for (std::size_t i = 0; i < num_dimensions(); ++i) {
            auto chunk_size = m_internal_chunk_size[i];
            if (outer_indices[i] == m_chunk_structure.size(i) - 1) {
              chunk_size = this->size(i) % chunk_size;
              if (chunk_size == 0) {
                chunk_size = m_internal_chunk_size[i];
              }
            }
            inner_ranges[i] = {0, chunk_size};
          }
          // inner loop iterates indices of current chunk
          for_loop(
              [&](auto const& inner_indices) {
                std::vector<std::size_t> global_indices(num_dimensions());

                for (std::size_t i = 0; i < num_dimensions(); ++i) {
                  global_indices[i] =
                      outer_indices[i] * m_internal_chunk_size[i] +
                      inner_indices[i];
                }
                iteration(global_indices);
              },
              inner_ranges);
        },
        outer_ranges);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
