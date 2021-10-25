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
struct chunked_multidim_array : dynamic_multidim_size<GlobalIndexOrder>{
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
  using parent_t::multi_index;
  using parent_t::in_range;
  using parent_t::num_components;
  using parent_t::num_dimensions;
  using parent_t::size;
  //----------------------------------------------------------------------------
 private:
  std::vector<size_t>                    m_internal_chunk_size;
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
#ifdef __cpp_concepts
  template <range SizeRange, range ChunkSizeRange>
      requires(is_integral<typename std::decay_t<SizeRange>::value_type>) &&
      (is_integral<typename std::decay_t<ChunkSizeRange>::value_type>)
#else
  template <typename SizeRange, typename ChunkSizeRange,
            enable_if<is_range<SizeRange, ChunkSizeRange> > = true>
#endif
          chunked_multidim_array(SizeRange&&      size,
                                 ChunkSizeRange&& chunk_size) {
    resize(std::forward<SizeRange>(size),
           std::forward<ChunkSizeRange>(chunk_size));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <range Range>
#else
  template <typename Range, enable_if<is_range<Range> > = true>
#endif
  chunked_multidim_array(Range&& data, std::vector<size_t> const& size,
                         std::vector<size_t> const& chunk_size) {
    resize(size, chunk_size);
    size_t i = 0;
    for (auto const& d : data) {
      (*this)[i++] = d;
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <can_read<this_t> Reader>
#else
  template <
      typename Reader,
      enable_if<is_same<void, std::void_t<decltype(std::declval<Reader>().read(
                                  std::declval<this_t>()))> > > = true>
#endif
  chunked_multidim_array(Reader&&                   reader,
                         std::vector<size_t> const& chunk_size) {
    m_internal_chunk_size = chunk_size;
    reader.read(*this);
  }
  //==============================================================================
#ifdef __cpp_concepts
  template <range SizeRange>
  requires(std::is_integral_v<typename std::decay_t<SizeRange>::value_type>)
#else
  template <
      typename SizeRange, enable_if<is_range<SizeRange> > = true,
      enable_if<is_integral<typename std::decay_t<SizeRange>::value_type> > =
          true>
#endif
      void resize(SizeRange&& size) {
    // apply full size
    parent_t::resize(size);

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
#ifdef __cpp_concepts
  template <range SizeRange, range ChunkSizeRange>
      requires(
          std::is_integral_v<typename std::decay_t<SizeRange>::value_type>) &&
      (std::is_integral_v<typename std::decay_t<ChunkSizeRange>::value_type>)
#else
  template <typename SizeRange, typename ChunkSizeRange,
            enable_if<is_range<SizeRange, ChunkSizeRange> >          = true,
            enable_if<is_integral<
                typename std::decay_t<SizeRange>::value_type,
                typename std::decay_t<ChunkSizeRange>::value_type> > = true>
#endif
  void resize(SizeRange&& size, ChunkSizeRange&& chunk_size) {
    m_internal_chunk_size.resize(chunk_size.size());
    std::copy(begin(chunk_size), end(chunk_size), begin(m_internal_chunk_size));
    size_t i = 0;
    for (auto& s : m_internal_chunk_size) {
      s = std::min<size_t>(s, size[i++]);
    }
    resize(std::forward<SizeRange>(size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if<is_integral<Size...> > = true>
#endif
  auto resize(Size const... sizes) -> void {
    return resize(std::vector{static_cast<size_t>(sizes)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename Tensor, integral Int, size_t N>
#else
  template <typename Tensor, typename Int, size_t N,
            enable_if<is_integral<Int> > = true>
#endif
  auto resize(base_tensor<Tensor, Int, N> const& v) -> void {
    assert(N == num_dimensions());
    std::vector<size_t> s(num_dimensions());
    for (size_t i = 0; i < N; ++i) {
      s[i] = v(i);
    }
    return resize(s);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Int, size_t N>
#else
  template <typename Int, size_t N, enable_if<is_integral<Int> > = true>
#endif
  auto resize(std::array<Int, N> const& v) {
    assert(N == num_dimensions());
    return resize(std::vector<size_t>(begin(v), end(v)));
  }
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <integral... Is, size_t... Seq>
#else
  template <typename... Is, size_t... Seq,
            enable_if<is_integral<Is...> > = true>
#endif
  auto plain_internal_chunk_index_from_global_indices(
      size_t plain_chunk_index, std::index_sequence<Seq...>,
      Is const... is) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(is) == m_chunks[plain_chunk_index]->num_dimensions());
    return m_chunks[plain_chunk_index]->plain_index(
        (is % m_internal_chunk_size[Seq])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto plain_internal_chunk_index_from_global_indices(
      size_t plain_chunk_index, Is const... is) const {
    assert(m_chunks[plain_chunk_index] != nullptr);
    assert(sizeof...(is) == m_chunks[plain_chunk_index]->num_dimensions());
    return plain_internal_chunk_index_from_global_indices(
        plain_chunk_index, std::make_index_sequence<sizeof...(is)>{},
        is...);
  }
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <integral... Is, size_t... Seq>
#else
  template <typename... Is, size_t... Seq,
            enable_if<is_integral<Is...> > = true>
#endif
  auto plain_chunk_index_from_global_indices(
      std::index_sequence<Seq...> /*seq*/, Is const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return m_chunk_structure.plain_index(
        (is / m_internal_chunk_size[Seq])...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto plain_chunk_index_from_global_indices(Is const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return plain_chunk_index_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <integral... Is, size_t... Seq>
#else
  template <typename... Is, size_t... Seq,
            enable_if<is_integral<Is...> > = true>
#endif
  auto internal_chunk_indices_from_global_indices(
      std::index_sequence<Seq...>, Is const... is) const {
    return std::array{
        static_cast<size_t>(is % m_internal_chunk_size[Seq])...};
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto internal_chunk_indices_from_global_indices(
      Is const... is) const {
    return internal_chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // template <integral Int>
  // auto internal_chunk_indices_from_global_indices(
  //    std::vector<Int> is) const {
  //  assert(size(is) == num_dimensions());
  //  for (size_t i = 0; i < num_dimensions(); ++i) {
  //    is[i] %= m_internal_chunk_size[i];
  //  }
  //  return is;
  //}
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <integral... Is, size_t... Seq>
#else
  template <typename... Is, size_t... Seq,
            enable_if<is_integral<Is...> > = true>
#endif
  auto chunk_indices_from_global_indices(std::index_sequence<Seq...>,
                                         Is const... is) const {
    return std::vector{
        (static_cast<size_t>(is) / m_internal_chunk_size[Seq])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto chunk_indices_from_global_indices(Is const... is) const {
    assert(sizeof...(is) == num_dimensions());
    return chunk_indices_from_global_indices(
        std::make_index_sequence<sizeof...(is)>{}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if<is_integral<Int> > = true>
#endif
  auto chunk_indices_from_global_indices(std::vector<Int> is) const {
    assert(size(is) == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      is[i] /= m_internal_chunk_size[i];
    }
    return is;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if<is_integral<Int> > = true>
#endif
  auto global_indices_from_chunk_indices(std::vector<Int> is) const {
    assert(is.size() == num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      is[i] *= m_internal_chunk_size[i];
    }
    return is;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... ChunkIndices>
#else
  template <typename... ChunkIndices, enable_if<is_integral<ChunkIndices...> > = true>
#endif
  auto plain_chunk_index_from_chunk_indices(
      ChunkIndices const... chunk_indices) const {
    assert(sizeof...(chunk_indices) == num_dimensions());
    return m_chunk_structure.plain_index(chunk_indices...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if<is_integral<Int> > = true>
#endif
  auto plain_chunk_index_from_chunk_indices(
      std::vector<Int> const& chunk_indices) const {
    assert(chunk_indices.size() == num_dimensions());
    return m_chunk_structure.plain_index(chunk_indices);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral ChunkIndex0, integral... ChunkIndices>
#else
  template <typename ChunkIndex0, typename... ChunkIndices,
            enable_if<is_integral<ChunkIndex0, ChunkIndices...> > = true>
#endif
  auto chunk_at(ChunkIndex0 const chunk_index0,
                ChunkIndices const... chunk_indices) const -> auto const& {
    if constexpr (sizeof...(chunk_indices) == 0) {
      return m_chunks[chunk_index0];
    } else {
      assert(sizeof...(chunk_indices) + 1 == num_dimensions());
      return m_chunks[plain_chunk_index_from_chunk_indices(chunk_index0,
                                                           chunk_indices...)];
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral ChunkIndex0, integral... ChunkIndices>
#else
  template <typename ChunkIndex0, typename... ChunkIndices,
            enable_if<is_integral<ChunkIndex0, ChunkIndices...> > = true>
#endif
  auto chunk_at_is_null(ChunkIndex0 const chunk_index0,
                        ChunkIndices const... chunk_indices) const {
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
    for (size_t i = 0; i < num_chunks(); ++i) {
      create_chunk_at(i);
    }
  }
  //----------------------------------------------------------------------------
  auto create_chunk_at(size_t const               plain_chunk_index,
                       std::vector<size_t> const& multi_indices) const -> auto const& {
    assert(multi_indices.size() == num_dimensions());
    std::vector<size_t> chunk_size = m_internal_chunk_size;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      if (multi_indices[i] == m_chunk_structure.size(i)-1) {
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
  auto create_chunk_at(size_t const plain_chunk_index) const -> auto const& {
    assert(plain_chunk_index < num_chunks());
    return create_chunk_at(plain_chunk_index,
                           m_chunk_structure.multi_index(plain_chunk_index));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral ChunkIndex0, integral ChunkIndex1,
            integral... ChunkIndices>
#else
  template <
      typename ChunkIndex0, typename ChunkIndex1, typename... ChunkIndices,
      enable_if<is_integral<ChunkIndex0, ChunkIndex1, ChunkIndices...> > = true>
#endif
  auto create_chunk_at(ChunkIndex0 const chunk_index0,
                       ChunkIndex1 const chunk_index1,
                       ChunkIndices const... chunk_indices) const -> auto const& {
    assert(sizeof...(chunk_indices) + 2 == num_dimensions());
    return create_chunk_at(
        plain_chunk_index_from_chunk_indices(chunk_index0, chunk_index1,
                                             chunk_indices...),
        std::vector<size_t>{chunk_index0, chunk_index1, chunk_indices...});
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral ChunkIndex0, integral... ChunkIndices>
#else
  template <typename ChunkIndex0, typename... ChunkIndices,
            enable_if<is_integral<ChunkIndex0, ChunkIndices...> > = true>
#endif
  auto destroy_chunk_at(ChunkIndex0 const chunk_index0,
                        ChunkIndices const... chunk_indices) const {
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
  auto chunk_size(size_t i) const { return m_chunk_structure.size(i); }
  //----------------------------------------------------------------------------
  auto internal_chunk_size() const { return m_internal_chunk_size; }
  auto internal_chunk_size(size_t const i) const {
    return m_internal_chunk_size[i];
  }
  //----------------------------------------------------------------------------
  auto operator[](size_t plain_index) -> T& {
    assert(plain_index < num_components());
    return at(multi_index(plain_index));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t plain_index) const -> T const& {
    assert(plain_index < num_components());
    return at(multi_index(plain_index));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto at(Is const... is) -> T& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    size_t const plain_index =
        plain_chunk_index_from_global_indices(is...);
    if (chunk_at_is_null(plain_index)) {
      create_chunk_at(plain_index);
    }
    size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, is...);
    return (*m_chunks[plain_index])[plain_internal_index];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto at(Is const... is) const -> T const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    size_t const plain_index =
        plain_chunk_index_from_global_indices(is...);
    if (chunk_at_is_null(plain_index)) {
      static constexpr T t{};
      return t;
    }
    size_t const plain_internal_index =
        plain_internal_chunk_index_from_global_indices(plain_index, is...);
    return (*m_chunks[plain_index])[plain_internal_index];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename Tensor, size_t N, integral S>
#else
  template <typename Tensor, size_t N, typename S,
            enable_if<is_integral<S> > = true>
#endif
  auto at(base_tensor<Tensor, S, N> const& is) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S, size_t N>
#else
  template <typename S, size_t N, enable_if<is_integral<S> > = true>
#endif
  auto at(std::array<S, N> const& is) -> T& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S>
#else
  template <typename S, enable_if<is_integral<S> > = true>
#endif
  auto at(std::vector<S> const& is) -> T& {
    assert(num_dimensions() == is.size());
    assert(in_range(is));
    size_t const plain_chunk_index = m_chunk_structure.plain_index(
        chunk_indices_from_global_indices(is));
    if (!m_chunks[plain_chunk_index]) {
      create_chunk_at(plain_chunk_index);
    }
    return m_chunks[plain_chunk_index]->at(
        internal_chunk_indices_from_global_indices(is));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S>
#else
  template <typename S, enable_if<is_integral<S> > = true>
#endif
  auto at(std::vector<S> const& is) const -> T const& {
    assert(num_dimensions() == is.size());
    assert(in_range(is));
    auto global_is = is;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      global_is[i] /= is[i];
    }
    size_t const plain_chunk_index = m_chunk_structure.plain_index(global_is);
    if (!m_chunks[plain_chunk_index]) {
      static const T t{};
      return t;
    }
    auto local_is = is;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      local_is[i] %= is[i];
    }
    return m_chunks[plain_chunk_index]->at(local_is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto operator()(Is const... is) -> T& {
    assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...> > = true>
#endif
  auto operator()(Is const... is) const -> T const& {
    assert(sizeof...(is) == num_dimensions());
    return at(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename Tensor, size_t N, integral S>
#else
  template <typename Tensor, size_t N, typename S,
            enable_if<is_integral<S> > = true>
#endif
  auto operator()(base_tensor<Tensor, S, N> const& is) -> T& {
    return at(is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S, size_t N>
#else
  template <typename S, size_t N, enable_if<is_integral<S> > = true>
#endif
  auto operator()(std::array<S, N> const& is) -> T& {
    return at(is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S, size_t N>
#else
  template <typename S, size_t N, enable_if<is_integral<S> > = true>
#endif
  auto operator()(std::array<S, N> const& is) const -> T const& {
    return at(is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral S, size_t N>
#else
  template <typename S, size_t N, enable_if<is_integral<S> > = true>
#endif
  auto at(std::array<S, N> const& is) const -> T const& {
    return invoke_unpacked(
        [this](auto const... is) -> decltype(auto) { return at(is...); },
        unpack(is));
  }
  ////----------------------------------------------------------------------------
  // auto unchunk() const {
  //  std::vector<T> data(num_components());
  //  for (size_t i = 0; i < num_components(); ++i) { data[i] = (*this)[i]; }
  //  return data;
  //}
  ////----------------------------------------------------------------------------
  // template <typename = void>
  // requires is_arithmetic<T>
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
#ifdef __cpp_concepts
  template <typename RandomEngine = std::mt19937_64>
  requires is_arithmetic<T>
#else
  template <typename RandomEngine        = std::mt19937_64,
            enable_if<is_arithmetic<T> > = true>
#endif
      auto randu(T min = 0, T max = 1,
                 RandomEngine&& random_engine = RandomEngine{
                     std::random_device{}()}) -> void {
    for (auto& chunk : m_chunks) {
      if (!chunk) {
        chunk = std::make_unique<chunk_t>();
      }
      chunk->randu(min, max, std::forward<RandomEngine>(random_engine));
    }
  }
#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename _T = T, enable_if<is_arithmetic<T> > = true>
#endif
      auto min_value() const {
    T min = std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) {
        min = std::min(min, *chunk->min_element());
      }
    }
    return min;
  }

#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename _T = T, enable_if<is_arithmetic<T> > = true>
#endif
      auto max_value() const {
    T max = -std::numeric_limits<T>::max();
    for (auto const& chunk : m_chunks) {
      if (chunk) {
        max = std::max(max, *chunk->max_element());
      }
    }
    return max;
  }

#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename _T = T, enable_if<is_arithmetic<T> > = true>
#endif
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

#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename _T = T, enable_if<is_arithmetic<T> > = true>
#endif
      auto normalize() -> void {
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
    std::vector<std::pair<size_t, size_t>> outer_ranges(num_dimensions());
    std::vector<std::pair<size_t, size_t>> inner_ranges(num_dimensions());

    for (size_t i = 0; i < num_dimensions(); ++i) {
      outer_ranges[i] = {0, m_chunk_structure.size(i)};
    }

    // outer loop iterates over chunks
    for_loop(
        [&](auto const& outer_indices) {
          for (size_t i = 0; i < num_dimensions(); ++i) {
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
                std::vector<size_t> global_indices(num_dimensions());

                for (size_t i = 0; i < num_dimensions(); ++i) {
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
