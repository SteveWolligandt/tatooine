#ifndef TATOOINE_LAZY_READER_H
#define TATOOINE_LAZY_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/index_order.h>
#include <mutex>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename DataSet, typename GlobalIndexOrder = x_fastest,
          typename LocalIndexOrder = GlobalIndexOrder>
struct lazy_reader
    : chunked_multidim_array<typename DataSet::value_type, GlobalIndexOrder, LocalIndexOrder> {
  using this_t     = lazy_reader<DataSet, GlobalIndexOrder, LocalIndexOrder>;
  using value_type = typename DataSet::value_type;
  using parent_t   = chunked_multidim_array<value_type, GlobalIndexOrder, LocalIndexOrder>;
  using parent_t::chunk_at;

  static auto default_value() -> value_type& {
    static value_type t{};
    return t;
  }

 private:
  DataSet                     m_dataset;
  mutable std::vector<bool>   m_read;
  mutable std::list<size_t>   m_chunks_loaded;
  size_t                      m_max_num_chunks_loaded   = 1024;
  bool                        m_limit_num_chunks_loaded = false;
  mutable std::mutex          m_mutex;

 public:
  lazy_reader(DataSet const& file, std::vector<size_t> chunk_size)
      : parent_t{std::vector<size_t>(chunk_size.size(), 0), chunk_size},
        m_dataset{file} {
    init(std::move(chunk_size));
  }
  //----------------------------------------------------------------------------
  lazy_reader(lazy_reader const& other)
      : parent_t{other},
        m_dataset{other.m_dataset},
        m_read{other.m_read},
        m_max_num_chunks_loaded{other.m_max_num_chunks_loaded},
        m_limit_num_chunks_loaded{other.m_limit_num_chunks_loaded} {
  }
  //----------------------------------------------------------------------------
 private:
  void init(std::vector<size_t> chunk_size) {
    std::lock_guard lock{m_mutex};
    auto s = m_dataset.size();
    this->resize(s, chunk_size);
    if constexpr (is_arithmetic<value_type>) {
      m_read.resize(this->num_chunks(), false);
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if<is_integral<Indices...>> = true>
#endif
  auto read_chunk(size_t& plain_index, Indices const... indices) const
      -> auto const& {
#ifndef NDEBUG
    if (!this->in_range(indices...)) {
      std::cerr << "not in range: ";
      ((std::cerr << indices << ", "), ...);
      std::cerr << '\n';
      std::cerr << '\n';
      std::cerr << "size is: ";
      for (auto s : this->size()) {
        std::cerr << s << ", ";
      }
      std::cerr << '\n';
    }
#endif
    assert(sizeof...(indices) == this->num_dimensions());
    assert(this->in_range(indices...));
    plain_index = this->plain_chunk_index_from_global_indices(indices...);

    if (this->chunk_at_is_null(plain_index) && !m_read[plain_index]) {
      // keep the number of loaded chunks between max_num_chunks_loaded/2 and
      // max_num_chunks_loaded
      if (m_limit_num_chunks_loaded &&
          num_chunks_loaded() > 0) {
        auto const it_begin = begin(m_chunks_loaded);
        auto const it_end   = next(it_begin, m_max_num_chunks_loaded / 2);
        for (auto it = it_begin; it != it_end; ++it) {
          m_read[*it] = false;
          this->destroy_chunk_at(*it);
        }
        m_chunks_loaded.erase(it_begin, it_end);
      }

      this->create_chunk_at(plain_index);
      m_read[plain_index] = true;
      m_chunks_loaded.push_back(plain_index);
      auto const offset = this->global_indices_from_chunk_indices(
          this->chunk_indices_from_global_indices(indices...));
      auto const s = this->internal_chunk_size();
      m_dataset.read_chunk(offset, s, *chunk_at(plain_index));

      //if constexpr (is_arithmetic<value_type>) {
      //  if (is_chunk_filled_with_zeros(plain_index)) {
      //    this->destroy_chunk_at(plain_index);
      //  }
      //}
    } else {
      // this will move the current adressed chunked at the end of loaded chunks
      m_chunks_loaded.splice(
          end(m_chunks_loaded), m_chunks_loaded,
          std::find(begin(m_chunks_loaded), end(m_chunks_loaded), plain_index));
    }
    return this->chunk_at(plain_index);
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if<is_integral<Indices...>> = true>
#endif
  auto at(Indices const... indices) const -> value_type const& {
    std::lock_guard lock{m_mutex};
    size_t      plain_index = 0;
    auto const& chunk       = read_chunk(plain_index, indices...);

    if (chunk != nullptr) {
      size_t const plain_internal_index =
          this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                               indices...);
      return (*chunk)[plain_internal_index];
    }
    auto& t = default_value();
    t       = value_type{};
    return t;
  }

 private:
#ifdef __cpp_concepts
  template <integral Index, size_t N, size_t... Seq>
#else
  template <typename Index, size_t N, size_t... Seq,
            enable_if<is_integral<Index>> = true>
#endif
  auto at(std::array<Index, N> const& indices,
          std::index_sequence<Seq...> /*seq*/) const -> value_type const& {
    return at(indices[Seq]...);
  }

 public:
#ifdef __cpp_concepts
  template <integral Index, size_t N>
#else
  template <typename Index, size_t N, enable_if<is_integral<Index>> = true>
#endif
  auto at(std::array<Index, N> const& indices) const -> value_type const& {
    return at(indices, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if<is_integral<Indices...>> = true>
#endif
  auto operator()(Indices const... indices) const -> value_type const& {
    assert(sizeof...(indices) == this->num_dimensions());
    return at(indices...);
  }
  //----------------------------------------------------------------------------
  auto is_chunk_filled_with_value(size_t const      plain_index,
                                  value_type const& value) const -> bool {
    auto const& chunk_data = chunk_at(plain_index)->data();
    for (auto const& v : chunk_data) {
      if (v != value) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<value_type>
#else
  template <typename V                           = value_type,
            enable_if<is_arithmetic<value_type>> = true>
#endif
      auto is_chunk_filled_with_zeros(size_t const plain_index) const -> bool {
    return is_chunk_filled_with_value(plain_index, 0);
  }
  //----------------------------------------------------------------------------
  auto set_max_num_chunks_loaded(size_t const max_num_chunks_loaded) {
    m_max_num_chunks_loaded = max_num_chunks_loaded;
  }
  //----------------------------------------------------------------------------
  auto limit_num_chunks_loaded(bool const l = true) {
    m_limit_num_chunks_loaded = l;
  }
  //----------------------------------------------------------------------------
  auto num_chunks_loaded() const {
    return size(m_chunks_loaded);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
