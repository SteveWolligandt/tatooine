#ifndef TATOOINE_LAZY_READER_H
#define TATOOINE_LAZY_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/index_order.h>
#include <mutex>
#include <list>
#include <map>
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
  struct chunks_it_comp {
    auto operator()(
        std::pair<size_t, typename std::list<size_t>::iterator> const& lhs,
        std::pair<size_t, typename std::list<size_t>::iterator> const& rhs) {
      return lhs.first < rhs.first;
    }
  };
  DataSet                   m_dataset;
  mutable std::vector<bool> m_read;
  mutable std::list<size_t> m_chunks_loaded;
  mutable std::map<size_t, typename std::list<size_t>::iterator> m_chunks_its;
  size_t             m_max_num_chunks_loaded   = 1024;
  bool               m_limit_num_chunks_loaded = false;
  mutable std::mutex m_chunks_loaded_mutex;
  mutable std::vector<std::unique_ptr<std::mutex>> m_mutexes;

 public:
  lazy_reader(DataSet const& file, std::vector<size_t> const& chunk_size)
      : parent_t{std::vector<size_t>(chunk_size.size(), 0), chunk_size},
        m_dataset{file} {
    init(chunk_size);
  }
  //----------------------------------------------------------------------------
  lazy_reader(lazy_reader const& other)
      : parent_t{other},
        m_dataset{other.m_dataset},
        m_read{other.m_read},
        m_max_num_chunks_loaded{other.m_max_num_chunks_loaded},
        m_limit_num_chunks_loaded{other.m_limit_num_chunks_loaded} {
    create_mutexes();
  }
  //----------------------------------------------------------------------------
 private:
  auto init(std::vector<size_t> const& chunk_size) -> void {
    auto s = m_dataset.size();
    this->resize(s, chunk_size);
    m_read.resize(this->num_chunks(), false);
    create_mutexes();
  }
  //----------------------------------------------------------------------------
  auto create_mutexes() {
    m_mutexes.resize(this->num_chunks());
    for (auto& mutex : m_mutexes) {
      mutex = std::make_unique<std::mutex>();
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if<is_integral<Indices...>> = true>
#endif
  auto read_chunk(size_t const plain_chunk_index, Indices const... indices) const
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

    if (this->chunk_at_is_null(plain_chunk_index) && !m_read[plain_chunk_index]) {
      {
        std::lock_guard chunks_loaded_lock{m_chunks_loaded_mutex};
        m_chunks_loaded.push_back(plain_chunk_index);
        m_chunks_its[plain_chunk_index] = prev(end(m_chunks_loaded));
        if (m_limit_num_chunks_loaded &&
            num_chunks_loaded() > m_max_num_chunks_loaded) {
          auto chunk_to_erase = begin(m_chunks_loaded);
          bool success        = false;
          while (!success) {
            ++chunk_to_erase;
            if (chunk_to_erase == end(m_chunks_loaded)) {
              chunk_to_erase = begin(m_chunks_loaded);
            }
            auto chunk_to_erase_lock =
                std::unique_lock{*m_mutexes[*chunk_to_erase], std::defer_lock};
            auto const is_locked = chunk_to_erase_lock.try_lock();
            if (is_locked) {
              m_read[*chunk_to_erase] = false;
              this->destroy_chunk_at(*chunk_to_erase);
              m_chunks_its.erase(*chunk_to_erase);
              m_chunks_loaded.erase(chunk_to_erase);
              success = true;
            }
          }
        }
      }

      this->create_chunk_at(plain_chunk_index);
      auto& chunk               = *this->chunk_at(plain_chunk_index);
      m_read[plain_chunk_index] = true;
      auto const offset         = this->global_indices_from_chunk_indices(
          this->chunk_indices_from_global_indices(indices...));
      for (size_t i = 0; i < offset.size(); ++i) {
        assert(offset[i] + chunk.size()[i] <= this->size()[i]);
      }
      m_dataset.read_chunk(offset, chunk.size(), chunk);

    } else {
      std::lock_guard lock{m_chunks_loaded_mutex};
      // this will move the current adressed chunked at the end of loaded
      // chunks
      m_chunks_loaded.splice(end(m_chunks_loaded), m_chunks_loaded,
                             m_chunks_its[plain_chunk_index]);
    }

    return this->chunk_at(plain_chunk_index);
  }
  //----------------------------------------------------------------------------
 public:
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if<is_integral<Indices...> > = true>
#endif
  auto at(Indices const... indices) const -> value_type const& {
    auto const      plain_chunk_index =
        this->plain_chunk_index_from_global_indices(indices...);
    std::lock_guard lock{*m_mutexes[plain_chunk_index]};
    auto const&       chunk = read_chunk(plain_chunk_index, indices...);

    if (chunk == nullptr) {
      auto const& t = default_value();
      return t;
    }
    return (*chunk)[this->plain_internal_chunk_index_from_global_indices(
        plain_chunk_index, indices...)];
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
  auto is_chunk_filled_with_value(size_t const      plain_chunk_index,
                                  value_type const& value) const -> bool {
    auto const& chunk_data = chunk_at(plain_chunk_index)->data();
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
      auto is_chunk_filled_with_zeros(size_t const plain_chunk_index) const
      -> bool {
    return is_chunk_filled_with_value(plain_chunk_index, 0);
  }
  //----------------------------------------------------------------------------
  auto set_max_num_chunks_loaded(size_t const max_num_chunks_loaded) {
    limit_num_chunks_loaded();
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
  //----------------------------------------------------------------------------
  auto chunks_loaded() const -> auto const& { return m_chunks_loaded; }
  //----------------------------------------------------------------------------
  auto chunk_is_loaded(size_t const plain_chunk_index) const {
    return std::find(begin(m_chunks_loaded), end(m_chunks_loaded),
                     plain_chunk_index) != end(m_chunks_loaded);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
