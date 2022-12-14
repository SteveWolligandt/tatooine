#ifndef TATOOINE_LAZY_READER_H
#define TATOOINE_LAZY_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/index_order.h>
#include <mutex>
#include <map>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename DataSet, typename GlobalIndexOrder = x_fastest,
          typename LocalIndexOrder = GlobalIndexOrder>
struct lazy_reader
    : chunked_multidim_array<typename DataSet::value_type, GlobalIndexOrder, LocalIndexOrder> {
  using this_type     = lazy_reader<DataSet, GlobalIndexOrder, LocalIndexOrder>;
  using value_type = typename DataSet::value_type;
  using parent_type   = chunked_multidim_array<value_type, GlobalIndexOrder, LocalIndexOrder>;
  using parent_type::chunk_at;

  static auto default_value() -> value_type& {
    static value_type t{};
    return t;
  }

 private:
  DataSet                     m_dataset;
  mutable std::vector<bool>   m_read;
  std::size_t                      m_max_num_chunks_loaded   = 1024;
  bool                        m_limit_num_chunks_loaded = false;
  mutable std::mutex          m_chunks_loaded_mutex;
  mutable std::vector<std::size_t> m_chunks_loaded;
  mutable std::vector<std::unique_ptr<std::mutex>> m_mutexes;

 public:
  lazy_reader(DataSet const& file, std::vector<std::size_t> const& chunk_size)
      : parent_type{std::vector<std::size_t>(chunk_size.size(), 0), chunk_size},
        m_dataset{file} {
    init(chunk_size);
  }
  //----------------------------------------------------------------------------
  lazy_reader(lazy_reader const& other)
      : parent_type{other},
        m_dataset{other.m_dataset},
        m_read{other.m_read}
        //, m_max_num_chunks_loaded{other.m_max_num_chunks_loaded}
        //, m_limit_num_chunks_loaded{other.m_limit_num_chunks_loaded}
  {
    create_mutexes();
  }
  //----------------------------------------------------------------------------
 private:
  auto init(std::vector<std::size_t> const& chunk_size) -> void {
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
  auto read_chunk(std::size_t const plain_chunk_index, integral auto const... indices) const
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
        auto chunks_loaded_lock = std::lock_guard{m_chunks_loaded_mutex};
        m_chunks_loaded.push_back(plain_chunk_index);
        //if (m_limit_num_chunks_loaded &&
        //    num_chunks_loaded() > m_max_num_chunks_loaded) {
        //  // TODO reimplement
        //}
      }

      this->create_chunk_at(plain_chunk_index);
      auto& chunk               = *this->chunk_at(plain_chunk_index);
      m_read[plain_chunk_index] = true;
      auto const offset         = this->global_indices_from_chunk_indices(
          this->chunk_indices_from_global_indices(indices...));
      for (std::size_t i = 0; i < offset.size(); ++i) {
        assert(offset[i] + chunk.size()[i] <= this->size()[i]);
      }
      m_dataset.read(offset, chunk.size(), chunk);

    }

    return this->chunk_at(plain_chunk_index);
  }
  //----------------------------------------------------------------------------
 public:
  auto at(integral auto const... indices) const -> value_type const& {
    auto const      plain_chunk_index =
        this->plain_chunk_index_from_global_indices(indices...);
    auto lock = std::lock_guard {*m_mutexes[plain_chunk_index]};
    auto const&       chunk = read_chunk(plain_chunk_index, indices...);

    if (chunk == nullptr) {
      auto const& t = default_value();
      return t;
    }
    return (*chunk)[this->plain_internal_chunk_index_from_global_indices(
        plain_chunk_index, indices...)];
  }

 private:
  template <integral Index, std::size_t N, std::size_t... Seq>
  auto at(std::array<Index, N> const& indices,
          std::index_sequence<Seq...> /*seq*/) const -> value_type const& {
    return at(indices[Seq]...);
  }

 public:
  template <integral Index, std::size_t N>
  auto at(std::array<Index, N> const& indices) const -> value_type const& {
    return at(indices, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... indices) const -> value_type const& {
    assert(sizeof...(indices) == this->num_dimensions());
    return at(indices...);
  }
  //----------------------------------------------------------------------------
  auto is_chunk_filled_with_value(std::size_t const      plain_chunk_index,
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
  auto is_chunk_filled_with_zeros(std::size_t const plain_chunk_index) const
      -> bool requires is_arithmetic<value_type> {
    return is_chunk_filled_with_value(plain_chunk_index, 0);
  }
  //----------------------------------------------------------------------------
  auto set_max_num_chunks_loaded(std::size_t const /*max_num_chunks_loaded*/) {
    //limit_num_chunks_loaded();
    //m_max_num_chunks_loaded = max_num_chunks_loaded;
  }
  //----------------------------------------------------------------------------
  auto limit_num_chunks_loaded(bool const /*l*/ = true) {
    //m_limit_num_chunks_loaded = l;
  }
  //----------------------------------------------------------------------------
  auto num_chunks_loaded() const {
    return size(m_chunks_loaded);
  }
  //----------------------------------------------------------------------------
  auto chunks_loaded() const -> auto const& { return m_chunks_loaded; }
  //----------------------------------------------------------------------------
  auto chunk_is_loaded(std::size_t const plain_chunk_index) const {
    return std::find(begin(m_chunks_loaded), end(m_chunks_loaded),
                     plain_chunk_index) != end(m_chunks_loaded);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
