#ifndef TATOOINE_LAZY_READER_H
#define TATOOINE_LAZY_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>

#include <mutex>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename DataSet>
struct lazy_reader : chunked_multidim_array<typename DataSet::value_type> {
  using this_t     = lazy_reader<DataSet>;
  using value_type = typename DataSet::value_type;
  using parent_t   = chunked_multidim_array<value_type>;
  using parent_t::chunk_at;

  static auto default_value() -> value_type& {
    static value_type t{};
    return t;
  }

 private:
  DataSet                                          m_dataset;
  mutable std::vector<bool>                        m_read;
  mutable std::vector<std::unique_ptr<std::mutex>> m_mutexes;

 public:
  lazy_reader(DataSet const& file, std::vector<size_t> chunk_size)
      : chunked_multidim_array<value_type>{std::vector<size_t>(
                                               chunk_size.size(), 0),
                                           chunk_size},
        m_dataset{file} {
    init(std::move(chunk_size));
  }
  //----------------------------------------------------------------------------
  lazy_reader(lazy_reader const& other) : parent_t{other}, m_dataset{other.m_dataset} {
    if constexpr (is_arithmetic<value_type>) {
      m_read.resize(this->num_chunks(), false);
      m_mutexes.resize(this->num_chunks());
      for (auto& mutex : m_mutexes) {
        mutex = std::make_unique<std::mutex>();
      }
    }
  }
  //----------------------------------------------------------------------------
 private:
  void init(std::vector<size_t> chunk_size) {
    auto s = m_dataset.size();
    std::reverse(begin(s), end(s));
    this->resize(s, chunk_size);
    if constexpr (is_arithmetic<value_type>) {
      m_read.resize(this->num_chunks(), false);
      m_mutexes.resize(this->num_chunks());
      for (auto& mutex : m_mutexes) {
        mutex = std::make_unique<std::mutex>();
      }
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
    static std::mutex m;
    if (!this->in_range(indices...)) {
      std::lock_guard lock{m};
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
    std::lock_guard lock{*m_mutexes[plain_index]};

    if constexpr (is_arithmetic<value_type>) {
      if (this->chunk_at_is_null(plain_index)) {
        if (!m_read[plain_index]) {
          m_read[plain_index] = true;
          this->create_chunk_at(plain_index);
          auto start_indices = this->global_indices_from_chunk_indices(
              this->chunk_indices_from_global_indices(indices...));
          auto s = this->internal_chunk_size();
          m_dataset.read_chunk(start_indices, s, *chunk_at(plain_index));

          if (is_chunk_filled_with_zeros(plain_index)) {
            this->destroy_chunk_at(plain_index);
          }
        }
      }
    } else {
      if (this->chunk_at_is_null(plain_index)) {
        this->create_chunk_at(plain_index);
        std::vector start_indices{static_cast<size_t>(indices)...};
        auto        s = this->internal_chunk_size();
        m_dataset.read_chunk(start_indices, this->internal_chunk_size());
      }
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
    size_t      plain_index = 0;
    auto const& chunk       = read_chunk(plain_index, indices...);

    if (chunk != nullptr) {
      size_t const plain_internal_index =
          this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                               indices...);
      return (*chunk)[plain_internal_index];
    } else {
      auto& t = default_value();
      t       = value_type{};
      return t;
    }
  }

 private:
#ifdef __cpp_concepts
  template <integral Index, size_t N, size_t... Seq>
#else
  template <typename Index, size_t N, size_t... Seq,
            enable_if<is_integral<Index...>> = true>
#endif
  auto at(std::array<Index, N> const& indices,
          std::index_sequence<Seq...> /*seq*/) const -> value_type const& {
    return at(indices[Seq]...);
  }

 public:
#ifdef __cpp_concepts
  template <integral Index, size_t N>
#else
  template <typename Index, size_t N, enable_if<is_integral<Index...>> = true>
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
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#ifdef TATOOINE_HAS_NETCDF_SUPPORT
#endif
#endif
