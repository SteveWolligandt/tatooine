#ifndef TATOOINE_LAZY_NETCDF_READER_H
#define TATOOINE_LAZY_NETCDF_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/netcdf.h>
#include <mutex>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
template <typename T>
struct lazy_reader : chunked_multidim_array<T> {
  using value_type = T;
  using parent_t   = chunked_multidim_array<T>;
  using parent_t::chunk_at;

 private:
  netcdf::variable<T>       m_var;
  mutable std::vector<bool> m_read;
  mutable size_t            m_num_active_chunks;
  mutable std::mutex        m_mutex;

 public:
  lazy_reader(std::string const& file_path, std::string const& var_name,
              std::vector<size_t> chunk_size)
      : chunked_multidim_array<T>{std::vector<size_t>(chunk_size.size(), 0),
                                  chunk_size},
        m_var{file{file_path, netCDF::NcFile::read}.variable<T>(var_name)} {
    init(std::move(chunk_size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  lazy_reader(netcdf::variable<T> const& var, std::vector<size_t> chunk_size)
      : chunked_multidim_array<T>{std::vector<size_t>(chunk_size.size(), 0),
                                  chunk_size},
        m_var{var} {
    init(std::move(chunk_size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  lazy_reader(netcdf::variable<T>&& var, std::vector<size_t> chunk_size)
      : chunked_multidim_array<T>{std::vector<size_t>(chunk_size.size(), 0),
                                  chunk_size},
        m_var{std::move(var)} {
    init(std::move(chunk_size));
  }
  //----------------------------------------------------------------------------
  lazy_reader(lazy_reader const& other) : parent_t{other}, m_var{other.m_var} {
    if constexpr (std::is_arithmetic_v<T>) {
      m_read.resize(this->num_chunks(), false);
    }
  }
  //----------------------------------------------------------------------------
 private:
  void init(std::vector<size_t> chunk_size) {
    auto s = m_var.size();
    std::reverse(begin(s), end(s));
    this->resize(s, chunk_size);
    if constexpr (std::is_arithmetic_v<T>) {
      m_read.resize(this->num_chunks(), false);
    }
  }
  //----------------------------------------------------------------------------
 public:
  auto at(integral auto const... indices) -> T& {
    assert(sizeof...(indices) == this->num_dimensions());
    assert(this->in_range(indices...));
    size_t const plain_index =
        this->plain_chunk_index_from_global_indices(indices...);

    static T t{};
    if constexpr (std::is_arithmetic_v<T>) {
      std::lock_guard lock{m_mutex};
      if (this->chunk_at_is_null(plain_index)) {
        if (m_read[plain_index]) {
          t = 0;
          return t;
        } else {
          m_read[plain_index] = true;
          // if (m_num_active_chunks > 50) {
          //  for (auto& chunk : this->m_chunks) {
          //    if (chunk != nullptr) { chunk.reset(); }
          //  }
          //  m_num_active_chunks = 0;
          //  for (auto& r:m_read) {r = false;}
          //}
          this->create_chunk_at(plain_index);
          ++m_num_active_chunks;
          auto start_indices = this->global_indices_from_chunk_indices(
              this->chunk_indices_from_global_indices(indices...));
          std::reverse(begin(start_indices), end(start_indices));
          auto s = this->internal_chunk_size();
          std::reverse(begin(s), end(s));
          m_var.read_chunk(start_indices, s, *chunk_at(plain_index));

          bool all_zero = true;
          for (auto const& v : chunk_at(plain_index)->data()) {
            if (v != 0) {
              all_zero = false;
              break;
            }
          }
          if (all_zero) {
            this->destroy_chunk_at(plain_index);
            t = 0;
            return t;
          }
        }
      }
    } else {
      if (this->chunk_at_is_null(plain_index)) {
        //if (m_num_active_chunks > 50) {
        //  for (auto& chunk : this->m_chunks) {
        //    if (chunk != nullptr) { chunk.reset(); }
        //  }
        //  m_num_active_chunks = 0;
        //}
        this->create_chunk_at(plain_index);
        ++m_num_active_chunks;
        std::vector start_indices{static_cast<size_t>(indices)...};
        std::reverse(begin(start_indices), end(start_indices));
        auto s = this->internal_chunk_size();
        m_var.read_chunk(start_indices, this->internal_chunk_size());
      }
    }

    size_t const plain_internal_index =
        this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                             indices...);
    assert(!this->chunk_at_is_null(plain_index));
    return (*chunk_at(plain_index))[plain_internal_index];
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == this->num_dimensions());
    assert(this->in_range(indices...));
    size_t const plain_index =
        this->plain_chunk_index_from_global_indices(indices...);

    static T t{};
    if constexpr (std::is_arithmetic_v<T>) {
      if (this->chunk_at_is_null(plain_index)) {
        std::lock_guard lock{m_mutex};
        if (m_read[plain_index]) {
          t = 0;
          return t;
        } else {
          m_read[plain_index] = true;
          //if (m_num_active_chunks > 50) {
          //  for (auto& chunk : this->m_chunks) {
          //    if (chunk != nullptr) { chunk.reset(); }
          //  }
          //  m_num_active_chunks = 0;
          //}
          this->create_chunk_at(plain_index);
          ++m_num_active_chunks;
          auto start_indices = this->global_indices_from_chunk_indices(
              this->chunk_indices_from_global_indices(indices...));
          std::reverse(begin(start_indices), end(start_indices));
          auto s = this->internal_chunk_size();
          std::reverse(begin(s), end(s));
          m_var.read_chunk(start_indices, s, *chunk_at(plain_index));

          bool all_zero = true;
          for (auto const& v : chunk_at(plain_index)->data()) {
            if (v != 0) {
              all_zero = false;
              break;
            }
            }
            if (all_zero) {
              this->destroy_chunk_at(plain_index);
              t = 0;
              return t;
            }
        }
      } else {
        if (this->chunk_at_is_null(plain_index)) {
          // if (m_num_active_chunks > 50) {
          //  for (auto& chunk : this->m_chunks) {
          //    if (chunk != nullptr) { chunk.reset(); }
          //  }
          //  m_num_active_chunks = 0;
          //}
          this->create_chunk_at(plain_index);
          ++m_num_active_chunks;
          std::vector start_indices{static_cast<size_t>(indices)...};
          std::reverse(begin(start_indices), end(start_indices));
          auto s = this->internal_chunk_size();
          m_var.read_chunk(start_indices, this->internal_chunk_size());
        }
      }
    }

    size_t const plain_internal_index =
        this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                             indices...);
    return (*chunk_at(plain_index))[plain_internal_index];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <integral Int, size_t N, size_t... Is>
  auto at(std::index_sequence<Is...>, std::array<Int, N> const& is) -> T& {
    return at(is[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N, size_t... Is>
  auto at(std::index_sequence<Is...>, std::array<Int, N> const& is) const
      -> T const& {
    return at(is[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <integral Int, size_t N>
  auto at(std::array<Int, N> const& is) -> T& {
    return at(std::make_index_sequence<N>{}, is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  auto at(std::array<Int, N> const& is) const -> T const& {
    return at(std::make_index_sequence<N>{}, is);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... indices) -> T& {
    assert(sizeof...(indices) == this->num_dimensions());
    return at(indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == this->num_dimensions());
    return at(indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  auto operator()(std::array<Int, N> const& is) -> T& {
    return at(std::make_index_sequence<N>{}, is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  auto operator()(std::array<Int, N> const& is) const -> T const& {
    return at(std::make_index_sequence<N>{}, is);
  }
};
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
#endif
