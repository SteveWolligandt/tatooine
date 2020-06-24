#ifndef TATOOINE_LAZY_NETCDF_READER_H
#define TATOOINE_LAZY_NETCDF_READER_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/netcdf.h>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
template <typename T>
struct lazy_reader : chunked_multidim_array<T> {
  using value_type = T;
  using parent_t = chunked_multidim_array<T>;
  using parent_t::chunk_at;

 private:
  netcdf::variable<T> m_var;
  std::vector<bool>   m_read;

 public:
  lazy_reader(std::string const& file_path, std::string const& var_name,
              std::vector<size_t> chunk_size)
      : chunked_multidim_array<T>{std::vector<size_t>(chunk_size.size(), 0),
                                  chunk_size},
        m_var{file{file_path, netCDF::NcFile::read}.variable<T>(var_name)} {
    init(std::move(chunk_size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  lazy_reader(netcdf::variable<T>const& var, std::vector<size_t> chunk_size)
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
      if (this->chunk_at_is_null(plain_index)) {
        if (m_read[plain_index]) {
          t = 0;
          return t;
        } else {
          m_read[plain_index] = true;
          this->create_chunk_at(plain_index);
          std::vector start_indices{static_cast<size_t>(indices)...};
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
        this->create_chunk_at(plain_index);
        std::vector start_indices{static_cast<size_t>(indices)...};
        std::reverse(begin(start_indices), end(start_indices));
        auto s = this->internal_chunk_size();
        m_var.read_chunk(start_indices, this->internal_chunk_size());
      }
    }

    size_t const plain_internal_index =
        this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                             indices...);
    return (*chunk_at(plain_index))[plain_internal_index];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto const... indices) const -> T const& {
    assert(sizeof...(indices) == this->num_dimensions());
    assert(this->in_range(indices...));
    size_t const plain_index =
        this->plain_chunk_index_from_global_indices(indices...);
    if (this->chunk_at_is_null(plain_index)) {
      static const T t{};
      return t;
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
  auto operator()(std::array<Int, N> const& is) -> T&{
    return at(std::make_index_sequence<N>{}, is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int, size_t N>
  auto operator()(std::array<Int, N> const& is) const -> T const&{
    return at(std::make_index_sequence<N>{}, is);
  }
};
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
#endif
