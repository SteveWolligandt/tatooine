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
  using this_t     = lazy_reader<T>;
  using parent_t   = chunked_multidim_array<T>;
  using value_type = T;
  using parent_t::chunk_at;

  static auto default_value() -> T& {
    static T t{};
    return t;
  }

 private:
  netcdf::variable<T>                              m_var;
  mutable std::vector<bool>                        m_read;
  mutable std::vector<std::unique_ptr<std::mutex>> m_mutexes;

 public:
  lazy_reader(std::filesystem::path const& path, std::string const& var_name,
              std::vector<size_t> chunk_size)
      : chunked_multidim_array<T>{std::vector<size_t>(chunk_size.size(), 0),
                                  chunk_size},
        m_var{file{path, netCDF::NcFile::read}.variable<T>(var_name)} {
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
      m_mutexes.resize(this->num_chunks());
      for (auto& mutex : m_mutexes) {
        mutex = std::make_unique<std::mutex>();
      }
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
      m_mutexes.resize(this->num_chunks());
      for (auto& mutex : m_mutexes) {
        mutex = std::make_unique<std::mutex>();
      }
    }
  }
  //----------------------------------------------------------------------------
  auto read_chunk(size_t& plain_index, integral auto const... indices) const
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

    if constexpr (std::is_arithmetic_v<T>) {
      if (this->chunk_at_is_null(plain_index)) {
        if (!m_read[plain_index]) {
          m_read[plain_index] = true;
          this->create_chunk_at(plain_index);
          auto start_indices = this->global_indices_from_chunk_indices(
              this->chunk_indices_from_global_indices(indices...));
          auto s = this->internal_chunk_size();
          std::reverse(begin(start_indices), end(start_indices));
          m_var.read_chunk(start_indices, s, *chunk_at(plain_index));

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
          std::reverse(begin(start_indices), end(start_indices));
        m_var.read_chunk(start_indices, this->internal_chunk_size());
      }
    }
    return this->chunk_at(plain_index);
  }
  //----------------------------------------------------------------------------
 public:
   auto at(integral auto const... indices) -> T& {
    size_t      plain_index = 0;
    auto & chunk       = read_chunk(plain_index, indices...);

    if (chunk != nullptr) {
      size_t const plain_internal_index =
          this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                               indices...);
      return (*chunk)[plain_internal_index];
    } else {
      auto& t = default_value();
      t       = T{};
      return t;
    }
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... indices) const -> T const& {
    size_t      plain_index = 0;
    auto const& chunk       = read_chunk(plain_index, indices...);

    if (chunk != nullptr) {
      size_t const plain_internal_index =
          this->plain_internal_chunk_index_from_global_indices(plain_index,
                                                               indices...);
      return (*chunk)[plain_internal_index];
    } else {
      auto& t = default_value();
      t       = T{};
      return t;
    }
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
  template <typename _T = value_type, enable_if_arithmetic<_T> = true>
  auto is_chunk_filled_with_zeros(size_t const plain_index) const -> bool {
    return is_chunk_filled_with_value(plain_index, 0);
  }
};
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
#endif
