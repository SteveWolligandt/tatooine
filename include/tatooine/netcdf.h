#ifndef TATOOINE_NETCDF_H
#define TATOOINE_NETCDF_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <cassert>
#include <memory>
#include <netcdf>
#include <numeric>
#include <vector>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
template <typename T>
auto to_nc_type();
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
auto to_nc_type<int>() {
  return netCDF::ncInt;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
auto to_nc_type<float>() {
  return netCDF::ncFloat;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
auto to_nc_type<double>() {
  return netCDF::ncDouble;
}
//==============================================================================
class group {};
//==============================================================================
class attribute {};
//==============================================================================
template <typename T>
class variable {
  mutable std::shared_ptr<netCDF::NcFile> m_file;
  mutable std::shared_ptr<std::mutex>     m_mutex;
  std::string                             m_variable_name;
  //============================================================================
 public:
  variable(std::shared_ptr<netCDF::NcFile>& file,
           std::shared_ptr<std::mutex>& mutex, std::string const& name)
      : m_file{file}, m_mutex{mutex}, m_variable_name{name} {}
  variable(variable const& other)     = default;
  variable(variable&& other) noexcept = default;
  //============================================================================
  auto write(T const* const arr) {
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).putVar(arr);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& arr) {
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).putVar(arr.data());
  }
  //----------------------------------------------------------------------------
  auto num_components() const {
    size_t c = 1;
    for (size_t i = 0; i < num_dimensions(); ++i) { c *= size(i); }
    return c;
  }
  //----------------------------------------------------------------------------
  auto read() const {
    dynamic_multidim_array<T, x_fastest> arr;
    read(arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read(dynamic_multidim_array<T, x_fastest>& arr) const {
    auto const n           = num_dimensions();
    bool       must_resize = n != arr.num_dimensions();
    if (!must_resize) {
      for (size_t i = 0; i < n; ++i) {
        if (arr.size(i) != size(i)) { break; }
      }
    }
    if (must_resize) {
      auto s = size();
      std::reverse(begin(s), end(s));
      arr.resize(s);
    }

    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).getVar(arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_chunked(size_t const chunk_size = 10) const {
    chunked_multidim_array<T, x_fastest> arr{
        std::vector<size_t>(num_dimensions(), 0),
        std::vector<size_t>(num_dimensions(), chunk_size)};
    read(arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunked(std::vector<size_t> const& chunk_size) const {
    chunked_multidim_array<T, x_fastest> arr{
        std::vector<size_t>(num_dimensions(), 0), chunk_size};
    read(arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read(chunked_multidim_array<T, x_fastest>& arr) const {
    bool            must_resize = arr.num_dimensions() != num_dimensions();
    if (!must_resize) {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        must_resize = size(i) != arr.size(num_dimensions() - i - 1);
        if (must_resize) { break; }
      }
    }
    if (must_resize) {
      auto s = size();
      std::reverse(begin(s), end(s));
      arr.resize(s);
    }

    for (auto const& chunk_indices : dynamic_multidim(arr.chunk_size())) {
      auto start_indices = arr.global_indices_from_chunk_indices(chunk_indices);
      auto const plain_chunk_index =
          arr.plain_chunk_index_from_chunk_indices(chunk_indices);

      if (arr.chunk_at_is_null(plain_chunk_index)) {
        arr.create_chunk_at(plain_chunk_index);
      }

      std::reverse(begin(start_indices), end(start_indices));
      auto s = arr.internal_chunk_size();
      std::reverse(begin(s), end(s));
      read_chunk(start_indices, s, *arr.chunk_at(plain_chunk_index));
      if constexpr (std::is_arithmetic_v<T>) {
        bool all_zero = true;
        for (auto const& v : arr.chunk_at(plain_chunk_index)->data()) {
          if (v != 0) {
            all_zero = false;
            break;
          }
        }
        if (all_zero) { arr.destroy_chunk_at(plain_chunk_index); }
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read(
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr) const {
    assert(sizeof...(Resolution) == num_dimensions());
    assert(std::vector{Resolution...} == size());
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).getVar(arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& arr) const {
    if (auto const n = num_components(); size(arr) != n) { arr.resize(n); }
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).getVar(arr.data());
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T>  arr(num_components());
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).getVar(arr.data());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_single(std::vector<size_t> const& start_indices) const {
    assert(size(start_indices) == num_dimensions());
    T t;
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar(start_indices, std::vector<size_t>(num_dimensions(), 1), &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_single(integral auto... is) const {
    assert(num_dimensions() == sizeof...(is));
    T t;
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar({static_cast<size_t>(is)...}, {((void)is, size_t(1))...}, &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> start_indices,
                  std::vector<size_t> counts) const {
    assert(start_indices.size() == counts.size());
    assert(start_indices.size() == num_dimensions());

    dynamic_multidim_array<T> arr(counts);
    std::reverse(begin(start_indices), end(start_indices));
    std::reverse(begin(counts), end(counts));
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar(start_indices, counts, arr.data_ptr());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const&            start_indices,
                  std::vector<size_t> const&            counts,
                  dynamic_multidim_array<T, x_fastest>& arr) const {
    if (num_dimensions() != arr.num_dimensions()) {
      arr.resize(counts);
    } else {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        if (arr.size(i) != size(i)) {
          arr.resize(counts);
          break;
        }
      }
    }
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar(start_indices, counts, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr,
      integral auto const... start_indices) const {
    static_assert(sizeof...(start_indices) == sizeof...(Resolution));
    assert(sizeof...(Resolution) == num_dimensions());
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar(std::vector{static_cast<size_t>(start_indices)...},
                std::vector{Resolution...}, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      std::vector<size_t> const&                                  start_indices,
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr) const {
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name)
        .getVar(start_indices, std::vector{Resolution...}, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  std::vector<T>&            arr) const {
    auto const      n = std::accumulate(begin(counts), end(counts), size_t(1),
                                   std::multiplies<size_t>{});
    if (size(arr) != n) { arr.resize(n); }
    std::lock_guard lock{*m_mutex};
    m_file->getVar(m_variable_name).getVar(start_indices, counts, arr.data());
  }
  //----------------------------------------------------------------------------
  auto is_null() const {
    std::lock_guard lock{*m_mutex};
    return m_file->getVar(m_variable_name).isNull();
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const {
    std::lock_guard lock{*m_mutex};
    return static_cast<size_t>(m_file->getVar(m_variable_name).getDimCount());
  }
  //----------------------------------------------------------------------------
  auto size(size_t i) const {
    std::lock_guard lock{*m_mutex};
    return m_file->getVar(m_variable_name).getDim(i).getSize();
  }
  //----------------------------------------------------------------------------
  auto dimension_name(size_t i) const {
    std::lock_guard lock{*m_mutex};
    return m_file->getVar(m_variable_name).getDim(i).getName();
  }
  //----------------------------------------------------------------------------
  auto size() const {
    std::vector<size_t> res;
    res.reserve(num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) { res.push_back(size(i)); }
    return res;
  }
  //----------------------------------------------------------------------------
  auto name() const {
    std::lock_guard lock{*m_mutex};
    return m_file->getVar(m_variable_name).getName();
  }
};
//==============================================================================
class size {};
//==============================================================================
class file {
  mutable std::shared_ptr<netCDF::NcFile> m_file;
  mutable std::shared_ptr<std::mutex> m_mutex;
  //============================================================================
 public:
  template <typename... Ts>
  file(std::string const& path, Ts&&... ts)
      : m_file{new netCDF::NcFile{path, std::forward<Ts>(ts)...}},
        m_mutex{std::make_shared<std::mutex>()} {}
  //============================================================================
  template <typename T>
  auto add_variable(std::string const&                variable_name,
                    std::vector<netCDF::NcDim> const& dims) {
    return netcdf::variable<T>{
        m_file, m_file->addVar(variable_name, to_nc_type<T>(), dims)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variable(std::string const& variable_name) const {
    return netcdf::variable<T>{m_file, m_mutex, variable_name};
  }
  //----------------------------------------------------------------------------
  auto add_dimension(std::string const& dimension_name, size_t const size) {
    return m_file->addDim(dimension_name, size);
  }
  //----------------------------------------------------------------------------
  auto dimensions() const { return m_file->getDims(); }
  //----------------------------------------------------------------------------
  auto attributes() const { return m_file->getAtts(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_file->getDimCount(); }
  //----------------------------------------------------------------------------
  auto size() const { return m_file->getDims(); }
  //----------------------------------------------------------------------------
  auto groups() const { return m_file->getGroups(); }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variables() const {
    std::vector<netcdf::variable<T>> vars;
    for (auto& [name, var] : m_file->getVars()) {
      if (var.getType() == to_nc_type<T>()) {
        vars.push_back(netcdf::variable<T>{m_file, m_mutex, std::move(var)});
      }
    }
    return vars;
  }
};
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
#endif
