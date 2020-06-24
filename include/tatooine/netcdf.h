#ifndef TATOOINE_NETCDF_H
#define TATOOINE_NETCDF_H
//==============================================================================
#include <tatooine/multidim_array.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/multidim.h>

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
  netCDF::NcVar                           m_variable;
  //============================================================================
 public:
  variable(std::shared_ptr<netCDF::NcFile>& file, netCDF::NcVar&& var)
      : m_file{file}, m_variable{std::move(var)} {}
  //============================================================================
  auto write(T const* const arr) { m_variable.putVar(arr); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& arr) { m_variable.putVar(arr.data()); }
  //----------------------------------------------------------------------------
  auto num_components() const {
    size_t c = 1;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      c *= dimension(i);
    }
    return c;
  }
  //----------------------------------------------------------------------------
  auto read() const {
    auto dims = dimensions();
    std::reverse(begin(dims), end(dims));
    dynamic_multidim_array<T, x_fastest> arr(dims);
    m_variable.getVar(arr.data_ptr());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read(dynamic_multidim_array<T, x_fastest>& arr) const {
    if (num_dimensions() != arr.num_dimensions()) {
      arr.resize(dimensions());
    } else {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        if (arr.size(i) != dimension(i)) {
          arr.resize(dimensions());
          break;
        }
      }
    }
    m_variable.getVar(arr.data_ptr());
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
    bool must_resize = arr.num_dimensions() != num_dimensions();
    if (!must_resize) {
     for (size_t i = 0; i < num_dimensions(); ++i) {
       must_resize = dimension(i) != arr.size(num_dimensions() - i - 1);
       if (must_resize) {break;}
     }
    }
    if (must_resize) {
      auto dims = dimensions();
      std::reverse(begin(dims), end(dims));
      arr.resize(dims); }

    for (auto const& chunk_indices : dynamic_multidim(arr.chunk_size())) {
      auto  start_indices =
          arr.global_indices_from_chunk_indices(chunk_indices);
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
        if (all_zero) {
          arr.destroy_chunk_at(plain_chunk_index);
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read(
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr) const {
    assert(sizeof...(Resolution) == num_dimensions());
    assert(std::vector{Resolution...} == dimensions());
    m_variable.getVar(arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& arr) const {
    if (auto const n = num_components(); size(arr) != n) { arr.resize(n); }
    m_variable.getVar(arr.data());
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T> arr(num_components());
    m_variable.getVar(arr.data());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_single(std::vector<size_t> const& start_indices) const {
    assert(size(start_indices) == num_dimensions());
    T t;
    m_variable.getVar(start_indices, std::vector<size_t>(num_dimensions(), 1),
                      &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_single(integral auto ... is) const {
    assert(num_dimensions() == sizeof...(is));
    T t;
    m_variable.getVar({static_cast<size_t>(is)...}, {((void)is, size_t(1))...},
                      &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t>  start_indices,
                  std::vector<size_t>  counts) const {
    assert(size(start_indices) == size(counts));
    assert(size(start_indices) == num_dimensions());

    dynamic_multidim_array<T> arr(counts);
    std::reverse(begin(start_indices), end(start_indices));
    std::reverse(begin(counts), end(counts));
    m_variable.getVar(start_indices, counts, arr.data_ptr());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  dynamic_multidim_array<T, x_fastest>& arr) const {
    if (num_dimensions() != arr.num_dimensions()) {
      arr.resize(counts);
    } else {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        if (arr.size(i) != dimension(i)) {
          arr.resize(counts);
          break;
        }
      }
    }
    m_variable.getVar(start_indices, counts, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr,
      integral auto const... start_indices) const {
    static_assert(sizeof...(start_indices) == sizeof...(Resolution));
    assert(sizeof...(Resolution) == num_dimensions());
    m_variable.getVar(std::vector{static_cast<size_t>(start_indices)...},
                      std::vector{Resolution...}, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      std::vector<size_t> const&                                 start_indices,
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr) const {
    m_variable.getVar(start_indices, std::vector{Resolution...},
                      arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  std::vector<T>&            arr) const {
    auto const n = std::accumulate(begin(counts), end(counts), size_t(1),
                                   std::multiplies<size_t>{});
    if (size(arr) != n) { arr.resize(n); }
    m_variable.getVar(start_indices, counts, arr.data());
  }
  //----------------------------------------------------------------------------
  auto is_null() const { return m_variable.isNull(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return static_cast<size_t>(m_variable.getDimCount()); }
  //----------------------------------------------------------------------------
  auto dimension(size_t i) const { return m_variable.getDim(i).getSize(); }
  //----------------------------------------------------------------------------
  auto dimension_name(size_t i) const { return m_variable.getDim(i).getName(); }
  //----------------------------------------------------------------------------
  auto dimensions() const {
    std::vector<size_t> res;
    res.reserve(num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      res.push_back(dimension(i));
    }
    return res;
  }
  //----------------------------------------------------------------------------
  auto name() const { return m_variable.getName(); }
};
//==============================================================================
class dimension {};
//==============================================================================
class file {
  mutable std::shared_ptr<netCDF::NcFile> m_file;
  //============================================================================
 public:
  template <typename... Ts>
  file(std::string const& path, Ts&&... ts)
      : m_file{new netCDF::NcFile{path, std::forward<Ts>(ts)...}} {}
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
    return netcdf::variable<T>{m_file, m_file->getVar(variable_name)};
  }
  //----------------------------------------------------------------------------
  auto add_dimension(std::string const& dimension_name, size_t const size) {
    return m_file->addDim(dimension_name, size);
  }
  //----------------------------------------------------------------------------
  auto attributes() const { return m_file->getAtts(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_file->getDimCount(); }
  //----------------------------------------------------------------------------
  auto dimensions() const { return m_file->getDims(); }
  //----------------------------------------------------------------------------
  auto groups() const { return m_file->getGroups(); }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variables() const {
    std::vector<netcdf::variable<T>> vars;
    for (auto& [name, var] : m_file->getVars()) {
      if (var.getType() == to_nc_type<T>()) {
        vars.push_back(netcdf::variable<T>{m_file, std::move(var)});
      }
    }
    return vars;
  }
};
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
#endif
