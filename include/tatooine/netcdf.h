#ifndef TATOOINE_NETCDF_H
#define TATOOINE_NETCDF_H
//==============================================================================
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
  netCDF::NcVar                           m_variable;
  //============================================================================
 public:
  variable(std::shared_ptr<netCDF::NcFile>& file, netCDF::NcVar&& var)
      : m_file{file}, m_variable{std::move(var)} {}
  //============================================================================
  auto write(T const* const data) { m_variable.putVar(data); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data) { m_variable.putVar(data.data()); }
  //----------------------------------------------------------------------------
  auto read() const {
    dynamic_multidim_array<T> data(dimensions());
    m_variable.getVar(data.data_ptr());
    return data;
  }
  //----------------------------------------------------------------------------
  auto read(dynamic_multidim_array<T>& data) const {
    if (num_dimensions() != data.num_dimensions()) {
      data.resize(dimensions());
    } else {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        if (data.size(i) != dimension(i)) {
          data.resize(dimensions());
          break;
        }
      }
    }
    m_variable.getVar(data.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename Indexing, typename MemLoc, size_t... Resolution>
  auto read(
      static_multidim_array<T, Indexing, MemLoc, Resolution...>& data) const {
    assert(sizeof...(Resolution) == num_dimensions());
    assert(std::vector{Resolution...} == dimensions());
    m_variable.getVar(data.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& data) const {
    if (auto const n = num_components(); size(data) != n) { data.resize(n); }
    m_variable.getVar(data.data());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts) const {
    assert(size(start_indices) == size(counts));
    assert(size(start_indices) == num_dimensions());

    dynamic_multidim_array<T> data(counts);
    m_variable.getVar(start_indices, counts, data.data_ptr());
    return data;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  dynamic_multidim_array<T>& data) const {
    if (num_dimensions() != data.num_dimensions()) {
      data.resize(counts);
    } else {
      for (size_t i = 0; i < num_dimensions(); ++i) {
        if (data.size(i) != dimension(i).getSize()) {
          data.resize(counts);
          break;
        }
      }
    }
    m_variable.getVar(start_indices, counts, data.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename Indexing, typename MemLoc, size_t... Resolution>
  auto read_chunk(
      static_multidim_array<T, Indexing, MemLoc, Resolution...>& data,
      integral auto const... start_indices) const {
    static_assert(sizeof...(start_indices) == sizeof...(Resolution));
    assert(sizeof...(Resolution) == num_dimensions());
    m_variable.getVar(std::vector{static_cast<size_t>(start_indices)...},
                      std::vector{Resolution...}, data.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename Indexing, typename MemLoc, size_t... Resolution>
  auto read_chunk(
      std::vector<size_t> const&                                 start_indices,
      static_multidim_array<T, Indexing, MemLoc, Resolution...>& data) const {
    m_variable.getVar(start_indices, std::vector{Resolution...},
                      data.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  std::vector<T>&            data) const {
    auto const n = std::accumulate(begin(counts), end(counts), size_t(1),
                                   std::multiplies<size_t>{});
    if (size(data) != n) { data.resize(n); }
    m_variable.getVar(start_indices, counts, data.data());
  }
  //----------------------------------------------------------------------------
  auto is_null() const { return m_variable.isNull(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_variable.getDimCount(); }
  //----------------------------------------------------------------------------
  auto dimension(size_t i) const { return m_variable.getDim(i).getSize(); }
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
  auto num_components() const {
    auto const dims = dimensions();
    return std::accumulate(begin(dims), end(dims), size_t(1),
                           std::multiplies<size_t>{});
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
