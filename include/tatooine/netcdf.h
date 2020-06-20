#ifndef TATOOINE_NETCDF_H
#define TATOOINE_NETCDF_H
//==============================================================================
#include <netcdf>
#include <memory>
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
  netCDF::NcVar                   m_variable;
  //============================================================================
 public:
  variable(std::shared_ptr<netCDF::NcFile>& file, netCDF::NcVar&& var)
      : m_file{file}, m_variable{std::move(var)} {}
  //============================================================================
  auto put(T const* const data) { m_variable.putVar(data); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto put(std::vector<T> const& data){
    m_variable.putVar(data.data());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t>& counts) const {
    auto n = std::accumulate(counts, std::multiplies<size_t>{}, size_t(1));
    std::vector<T> data();

    m_variable.getVar(start_indices, counts, data);
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t>& counts, std::vector<T>& data) const {
    m_variable.getVar(start_indices, counts, data);
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t>& counts, T* const data) const {
    m_variable.getVar(start_indices, counts, data);
  }
  //----------------------------------------------------------------------------
  auto to_vector() const {
    size_t size = 1;
    for (auto const& dim : dimensions()) { size *= dim.getSize(); }
    std::vector<T> data(size);
    m_variable.getVar(data.data());
    return data;
  }
  //----------------------------------------------------------------------------
  auto is_null() const { return m_variable.isNull(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_variable.getDimCount(); }
  //----------------------------------------------------------------------------
  auto dimension(int i) const { return m_variable.getDim(i); }
  //----------------------------------------------------------------------------
  auto dimensions() const { return m_variable.getDims(); }
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
    return variable<T>{
        m_file, m_file->addVar(variable_name, to_nc_type<T>(), dims)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto get_variable(std::string const& variable_name) const {
    return variable<T>{m_file, m_file->getVar(variable_name)};
  }
  //----------------------------------------------------------------------------
  auto add_dimension(std::string const& dimension_name, size_t const size) {
    return m_file->addDim(dimension_name, size);
  }
  //----------------------------------------------------------------------------
  auto attributes() const {
    return m_file->getAtts();
  }
  //----------------------------------------------------------------------------
  auto dimensions() const {
    return m_file->getDims();
  }
  //----------------------------------------------------------------------------
  auto groups() const {
    return m_file->getGroups();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variables() const {
    std::vector<variable<T>> vars;
    for (auto& [name, var] : m_file->getVars()) {
      if (var.getType() == to_nc_type<T>()) {
        vars.push_back(variable<T>{m_file, std::move(var)});
      }
    }
    return vars;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variable(std::string const& name) const {
    return variable<T>{m_file, m_file->getVar(name)};
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
