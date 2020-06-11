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
template <>
auto to_nc_type<int>() {return netCDF::ncInt;}
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
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
