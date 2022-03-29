#ifdef TATOOINE_NETCDF_AVAILABLE
#ifndef TATOOINE_NETCDF_H
#define TATOOINE_NETCDF_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <cassert>
#include <tatooine/filesystem.h>
#include <memory>
#include <mutex>
#include <netcdf>
#include <numeric>
#include <vector>
//==============================================================================
namespace tatooine::netcdf {
//==============================================================================
template <typename T>
struct nc_type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct nc_type<int> {
  static auto value() { return netCDF::ncInt; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct nc_type<float> {
  static auto value() { return netCDF::ncFloat; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct nc_type<double> {
  static auto value() { return netCDF::ncDouble; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
struct nc_type<mat<T, M, N>> {
  static auto value() { return nc_type<T>::value(); }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
struct nc_type<vec<T, N>> {
  static auto value() { return nc_type<T>::value(); }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t... Dims>
struct nc_type<tensor<T, Dims...>> {
  static auto value() { return nc_type<T>::value(); }
};
//==============================================================================
class group {};
//==============================================================================
class attribute {};
//==============================================================================
template <typename T>
class variable {
 public:
  using this_type     = variable<T>;
  using value_type = T;

 private:
  mutable std::shared_ptr<netCDF::NcFile> m_file;
  mutable std::shared_ptr<std::mutex>     m_mutex;
  netCDF::NcVar                           m_var;
  //============================================================================
 public:
  variable(std::shared_ptr<netCDF::NcFile>& file,
           std::shared_ptr<std::mutex>& mutex, netCDF::NcVar const& var)
      : m_file{file}, m_mutex{mutex}, m_var{var} {}
  //----------------------------------------------------------------------------
  variable(variable const&)     = default;
  variable(variable&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(variable const&) -> variable& = default;
  auto operator=(variable&&) noexcept -> variable& = default;
  //============================================================================
  auto write(std::vector<size_t> const& is, T const& t) {
    std::lock_guard lock{*m_mutex};
    // std::reverse(begin(is), end(is));
    return m_var.putVar(is, t);
  }
  auto write(std::vector<size_t> const& is, std::vector<size_t> const& count,
             T const* const arr) {
    std::lock_guard lock{*m_mutex};
    // std::reverse(begin(is), end(is));
    // std::reverse(begin(count), end(count));
    return m_var.putVar(is, count, arr);
  }
  auto write(T const* const arr) {
    std::lock_guard lock{*m_mutex};
    return m_var.putVar(arr);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& arr) {
    std::lock_guard lock{*m_mutex};
    return m_var.putVar(arr.data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(range auto const r) { return write(std::vector(begin(r), end(r))); }
  //----------------------------------------------------------------------------
  auto num_components() const {
    size_t c = 1;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      c *= size(i);
    }
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
        if (arr.size(i) != size(i)) {
          break;
        }
      }
    }
    if (must_resize) {
      auto s = size();
      //std::reverse(begin(s), end(s));
      arr.resize(s);
    }

    std::lock_guard lock{*m_mutex};
    m_var.getVar(arr.data());
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
        must_resize = size(i) != arr.size(num_dimensions() - i - 1);
        if (must_resize) {
          break;
        }
      }
    }
    if (must_resize) {
      auto s = size();
      //std::reverse(begin(s), end(s));
      arr.resize(s);
    }

    for (auto const& chunk_indices : dynamic_multidim(arr.chunk_size())) {
      auto start_indices = arr.global_indices_from_chunk_indices(chunk_indices);
      auto const plain_chunk_index =
          arr.plain_chunk_index_from_chunk_indices(chunk_indices);

      if (arr.chunk_at_is_null(plain_chunk_index)) {
        arr.create_chunk_at(plain_chunk_index);
      }

      //std::reverse(begin(start_indices), end(start_indices));
      auto s = arr.internal_chunk_size();
      //std::reverse(begin(s), end(s));
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
    assert(std::vector{Resolution...} == size());
    std::lock_guard lock{*m_mutex};
    m_var.getVar(arr.data());
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& arr) const {
    if (auto const n = num_components(); arr.size() != n) {
      arr.resize(n);
    }
    std::lock_guard lock{*m_mutex};
    m_var.getVar(arr.data());
  }
  //----------------------------------------------------------------------------
  auto read(T* const ptr) const {
    std::lock_guard lock{*m_mutex};
    m_var.getVar(ptr);
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T>  arr(num_components());
    std::lock_guard lock{*m_mutex};
    m_var.getVar(arr.data());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_single(std::vector<size_t> const& start_indices) const {
    assert(size(start_indices) == num_dimensions());
    T               t;
    std::lock_guard lock{*m_mutex};
    m_var.getVar(start_indices, std::vector<size_t>(num_dimensions(), 1), &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_single(integral auto const... is) const {
    assert(num_dimensions() == sizeof...(is));
    T               t;
    std::lock_guard lock{*m_mutex};
    m_var.getVar({static_cast<size_t>(is)...}, {((void)is, size_t(1))...}, &t);
    return t;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> start_indices,
                  std::vector<size_t> counts) const {
    assert(start_indices.size() == counts.size());
    assert(start_indices.size() == num_dimensions());

    dynamic_multidim_array<T> arr(counts);
    //std::reverse(begin(start_indices), end(start_indices));
    //std::reverse(begin(counts), end(counts));
    std::lock_guard lock{*m_mutex};
    m_var.getVar(start_indices, counts, arr.data());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> start_indices, std::vector<size_t> counts,
                  T* ptr) const {
    assert(start_indices.size() == counts.size());
    assert(start_indices.size() == num_dimensions());

    //std::reverse(begin(start_indices), end(start_indices));
    //std::reverse(begin(counts), end(counts));
    std::lock_guard lock{*m_mutex};
    m_var.getVar(start_indices, counts, ptr);
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
    m_var.getVar(start_indices, counts, arr.data());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr,
      integral auto const... start_indices) const {
    static_assert(sizeof...(start_indices) == sizeof...(Resolution));
    assert(sizeof...(Resolution) == num_dimensions());
    std::lock_guard lock{*m_mutex};
    m_var.getVar(std::vector{static_cast<size_t>(start_indices)...},
                 std::vector{Resolution...}, arr.data());
  }
  //----------------------------------------------------------------------------
  template <typename MemLoc, size_t... Resolution>
  auto read_chunk(
      std::vector<size_t> const&                                  start_indices,
      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr) const {
    std::lock_guard lock{*m_mutex};
    m_var.getVar(start_indices, std::vector{Resolution...}, arr.data());
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> const& start_indices,
                  std::vector<size_t> const& counts,
                  std::vector<T>&            arr) const {
    auto const n = std::accumulate(begin(counts), end(counts), size_t(1),
                                   std::multiplies<size_t>{});
    if (size(arr) != n) {
      arr.resize(n);
    }
    std::lock_guard lock{*m_mutex};
    m_var.getVar(start_indices, counts, arr.data());
  }
  //----------------------------------------------------------------------------
  auto is_null() const {
    std::lock_guard lock{*m_mutex};
    return m_var.isNull();
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const {
    std::lock_guard lock{*m_mutex};
    return static_cast<size_t>(m_var.getDimCount());
  }
  //----------------------------------------------------------------------------
  auto size(size_t i) const {
    std::lock_guard lock{*m_mutex};
    return m_var.getDim(i).getSize();
  }
  //----------------------------------------------------------------------------
  auto dimension_name(size_t i) const {
    std::lock_guard lock{*m_mutex};
    return m_var.getDim(i).getName();
  }
  //----------------------------------------------------------------------------
  auto size() const {
    std::vector<size_t> res;
    res.reserve(num_dimensions());
    for (size_t i = 0; i < num_dimensions(); ++i) {
      res.push_back(size(i));
    }
    return res;
  }
  //----------------------------------------------------------------------------
  auto name() const {
    std::lock_guard lock{*m_mutex};
    return m_var.getName();
  }
};
//==============================================================================
class size {};
//==============================================================================
class file {
  mutable std::shared_ptr<netCDF::NcFile> m_file;
  mutable std::shared_ptr<std::mutex>     m_mutex;
  //============================================================================
 public:
  template <typename... Ts>
  file(filesystem::path const& path, Ts&&... ts)
      : m_file{new netCDF::NcFile{path.string(), std::forward<Ts>(ts)...}},
        m_mutex{std::make_shared<std::mutex>()} {}
  //============================================================================
  template <typename T>
  auto add_variable(std::string const&   variable_name,
                    netCDF::NcDim const& dim) {
    return netcdf::variable<T>{
        m_file, m_mutex,
        m_file->addVar(variable_name, nc_type<T>::value(), dim)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_variable(std::string const&                variable_name,
                    std::vector<netCDF::NcDim> const& dims) {
    return netcdf::variable<T>{
        m_file, m_mutex,
        m_file->addVar(variable_name, nc_type<T>::value(), dims)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto variable(std::string const& variable_name) const {
    return netcdf::variable<T>{m_file, m_mutex, m_file->getVar(variable_name)};
  }
  //----------------------------------------------------------------------------
  auto add_dimension(std::string const& dimension_name) {
    return m_file->addDim(dimension_name);
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
      if (var.getType() == nc_type<T>::value()) {
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
#endif
