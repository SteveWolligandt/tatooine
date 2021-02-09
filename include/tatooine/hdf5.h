#ifdef TATOOINE_HAS_HDF5_SUPPORT
#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H
//==============================================================================
#include <H5Cpp.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/concepts.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <cassert>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>
//==============================================================================
namespace tatooine::hdf5 {
//==============================================================================
template <typename T>
struct h5_type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct h5_type<int> {
  static auto value() { return H5::PredType::NATIVE_INT; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct h5_type<float> {
  static auto value() { return H5::PredType::NATIVE_FLOAT; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct h5_type<double> {
  static auto value() { return H5::PredType::NATIVE_DOUBLE; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
struct h5_type<mat<T, M, N>> {
  static auto value() { return h5_type<T>::value(); }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
struct h5_type<vec<T, N>> {
  static auto value() { return h5_type<T>::value(); }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t... Dims>
struct h5_type<tensor<T, Dims...>> {
  static auto value() { return h5_type<T>::value(); }
};
//==============================================================================
template <typename T>
class dataset {
 public:
  using this_t     = dataset<T>;
  using value_type = T;

 private:
  mutable std::shared_ptr<H5::H5File> m_file;
  mutable std::shared_ptr<std::mutex> m_mutex;
  H5::DataSet                         m_dataset;
  std::string                         m_name;
  //============================================================================
 public:
  dataset(std::shared_ptr<H5::H5File>& file, std::shared_ptr<std::mutex>& mutex,
          H5::DataSet const& var, std::string const& name)
      : m_file{file}, m_mutex{mutex}, m_dataset{var}, m_name{name} {}
  //----------------------------------------------------------------------------
  dataset(dataset const&)     = default;
  dataset(dataset&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dataset const&) -> dataset& = default;
  auto operator=(dataset&&) noexcept -> dataset& = default;
  //============================================================================
  //auto write(std::vector<size_t> const& is, T const& t) {
  //  std::lock_guard lock{*m_mutex};
  //  m_dataset.putVar(is, t);
  //}
  //auto write(std::vector<size_t> const& is, std::vector<size_t> const& count,
  //           T const* const arr) {
  //  std::lock_guard lock{*m_mutex};
  //  // std::reverse(begin(is), end(is));
  //  // std::reverse(begin(count), end(count));
  //  return m_dataset.putVar(is, count, arr);
  //}
  //auto write(T const* const arr) {
  //  std::lock_guard lock{*m_mutex};
  //  return m_dataset.putVar(arr);
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(T const* data) {
    std::lock_guard lock{*m_mutex};
    return m_dataset.write(data);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data) {
    std::lock_guard lock{*m_mutex};
    return m_dataset.write(data.data(), h5_type<T>::value());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  auto write(std::array<T, N> const& data) {
    std::lock_guard lock{*m_mutex};
    return m_dataset.write(data.data(), h5_type<T>::value());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Range>
#else
  template <typename Range, enable_if<is_range<Range>> = true>
#endif
  auto write(Range r) {
    write(std::vector(begin(r), end(r)));
  }
  //----------------------------------------------------------------------------
  auto read() const {
    dynamic_multidim_array<T, x_fastest> arr;
    read(arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read(dynamic_multidim_array<T, x_fastest>& arr) const {
    auto const ds          = data_space();
    auto const num_dims    = ds.getSimpleExtentNdims();
    auto       size        = std::make_unique<hsize_t[]>(num_dims);
    ds.getSimpleExtentDims(size.get());
    bool       must_resize = num_dims != arr.num_dimensions();
    if (!must_resize) {
      for (size_t i = 0; i < num_dims; ++i) {
        if (arr.size(i) != size[i]) {
          break;
        }
      }
    }
    if (must_resize) {
      // std::reverse(begin(s), end(s));
      arr.resize(std::vector<size_t>(size.get(), size.get() + num_dims));
    }

    std::lock_guard lock{*m_mutex};
    m_dataset.read(arr.data_ptr(), h5_type<T>::value());
  }
  //----------------------------------------------------------------------------
  //  auto read_chunked(size_t const chunk_size = 10) const {
  //    chunked_multidim_array<T, x_fastest> arr{
  //        std::vector<size_t>(num_dimensions(), 0),
  //        std::vector<size_t>(num_dimensions(), chunk_size)};
  //    read(arr);
  //    return arr;
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read_chunked(std::vector<size_t> const& chunk_size) const {
  //    chunked_multidim_array<T, x_fastest> arr{
  //        std::vector<size_t>(num_dimensions(), 0), chunk_size};
  //    read(arr);
  //    return arr;
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read(chunked_multidim_array<T, x_fastest>& arr) const {
  //    bool must_resize = arr.num_dimensions() != num_dimensions();
  //    if (!must_resize) {
  //      for (size_t i = 0; i < num_dimensions(); ++i) {
  //        must_resize = size(i) != arr.size(num_dimensions() - i - 1);
  //        if (must_resize) {
  //          break;
  //        }
  //      }
  //    }
  //    if (must_resize) {
  //      auto s = size();
  //      // std::reverse(begin(s), end(s));
  //      arr.resize(s);
  //    }
  //
  //    for (auto const& chunk_indices : dynamic_multidim(arr.chunk_size())) {
  //      auto start_indices =
  //      arr.global_indices_from_chunk_indices(chunk_indices); auto const
  //      plain_chunk_index =
  //          arr.plain_chunk_index_from_chunk_indices(chunk_indices);
  //
  //      if (arr.chunk_at_is_null(plain_chunk_index)) {
  //        arr.create_chunk_at(plain_chunk_index);
  //      }
  //
  //      // std::reverse(begin(start_indices), end(start_indices));
  //      auto s = arr.internal_chunk_size();
  //      // std::reverse(begin(s), end(s));
  //      read_chunk(start_indices, s, *arr.chunk_at(plain_chunk_index));
  //      if constexpr (std::is_arithmetic_v<T>) {
  //        bool all_zero = true;
  //        for (auto const& v : arr.chunk_at(plain_chunk_index)->data()) {
  //          if (v != 0) {
  //            all_zero = false;
  //            break;
  //          }
  //        }
  //        if (all_zero) {
  //          arr.destroy_chunk_at(plain_chunk_index);
  //        }
  //      }
  //    }
  //  }
  //  //----------------------------------------------------------------------------
  //  template <typename MemLoc, size_t... Resolution>
  //  auto read(
  //      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr)
  //      const
  //      {
  //    assert(sizeof...(Resolution) == num_dimensions());
  //    assert(std::vector{Resolution...} == size());
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(arr.data_ptr());
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read(std::vector<T>& arr) const {
  //    if (auto const n = num_components(); arr.size() != n) {
  //      arr.resize(n);
  //    }
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(arr.data());
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read(T* const ptr) const {
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(ptr);
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read_as_vector() const {
  //    std::vector<T>  arr(num_components());
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(arr.data());
  //    return arr;
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read_single(std::vector<size_t> const& start_indices) const {
  //    assert(size(start_indices) == num_dimensions());
  //    T               t;
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(start_indices, std::vector<size_t>(num_dimensions(),
  //    1), &t); return t;
  //  }
  //  //----------------------------------------------------------------------------
  //#ifdef __cpp_concepts
  //  template <integral... Is>
  //#else
  //  template <typename... Is, enable_if<is_integral<Is...>> = true>
  //#endif
  //  auto read_single(Is const... is) const {
  //    assert(num_dimensions() == sizeof...(is));
  //    T               t;
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar({static_cast<size_t>(is)...}, {((void)is,
  //    size_t(1))...}, &t); return t;
  //  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> start_indices, std::vector<size_t> counts,
                  dynamic_multidim_array<T, x_fastest>& arr) const {
    read_chunk(std::vector<hsize_t>(begin(start_indices), end(start_indices)),
               std::vector<hsize_t>(begin(counts), end(counts)), arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<size_t> start_indices,
                  std::vector<size_t> counts) const {
    return read_chunk(
        std::vector<hsize_t>(begin(start_indices), end(start_indices)),
        std::vector<hsize_t>(begin(counts), end(counts)));
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<hsize_t> start_indices,
                  std::vector<hsize_t> counts) const {
    dynamic_multidim_array<T, x_fastest> arr;
    read_chunk(start_indices, counts, arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read_chunk(std::vector<hsize_t> start_indices,
                  std::vector<hsize_t> counts,
                  dynamic_multidim_array<T, x_fastest>& arr) const {
    assert(start_indices.size() == counts.size());

    auto const ds       = data_space();
    auto const num_dims = ds.getSimpleExtentNdims();
    auto size =  std::make_unique<hsize_t[]>(num_dims);
    data_space().getSimpleExtentDims(size.get());

    // Write a subset of data to the dataset, then read the
    // entire dataset back from the file.

    assert(start_indices.size() == num_dims);
    if (num_dims != static_cast<int>(arr.num_dimensions())) {
      arr.resize(counts);
    } else {
      for (int i = 0; i < num_dims; ++i) {
        if (arr.size(i) != size[i]) {
          arr.resize(counts);
          break;
        }
      }
    }

    auto stride = std::make_unique<hsize_t[]>(num_dims);
    auto block  = std::make_unique<hsize_t[]>(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      stride[i] = block[i] = 1;
    }

    ds.selectHyperslab(H5S_SELECT_SET, counts.data(), start_indices.data(),
                       stride.get(), block.get());
    H5::DataSpace mem_space{num_dims, counts.data()};

    {
      std::lock_guard lock{*m_mutex};
      m_dataset.read(arr.data_ptr(), h5_type<T>::value(), mem_space, ds);
    }
    return arr;
  }
  //----------------------------------------------------------------------------
  //  auto read_chunk(std::vector<size_t> start_indices, std::vector<size_t>
  //  counts,
  //                  T* ptr) const {
  //    assert(start_indices.size() == counts.size());
  //    assert(start_indices.size() == num_dimensions());
  //
  //    // std::reverse(begin(start_indices), end(start_indices));
  //    // std::reverse(begin(counts), end(counts));
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(start_indices, counts, ptr);
  //  }
  //----------------------------------------------------------------------------
  //#ifdef __cpp_concepts
  //  template <typename MemLoc, size_t... Resolution, integral...
  //  StartIndices>
  //#else
  //  template <typename MemLoc, size_t... Resolution, typename...
  //  StartIndices,
  //            enable_if<is_integral<StartIndices...>> = true>
  //#endif
  //  auto read_chunk(
  //      static_multidim_array<T, x_fastest, MemLoc, Resolution...>& arr,
  //      StartIndices const... start_indices) const {
  //    static_assert(sizeof...(start_indices) == sizeof...(Resolution));
  //    assert(sizeof...(Resolution) == num_dimensions());
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(std::vector{static_cast<size_t>(start_indices)...},
  //                 std::vector{Resolution...}, arr.data_ptr());
  //  }
  //  //----------------------------------------------------------------------------
  //  template <typename MemLoc, size_t... Resolution>
  //  auto read_chunk(
  //      std::vector<size_t> const& start_indices, static_multidim_array<T,
  //      x_fastest, MemLoc, Resolution...>& arr) const {
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(start_indices, std::vector{Resolution...},
  //    arr.data_ptr());
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read_chunk(std::vector<size_t> const& start_indices,
  //                  std::vector<size_t> const& counts,
  //                  std::vector<T>&            arr) const {
  //    auto const n = std::accumulate(begin(counts), end(counts), size_t(1),
  //                                   std::multiplies<size_t>{});
  //    if (size(arr) != n) {
  //      arr.resize(n);
  //    }
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(start_indices, counts, arr.data());
  //  }
  //  //----------------------------------------------------------------------------
  //  auto is_null() const {
  //    std::lock_guard lock{*m_mutex};
  //    return m_dataset.isNull();
  //  }
  //----------------------------------------------------------------------------
  auto data_space() const {
    std::lock_guard lock{*m_mutex};
    return m_dataset.getSpace();
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return data_space().getSimpleExtentNdims(); }
  //----------------------------------------------------------------------------
  auto size(size_t i) const {
    return size()[i];
  }
  //----------------------------------------------------------------------------
  auto size() const {
    auto const n = data_space().getSimpleExtentNdims();
    auto size =  std::make_unique<hsize_t[]>(n);
    data_space().getSimpleExtentDims(size.get());
    return std::vector<size_t>(size.get(), size.get() + n);
  }
  //  //----------------------------------------------------------------------------
  //  auto name() const {
  //    std::lock_guard lock{*m_mutex};
  //    return m_dataset.getName();
  //  }
  auto read_lazy(std::vector<size_t> const& chunk_size) {
    return lazy_reader<this_t>{*this, chunk_size};
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& { return m_name; }
};
//==============================================================================
class size {};
//==============================================================================
class file {
  mutable std::shared_ptr<H5::H5File> m_file;
  mutable std::shared_ptr<std::mutex> m_mutex;
  //============================================================================
 public:
  template <typename... Ts>
  file(std::filesystem::path const& path, Ts&&... ts)
      : m_file{new H5::H5File(path.string(), std::forward<Ts>(ts)...)},
        m_mutex{std::make_shared<std::mutex>()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  file(std::string const& path, Ts&&... ts)
      : m_file{new H5::H5File(path, std::forward<Ts>(ts)...)},
        m_mutex{std::make_shared<std::mutex>()} {}
  //============================================================================
  template <typename T, integral... Size>
  auto add_dataset(std::string const& dataset_name, Size... size) {
    H5::AtomType data_type{h5_type<T>::value()};
    data_type.setOrder(H5T_ORDER_LE);
    hsize_t dimsf[]{static_cast<hsize_t>(size)...};  // data set dimensions
    return hdf5::dataset<T>{
        m_file, m_mutex,
        m_file->createDataSet(dataset_name, data_type,
                              H5::DataSpace{sizeof...(Size), dimsf}), dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(char const* dataset_name) const {
    return hdf5::dataset<T>{m_file, m_mutex, m_file->openDataSet(dataset_name), dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(std::string const& dataset_name) const {
    return hdf5::dataset<T>{m_file, m_mutex, m_file->openDataSet(dataset_name), dataset_name};
  }
  //----------------------------------------------------------------------------
  //template <typename T>
  //auto datasets() const {
  //  std::map<std::string, hdf5::dataset<T>> vars;
  //  for (auto& [name, var] : m_file->getDataSets()) {
  //    if (var.getType() == h5_type<T>::value()) {
  //      vars[name] = hdf5::dataset<T>{m_file, m_mutex, std::move(var)};
  //    }
  //  }
  //  return vars;
  //}
};
//==============================================================================
}  // namespace tatooine::hdf5
//==============================================================================
#endif
#endif
