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
class attribute {
 public:
  using this_t = attribute;

 private:
  mutable std::shared_ptr<H5::H5File> m_file;
  mutable std::shared_ptr<std::mutex> m_mutex;
  H5::Attribute                       m_attribute;
  std::string                         m_name;

 public:
  attribute(std::shared_ptr<H5::H5File>& file,
            std::shared_ptr<std::mutex>& mutex, H5::Attribute const& attribute,
            std::string const& name)
      : m_file{file}, m_mutex{mutex}, m_attribute{attribute}, m_name{name} {}
  //----------------------------------------------------------------------------
  attribute(attribute const&)     = default;
  attribute(attribute&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(attribute const&) -> attribute& = default;
  auto operator=(attribute&&) noexcept -> attribute& = default;
  //============================================================================
  template <typename T>
  auto read() {
    std::vector<T> t;
    auto           s = m_attribute.getInMemDataSize();
    t.resize(s / sizeof(T));
    m_attribute.read(h5_type<T>::value(), t.data());
    return t;
  }
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
    auto const   ds       = data_space();
    size_t const num_dims = ds.getSimpleExtentNdims();
    auto    size     = std::make_unique<hsize_t[]>(num_dims);
    ds.getSimpleExtentDims(size.get());
    std::reverse(size.get(), size.get() + num_dims);
    bool must_resize = num_dims != arr.num_dimensions();
    if (!must_resize) {
      for (size_t i = 0; i < num_dims; ++i) {
        if (arr.size(i) != size[i]) {
          must_resize = true;
          break;
        }
      }
    }
    if (must_resize) {
      arr.resize(std::vector<size_t>(size.get(), size.get() + num_dims));
    }

    std::lock_guard lock{*m_mutex};
    m_dataset.read(arr.data_ptr(), h5_type<T>::value());
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T> data;
    read(data);
    return data;
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& data) const {
    auto const ds       = data_space();
    auto const num_dims = ds.getSimpleExtentNdims();
    auto       size     = std::make_unique<hsize_t[]>(num_dims);
    ds.getSimpleExtentDims(size.get());
    size_t num_entries = 1;
    for (size_t i = 0; i < num_dims; ++i) {
      num_entries *= size[i];
    }
    if (data.size() != num_entries) {
      data.resize(num_entries);
    }

    std::lock_guard lock{*m_mutex};
    m_dataset.read(data.data(), h5_type<T>::value());
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
  //      auto offset =
  //      arr.global_indices_from_chunk_indices(chunk_indices); auto const
  //      plain_chunk_index =
  //          arr.plain_chunk_index_from_chunk_indices(chunk_indices);
  //
  //      if (arr.chunk_at_is_null(plain_chunk_index)) {
  //        arr.create_chunk_at(plain_chunk_index);
  //      }
  //
  //      // std::reverse(begin(offset), end(offset));
  //      auto s = arr.internal_chunk_size();
  //      // std::reverse(begin(s), end(s));
  //      read_chunk(offset, s, *arr.chunk_at(plain_chunk_index));
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
  //  auto read_single(std::vector<size_t> const& offset) const {
  //    assert(size(offset) == num_dimensions());
  //    T               t;
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(offset, std::vector<size_t>(num_dimensions(),
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
  template <typename Ordering>
  auto read_chunk(std::vector<size_t> const&            offset,
                  std::vector<size_t> const&            count,
                  dynamic_multidim_array<T, Ordering>& arr) const {
    read_chunk(std::vector<hsize_t>(begin(offset), end(offset)),
               std::vector<hsize_t>(begin(count), end(count)), arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename Ordering = x_fastest>
  auto read_chunk(std::vector<size_t> const& offset,
                  std::vector<size_t> const& count) const {
    return read_chunk<Ordering>(
        std::vector<hsize_t>(begin(offset), end(offset)),
        std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  template <typename Ordering = x_fastest>
  auto read_chunk(std::vector<hsize_t> const& offset,
                  std::vector<hsize_t> const& count) const {
    dynamic_multidim_array<T, Ordering> arr;
    read_chunk(offset, count, arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename Ordering>
  auto read_chunk(std::vector<hsize_t>            offset,
                  std::vector<hsize_t>            count,
                  dynamic_multidim_array<T, Ordering>& arr) const {
    assert(offset.size() == count.size());

    auto dataspace = [this]() {
      std::lock_guard lock{*m_mutex};
      return m_dataset.getSpace();
    }();
    int rank = dataspace.getSimpleExtentNdims();
    if (static_cast<unsigned int>(rank) != arr.num_dimensions()) {
      arr.resize(count);
    } else {
      for (int i = 0; i < rank; ++i) {
        if (arr.size(i) != count[i]) {
          arr.resize(count);
          break;
        }
      }
    }
    std::reverse(begin(count), end(count));
    std::reverse(begin(offset), end(offset));
    {
      dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
      H5::DataSpace   memspace(rank, count.data());
      std::lock_guard lock{*m_mutex};
      m_dataset.read(arr.data_ptr(), h5_type<T>::value(), memspace, dataspace);
    }
    return arr;
  }
  //----------------------------------------------------------------------------
  //  auto read_chunk(std::vector<size_t> offset, std::vector<size_t>
  //  count,
  //                  T* ptr) const {
  //    assert(offset.size() == count.size());
  //    assert(offset.size() == num_dimensions());
  //
  //    // std::reverse(begin(offset), end(offset));
  //    // std::reverse(begin(count), end(count));
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(offset, count, ptr);
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
  //      StartIndices const... offset) const {
  //    static_assert(sizeof...(offset) == sizeof...(Resolution));
  //    assert(sizeof...(Resolution) == num_dimensions());
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(std::vector{static_cast<size_t>(offset)...},
  //                 std::vector{Resolution...}, arr.data_ptr());
  //  }
  //  //----------------------------------------------------------------------------
  //  template <typename MemLoc, size_t... Resolution>
  //  auto read_chunk(
  //      std::vector<size_t> const& offset, static_multidim_array<T,
  //      x_fastest, MemLoc, Resolution...>& arr) const {
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(offset, std::vector{Resolution...},
  //    arr.data_ptr());
  //  }
  //  //----------------------------------------------------------------------------
  //  auto read_chunk(std::vector<size_t> const& offset,
  //                  std::vector<size_t> const& count,
  //                  std::vector<T>&            arr) const {
  //    auto const n = std::accumulate(begin(count), end(count), size_t(1),
  //                                   std::multiplies<size_t>{});
  //    if (size(arr) != n) {
  //      arr.resize(n);
  //    }
  //    std::lock_guard lock{*m_mutex};
  //    m_dataset.getVar(offset, count, arr.data());
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
  //----------------------------------------------------------------------------
  //  auto name() const {
  //    std::lock_guard lock{*m_mutex};
  //    return m_dataset.getName();
  //  }
  auto read_lazy(std::vector<size_t> const& chunk_size) {
    return lazy_reader<this_t>{*this, chunk_size};
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& { return m_name; }
  //----------------------------------------------------------------------------
};
//==============================================================================
class group {
 public:
  using this_t = group;

 private:
  mutable std::shared_ptr<H5::H5File> m_file;
  mutable std::shared_ptr<std::mutex> m_mutex;
  H5::Group                           m_group;
  std::string                         m_name;

 public:
  group(std::shared_ptr<H5::H5File>& file, std::shared_ptr<std::mutex>& mutex,
          H5::Group const& group, std::string const& name)
      : m_file{file}, m_mutex{mutex}, m_group{group}, m_name{name} {}
  //----------------------------------------------------------------------------
  group(group const&)     = default;
  group(group&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(group const&) -> group& = default;
  auto operator=(group&&) noexcept -> group& = default;
  //============================================================================
  auto attribute(std::string const& attribute_name) {
    return hdf5::attribute{
        m_file, m_mutex, m_group.openAttribute(attribute_name), attribute_name};
  }
  //============================================================================
  template <typename T>
  auto dataset(std::string const& dataset_name) {
    return hdf5::dataset<T>{
        m_file, m_mutex, m_group.openDataSet(dataset_name), dataset_name};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename T, integral... Size>
#else
  template <typename T, typename... Size, enable_if<is_integral<Size...>> = true>
#endif
  auto add_dataset(std::string const& dataset_name, Size... size) {
    H5::AtomType data_type{h5_type<T>::value()};
    hsize_t      dimsf[]{static_cast<hsize_t>(size)...};  // data set dimensions
    std::reverse(dimsf, dimsf + sizeof...(Size));
    return hdf5::dataset<T>{
        m_file, m_mutex,
        m_group.createDataSet(dataset_name, data_type,
                              H5::DataSpace{sizeof...(Size), dimsf}),
        dataset_name};
  }
};
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Ts>
  file(char const* path, Ts&&... ts)
      : m_file{new H5::H5File(path, std::forward<Ts>(ts)...)},
        m_mutex{std::make_shared<std::mutex>()} {}
  //============================================================================
  auto group(std::string const& group_name) {
    return hdf5::group{m_file, m_mutex, m_file->openGroup(group_name),
                       group_name};
  }
  //============================================================================
#ifdef __cpp_concepts
  template <typename T, integral... Size>
#else
  template <typename T, typename... Size, enable_if<is_integral<Size...>> = true>
#endif
  auto add_dataset(std::string const& dataset_name, Size... size) {
    H5::AtomType data_type{h5_type<T>::value()};
    hsize_t      dimsf[]{static_cast<hsize_t>(size)...};  // data set dimensions
    std::reverse(dimsf, dimsf + sizeof...(Size));
    return hdf5::dataset<T>{
        m_file, m_mutex,
        m_file->createDataSet(dataset_name, data_type,
                              H5::DataSpace{sizeof...(Size), dimsf}),
        dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(char const* dataset_name) const {
    return hdf5::dataset<T>{m_file, m_mutex, m_file->openDataSet(dataset_name),
                            dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(std::string const& dataset_name) const {
    return hdf5::dataset<T>{m_file, m_mutex, m_file->openDataSet(dataset_name),
                            dataset_name};
  }
  //----------------------------------------------------------------------------
  // template <typename T>
  // auto datasets() const {
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
