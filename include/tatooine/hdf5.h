#ifdef TATOOINE_HDF5_AVAILABLE
#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H
//==============================================================================
#include <hdf5.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/concepts.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <cassert>
#include <tatooine/filesystem.h>
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
  static auto value() { return H5T_NATIVE_INT; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct h5_type<float> {
  static auto value() { return H5T_NATIVE_FLOAT; }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct h5_type<double> {
  static auto value() { return H5T_NATIVE_DOUBLE; }
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
//class attribute {
// public:
//  using this_t = attribute;
//
// private:
//  mutable std::shared_ptr<H5::H5File> m_file_id;
//  mutable std::shared_ptr<std::mutex> m_mutex;
//  H5::Attribute                       m_attribute;
//  std::string                         m_name;
//
// public:
//  attribute(std::shared_ptr<H5::H5File>& file_id,
//            std::shared_ptr<std::mutex>& mutex, H5::Attribute const& attribute,
//            std::string const& name)
//      : m_file_im_file_id{file_id}, m_mutex{mutex}, m_attribute{attribute}, m_name{name} {}
//  //----------------------------------------------------------------------------
//  attribute(attribute const&)     = default;
//  attribute(attribute&&) noexcept = default;
//  //----------------------------------------------------------------------------
//  auto operator=(attribute const&) -> attribute& = default;
//  auto operator=(attribute&&) noexcept -> attribute& = default;
//  //============================================================================
//  template <typename T>
//  auto read() {
//    std::vector<T> t;
//    auto           s = m_attribute.getInMemDataSize();
//    t.resize(s / sizeof(T));
//    m_attribute.read(h5_type<T>::value(), t.data());
//    return t;
//  }
//};
//==============================================================================
template <typename T>
class dataset {
 public:
  using this_t     = dataset<T>;
  using value_type = T;

 private:
  mutable std::shared_ptr<std::mutex> m_mutex;
  std::unique_ptr<hid_t>              m_file_id;
  std::unique_ptr<hid_t>              m_dataset_id;
  std::string                         m_name;
  //============================================================================
 public:
  dataset(std::shared_ptr<std::mutex>&  mutex,
          std::unique_ptr<hid_t> const& file_id, hid_t const dataset_id,
          std::string const& name)
      : m_mutex{mutex},
        m_file_id{std::make_unique<hid_t>(*file_id)},
        m_dataset_id{std::make_unique<hid_t>(dataset_id)},
        m_name{name} {}
  //----------------------------------------------------------------------------
  dataset(dataset const& other)
      : m_mutex{other.m_mutex},
        m_file_id{std::make_unique<hid_t>(*other.m_file_id)},
        m_dataset_id{std::make_unique<hid_t>(*other.m_dataset_id)},
        m_name{other.m_name} {
    H5Iinc_ref(*m_file_id);
    H5Iinc_ref(*m_dataset_id);
  }
  //----------------------------------------------------------------------------
  dataset(dataset&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dataset const&other) -> dataset& {
    m_mutex      = other.m_mutex;
    if (m_file_id != nullptr) {
      H5Fclose(*m_file_id);
    }
    if (m_dataset_id != nullptr) {
      H5Dclose(*m_dataset_id);
    }
    m_file_id    = std::make_unique<hid_t>(*other.m_file_id);
    m_dataset_id = std::make_unique<hid_t>(*other.m_dataset_id);
    m_name       = other.m_name;
    H5Iinc_ref(*m_file_id);
    H5Iinc_ref(*m_dataset_id);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(dataset&&) noexcept -> dataset& = default;
  //----------------------------------------------------------------------------
  ~dataset() {
    if (m_file_id != nullptr) {
      H5Fclose(*m_file_id);
    }
    if (m_dataset_id != nullptr) {
      H5Dclose(*m_dataset_id);
    }
  }
  //============================================================================
  auto write(T const* data) {
    std::lock_guard lock{*m_mutex};
     H5Dwrite(*m_dataset_id, h5_type<T>::value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
              data);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data) {
    std::lock_guard lock{*m_mutex};
    H5Dwrite(*m_dataset_id, h5_type<T>::value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
             data.data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  auto write(std::array<T, N> const& data) {
    std::lock_guard lock{*m_mutex};
    H5Dwrite(*m_dataset_id, h5_type<T>::value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
             data.data());
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
    hid_t      dspace = H5Dget_space(*m_dataset_id);
    auto const num_dims      = H5Sget_simple_extent_ndims(dspace);
    auto       size   = std::make_unique<hsize_t[]>(num_dims);
    H5Sget_simple_extent_dims(dspace, size.get(), nullptr);
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
    H5Dread(*m_dataset_id, h5_type<T>::value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
            arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T> data;
    read(data);
    return data;
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<T>& data) const {
    hid_t      dspace = H5Dget_space(*m_dataset_id);
    auto const num_dims      = H5Sget_simple_extent_ndims(dspace);
    auto       size   = std::make_unique<hsize_t[]>(num_dims);
    H5Sget_simple_extent_dims(dspace, size.get(), nullptr);
    size_t num_entries = 1;
    for (size_t i = 0; i < num_dims; ++i) {
      num_entries *= size[i];
    }
    if (data.size() != num_entries) {
      data.resize(num_entries);
    }

    std::lock_guard lock{*m_mutex};
    H5Dread(*m_dataset_id, h5_type<T>::value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
            data.data());
  }
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
  auto read_chunk(std::vector<hsize_t> offset, std::vector<hsize_t> count,
                  dynamic_multidim_array<T, Ordering>& arr) const {
    std::lock_guard lock{*m_mutex};
    assert(offset.size() == count.size());

    hid_t      dspace = H5Dget_space(*m_dataset_id);
    auto const rank      = H5Sget_simple_extent_ndims(dspace);
    auto       size   = std::make_unique<hsize_t[]>(rank);
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
    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, offset.data(), nullptr,
                        count.data(), nullptr);
    auto memspace = H5Screate_simple(rank, count.data(), nullptr);
    H5Dread(*m_dataset_id, h5_type<T>::value(), memspace, dspace, H5P_DEFAULT,
            arr.data_ptr());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto num_dimensions() const {
    return H5Sget_simple_extent_ndims(H5Dget_space(*m_dataset_id));
  }
  //----------------------------------------------------------------------------
  auto size(size_t i) const {
    return size()[i];
  }
  //----------------------------------------------------------------------------
  auto size() const {
    hid_t      dspace = H5Dget_space(*m_dataset_id);
    auto const num_dims      = H5Sget_simple_extent_ndims(dspace);
    auto       size   = std::make_unique<hsize_t[]>(num_dims);
    H5Sget_simple_extent_dims(dspace, size.get(), nullptr);
    return std::vector<size_t>(size.get(), size.get() + num_dims);
  }
  //----------------------------------------------------------------------------
  auto read_lazy(std::vector<size_t> const& chunk_size) {
    return lazy_reader<this_t>{*this, chunk_size};
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& { return m_name; }
};
//==============================================================================
//class group {
// publicN:
//  using this_t = group;
//
// private:
//  mutable std::shared_ptr<H5::H5File> m_file_id;
//  mutable std::shared_ptr<std::mutex> m_mutex;
//  H5::Group                           m_group;
//  std::string                         m_name;
//
// public:
//  group(std::shared_ptr<H5::H5File>& file_id, std::shared_ptr<std::mutex>& mutex,
//          H5::Group const& group, std::string const& name)
//      : m_file_id{file_id}, m_mutex{mutex}, m_group{group}, m_name{name} {}
//  //----------------------------------------------------------------------------
//  group(group const&)     = default;
//  group(group&&) noexcept = default;
//  //----------------------------------------------------------------------------
//  auto operator=(group const&) -> group& = default;
//  auto operator=(group&&) noexcept -> group& = default;
//  //============================================================================
//  auto attribute(std::string const& attribute_name) {
//    return hdf5::attribute{
//        m_file_id, m_mutex, m_group.openAttribute(attribute_name), attribute_name};
//  }
//  //============================================================================
//  template <typename T>
//  auto dataset(std::string const& dataset_name) {
//    return hdf5::dataset<T>{
//        m_file_id, m_mutex, m_group.openDataSet(dataset_name), dataset_name};
//  }
//  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <typename T, integral... Size>
//#else
//  template <typename T, typename... Size, enable_if<is_integral<Size...>> = true>
//#endif
//  auto add_dataset(std::string const& dataset_name, Size... size) {
//    H5::AtomType data_type{h5_type<T>::value()};
//    hsize_t      dimsf[]{static_cast<hsize_t>(size)...};  // data set dimensions
//    std::reverse(dimsf, dimsf + sizeof...(Size));
//    return hdf5::dataset<T>{
//        m_file_id, m_mutex,
//        m_group.createDataSet(dataset_name, data_type,
//                              H5::DataSpace{sizeof...(Size), dimsf}),
//        dataset_name};
//  }
//};
//==============================================================================
class file {
  std::unique_ptr<hid_t> m_file_id;
  mutable std::shared_ptr<std::mutex> m_mutex;
  //============================================================================
 public:
  file(filesystem::path const& path) : m_mutex{std::make_shared<std::mutex>()} {
    try {
      m_file_id = std::make_unique<hid_t>(
          H5Fopen(path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    } catch (...) {
      m_file_id = std::make_unique<hid_t>(H5Fcreate(
          path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  file(std::string const& path) : m_mutex{std::make_shared<std::mutex>()} {
    try {
      m_file_id = std::make_unique<hid_t>(
          H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    } catch (...) {
      m_file_id = std::make_unique<hid_t>(
          H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  file(char const* path) : m_mutex{std::make_shared<std::mutex>()} {
    try {
      m_file_id =
          std::make_unique<hid_t>(H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT));
    } catch (...) {
      m_file_id = std::make_unique<hid_t>(
          H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
    }
  }
  //----------------------------------------------------------------------------
  file(file const& other)
      : m_mutex{other.m_mutex},
        m_file_id{std::make_unique<hid_t>(*other.m_file_id)} {
    H5Iinc_ref(*m_file_id);
  }
  //----------------------------------------------------------------------------
  file(file&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(file const&other) -> file& {
    m_mutex      = other.m_mutex;
    if (m_file_id != nullptr) {
      H5Fclose(*m_file_id);
    }
    m_file_id    = std::make_unique<hid_t>(*other.m_file_id);
    H5Iinc_ref(*m_file_id);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(file&&) noexcept -> file& = default;
  //----------------------------------------------------------------------------
  ~file() {
    if (m_file_id != nullptr) {
      H5Fclose(*m_file_id);
    }
  }
  //============================================================================
  // auto group(std::string const& group_name) {
  //  return hdf5::group{m_file_id, m_mutex, m_file_id->openGroup(group_name),
  //                     group_name};
  //}
  //============================================================================
#ifdef __cpp_concepts
  template <typename T, integral... Size>
#else
  template <typename T, typename... Size,
            enable_if<is_integral<Size...>> = true>
#endif
  auto add_dataset(std::string const& dataset_name, Size... size) {
    hsize_t dimsf[]{static_cast<hsize_t>(size)...};  // data set dimensions
    std::reverse(dimsf, dimsf + sizeof...(Size));
    return hdf5::dataset<T>{
        m_mutex, m_file_id,
        H5Dcreate2(*m_file_id, dataset_name.c_str(), H5T_STD_I32BE,
                   H5Screate_simple(sizeof...(Size), dimsf, nullptr),
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(char const* dataset_name) const {
    return hdf5::dataset<T>{m_mutex, m_file_id,
                            H5Dopen(*m_file_id, dataset_name, H5P_DEFAULT),
                            dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(std::string const& dataset_name) const {
    return hdf5::dataset<T>{
        m_mutex, m_file_id,
        H5Dopen(*m_file_id, dataset_name.c_str(), H5P_DEFAULT), dataset_name};
  }
};
//==============================================================================
}  // namespace tatooine::hdf5
//==============================================================================
#endif
#else
#pragma message(including <tatooine/hdf5.h> without HDF5 support.)
#endif
