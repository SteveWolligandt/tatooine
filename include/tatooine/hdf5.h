//#if @TATOOINE_HDF5_AVAILABLE@
#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/hdf5/type.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <tatooine/hdf5_include.h>
#include <boost/range/algorithm/reverse.hpp>
#include <cassert>
#include <memory>
#include <numeric>
#include <vector>
//==============================================================================
namespace tatooine::hdf5 {
//==============================================================================
static constexpr auto unlimited = H5S_UNLIMITED;
//==============================================================================
struct api {
 private:
  H5E_auto_t m_old_func        = nullptr;
  void*      m_old_client_data = nullptr;
  //----------------------------------------------------------------------------
  api() { H5Eget_auto(H5E_DEFAULT, &m_old_func, &m_old_client_data); }

 public:
  //----------------------------------------------------------------------------
  auto disable_error_printing() -> void {
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
  }
  //----------------------------------------------------------------------------
  auto enable_error_printing() -> void {
    H5Eset_auto(H5E_DEFAULT, m_old_func, m_old_client_data);
  }
  //----------------------------------------------------------------------------
  static auto get() -> auto& {
    static auto obj = api{};
    return obj;
  }
  //----------------------------------------------------------------------------
  static auto my_hdf5_error_handler(void * /*error_data*/) -> herr_t {
    std::cerr << "An HDF5 error was detected. Bye.\n";
    exit(1);
  }
};
//==============================================================================
struct property_list {
  //============================================================================
  // FACTORIES
  //============================================================================
  static auto dataset_creation() { return property_list{H5P_DATASET_CREATE}; }
  //============================================================================
  // MEMBERS
  //============================================================================
 private:
  hid_t m_id;

  //============================================================================
  // CTORS
  //============================================================================
 public:
  property_list(hid_t cls_id) : m_id{H5Pcreate(cls_id)} {}
  property_list(property_list const& other) : m_id{H5Pcopy(other.m_id)} {}
  ~property_list() { H5Pclose(m_id); }

  //============================================================================
  // GETTERS
  //============================================================================
  auto id() { return m_id; }

  //============================================================================
  // METHODS
  //============================================================================
  template <typename... Size, enable_if_integral<Size...> = true>
  auto set_chunk(Size const... size) {
    auto dim = std::array{static_cast<hsize_t>(size)...};
    H5Pset_chunk(m_id, sizeof...(Size), dim.data());
  }
};
//==============================================================================
struct dataspace {
  //============================================================================
  // MEMBERS
  //============================================================================
 private:
  hid_t m_id;

  //============================================================================
  // CTORS
  //============================================================================
 public:
  dataspace(dataspace const& other) : m_id{H5Scopy(other.m_id)} {}
  //----------------------------------------------------------------------------
  explicit dataspace(hid_t const id) : m_id{id} {}
  //----------------------------------------------------------------------------
  template <typename... Size, enable_if_integral<Size...> = true>
  explicit dataspace(Size const... size)
      : dataspace{std::array{static_cast<hsize_t>(size)...}} {}
  //----------------------------------------------------------------------------
  explicit dataspace(std::vector<hsize_t> const& cur_resolution)
      : m_id{H5Screate_simple(cur_resolution.size(), cur_resolution.data(),
                              nullptr)} {}
  //----------------------------------------------------------------------------
  dataspace(std::vector<hsize_t> const& cur_resolution,
            std::vector<hsize_t> const& max_resolution)
      : m_id{H5Screate_simple(cur_resolution.size(), cur_resolution.data(),
                              max_resolution.data())} {}
  //----------------------------------------------------------------------------
  template <std::size_t N>
  explicit dataspace(std::array<hsize_t, N> const& cur_resolution)
      : m_id{H5Screate_simple(N, cur_resolution.data(), nullptr)} {}
  //----------------------------------------------------------------------------
  template <std::size_t N>
  dataspace(std::array<hsize_t, N> const& cur_resolution,
            std::array<hsize_t, N> const& maxdims)
      : m_id{H5Screate_simple(N, cur_resolution.data()), maxdims.data()} {}
  //----------------------------------------------------------------------------
  ~dataspace() { H5Sclose(m_id); }
  //============================================================================
  // GETTERS
  //============================================================================
  auto id() { return m_id; }

  //============================================================================
  // METHODS
  //============================================================================
  auto rank() const { return H5Sget_simple_extent_ndims(m_id); }
  //------------------------------------------------------------------------------
  auto current_resolution() const {
    std::vector<hsize_t> cur_res(rank());
    H5Sget_simple_extent_dims(m_id, cur_res.data(), nullptr);
    return cur_res; 
  }
  //------------------------------------------------------------------------------
  auto max_resolution() const {
    std::vector<hsize_t> max_res(rank());
    H5Sget_simple_extent_dims(m_id, nullptr, max_res.data());
    return max_res; 
  }
  //------------------------------------------------------------------------------
  auto current_and_max_resolution() const {
    auto ret =
        std::tuple{std::vector<hsize_t>(rank()), std::vector<hsize_t>(rank())};
    H5Sget_simple_extent_dims(m_id, std::get<0>(ret).data(),
                              std::get<1>(ret).data());
    return ret;
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(std::vector<hsize_t> const& offset,
                        std::vector<hsize_t> const& count) {
    H5Sselect_hyperslab(m_id, H5S_SELECT_SET, offset.data(), nullptr,
                        count.data(), nullptr);
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(hsize_t const offset, hsize_t const count) {
    H5Sselect_hyperslab(m_id, H5S_SELECT_SET, &offset, nullptr, &count,
                        nullptr);
  }
  //------------------------------------------------------------------------------
  auto is_unlimited() const {
    bool has_unlimited_dimension = false;
    auto max                     = max_resolution();
    for (auto dim : max) {
      if (dim == unlimited) {
        has_unlimited_dimension = true;
        break;
      }
    }
    return has_unlimited_dimension;
  }
};
//==============================================================================
class attribute {
 public:
  using this_t = attribute;

 private:
  std::unique_ptr<hid_t> m_parent_id;
  std::string            m_name;

 public:
  attribute(std::unique_ptr<hid_t> const& file_id, std::string const& name)
      : m_parent_id{std::make_unique<hid_t>(*file_id)}, m_name{name} {
    // H5Iinc_ref(*m_parent_id);
  }
  //----------------------------------------------------------------------------
  attribute(attribute const& other)
      : m_parent_id{std::make_unique<hid_t>(*other.m_parent_id)},
        m_name{other.m_name} {
    // H5Iinc_ref(*m_parent_id);
  }
  //----------------------------------------------------------------------------
  attribute(attribute&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(attribute const& other) -> attribute& {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
    m_parent_id = std::make_unique<hid_t>(*other.m_parent_id);
    m_name      = other.m_name;
    // H5Iinc_ref(*m_parent_id);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(attribute&&) noexcept -> attribute& = default;
  //----------------------------------------------------------------------------
  ~attribute() {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
  }
  //============================================================================
  auto read() {
    // T data;
    // auto           s = m_attribute.getInMemDataSize();
    // t.resize(s / sizeof(T));
    // m_attribute.read(type_id<T>(), t.data());
    // return t;
  }
  //============================================================================
  auto write(std::string const& s) {
    auto dataspace_id = H5Screate(H5S_SCALAR);
    auto type_id      = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, s.size());
    api::get().disable_error_printing();
    auto attribute_id = H5Acreate(*m_parent_id, m_name.data(), type_id,
                                  dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    api::get().enable_error_printing();
    if (attribute_id <= 0) {
      attribute_id = H5Aopen(*m_parent_id, m_name.data(), H5P_DEFAULT);
    }

    if (attribute_id >= 0) {
      H5Awrite(attribute_id, type_id, s.data());
      H5Aclose(attribute_id);
    }

    H5Sclose(dataspace_id);
    H5Tclose(type_id);
  }
  //----------------------------------------------------------------------------
  auto operator=(std::string const& s) -> attribute& {
    write(s);
    return *this;
  }
};
//==============================================================================
template <typename T>
class dataset {
 public:
  using this_t     = dataset<T>;
  using value_type = T;

 private:
  std::unique_ptr<hid_t> m_parent_id;
  std::unique_ptr<hid_t> m_dataset_id;
  std::string            m_name;
  //============================================================================
 public:
  dataset(std::unique_ptr<hid_t> const& file_id, hid_t const dataset_id,
          std::string const& name)
      : m_parent_id{std::make_unique<hid_t>(*file_id)},
        m_dataset_id{std::make_unique<hid_t>(dataset_id)},
        m_name{name} {
    // H5Iinc_ref(*m_parent_id);
  }
  //----------------------------------------------------------------------------
  dataset(dataset const& other)
      : m_parent_id{std::make_unique<hid_t>(*other.m_parent_id)},
        m_dataset_id{std::make_unique<hid_t>(*other.m_dataset_id)},
        m_name{other.m_name} {
    // H5Iinc_ref(*m_parent_id);
    H5Iinc_ref(*m_dataset_id);
  }
  //----------------------------------------------------------------------------
  dataset(dataset&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dataset const& other) -> dataset& {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
    if (m_dataset_id != nullptr) {
      H5Dclose(*m_dataset_id);
    }
    m_parent_id  = std::make_unique<hid_t>(*other.m_parent_id);
    m_dataset_id = std::make_unique<hid_t>(*other.m_dataset_id);
    m_name       = other.m_name;
    // H5Iinc_ref(*m_parent_id);
    H5Iinc_ref(*m_dataset_id);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(dataset&&) noexcept -> dataset& = default;
  //----------------------------------------------------------------------------
  ~dataset() {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
    if (m_dataset_id != nullptr) {
      H5Dclose(*m_dataset_id);
    }
  }
  //============================================================================
  auto resize(hsize_t const extent) {
    H5Dset_extent(*m_dataset_id, &extent);
  }
  //----------------------------------------------------------------------------
  auto resize(std::vector<hsize_t> const& extent) {
    H5Dset_extent(*m_dataset_id, extent.data());
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  auto resize(std::array<hsize_t, N> const& extent) {
    H5Dset_extent(*m_dataset_id, extent.data());
  }
  //----------------------------------------------------------------------------
  template <typename Integral, enable_if_integral<Integral>>
  auto resize(std::vector<Integral> const& extent) {
    resize(std::vector<hsize_t>(begin(extent), end(extent)));
  }
  //----------------------------------------------------------------------------
  template <typename Integral, size_t N, enable_if_integral<Integral>>
  auto resize(std::array<Integral, N> const& extent) {
    resize(std::array<hsize_t, N>(begin(extent), end(extent)));
  }
  //----------------------------------------------------------------------------
  auto resize_if_necessary(hsize_t const requested_size) {
    auto ds                       = dataspace();
    auto const [cur_res, max_res] = ds.current_and_max_resolution();
    assert(cur_res.size() == 1);
    if (requested_size > cur_res[0] && max_res[0] == unlimited) {
      resize(requested_size);
    }
  }
  //----------------------------------------------------------------------------
  auto resize_if_necessary(std::vector<hsize_t> const& requested_size) {
    auto ds                       = dataspace();
    auto const [cur_res, max_res] = ds.current_and_max_resolution();
    assert(cur_res.size() == requested_size.size());
    bool must_resize              = true;
    for (size_t i = 0; i < cur_res.size(); ++i) {
      if (requested_size[i] > cur_res[i] && max_res[i] == unlimited) {
        must_resize = true;
        break;
      }
    }
    if (must_resize) {
      resize(requested_size);
    }
  }
  //============================================================================
  auto clear() {
    auto s = std::vector<hsize_t>(dataspace().rank(), 0);
    resize(s);
  }
  //============================================================================
  auto write(T const* data) -> void {
    H5Dwrite(*m_dataset_id, type_id<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data) -> void {
    resize_if_necessary(data.size());
    H5Dwrite(*m_dataset_id, type_id<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
             data.data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data, hsize_t const offset) -> void {
    resize_if_necessary(data.size() + offset);
    write(data.data(), std::vector{offset},
          std::vector{static_cast<hsize_t>(data.size())});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(std::vector<T> const& data) -> void {
    auto cur_res = dataspace().current_resolution();
    write(data, cur_res[0]);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(T const& data) -> void {
    auto cur_res = dataspace().current_resolution();
    resize_if_necessary(cur_res[0] + 1);
    write(&data, std::vector{cur_res[0]}, std::vector{hsize_t(1)});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  auto write(std::array<T, N> const& data) -> void {
    resize_if_necessary(data.size());
    H5Dwrite(*m_dataset_id, type_id<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
             data.data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//  template <range Range>
//#else
//  template <typename Range, enable_if_range<Range> = true>
//#endif
//  auto write(Range&& r) -> void {
//    write(std::vector(begin(r), end(r)));
//  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto write(dynamic_multidim_array<T, IndexOrder> const& data) -> void {
    write(data.data_ptr());
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto write(dynamic_multidim_array<T, IndexOrder> const& data,
             std::vector<size_t> const&                   offset) -> void {
    assert(data.num_dimensions() == offset.size());
    auto const size = data.size();
    auto total_size = size;
    for (size_t i = 0; i < size.size(); ++i) {
      total_size[i] += offset[i];
    }
    resize_if_necessary(total_size);
    write(data.data_ptr(), std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(begin(size), end(size)));
  }
//  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <typename IndexOrder, integral... Is>
//#else
//  template <typename IndexOrder, typename... Is,
//            enable_if_arithmetic<Is...> = true>
//#endif
//  auto write(dynamic_multidim_array<T, IndexOrder> const& data, Is const...
//  is)
//      -> void {
//    auto const s = data.size();
//    write<IndexOrder>(data.data_ptr(),
//                      std::vector<hsize_t>{static_cast<hsize_t>(is)...},
//                      std::vector<hsize_t>(begin(s), end(s)));
//  }
//----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename IndexOrder = x_fastest, integral... Is>
#else
  template <typename IndexOrder         = x_fastest, typename... Is,
            enable_if_arithmetic<Is...> = true>
#endif
  auto write(T const& data, Is const... is) -> void {
    write<IndexOrder>(&data, std::vector<hsize_t>{static_cast<hsize_t>(is)...},
                      std::vector<hsize_t>(sizeof...(Is), 1));
  }
  //----------------------------------------------------------------------------
  auto write(T const& data, std::vector<size_t> const& offset) -> void {
    write(&data, std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(size(offset), 1));
  }
  //----------------------------------------------------------------------------
  auto write(T const& data, std::vector<hsize_t> offset) -> void {
    write(&data, std::move(offset), std::vector<hsize_t>(size(offset), 1));
  }
  //----------------------------------------------------------------------------
  auto write(std::vector<T> const& data, std::vector<size_t> const& offset,
             std::vector<size_t> const& count) -> void {
    write(data.data(), std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  auto write(T const* data, std::vector<size_t> const& offset,
             std::vector<size_t> const& count) -> void {
    write(data, std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  auto write(std::vector<T> const& data, std::vector<hsize_t> offset,
             std::vector<hsize_t> count) -> void {
    write(data.data(), std::move(offset), std::move(count));
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto write(T const* data, std::vector<hsize_t> offset,
             std::vector<hsize_t> count) -> void {
    assert(offset.size() == count.size());

    auto dataset_space = dataspace();
    boost::reverse(offset);
    boost::reverse(count);
    dataset_space.select_hyperslab(offset, count);
    auto memory_space = hdf5::dataspace{count};
    H5Dwrite(*m_dataset_id, type_id<T>(), memory_space.id(), dataset_space.id(),
             H5P_DEFAULT, data);
  }
  //============================================================================
  auto read(hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id,
            T* buf) const -> void {
    H5Dread(*m_dataset_id, type_id<T>(), mem_space_id, file_space_id,
            xfer_plist_id, buf);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read() const {
    dynamic_multidim_array<T, IndexOrder> arr;
    read(arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto read(dynamic_multidim_array<T, IndexOrder>& arr) const {
    auto       dataset_space = dataspace();
    auto const rank          = dataset_space.rank();
    auto       size          = dataset_space.current_resolution();
    boost::reverse(size);
    bool must_resize = (unsigned int)rank != arr.num_dimensions();
    if (!must_resize) {
      for (int i = 0; i < rank; ++i) {
        if (arr.size(i) != size[i]) {
          must_resize = true;
          break;
        }
      }
    }
    if (must_resize) {
      arr.resize(std::vector<size_t>(begin(size), end(size)));
    }
    read(H5S_ALL, H5S_ALL, H5P_DEFAULT, arr.data_ptr());
  }
  //----------------------------------------------------------------------------
  auto read_as_vector() const {
    std::vector<T> data;
    read(data);
    return data;
  }
  //----------------------------------------------------------------------------
  auto read(std::vector<hsize_t> const& offset,
            std::vector<hsize_t> const& count, std::vector<T>& data) const {
    assert(offset.size() == count.size());
    boost::reverse(offset);
    boost::reverse(count);

    auto dataset_space = dataspace();
    dataset_space.select_hyperslab(offset, count);

    auto memory_space = hdf5::dataspace{count};
    read(memory_space.id(), dataset_space.id(),
            H5P_DEFAULT, data.data());
  }
  //----------------------------------------------------------------------------
  auto read(hsize_t const offset, hsize_t const count,
            std::vector<T>& data) const {
    assert(count == data.size());
    auto dataset_space = dataspace();
    dataset_space.select_hyperslab(offset, count);

    auto memory_space = hdf5::dataspace{count};
    read(memory_space.id(), dataset_space.id(),
            H5P_DEFAULT, data.data());
  }
  //----------------------------------------------------------------------------
  template <typename Int0, typename Int1, enable_if_integral<Int0, Int1> = true>
  auto read_as_vector(std::vector<Int0> const& offset,
                      std::vector<Int1> const& count) const {
    return read_as_vector(std::vector<hsize_t>(begin(offset), end(offset)),
                          std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read(std::vector<T>& data) const {
    hid_t      dataset_space = H5Dget_space(*m_dataset_id);
    auto const num_dims      = H5Sget_simple_extent_ndims(dataset_space);
    auto       size          = std::make_unique<hsize_t[]>(num_dims);
    H5Sget_simple_extent_dims(dataset_space, size.get(), nullptr);
    std::reverse(size.get(), size.get() + num_dims);
    size_t num_entries = 1;
    for (int i = 0; i < num_dims; ++i) {
      num_entries *= size[i];
    }
    if (data.size() != num_entries) {
      data.resize(num_entries);
    }

    read(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(dataset_space);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto read_chunk(std::vector<size_t> const&             offset,
                  std::vector<size_t> const&             count,
                  dynamic_multidim_array<T, IndexOrder>& arr) const {
    read_chunk(std::vector<hsize_t>(begin(offset), end(offset)),
               std::vector<hsize_t>(begin(count), end(count)), arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read_chunk(std::vector<size_t> const& offset,
                  std::vector<size_t> const& count) const {
    return read_chunk<IndexOrder>(
        std::vector<hsize_t>(begin(offset), end(offset)),
        std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read_chunk(std::vector<hsize_t> const& offset,
                  std::vector<hsize_t> const& count) const {
    dynamic_multidim_array<T, IndexOrder> arr;
    read_chunk(offset, count, arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto read_chunk(std::vector<hsize_t> offset, std::vector<hsize_t> count,
                  dynamic_multidim_array<T, IndexOrder>& arr) const -> auto& {
    assert(offset.size() == count.size());

    std::vector<hsize_t> count_without_ones;

    hid_t  dataset_space = H5Dget_space(*m_dataset_id);
    size_t rank          = 0;
    for (size_t i = 0; i < count.size(); ++i) {
      if (count[i] > 1) {
        ++rank;
        count_without_ones.push_back(count[i]);
      }
    }
    size_t num_dimensions = 0;
    for (size_t i = 0; i < arr.num_dimensions(); ++i) {
      if (arr.size(i) > 1) {
        ++num_dimensions;
      }
    }

    if (rank != num_dimensions) {
      arr.resize(count);
    } else {
      for (size_t i = 0; i < rank; ++i) {
        if (arr.size(i) != count_without_ones[i]) {
          arr.resize(count_without_ones);
          break;
        }
      }
    }
    boost::reverse(offset);
    boost::reverse(count);
    H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, offset.data(), nullptr,
                        count.data(), nullptr);

    auto memory_space =
        H5Screate_simple(static_cast<int>(rank), count.data(), nullptr);
    read( memory_space, dataset_space,
            H5P_DEFAULT, arr.data_ptr());
    H5Sclose(dataset_space);
    H5Sclose(memory_space);
    return arr;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto read(Is const... is) const {
    std::vector<hsize_t> offset{static_cast<hsize_t>(is)...};
    std::vector<hsize_t> count(sizeof...(Is), 1);

    hid_t      dataset_space = H5Dget_space(*m_dataset_id);
    auto const rank          = H5Sget_simple_extent_ndims(dataset_space);
    auto       size          = std::make_unique<hsize_t[]>(rank);
    boost::reverse(offset);
    H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, offset.data(), nullptr,
                        count.data(), nullptr);
    auto memory_space = H5Screate_simple(rank, count.data(), nullptr);
    T    data;
    Dread(memory_space, dataset_space,
            H5P_DEFAULT, &data);
    H5Sclose(dataset_space);
    H5Sclose(memory_space);
    return data;
  }
  //----------------------------------------------------------------------------
  auto operator[](hsize_t const i) const { return read(i); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const {
    auto dataset_space = H5Dget_space(*m_dataset_id);
    auto ndims         = H5Sget_simple_extent_ndims(dataset_space);
    H5Sclose(dataset_space);
    return ndims;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto size(size_t i) const {
    return size<IndexOrder>()[i];
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto size() const {
    hid_t      dataset_space = H5Dget_space(*m_dataset_id);
    auto const num_dims      = H5Sget_simple_extent_ndims(dataset_space);
    auto       size          = std::make_unique<hsize_t[]>(num_dims);
    H5Sget_simple_extent_dims(dataset_space, size.get(), nullptr);
    std::vector<size_t> s(size.get(), size.get() + num_dims);
    boost::reverse(s);
    H5Sclose(dataset_space);
    return s;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read_lazy(std::vector<size_t> const& chunk_size) {
    return lazy_reader<this_t, IndexOrder>{*this, chunk_size};
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& { return m_name; }
  //----------------------------------------------------------------------------
  auto attribute(std::string const& name) const {
    return hdf5::attribute{m_dataset_id, name};
  }
  auto dataspace() const {
    return hdf5::dataspace{H5Dget_space(*m_dataset_id)};
  }
  auto flush() {
    H5Dflush(*m_dataset_id);
  }
};
//==============================================================================
class group {
 public:
  using this_t = group;

 private:
  std::unique_ptr<hid_t> m_parent_id;
  std::unique_ptr<hid_t> m_group_id;
  std::string            m_name;
  //============================================================================
 public:
  group(std::unique_ptr<hid_t> const& file_id, hid_t const group_id,
        std::string const& name)
      : m_parent_id{std::make_unique<hid_t>(*file_id)},
        m_group_id{std::make_unique<hid_t>(group_id)},
        m_name{name} {
    // H5Iinc_ref(*m_parent_id);
  }
  //----------------------------------------------------------------------------
  group(group const& other)
      : m_parent_id{std::make_unique<hid_t>(*other.m_parent_id)},
        m_group_id{std::make_unique<hid_t>(*other.m_group_id)},
        m_name{other.m_name} {
    // H5Iinc_ref(*m_parent_id);
    H5Iinc_ref(*m_group_id);
  }
  //----------------------------------------------------------------------------
  group(group&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(group const& other) -> group& {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
    if (m_group_id != nullptr) {
      H5Gclose(*m_group_id);
    }
    m_parent_id = std::make_unique<hid_t>(*other.m_parent_id);
    m_group_id  = std::make_unique<hid_t>(*other.m_group_id);
    m_name      = other.m_name;
    // H5Iinc_ref(*m_parent_id);
    H5Iinc_ref(*m_group_id);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(group&&) noexcept -> group& = default;
  //----------------------------------------------------------------------------
  ~group() {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(*m_parent_id);
    //}
    if (m_group_id != nullptr) {
      H5Gclose(*m_group_id);
    }
  }
  auto attribute(std::string const& name) const {
    return hdf5::attribute{m_group_id, name};
  }
  //============================================================================
#ifdef __cpp_concepts
  template <typename T, typename IndexOrder = x_fastest, integral... Size>
#else
  template <typename T, typename IndexOrder = x_fastest, typename... Size,
            enable_if_integral<Size...> = true>
#endif
  auto add_dataset(std::string const& dataset_name, Size const... size) {
    static_assert(sizeof...(Size) > 0,
                  "hdf5::dataset needs at least 1 dimension.");
    auto dimsf = std::array{static_cast<hsize_t>(size)...};
    boost::reverse(dimsf);
    bool has_unlimited_dimension = false;
    for (auto dim : dimsf) {
      if (dim == unlimited) {
        has_unlimited_dimension = true;
        break;
      }
    }
      hid_t dataset_id{};
    if (has_unlimited_dimension) {
      auto maxdims = dimsf;
      for (auto& dim : dimsf) {
        if (dim == unlimited) {
          dim = 0;
        }
      }
      auto dataset_space =
          H5Screate_simple(sizeof...(Size), dimsf.data(), maxdims.data());
      // api::get().disable_error_printing();
      auto  plist = property_list::dataset_creation();
      plist.set_chunk(
          (static_cast<hsize_t>(size) == unlimited ? 64 : size)...);
      dataset_id =
          H5Dcreate(*m_group_id, dataset_name.data(), type_id<T>(),
                    dataset_space, H5P_DEFAULT, plist.id(), H5P_DEFAULT);
      H5Sclose(dataset_space);
    } else {
      auto dataset_space =
          H5Screate_simple(sizeof...(Size), dimsf.data(), nullptr);
      // api::get().disable_error_printing();
      dataset_id =
          H5Dcreate(*m_group_id, dataset_name.data(), type_id<T>(),
                    dataset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Sclose(dataset_space);
    }
    //api::get().enable_error_printing();
    if (dataset_id >= 0) {
      return hdf5::dataset<T>{m_group_id, dataset_id, dataset_name};
    }
    return dataset<T>(dataset_name.data());
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(char const* dataset_name) const {
    return hdf5::dataset<T>{m_group_id,
                            H5Dopen(*m_group_id, dataset_name, H5P_DEFAULT),
                            dataset_name};
  }
  //============================================================================
  auto sub_group(char const* name) {
    hid_t group_id;
    api::get().disable_error_printing();
    group_id =
        H5Gcreate2(*m_group_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    api::get().enable_error_printing();
    if (group_id <= 0) {
      group_id = H5Gopen2(*m_group_id, name, H5P_DEFAULT);
    }
    return hdf5::group{m_group_id, group_id, name};
  }
  //----------------------------------------------------------------------------
  auto sub_group(std::string const& name) { return sub_group(name.data()); }
};
//==============================================================================
class file {
  std::unique_ptr<hid_t> m_file_id;
  //============================================================================
 public:
  file(filesystem::path const& path) : file{path.c_str()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  file(std::string const& path) : file{path.data()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  file(char const* path) { open(path); }
  //----------------------------------------------------------------------------
  file(file const& other)
      : m_file_id{std::make_unique<hid_t>(*other.m_file_id)} {
    H5Iinc_ref(*m_file_id);
  }
  //----------------------------------------------------------------------------
  file(file&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(file const& other) -> file& {
    if (m_file_id != nullptr) {
      H5Fclose(*m_file_id);
    }
    m_file_id = std::make_unique<hid_t>(*other.m_file_id);
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

 private:
  auto open(char const* path) -> void {
    if (filesystem::exists(filesystem::path{path})) {
      m_file_id =
          std::make_unique<hid_t>(H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT));
    } else {
      m_file_id = std::make_unique<hid_t>(
          H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
    }
  }

 public:
  //============================================================================
  // auto group(std::string const& group_name) {
  //  return hdf5::group{m_file_id, m_file_id->openGroup(group_name),
  //                     group_name};
  //}
  //============================================================================
#ifdef __cpp_concepts
  template <typename T, typename IndexOrder = x_fastest, integral... Size>
#else
  template <typename T, typename IndexOrder = x_fastest, typename... Size,
            enable_if_integral<Size...> = true>
#endif
  auto add_dataset(std::string const& dataset_name, Size... size) {

    static_assert(sizeof...(Size) > 0,
                  "hdf5::dataset needs at least 1 dimension.");
    auto dimsf = std::array{static_cast<hsize_t>(size)...};
    boost::reverse(dimsf);
    bool has_unlimited_dimension = false;
    for (auto dim : dimsf) {
      if (dim == unlimited) {
        has_unlimited_dimension = true;
        break;
      }
    }
    hid_t dataset_id{};
    if (has_unlimited_dimension) {
      auto maxdims = dimsf;
      for (auto& dim : dimsf) {
        if (dim == unlimited) {
          dim = 0;
        }
      }
      auto dataset_space =
          H5Screate_simple(sizeof...(Size), dimsf.data(), maxdims.data());
      // api::get().disable_error_printing();
      auto  plist = property_list::dataset_creation();
      plist.set_chunk(
          (static_cast<hsize_t>(size) == unlimited ? 100 : size)...);
      dataset_id =
          H5Dcreate(*m_file_id, dataset_name.data(), type_id<T>(),
                    dataset_space, H5P_DEFAULT, plist.id(), H5P_DEFAULT);
      H5Sclose(dataset_space);
    } else {
      auto dataset_space =
          H5Screate_simple(sizeof...(Size), dimsf.data(), nullptr);
      // api::get().disable_error_printing();
      dataset_id =
          H5Dcreate(*m_file_id, dataset_name.data(), type_id<T>(),
                    dataset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Sclose(dataset_space);
    }
    //api::get().enable_error_printing();
    if (dataset_id >= 0) {
      return hdf5::dataset<T>{m_file_id, dataset_id, dataset_name};
    }
    return dataset<T>(dataset_name.data());
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(char const* dataset_name) const {
    return hdf5::dataset<T>{m_file_id,
                            H5Dopen(*m_file_id, dataset_name, H5P_DEFAULT),
                            dataset_name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto dataset(std::string const& dataset_name) const {
    return hdf5::dataset<T>{
        m_file_id, H5Dopen(*m_file_id, dataset_name.data(), H5P_DEFAULT),
        dataset_name};
  }
  //============================================================================
  auto group(char const* name) {
    hid_t group_id;
    api::get().disable_error_printing();
    group_id =
        H5Gcreate2(*m_file_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    api::get().enable_error_printing();
    if (group_id <= 0) {
      group_id = H5Gopen2(*m_file_id, name, H5P_DEFAULT);
    }
    return hdf5::group{m_file_id, group_id, name};
  }
  //----------------------------------------------------------------------------
  auto group(std::string const& name) { return group(name.data()); }
  //============================================================================
  auto attribute(std::string const& name) const {
    return hdf5::attribute{m_file_id, name};
  }
};
//==============================================================================
}  // namespace tatooine::hdf5
//==============================================================================
#endif
//#else
//#pragma message(including <tatooine / hdf5.h> without HDF5 support.)
//#endif
