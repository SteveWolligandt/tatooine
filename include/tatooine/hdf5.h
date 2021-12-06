//#if @TATOOINE_HDF5_AVAILABLE@
#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H
//==============================================================================
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/filesystem.h>
#include <tatooine/hdf5/type.h>
#include <tatooine/hdf5_include.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_array.h>

#include <boost/range/algorithm/reverse.hpp>
#include <boost/range/numeric.hpp>
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
  static auto get() -> auto& {
    static auto obj = api{};
    return obj;
  }
  //----------------------------------------------------------------------------
 public:
  static auto disable_error_printing() -> void {
    get()._disable_error_printing();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto _disable_error_printing() -> void {
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
  }
  //----------------------------------------------------------------------------
 public:
  static auto enable_error_printing() -> void {
    get()._enable_error_printing();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  auto _enable_error_printing() -> void {
    H5Eset_auto(H5E_DEFAULT, m_old_func, m_old_client_data);
  }
  //----------------------------------------------------------------------------
  static auto error_handler(void* /*error_data*/) -> herr_t {
    std::cerr << "An HDF5 error was detected. Bye.\n";
    exit(1);
  }
};
//==============================================================================
struct id_holder {
  //============================================================================
 private:
  hid_t m_id = hid_t{};

 public:
  explicit id_holder(hid_t const id) : m_id{id} {}
  auto id() const { return m_id; }
  auto set_id(hid_t const id) { m_id = id; }
};
//==============================================================================
struct property_list : id_holder {
  //============================================================================
  // FACTORIES
  //============================================================================
  static auto dataset_creation() { return property_list{H5P_DATASET_CREATE}; }

  //============================================================================
  // CTORS
  //============================================================================
 public:
  explicit property_list(hid_t cls_id = H5P_DEFAULT)
      : id_holder{cls_id == H5P_DEFAULT ? H5P_DEFAULT : H5Pcreate(cls_id)} {}
  property_list(property_list const& other) : id_holder{H5Pcopy(other.id())} {}
  auto operator=(property_list const& other) -> property_list& {
    close();
    set_id(H5Pcopy(other.id()));
    return *this;
  }
  ~property_list() { close(); }

  //============================================================================
  // GETTERS
  //============================================================================
  [[nodiscard]] auto is_default() const { return id() == H5P_DEFAULT; }

  //============================================================================
  // METHODS
  //============================================================================
  auto close() -> void {
    if (!is_default()) {
      H5Pclose(id());
    }
  }
  //----------------------------------------------------------------------------
  template <typename... Size, enable_if_integral<Size...> = true>
  auto set_chunk(Size const... size) {
    auto dim = std::array{static_cast<hsize_t>(size)...};
    H5Pset_chunk(id(), sizeof...(Size), dim.data());
  }
};
//==============================================================================
struct dataspace : id_holder {
  //============================================================================
  // CTORS
  //============================================================================
 public:
  dataspace(dataspace const& other) : id_holder{H5Scopy(other.id())} {}
  //----------------------------------------------------------------------------
  explicit dataspace(hid_t const id) : id_holder{id} {}
  //----------------------------------------------------------------------------
  template <typename... Size, enable_if_integral<Size...> = true>
  explicit dataspace(Size const... size)
      : dataspace{std::array{static_cast<hsize_t>(size)...}} {}
  //----------------------------------------------------------------------------
  explicit dataspace(std::vector<hsize_t> cur_resolution)
      : id_holder{H5Screate_simple(cur_resolution.size(),
                                   boost::reverse(cur_resolution).data(),
                                   nullptr)} {}
  //----------------------------------------------------------------------------
  dataspace(std::vector<hsize_t> cur_resolution,
            std::vector<hsize_t> max_resolution)
      : id_holder{H5Screate_simple(cur_resolution.size(),
                                   boost::reverse(cur_resolution).data(),
                                   boost::reverse(max_resolution).data())} {}
  //----------------------------------------------------------------------------
  template <std::size_t N>
  explicit dataspace(std::array<hsize_t, N> cur_resolution)
      : id_holder{H5Screate_simple(N, boost::reverse(cur_resolution).data(),
                                   nullptr)} {}
  //----------------------------------------------------------------------------
  template <std::size_t N>
  dataspace(std::array<hsize_t, N> cur_resolution,
            std::array<hsize_t, N> maxdims)
      : id_holder{H5Screate_simple(N, boost::reverse(cur_resolution).data(),
                                   boost::reverse(maxdims).data())} {}
  //----------------------------------------------------------------------------
  ~dataspace() { H5Sclose(id()); }

  //============================================================================
  // METHODS
  //============================================================================
  auto rank() const { return H5Sget_simple_extent_ndims(id()); }
  //------------------------------------------------------------------------------
  auto current_resolution() const {
    std::vector<hsize_t> cur_res(rank());
    H5Sget_simple_extent_dims(id(), cur_res.data(), nullptr);
    return boost::reverse(cur_res);
  }
  //------------------------------------------------------------------------------
  auto max_resolution() const {
    std::vector<hsize_t> max_res(rank());
    H5Sget_simple_extent_dims(id(), nullptr, max_res.data());
    return boost::reverse(max_res);
  }
  //------------------------------------------------------------------------------
  auto current_and_max_resolution() const {
    auto ret =
        std::tuple{std::vector<hsize_t>(rank()), std::vector<hsize_t>(rank())};
    H5Sget_simple_extent_dims(id(), std::get<0>(ret).data(),
                              std::get<1>(ret).data());
    boost::reverse(std::get<0>(ret));
    boost::reverse(std::get<1>(ret));
    return ret;
  }
  //------------------------------------------------------------------------------
  auto num_elements() const { return H5Sget_simple_extent_npoints(id()); }
  auto selected_num_elements() const { return H5Sget_select_npoints(id()); }
  auto num_hyperslab_blocks() const {
    return H5Sget_select_hyper_nblocks(id());
  }
  auto selected_num_hyperslab_blocks() const {
    return H5Sget_select_elem_npoints(id());
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(hsize_t const* start, hsize_t const* stride,
                        hsize_t const* count, hsize_t const* block = nullptr) {
    H5Sselect_hyperslab(id(), H5S_SELECT_SET, start, stride, count, block);
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(H5S_seloper_t op, hsize_t const* start,
                        hsize_t const* stride, hsize_t const* count,
                        hsize_t const* block = nullptr) {
    H5Sselect_hyperslab(id(), op, start, stride, count, block);
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(std::vector<hsize_t> offset,
                        std::vector<hsize_t> stride,
                        std::vector<hsize_t> count) {
    boost::reverse(offset);
    boost::reverse(stride);
    boost::reverse(count);
    select_hyperslab(offset.data(), stride.data(), count.data());
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(std::vector<hsize_t> offset,
                        std::vector<hsize_t> count) {
    boost::reverse(offset);
    boost::reverse(count);
    select_hyperslab(offset.data(), nullptr, count.data());
  }
  //------------------------------------------------------------------------------
  auto select_hyperslab(hsize_t const offset, hsize_t const count) {
    select_hyperslab(&offset, nullptr, &count);
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
  //----------------------------------------------------------------------------
  auto name() const {
    auto n = std::string(H5Aget_name(id(), 0, nullptr), ' ');
    H5Aget_name(id(), n.size(), n.data());
    return n;
  }
  //----------------------------------------------------------------------------
  auto info() const {
    auto i = H5A_info_t{};
    H5Aget_info(id(), &i);
    return i;
  }
};
//==============================================================================
struct attribute {
 public:
  using this_t = attribute;

 private:
  hid_t       m_parent_id;
  std::string m_name;

 public:
  attribute(hid_t const parent_id, std::string name)
      : m_parent_id{parent_id}, m_name{std::move(name)} {
    H5Iinc_ref(m_parent_id);
  }
  //----------------------------------------------------------------------------
  attribute(attribute&&) noexcept = default;
  auto operator=(attribute&&) noexcept -> attribute& = default;
  //----------------------------------------------------------------------------
  ~attribute() { H5Idec_ref(m_parent_id); }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read_as() const {
    T    attr;
    auto attr_id = H5Aopen(m_parent_id, m_name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) {
      throw std::runtime_error{"Attribute not found."};
    }
    auto t = H5Aget_type(attr_id);
    if constexpr (is_same<T, std::string>) {
      attr = T(H5Tget_size(t) - 1, ' ');
      H5Aread(attr_id, t, attr.data());
    } else {
      H5Aread(attr_id, t, &attr);
    }
    H5Tclose(t);
    H5Aclose(attr_id);
    return attr;
  }
  //----------------------------------------------------------------------------
  auto read_as_string() const { return read_as<std::string>(); }
  //----------------------------------------------------------------------------
  auto read_int() const { return read_as<int>(); }
  //----------------------------------------------------------------------------
  auto read_float() const { return read_as<float>(); }
  //----------------------------------------------------------------------------
  auto read_double() const { return read_as<double>(); }
  //----------------------------------------------------------------------------
  auto read_int8() const { return read_as<std::int8_t>(); }
  //----------------------------------------------------------------------------
  auto read_uint8() const { return read_as<std::uint8_t>(); }
  //----------------------------------------------------------------------------
  auto read_int16() const { return read_as<std::int16_t>(); }
  //----------------------------------------------------------------------------
  auto read_uint16() const { return read_as<std::uint16_t>(); }
  //----------------------------------------------------------------------------
  auto read_int32() const { return read_as<std::int32_t>(); }
  //----------------------------------------------------------------------------
  auto read_uint32() const { return read_as<std::uint32_t>(); }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write(T const& val) {
    if (H5Aexists(m_parent_id, m_name.c_str()) > 0) {
      H5Adelete(m_parent_id, m_name.c_str());
    }

    auto ds   = hdf5::dataspace{H5Screate(H5S_SCALAR)};
    auto type = hdf5::type<T>{};
    if constexpr (is_same<T, std::string>) {
      type.set_size(val.size() + 1);
    } else if constexpr (is_same<T, char const*>) {
      type.set_size(std::strlen(val));
    }
    api::enable_error_printing();
    auto attr_id = H5Acreate(m_parent_id, m_name.c_str(), type.id(), ds.id(),
                             H5P_DEFAULT, H5P_DEFAULT);
    api::disable_error_printing();
    if (attr_id >= 0) {
      if constexpr (is_same<T, std::string>) {
        H5Awrite(attr_id, type.id(), val.data());
      } else if constexpr (is_same<T, char const*>) {
        H5Awrite(attr_id, type.id(), val);
      } else {
        H5Awrite(attr_id, type.id(), &val);
      }
      H5Aclose(attr_id);
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto operator=(T const& val) -> attribute& {
    write(val);
    return *this;
  }
};
//==============================================================================
template <typename IDHolder>
struct attribute_creator {
  auto as_id_holder() -> auto& { return *static_cast<IDHolder*>(this); }
  auto as_id_holder() const -> auto const& {
    return *static_cast<IDHolder const*>(this);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto attribute(char const* name) const {
    return hdf5::attribute{as_id_holder().id(), name};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto attribute(std::string const& name) const {
    return attribute(name.c_str());
  }
};
//==============================================================================
template <typename T>
struct dataset : id_holder, attribute_creator<dataset<T>> {
 public:
  using this_t     = dataset<T>;
  using value_type = T;

 private:
  hid_t       m_parent_id;
  std::string m_name;
  //============================================================================
 public:
  template <typename... Size>
  dataset(hid_t const parent_id, std::string const& name, Size const... size)
      : id_holder{-1}, m_parent_id{parent_id}, m_name{name} {
    if constexpr (sizeof...(Size) > 0) {
      auto dims                    = std::array{static_cast<hsize_t>(size)...};
      bool has_unlimited_dimension = false;
      for (auto dim : dims) {
        if (dim == unlimited) {
          has_unlimited_dimension = true;
          break;
        }
      }
      auto plist   = property_list{};
      auto maxdims = dims;
      if (has_unlimited_dimension) {
        for (auto& dim : dims) {
          if (dim == unlimited) {
            dim = 0;
          }
        }
        plist = property_list::dataset_creation();
        plist.set_chunk(
            (static_cast<hsize_t>(size) == unlimited ? 100 : size)...);
      }
      auto ds = hdf5::dataspace{dims, maxdims};
      set_id(H5Dcreate(parent_id, name.data(), type_id<T>(), ds.id(),
                       H5P_DEFAULT, plist.id(), H5P_DEFAULT));
      if (id() < 0) {
        set_id(H5Dopen(m_parent_id, name.c_str(), H5P_DEFAULT));
        resize(size...);
      }
    } else {
      set_id(H5Dopen(m_parent_id, name.c_str(), H5P_DEFAULT));
    }
  }
  //----------------------------------------------------------------------------
  dataset(dataset const& other)
      : id_holder{other.id()},
        m_parent_id{other.m_parent_id},
        m_name{other.m_name} {}
  //----------------------------------------------------------------------------
  dataset(dataset&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dataset const& other) -> dataset& {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(m_parent_id);
    //}
    H5Dclose(id());
    set_id(other.id());
    m_parent_id = other.m_parent_id;
    m_name      = other.m_name;
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(dataset&&) noexcept -> dataset& = default;
  //----------------------------------------------------------------------------
  ~dataset() {
    // if (m_parent_id != nullptr) {
    //  H5Fclose(m_parent_id);
    //}
    H5Dclose(id());
  }
  //============================================================================
  auto resize(hsize_t const extent) { H5Dset_extent(id(), &extent); }
  //----------------------------------------------------------------------------
  auto resize(std::vector<hsize_t> const& extent) {
    H5Dset_extent(id(), extent.data());
  }
  //----------------------------------------------------------------------------
  template <std::size_t N>
  auto resize(std::array<hsize_t, N> const& extent) {
    H5Dset_extent(id(), extent.data());
  }
  //----------------------------------------------------------------------------
  template <typename Integral, enable_if_integral<Integral>>
  auto resize(std::vector<Integral> const& extent) {
    resize(std::vector<hsize_t>(begin(extent), end(extent)));
  }
  //----------------------------------------------------------------------------
  template <typename Integral, std::size_t N, enable_if_integral<Integral>>
  auto resize(std::array<Integral, N> const& extent) {
    resize(std::array<hsize_t, N>(begin(extent), end(extent)));
  }
  //----------------------------------------------------------------------------
  template <typename... Size>
  auto resize(Size const... size) {
    resize(std::array{static_cast<hsize_t>(size)...});
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
    bool must_resize = true;
    for (std::size_t i = 0; i < cur_res.size(); ++i) {
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
  auto write(hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id,
             const void* buf) const -> void {
    H5Dwrite(id(), type_id<T>(), mem_space_id, file_space_id, xfer_plist_id, buf);
  }
  //----------------------------------------------------------------------------
  auto write(T const* data) -> void {
    write(H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto write(std::vector<T> const& data) -> void {
    resize_if_necessary(data.size());
    write(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
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
  template <std::size_t N>
  auto write(std::array<T, N> const& data) -> void {
    resize_if_necessary(data.size());
    write(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //  template <range Range>
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
             std::vector<std::size_t> const&              offset) -> void {
    assert(data.num_dimensions() == offset.size());
    auto const size       = data.size();
    auto       total_size = size;
    for (std::size_t i = 0; i < size.size(); ++i) {
      total_size[i] += offset[i];
    }
    resize_if_necessary(total_size);
    write(data.data_ptr(), std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(begin(size), end(size)));
  }
//  //----------------------------------------------------------------------------
//  template <typename IndexOrder, integral... Is>
//  auto write(dynamic_multidim_array<T, IndexOrder> const& data, Is const...
//  is)
//      -> void {
//    auto const s = data.size();
//    write<IndexOrder>(data.data_ptr(),
//                      std::vector<hsize_t>{static_cast<hsize_t>(is)...},
//                      std::vector<hsize_t>(begin(s), end(s)));
//  }
//----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest, integral... Is>
  auto write(T const& data, Is const... is) -> void {
    write<IndexOrder>(&data, std::vector<hsize_t>{static_cast<hsize_t>(is)...},
                      std::vector<hsize_t>(sizeof...(Is), 1));
  }
  //----------------------------------------------------------------------------
  auto write(T const& data, std::vector<std::size_t> const& offset) -> void {
    write(&data, std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(size(offset), 1));
  }
  //----------------------------------------------------------------------------
  auto write(T const& data, std::vector<hsize_t> offset) -> void {
    write(&data, std::move(offset), std::vector<hsize_t>(size(offset), 1));
  }
  //----------------------------------------------------------------------------
  auto write(std::vector<T> const& data, std::vector<std::size_t> const& offset,
             std::vector<std::size_t> const& count) -> void {
    write(data.data(), std::vector<hsize_t>(begin(offset), end(offset)),
          std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  auto write(T const* data, std::vector<std::size_t> const& offset,
             std::vector<std::size_t> const& count) -> void {
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
    dataset_space.select_hyperslab(offset, count);
    auto memory_space = hdf5::dataspace{count};
    write(memory_space.id(), dataset_space.id(), H5P_DEFAULT,
          data);
  }
  //============================================================================
  auto read(hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id,
            T* buf) const -> void {
    H5Dread(id(), type_id<T>(), mem_space_id, file_space_id, xfer_plist_id,
            buf);
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
    bool       must_resize   = (unsigned int)rank != arr.num_dimensions();
    if (!must_resize) {
      for (int i = 0; i < rank; ++i) {
        if (arr.size(i) != size[i]) {
          must_resize = true;
          break;
        }
      }
    }
    if (must_resize) {
      arr.resize(std::vector<std::size_t>(begin(size), end(size)));
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

    auto dataset_space = dataspace();
    dataset_space.select_hyperslab(offset, count);

    auto memory_space = hdf5::dataspace{count};
    read(memory_space.id(), dataset_space.id(), H5P_DEFAULT, data.data());
  }
  //----------------------------------------------------------------------------
  auto read(hsize_t const offset, hsize_t const count,
            std::vector<T>& data) const {
    assert(count == data.size());
    auto dataset_space = dataspace();
    dataset_space.select_hyperslab(offset, count);

    auto memory_space = hdf5::dataspace{count};
    read(memory_space.id(), dataset_space.id(), H5P_DEFAULT, data.data());
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
    std::size_t num_entries =
        boost::accumulate(dataspace().current_resolution(), std::size_t(1),
                          std::multiplies<std::size_t>{});
    if (data.size() != num_entries) {
      data.resize(num_entries);
    }

    read(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto read(std::vector<std::size_t> const&        offset,
            std::vector<std::size_t> const&        count,
            dynamic_multidim_array<T, IndexOrder>& arr) const {
    read(std::vector<hsize_t>(begin(offset), end(offset)),
         std::vector<hsize_t>(begin(count), end(count)), arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read(std::vector<std::size_t> const& offset,
            std::vector<std::size_t> const& count) const {
    return read<IndexOrder>(std::vector<hsize_t>(begin(offset), end(offset)),
                            std::vector<hsize_t>(begin(count), end(count)));
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto read(std::vector<hsize_t> const& offset,
            std::vector<hsize_t> const& count) const {
    dynamic_multidim_array<T, IndexOrder> arr;
    read(offset, count, arr);
    return arr;
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder>
  auto read(std::vector<hsize_t> offset, std::vector<hsize_t> count,
            dynamic_multidim_array<T, IndexOrder>& arr) const -> auto& {
    assert(offset.size() == count.size());

    std::vector<hsize_t> count_without_ones;

    auto        ds   = dataspace();
    std::size_t rank = 0;
    for (std::size_t i = 0; i < count.size(); ++i) {
      if (count[i] > 1) {
        ++rank;
        count_without_ones.push_back(count[i]);
      }
    }
    std::size_t num_dimensions = 0;
    for (std::size_t i = 0; i < arr.num_dimensions(); ++i) {
      if (arr.size(i) > 1) {
        ++num_dimensions;
      }
    }

    if (rank != num_dimensions) {
      arr.resize(count);
    } else {
      for (std::size_t i = 0; i < rank; ++i) {
        if (arr.size(i) != count_without_ones[i]) {
          arr.resize(count_without_ones);
          break;
        }
      }
    }
    ds.select_hyperslab(offset, count);
    auto memory_space = hdf5::dataspace{count};
    read(memory_space.id(), ds.id(), H5P_DEFAULT, arr.data_ptr());
    return arr;
  }
  //----------------------------------------------------------------------------
  auto read(integral auto const... is) const {
    auto offset = std::vector<hsize_t>{static_cast<hsize_t>(is)...};
    auto count  = std::vector<hsize_t>(sizeof...(is), 1);

    auto ds = dataspace();
    ds.select_hyperslab(offset, count);
    auto ms = hdf5::dataspace{count};
    T    data;
    read(ms.id(), ds.id(), H5P_DEFAULT, &data);
    return data;
  }
  //----------------------------------------------------------------------------
  auto operator[](hsize_t const i) const { return read(i); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const {
    auto dataset_space = H5Dget_space(id());
    auto ndims         = H5Sget_simple_extent_ndims(dataset_space);
    H5Sclose(dataset_space);
    return ndims;
  }
  //----------------------------------------------------------------------------
  auto read_lazy(std::vector<std::size_t> const& chunk_size) {
    return lazy_reader<this_t>{*this, chunk_size};
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& { return m_name; }
  //----------------------------------------------------------------------------
  auto dataspace() const { return hdf5::dataspace{H5Dget_space(id())}; }
  auto flush() { H5Dflush(id()); }
  //----------------------------------------------------------------------------
  auto size() const { return dataspace().current_resolution(); }
};
//==============================================================================
template <typename IDHolder>
struct dataset_creator {
  auto as_id_holder() -> auto& { return *static_cast<IDHolder*>(this); }
  auto as_id_holder() const -> auto const& {
    return *static_cast<IDHolder const*>(this);
  }
  template <typename T, typename IndexOrder = x_fastest>
  auto create_dataset(std::string const& name, integral auto const... size) {
    return hdf5::dataset<T>{as_id_holder().id(), name, size...};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  [[nodiscard]] auto dataset(char const* name) const {
    return hdf5::dataset<T>{as_id_holder().id(), name};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  [[nodiscard]] auto dataset(std::string const& name) const {
    return dataset<T>(name.c_str());
  }
};
//==============================================================================
struct group;
//==============================================================================
template <typename IDHolder>
struct group_creator {
  auto as_id_holder() -> auto& { return *static_cast<IDHolder*>(this); }
  auto as_id_holder() const -> auto const& {
    return *static_cast<IDHolder const*>(this);
  }
  //============================================================================
  auto group(char const* name) {
    return hdf5::group{as_id_holder().id(), name};
  }
  //----------------------------------------------------------------------------
  auto group(std::string const& name) { return group(name.data()); }
};
//==============================================================================
template <typename IDHolder>
struct node : id_holder,
              dataset_creator<IDHolder>,
              attribute_creator<IDHolder>,
              group_creator<IDHolder> {
  node(hid_t const id) : id_holder{id} {}
};
//==============================================================================
struct group : node<group> {
 public:
  using this_t = group;

 private:
  std::string m_name;
  //============================================================================
 public:
  group(hid_t const parent_id, char const* name)
      : node{H5Gopen(parent_id, name, H5P_DEFAULT)}, m_name{name} {
    if (id() < 0) {
      set_id(H5Gcreate(parent_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }
  }
  //----------------------------------------------------------------------------
  group(hid_t const parent_id, std::string const& name)
      : group{parent_id, name.c_str()} {}
  //----------------------------------------------------------------------------
  group(group&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(group const& other) -> group& {
    H5Gclose(id());
    set_id(other.id());
    m_name = other.m_name;
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(group&&) noexcept -> group& = default;
  //----------------------------------------------------------------------------
  ~group() { H5Gclose(id()); }
  //============================================================================
  auto sub_group(char const* name) { return node<group>::group(name); }
  auto sub_group(std::string const& name) { return node<group>::group(name); }
};
//==============================================================================
struct file : node<file> {
 public:
  explicit file(filesystem::path const& path) : file{path.c_str()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit file(std::string const& path) : file{path.data()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit file(char const* path) : node{-1} {
    api::disable_error_printing();
    open(path);
  }
  //----------------------------------------------------------------------------
  file(file const& other) : node{other.id()} {}
  //----------------------------------------------------------------------------
  file(file&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(file const& other) -> file& {
    H5Fclose(id());
    set_id(other.id());
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(file&&) noexcept -> file& = default;
  //----------------------------------------------------------------------------
  ~file() { H5Fclose(id()); }

 private:
  auto open(char const* path) -> void {
    if (filesystem::exists(filesystem::path{path})) {
      set_id(H5Fopen(path, H5F_ACC_RDWR, H5P_DEFAULT));
    } else {
      set_id(H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT));
    }
  }

 public:
  //----------------------------------------------------------------------------
  auto node_exists(char const* name,
                   hid_t link_access_property_list_id = H5P_DEFAULT) const {
    return H5Lexists(id(), name, link_access_property_list_id);
  }
  //----------------------------------------------------------------------------
  auto node_exists(std::string const& name,
                   hid_t link_access_property_list_id = H5P_DEFAULT) const {
    return node_exists(name.c_str(), link_access_property_list_id);
  }
};
//==============================================================================
}  // namespace tatooine::hdf5
//==============================================================================
#endif
//#else
//#pragma message(including <tatooine / hdf5.h> without HDF5 support.)
//#endif
