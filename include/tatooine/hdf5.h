#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H

#include <H5Cpp.h>

#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "tensor.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine::h5 {
//==============================================================================
template <typename T>
struct multi_array {
  std::vector<hsize_t> m_resolution;
  std::vector<T>  m_data;
  //============================================================================
  multi_array()                                    = default;
  multi_array(const multi_array& other)            = default;
  multi_array(multi_array&& other)                 = default;
  multi_array& operator=(const multi_array& other) = default;
  multi_array& operator=(multi_array&& other)      = default;
  //============================================================================
  multi_array(const std::vector<hsize_t>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution), hsize_t{1},
                               std::multiplies<hsize_t>{}),
               0) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  multi_array(const std::array<hsize_t, NDims>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution), hsize_t{1},
                               std::multiplies<hsize_t>{}),
               0) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  multi_array(const std::array<hsize_t, NDims>& p_resolution,
              const T&                          initial = T{})
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               hsize_t{1}, std::multiplies<hsize_t>{}),
               initial) {}
  //----------------------------------------------------------------------------
  multi_array(const std::vector<hsize_t>& p_resolution,
              const std::vector<T>&  p_data)
      : m_resolution{begin(p_resolution), end(p_resolution)}, m_data(p_data) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  multi_array(const std::array<hsize_t, NDims>& p_resolution,
              const std::vector<T>&        p_data)
      : m_resolution{begin(p_resolution), end(p_resolution)}, m_data(p_data) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  multi_array(const H5::DataSet&                dataset,
              const std::array<hsize_t, NDims>& p_offset,
              const std::array<hsize_t, NDims>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution), hsize_t{1},
                               std::multiplies<hsize_t>{})) {
    H5::DataSpace chunkspace(number_of_dimensions(), m_resolution.data());
    auto          filespace = dataset.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, m_resolution.data(), p_offset.data());
    dataset.read(m_data.data(), dataset.getFloatType(), chunkspace, filespace);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, size_t... Cs>
  const auto& at(std::index_sequence<Cs...>, Is... p_is) const {
    assert(sizeof...(Is) == number_of_dimensions());
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < is.size(); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      multiplier *= m_resolution[is.size() - 1 - i];
    }
    return m_data[idx];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, size_t... Cs>
  auto& at(std::index_sequence<Cs...>, Is... p_is) {
    assert(sizeof...(Is) == number_of_dimensions());
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < is.size(); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      multiplier *= m_resolution[is.size() - 1 - i];
    }
    return m_data[idx];
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  const auto& operator()(Is... is) const {
    assert(sizeof...(Is) == number_of_dimensions());
    return at(std::make_index_sequence<sizeof...(Is)>{}, is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  auto& operator()(Is... is) {
    assert(sizeof...(Is) == number_of_dimensions());
    return at(std::make_index_sequence<sizeof...(Is)>{}, is...);
  }
  //----------------------------------------------------------------------------
  auto& operator[](size_t i) { return m_data[i]; }
  const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  auto number_of_elements() const {
    return std::accumulate(begin(m_resolution), end(m_resolution), hsize_t{1},
                           std::multiplies<hsize_t>{});
  }
  //----------------------------------------------------------------------------
  auto number_of_dimensions() const { return m_resolution.size(); }
  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution; }
  //----------------------------------------------------------------------------
  auto resolution(size_t i) const { return m_resolution[i]; }
  //----------------------------------------------------------------------------
  T* data() {
    return m_data.data();
  }
  const T* data() const { return m_data.data(); }
  void     write_vtk(const std::string& filepath, const vec<double, 3>& origin,
                     const vec<double, 3>& spacing,
                     const std::string&    data_name = "tatooine data") {
    vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
    if (writer.is_open()) {
      writer.set_title("tatooine");
      writer.write_header();

      writer.write_dimensions(m_resolution[0], m_resolution[1],
                              m_resolution[2]);
      writer.write_origin(origin(0), origin(1), origin(2));
      writer.write_spacing(spacing(0), spacing(1), spacing(2));
      writer.write_point_data(number_of_elements());

      writer.write_scalars(data_name, m_data);
      writer.close();
    }
  }
};
//==============================================================================
struct dataset {
  std::shared_ptr<H5::H5File> m_file;
  H5::DataSet                 m_dataset;

  //----------------------------------------------------------------------------
  dataset(const std::shared_ptr<H5::H5File>& p_file,
          const std::string&                 p_dataset_name)
      : m_file{p_file}, m_dataset{m_file->openDataSet(p_dataset_name)} {}
  //----------------------------------------------------------------------------
  dataset(const std::shared_ptr<H5::H5File>& p_file,
          const std::string_view&            p_dataset_name)
      : m_file{p_file}, m_dataset{m_file->openDataSet(p_dataset_name.data())} {}
  //----------------------------------------------------------------------------
  dataset(const std::shared_ptr<H5::H5File>& p_file,
          const char*            p_dataset_name)
      : m_file{p_file}, m_dataset{m_file->openDataSet(p_dataset_name)} {}
  //----------------------------------------------------------------------------
  auto resolution() const {
    auto                 dataspace = m_dataset.getSpace();
    std::vector<hsize_t> dims(dataspace.getSimpleExtentNdims());
    dataspace.getSimpleExtentDims(dims.data(), nullptr);
    return dims;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read() const {
    multi_array<T> mem{resolution()};
    if constexpr (std::is_floating_point_v<T>) {
      m_dataset.read(mem.data(), m_dataset.getFloatType());
    }
    if constexpr (std::is_integral_v<T>) {
      m_dataset.read(mem.data(), m_dataset.getIntType());
    }
    return mem;
  }
  //----------------------------------------------------------------------------
  template <typename T, size_t NDims>
  auto read_chunk(const std::array<hsize_t, NDims>& p_offset,
                  const std::array<hsize_t, NDims>& p_resolution) const {
    return multi_array<T>{m_dataset, p_offset, p_resolution};
  }
};
//==============================================================================
struct attribute {
  std::shared_ptr<H5::H5File> m_file;
  H5::Attribute               m_attr;
  //============================================================================
  attribute(const std::shared_ptr<H5::H5File>& p_file,
            const H5::Group& p_group,
            const std::string& p_attribute_name)
      : m_file{p_file}, m_attr{p_group.openAttribute(p_attribute_name)} {}
  //----------------------------------------------------------------------------
  attribute(const std::shared_ptr<H5::H5File>& p_file,
            const H5::Group& p_group,
            const std::string_view& p_attribute_name)
      : m_file{p_file}, m_attr{p_group.openAttribute(p_attribute_name.data())} {}
  //----------------------------------------------------------------------------
  attribute(const std::shared_ptr<H5::H5File>& p_file,
            const H5::Group& p_group,
            const char*& p_attribute_name)
      : m_file{p_file}, m_attr{p_group.openAttribute(p_attribute_name)} {}
  //----------------------------------------------------------------------------
  template <typename T>
  auto read() {
    multi_array<T> mem(resolution());
    m_attr.read(m_attr.getDataType(), mem.data());
    return mem;
  }
  //----------------------------------------------------------------------------
  std::vector<hsize_t> resolution() const {
    auto                 dataspace = m_attr.getSpace();
    std::vector<hsize_t> dims(dataspace.getSimpleExtentNdims());
    dataspace.getSimpleExtentDims(dims.data(), nullptr);
    return dims;
  }
  //----------------------------------------------------------------------------
  auto number_of_elements() const {
    auto           dims   = resolution();
    return std::accumulate(begin(dims), end(dims), hsize_t{1},
                           std::multiplies<hsize_t>{});
  }
};
//==============================================================================
struct group {
  std::shared_ptr<H5::H5File> m_file;
  H5::Group                   m_group;
  //============================================================================
  group(const std::shared_ptr<H5::H5File>& p_file,
        const std::string&                 p_group_name)
      : m_file{p_file}, m_group{m_file->openGroup(p_group_name)} {}
  //----------------------------------------------------------------------------
  group(const std::shared_ptr<H5::H5File>& p_file,
        const std::string_view&            p_group_name)
      : m_file{p_file}, m_group{m_file->openGroup(p_group_name.data())} {}
  //----------------------------------------------------------------------------
  group(const std::shared_ptr<H5::H5File>& p_file,
        const char*            p_group_name)
      : m_file{p_file}, m_group{m_file->openGroup(p_group_name)} {}
  //----------------------------------------------------------------------------
  auto attribute(const std::string& attr_name) const {
    return tatooine::h5::attribute{m_file, m_group, attr_name};
  }
  //----------------------------------------------------------------------------
  auto attribute(const std::string_view& attr_name) const {
    return tatooine::h5::attribute{m_file, m_group, attr_name};
  }
  //----------------------------------------------------------------------------
  auto attribute(const char* attr_name) const {
    return tatooine::h5::attribute{m_file, m_group, attr_name};
  }
};
//==============================================================================
struct ifile {
  std::shared_ptr<H5::H5File> m_file;
  //----------------------------------------------------------------------------
  ifile(const std::string& p_filepath) : ifile{p_filepath.data()} {}
  //----------------------------------------------------------------------------
  ifile(const std::string_view& p_filepath) : ifile{p_filepath.data()} {}
  //----------------------------------------------------------------------------
  ifile(const char* p_filepath)
      : m_file{std::make_shared<H5::H5File>(p_filepath, H5F_ACC_RDONLY)} {}
  //----------------------------------------------------------------------------
  auto dataset(const std::string& dataset_name) const {
    return tatooine::h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto dataset(const std::string_view& dataset_name) const {
    return tatooine::h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto dataset(const char* dataset_name) const {
    return tatooine::h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto group(const std::string& group_name) const {
    return tatooine::h5::group(m_file, group_name);
  }
  //----------------------------------------------------------------------------
  auto group(const std::string_view& group_name) const {
    return tatooine::h5::group(m_file, group_name);
  }
  //----------------------------------------------------------------------------
  auto group(const char* group_name) const {
    return tatooine::h5::group(m_file, group_name);
  }
};

//==============================================================================
}  // namespace tatooine::h5
//==============================================================================

#endif
