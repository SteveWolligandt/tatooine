#ifndef TATOOINE_HDF5_H
#define TATOOINE_HDF5_H
//==============================================================================
#include <H5Cpp.h>

#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "tensor.h"
#include "linspace.h"
#include "vtk_legacy.h"
//==============================================================================
namespace tatooine::h5 {
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
  dataset(const std::shared_ptr<H5::H5File>& p_file, const char* p_dataset_name)
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
    dynamic_multidim_array<T, x_slowest> mem{resolution()};
    if constexpr (std::is_floating_point_v<T>) {
      m_dataset.read(mem.data_ptr(), m_dataset.getFloatType());
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
    dynamic_multidim_array<T, x_slowest> data{p_resolution};

    H5::DataSpace chunkspace(NDims, p_resolution.data());
    auto          filespace = m_dataset.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, p_resolution.data(), p_offset.data());
    m_dataset.read(data.data_ptr(), m_dataset.getFloatType(), chunkspace,
                   filespace);
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename T, size_t NDims>
  auto read_chunk(const std::array<size_t, NDims>& p_offset,
                  const std::array<size_t, NDims>& p_resolution) const {
    auto casted_offset = make_array<hsize_t, NDims>();
    auto casted_resolution = make_array<hsize_t, NDims>();

    for (size_t i = 0; i < NDims; ++i) {
      casted_offset[i]     = static_cast<hsize_t>(p_offset[i]);
      casted_resolution[i] = static_cast<hsize_t>(p_resolution[i]);
    }
    return read_chunk<T>(casted_offset, casted_resolution);
  }
};
//==============================================================================
struct attribute {
  std::shared_ptr<H5::H5File> m_file;
  H5::Attribute               m_attr;
  //============================================================================
  attribute(const std::shared_ptr<H5::H5File>& p_file, const H5::Group& p_group,
            const std::string& p_attribute_name)
      : m_file{p_file}, m_attr{p_group.openAttribute(p_attribute_name)} {}
  //----------------------------------------------------------------------------
  attribute(const std::shared_ptr<H5::H5File>& p_file, const H5::Group& p_group,
            const std::string_view& p_attribute_name)
      : m_file{p_file},
        m_attr{p_group.openAttribute(p_attribute_name.data())} {}
  //----------------------------------------------------------------------------
  attribute(const std::shared_ptr<H5::H5File>& p_file, const H5::Group& p_group,
            const char*& p_attribute_name)
      : m_file{p_file}, m_attr{p_group.openAttribute(p_attribute_name)} {}
  //----------------------------------------------------------------------------
  template <typename T>
  auto read() {
    dynamic_multidim_array<T, x_slowest> mem(resolution());
    m_attr.read(m_attr.getDataType(), mem.data());
    return mem;
  }
  //----------------------------------------------------------------------------
  auto resolution() const {
    auto                 dataspace = m_attr.getSpace();
    std::vector<hsize_t> dims(dataspace.getSimpleExtentNdims());
    dataspace.getSimpleExtentDims(dims.data(), nullptr);
    return dims;
  }
  //----------------------------------------------------------------------------
  auto number_of_elements() const {
    auto dims = resolution();
    return std::accumulate(begin(dims), end(dims), size_t{1},
                           std::multiplies<size_t>{});
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
  group(const std::shared_ptr<H5::H5File>& p_file, const char* p_group_name)
      : m_file{p_file}, m_group{m_file->openGroup(p_group_name)} {}
  //----------------------------------------------------------------------------
  auto attribute(const std::string& attr_name) const {
    return h5::attribute{m_file, m_group, attr_name};
  }
  //----------------------------------------------------------------------------
  auto attribute(const std::string_view& attr_name) const {
    return h5::attribute{m_file, m_group, attr_name};
  }
  //----------------------------------------------------------------------------
  auto attribute(const char* attr_name) const {
    return h5::attribute{m_file, m_group, attr_name};
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
    return h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto dataset(const std::string_view& dataset_name) const {
    return h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto dataset(const char* dataset_name) const {
    return h5::dataset(m_file, dataset_name);
  }
  //----------------------------------------------------------------------------
  auto group(const std::string& group_name) const {
    return h5::group(m_file, group_name);
  }
  //----------------------------------------------------------------------------
  auto group(const std::string_view& group_name) const {
    return h5::group(m_file, group_name);
  }
  //----------------------------------------------------------------------------
  auto group(const char* group_name) const {
    return h5::group(m_file, group_name);
  }
};
//==============================================================================
}  // namespace tatooine::h5
//==============================================================================
#endif
