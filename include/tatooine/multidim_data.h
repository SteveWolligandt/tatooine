#ifndef TATOOINE_MULTIDIM_DATA_H
#define TATOOINE_MULTIDIM_DATA_H
//==============================================================================
#include <array>
#include <vector>

#include "multidim_index.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t... Dims>
struct static_multidim_data {
  
};

//==============================================================================
template <typename T, typename Indexing>
class dynamic_multidim_array {
  //============================================================================
  // members
  //============================================================================
  std::vector<size_t> m_resolution;
  std::vector<T>      m_data;
  //============================================================================
  // ctors
  //============================================================================
 public:
  dynamic_multidim_array()                                    = default;
  dynamic_multidim_array(const dynamic_multidim_array& other) = default;
  dynamic_multidim_array(dynamic_multidim_array&& other)      = default;
  dynamic_multidim_array& operator=(const dynamic_multidim_array& other) =
      default;
  dynamic_multidim_array& operator=(dynamic_multidim_array&& other) = default;
  //============================================================================
  dynamic_multidim_array(const std::vector<size_t>& p_resolution)
      : m_resolution{p_resolution},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               size_t{1}, std::multiplies<size_t>{})) {}
  //----------------------------------------------------------------------------
  dynamic_multidim_array(const std::vector<size_t>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               size_t{1}, std::multiplies<size_t>{})) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  dynamic_multidim_array(const std::array<size_t, NDims>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               size_t{1}, std::multiplies<size_t>{})) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  dynamic_multidim_array(const std::array<size_t, NDims>& p_resolution)
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               size_t{1}, std::multiplies<size_t>{})) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  dynamic_multidim_array(const std::array<size_t, NDims>& p_resolution,
              const T&                          initial = T{})
      : m_resolution{begin(p_resolution), end(p_resolution)},
        m_data(std::accumulate(begin(p_resolution), end(p_resolution),
                               size_t{1}, std::multiplies<size_t>{}),
               initial) {}
  //----------------------------------------------------------------------------
  dynamic_multidim_array(const std::vector<size_t>& p_resolution,
              const std::vector<T>&       p_data)
      : m_resolution{begin(p_resolution), end(p_resolution)}, m_data(p_data) {}
  //----------------------------------------------------------------------------
  template <size_t NDims>
  dynamic_multidim_array(const std::array<size_t, NDims>& p_resolution,
              const std::vector<T>&             p_data)
      : m_resolution{begin(p_resolution), end(p_resolution)}, m_data(p_data) {}
  //============================================================================
  // methods
  //============================================================================
  template <typename... Is, size_t... Cs>
  const auto& at(std::index_sequence<Cs...>, Is... p_is) const {
    assert(sizeof...(Is) == num_dims());
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
    assert(sizeof...(Is) == num_dims());
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
    assert(sizeof...(Is) == num_dims());
    return at(std::make_index_sequence<sizeof...(Is)>{}, is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  auto& operator()(Is... is) {
    assert(sizeof...(Is) == num_dims());
    return at(std::make_index_sequence<sizeof...(Is)>{}, is...);
  }
  //----------------------------------------------------------------------------
  auto&       operator[](size_t i) { return m_data[i]; }
  const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  auto num_elements() const {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t{1},
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
  auto num_dims() const { return m_resolution.size(); }
  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution; }
  //----------------------------------------------------------------------------
  auto resolution(size_t i) const { return m_resolution[i]; }
  //----------------------------------------------------------------------------
  T*       data() { return m_data.data(); }
  const T* data() const { return m_data.data(); }
  //----------------------------------------------------------------------------
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
      writer.write_point_data(num_elements());

      writer.write_scalars(data_name, m_data);
      writer.close();
    }
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
