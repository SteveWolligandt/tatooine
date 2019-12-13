#ifndef TATOOINE_MULTIDIM_ARRAY_H
#define TATOOINE_MULTIDIM_ARRAY_H
//==============================================================================
#include <array>
#include <vector>

#include "multidim_resolution.h"
#include "random.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Indexing, size_t... Resolution>
struct static_multidim_array {
  //============================================================================
  // typedefs
  //============================================================================
  using res_t = static_multidim_resolution<Indexing, Resolution...>;

  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_elements() { return res_t::num_elements(); }
  static constexpr auto num_dimensions() { return res_t::num_dimensions(); }
  static constexpr auto resolution() { return res_t::resolution(); }
  static constexpr auto size(size_t i) { return res_t::size(i); }

  //============================================================================
  // members
  //============================================================================
 private:
  std::array<T, num_elements()> m_data;

  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr static_multidim_array(const static_multidim_array& other) = default;
  constexpr static_multidim_array(static_multidim_array&& other)      = default;
  constexpr static_multidim_array& operator=(
      const static_multidim_array& other) = default;
  constexpr static_multidim_array& operator=(static_multidim_array&& other) =
      default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  constexpr static_multidim_array(
      const static_multidim_array<OtherT, OtherIndexing, Resolution...>& other)
      : m_data(make_array<T, num_elements()>()) {
    for (auto is : multidim({0, Resolution}...)) { at(is) = other(is); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  constexpr static_multidim_array& operator=(
      const static_multidim_array<OtherT, OtherIndexing, Resolution...>&
          other) {
    for (auto is : tatooine::multidim{{size_t(0), Resolution}...}) {
      at(is) = other(is);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... Ts,
            std::enable_if_t<sizeof...(Ts) == num_elements(), bool> = true>
  constexpr static_multidim_array(Ts&&... ts) : m_data{static_cast<T>(ts)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array(const T& initial = T{})
      : m_data(make_array<T, num_elements()>(initial)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static_multidim_array(const std::vector<T>& data)
      : m_data(begin(data), end(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array(const std::array<T, num_elements()>& data)
      : m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array(std::array<T, num_elements()>&& data)
      : m_data(std::move(data)) {}
  //============================================================================
  // methods
  //============================================================================
 public:
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr const auto& at(Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(res_t::in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto& at(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(res_t::in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  constexpr const auto& at(const std::array<size_t, num_dimensions()>& is) const {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  constexpr auto& at(const std::array<size_t, num_dimensions()>& is) {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr const auto& operator()(Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(res_t::in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto& operator()(Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    assert(res_t::in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  constexpr const auto& operator()(
      const std::array<size_t, num_dimensions()>& is) const {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto& operator()(const std::array<size_t, num_dimensions()>& is) {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_data[i]; }
  constexpr const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  constexpr T*       data() { return m_data.data(); }
  constexpr const T* data() const { return m_data.data(); }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto plain_idx(Is... is) const {
    return res_t::plain_idx(is...);
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64, typename _T = T,
            enable_if_arithmetic<_T> = true>
  void randu(T min = 0, T max = 1,
             RandEng&& eng = RandEng{std::random_device{}()}) {
    random_uniform<T, RandEng> rand(min, max, eng);
    std::generate(begin(m_data), end(m_data),
                  [&rand, &eng] { return rand(eng); });
  }
  //----------------------------------------------------------------------------
  auto indices() const {
    return res_t::indices();
  }
};

//==============================================================================
template <typename T, typename Indexing>
class dynamic_multidim_array {
  //============================================================================
  // members
  //============================================================================
  dynamic_multidim_resolution<Indexing> m_resolution;
  std::vector<T>                        m_data;
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
  ////----------------------------------------------------------------------------
  //template <typename OtherT, typename OtherIndexing>
  //constexpr dynamic_multidim_array(
  //    const dynamic_multidim_array<OtherT, OtherIndexing>& other)
  //    : m_resolution{other.m_resolution} {
  //  for (auto is : multidim(m_resolution)) { at(is) = other(is); }
  //}
  ////----------------------------------------------------------------------------
  //template <typename OtherT, typename OtherIndexing>
  //constexpr dynamic_multidim_array& operator=(
  //    const dynamic_multidim_array<OtherT, OtherIndexing>&
  //        other) {
  //  m_resolution = other.m_resolution;
  //  for (auto is : multidim(m_resolution)) { at(is) = other(is); }
  //  return *this;
  //}
  //============================================================================
  dynamic_multidim_array(const std::vector<size_t>& resolution,
                         const T&                   initial = T{})
      : m_resolution{resolution},
        m_data(m_resolution.num_elements(), initial) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  dynamic_multidim_array(const std::vector<size_t>& resolution,
                         const std::vector<T>&      data)
      : m_resolution{begin(resolution), end(resolution)}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  dynamic_multidim_array(const std::vector<size_t>& resolution,
                         std::vector<T>&&           data)
      : m_resolution{begin(resolution), end(resolution)},
        m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <size_t N>
  dynamic_multidim_array(const std::array<size_t, N>& resolution,
                         const T&                     initial = T{})
      : m_resolution{resolution},
        m_data(m_resolution.num_elements(), initial) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  dynamic_multidim_array(const std::array<size_t, N>& resolution,
                         const std::vector<T>&        data)
      : m_resolution{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  dynamic_multidim_array(const std::array<size_t, N>& resolution,
                         std::vector<T>&&             data)
      : m_resolution{resolution}, m_data(std::move(data)) {}
  //============================================================================
  // methods
  //============================================================================
  template <typename... Is, enable_if_integral<Is...> = true>
  auto& at(Is... is) {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& at(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto& operator()(Is... is) {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& operator()(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto&       operator[](size_t i) { return m_data[i]; }
  const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  template <typename... Resolution>
  void resize(Resolution... resolution) {
    m_resolution.resize(resolution...);
    m_data.resize(num_elements());
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto plain_idx(Is... is) const {
    return m_resolution.plain_idx(is...);
  }
  //----------------------------------------------------------------------------
  auto num_elements() const { return m_resolution.num_elements(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_resolution.num_dimensions(); }
  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution.resolution(); }
  //----------------------------------------------------------------------------
  auto size(size_t i) const { return m_resolution.size[i]; }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto in_range(Is... is) const {
    return m_resolution.in_range(is...);
  }
  //----------------------------------------------------------------------------
  T*       data() { return m_data.data(); }
  const T* data() const { return m_data.data(); }
};
//==============================================================================
// deduction guides
//==============================================================================
#if has_cxx17_support()
template <typename T, typename Indexing>
dynamic_multidim_array(const dynamic_multidim_array<T, Indexing>&)
    ->dynamic_multidim_array<T, Indexing>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename Indexing>
dynamic_multidim_array(dynamic_multidim_array<T, Indexing> &&)
    ->dynamic_multidim_array<T, Indexing>;
//----------------------------------------------------------------------------
template <typename T>
dynamic_multidim_array(const std::vector<size_t>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
dynamic_multidim_array(const std::vector<size_t>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
dynamic_multidim_array(const std::vector<size_t>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
//----------------------------------------------------------------------------
template <typename T, size_t N>
dynamic_multidim_array(const std::array<size_t, N>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
dynamic_multidim_array(const std::array<size_t, N>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
dynamic_multidim_array(const std::array<size_t, N>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
#endif

////==============================================================================
//#include "vtk_legacy.h"
////==============================================================================
//template <typename T, typename Indexing, size_t... Resolution>
//void write_vtk(const static_multidim_array<T, Indexing, Resolution...>& arr,
//               const std::string& filepath, const vec<double, 3>& origin,
//               const vec<double, 3>& spacing,
//               const std::string&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    writer.write_dimensions(m_resolution[0], m_resolution[1], m_resolution[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(num_elements());
//
//    writer.write_scalars(data_name, m_data);
//    writer.close();
//  }
//}
////------------------------------------------------------------------------------
//template <typename T, typename Indexing>
//void write_vtk(const dynamic_multidim_array<T, Indexing>& arr,
//               const std::string& filepath, const vec<double, 3>& origin,
//               const vec<double, 3>& spacing,
//               const std::string&    data_name = "tatooine data") {
      //writer.set_title("tatooine");
      //writer.write_header();
      //
      //writer.write_dimensions(m_resolution[0], m_resolution[1],
      //                        m_resolution[2]);
      //writer.write_origin(origin(0), origin(1), origin(2));
      //writer.write_spacing(spacing(0), spacing(1), spacing(2));
      //writer.write_point_data(num_elements());
      //
      //writer.write_scalars(data_name, m_data);
      //writer.close();
//  }
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
