#ifndef TATOOINE_MULTIDIM_ARRAY_H
#define TATOOINE_MULTIDIM_ARRAY_H
//==============================================================================
#include <array>
#include <vector>

#include "multidim_resolution.h"
#include "random.h"
#include "linspace.h"
//==============================================================================
namespace tatooine {
//==============================================================================
struct heap {};
struct stack {};

template <typename T, typename Indexing, typename MemLoc, size_t... Resolution>
struct static_multidim_array {
  //============================================================================
  // assertions
  //============================================================================
  static_assert(std::is_same_v<MemLoc, heap> || std::is_same_v<MemLoc, stack>,
                "MemLoc must either be tatooine::heap or tatooine::stack");
  //============================================================================
  // typedefs
  //============================================================================
  using res_t  = static_multidim_resolution<Indexing, Resolution...>;
  using container_t =
      std::conditional_t<std::is_same_v<MemLoc, stack>,
                         std::array<T, res_t::num_elements()>, std::vector<T>>;

  //============================================================================
  // static methods
  //============================================================================
 public:
  static constexpr auto num_elements() { return res_t::num_elements(); }
  static constexpr auto num_dimensions() { return res_t::num_dimensions(); }
  static constexpr auto resolution() { return res_t::resolution(); }
  static constexpr auto size(size_t i) { return res_t::size(i); }

 private:
  static constexpr auto init_data(T init = T{}) {
    if constexpr (std::is_same_v<MemLoc, stack>) {
      return make_array<T, num_elements()>(init);
    } else {
      return std::vector(num_elements(), init);
    }
  }
   //============================================================================
   // members
   //============================================================================
  private:
   container_t m_data;

   //============================================================================
   // ctors
   //============================================================================
  public:
   constexpr static_multidim_array(const static_multidim_array& other) =
       default;
   constexpr static_multidim_array(static_multidim_array&& other) = default;
   constexpr static_multidim_array& operator                      =(
       const static_multidim_array& other) = default;
   constexpr static_multidim_array& operator=(static_multidim_array&& other) =
       default;
   //----------------------------------------------------------------------------
   template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
   constexpr static_multidim_array(
       const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                   Resolution...>& other)
       : m_data(init_data()) {
     for (auto is : multidim({0, Resolution}...)) { at(is) = other(is); }
   }
  //----------------------------------------------------------------------------
   template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
   constexpr static_multidim_array& operator=(
       const static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                   Resolution...>& other) {
     for (auto is : tatooine::static_multidim{Resolution...}) {
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
      : m_data(init_data(initial)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static_multidim_array(const std::vector<T>& data)
      : m_data(begin(data), end(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array(const std::array<T, num_elements()>& data)
      : m_data(begin(data), end(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _MemLoc                                       = MemLoc,
            std::enable_if_t<std::is_same_v<_MemLoc, stack>, bool> = true>
  constexpr static_multidim_array(std::array<T, num_elements()>&& data)
      : m_data(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _MemLoc                                      = MemLoc,
            std::enable_if_t<std::is_same_v<_MemLoc, heap>, bool> = true>
  constexpr static_multidim_array(std::vector<T>&& data)
      : m_data(std::move(data)) {
    assert(num_elements() == data.size());
  }
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
  template <typename UInt, enable_if_integral<UInt> = true>
  constexpr const auto& at(const std::array<UInt, num_dimensions()>& is) const {
    return invoke_unpacked(
        [&](auto... is) -> decltype(auto) { return at(is...); }, unpack(is));
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_integral<UInt> = true>
  constexpr auto& at(const std::array<UInt, num_dimensions()>& is) {
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
  template <typename UInt, enable_if_integral<UInt> = true>
  constexpr const auto& operator()(
      const std::array<UInt, num_dimensions()>& is) const {
    return at(is);
  }
  //----------------------------------------------------------------------------
  template <typename UInt, enable_if_integral<UInt> = true>
  constexpr auto& operator()(const std::array<UInt, num_dimensions()>& is) {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto&       operator[](size_t i) { return m_data[i]; }
  constexpr const auto& operator[](size_t i) const { return m_data[i]; }
  //----------------------------------------------------------------------------
  constexpr auto&       data() { return m_data; }
  constexpr const auto& data() const { return m_data; }
  //----------------------------------------------------------------------------
  constexpr T*       data_ptr() { return m_data.data(); }
  constexpr const T* data_ptr() const { return m_data.data(); }
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
  // typedefs
  //============================================================================
  using res_t = dynamic_multidim_resolution<Indexing>;
  using container_t = std::vector<T>;
  //============================================================================
  // members
  //============================================================================
  res_t       m_resolution;
  container_t m_data;
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
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  constexpr dynamic_multidim_array(
      const dynamic_multidim_array<OtherT, OtherIndexing>& other)
      : m_resolution{other.m_resolution} {
    for (auto is : indices()) { at(is) = other(is); }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  constexpr dynamic_multidim_array& operator=(
      const dynamic_multidim_array<OtherT, OtherIndexing>&
          other) {
    if (m_resolution != other.dyn_resolution()) {
      resize(other.resolution());
    }
    m_resolution = other.dyn_resolution();
    for (auto is : indices()) { at(is) = other(is); }
    return *this;
  }
  //============================================================================
  template <typename UInt, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<UInt>& resolution,
                         const T&                   initial = T{})
      : m_resolution{resolution},
        m_data(m_resolution.num_elements(), initial) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<UInt>& resolution,
                         const std::vector<T>&      data)
      : m_resolution{begin(resolution), end(resolution)}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::vector<UInt>& resolution,
                         std::vector<T>&&           data)
      : m_resolution{begin(resolution), end(resolution)},
        m_data(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::array<UInt, N>& resolution,
                         const T&                     initial = T{})
      : m_resolution{resolution},
        m_data(m_resolution.num_elements(), initial) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::array<UInt, N>& resolution,
                         const std::vector<T>&        data)
      : m_resolution{resolution}, m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  dynamic_multidim_array(const std::array<UInt, N>& resolution,
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& at(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_idx(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  auto& at(const std::vector<UInt>& is) {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  const auto& at(const std::vector<UInt>& is) const {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  auto& at(const std::array<UInt, N>& is) {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  const auto& at(const std::array<UInt, N>& is) const {
    assert(N == num_dimensions());
    assert(in_range(is));
    return m_data[plain_idx(is)];
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto& operator()(Is... is) {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  const auto& operator()(Is... is) const {
    assert(sizeof...(Is) == num_dimensions());
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  auto& operator()(const std::vector<UInt>& is) {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  const auto& operator()(const std::vector<UInt>& is) const {
    assert(is.size() == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  auto& operator()(const std::array<UInt, N>& is) {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  const auto& operator()(const std::array<UInt, N>& is) const {
    assert(N == num_dimensions());
    assert(in_range(is));
    return at(is);
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  void resize(const std::vector<UInt>& res, const T value = T{}) {
    m_resolution.resize(res);
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void resize(std::vector<size_t>&& res, const T value = T{}) {
    m_resolution.resize(std::move(res));
    m_data.resize(num_elements(), value);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  void resize(const std::array<UInt, N>& res, const T value = T{}) {
    m_resolution.resize(res);
    m_data.resize(num_elements(), value);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto plain_idx(Is... is) const {
    return m_resolution.plain_idx(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, enable_if_integral<UInt> = true>
  auto plain_idx(const std::vector<UInt>& is) const {
    return m_resolution.plain_idx(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename UInt, size_t N, enable_if_integral<UInt> = true>
  auto plain_idx(const std::array<UInt, N>& is) const {
    return m_resolution.plain_idx(is);
  }
  //----------------------------------------------------------------------------
  auto num_elements() const { return m_resolution.num_elements(); }
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_resolution.num_dimensions(); }
  //----------------------------------------------------------------------------
  const auto& resolution() const { return m_resolution.resolution(); }
  const auto& dyn_resolution() const { return m_resolution; }
  //----------------------------------------------------------------------------
  auto size(size_t i) const { return m_resolution.size[i]; }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  auto in_range(Is... is) const {
    return m_resolution.in_range(is...);
  }
  //----------------------------------------------------------------------------
  constexpr auto&       data() { return m_data; }
  constexpr const auto& data() const { return m_data; }
  //----------------------------------------------------------------------------
  constexpr T*       data_ptr() { return m_data.data(); }
  constexpr const T* data_ptr() const { return m_data.data(); }
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
    return m_resolution.indices();
  }
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
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt>
dynamic_multidim_array(const std::vector<UInt>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
//----------------------------------------------------------------------------
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const T& initial)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, const std::vector<T>&)
    ->dynamic_multidim_array<T, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename UInt, size_t N>
dynamic_multidim_array(const std::array<UInt, N>&, std::vector<T> &&)
    ->dynamic_multidim_array<T, x_fastest>;
#endif

//==============================================================================
template <typename T0, typename T1, typename MemLoc0, typename MemLoc1,
          typename FReal>
auto interpolate(const dynamic_multidim_array<T0, MemLoc0>& arr0,
                 const dynamic_multidim_array<T1, MemLoc1>& arr1,
                 FReal                                      factor) {
  assert(arr0.dyn_resolution() == arr1.dyn_resolution());
  dynamic_multidim_array<promote_t<T0, T1>, MemLoc0> interpolated{arr0};

  // TODO not only 3d
#ifndef NDEBUG
#pragma omp parallel for collapse(3)
#endif
  for (size_t iz = 0; iz < interpolated.resolution(2); ++iz) {
    for (size_t iy = 0; iy < interpolated.resolution(1); ++iy) {
      for (size_t ix = 0; ix < interpolated.resolution(0); ++ix) {
        interpolated(ix, iy, iz) =
            interpolated(ix, iy, iz) * (1 - factor) + arr1(ix, iy, iz) * factor;
      }
    }
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename MemLoc0, typename MemLoc1,
          typename LinReal, typename TReal>
auto interpolate(const dynamic_multidim_array<T0, MemLoc0>& arr0,
                 const dynamic_multidim_array<T1, MemLoc1>& arr1,
                 const linspace<LinReal>& ts, TReal t) {
  return interpolate(arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
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
