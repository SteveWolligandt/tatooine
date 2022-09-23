#ifndef TATOOINE_DYNAMIC_MULTIDIM_ARRAY_H
#define TATOOINE_DYNAMIC_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_order.h>
#include <tatooine/linspace.h>
#include <tatooine/make_array.h>
#include <tatooine/multidim.h>
#include <tatooine/dynamic_multidim_size.h>
#include <tatooine/png.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename ValueType, typename IndexOrder = x_fastest>
class dynamic_multidim_array : public dynamic_multidim_size<IndexOrder> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type     = ValueType;
  using index_order    = IndexOrder;
  using this_type      = dynamic_multidim_array<value_type, index_order>;
  using parent_type    = dynamic_multidim_size<index_order>;
  using container_type = std::vector<value_type>;
  using parent_type::in_range;
  using parent_type::indices;
  using parent_type::num_components;
  using parent_type::num_dimensions;
  using parent_type::plain_index;
  using parent_type::size;
  //============================================================================
  // members
  //============================================================================
  container_type m_data_container;
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto const... size) {
    return this_type{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  static auto zeros(Size&& size) {
    return this_type{tag::zeros, std::forward<Size>(size)};
  }
  //------------------------------------------------------------------------------
  static auto ones(integral auto const... size) {
    return this_type{tag::ones, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  static auto ones(Size&& size) {
    return this_type{tag::ones, std::forward<Size>(size)};
  }
  //----------------------------------------------------------------------------
  template <integral_range Size, typename RandEng = std::mt19937_64>
  static auto randu(value_type const min, value_type const max, Size&& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<value_type, RandEng>{min, max, std::forward<RandEng>(eng)},
        std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size, typename RandEng = std::mt19937_64>
  static auto randu(Size&& size, value_type const min = 0, value_type const max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<value_type, RandEng>{min, max, std::forward<RandEng>(eng)},
        std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randu(integral auto const... is) {
    return this_type{random::uniform<value_type, RandEng>{
                         value_type(0), value_type(1), RandEng{std::random_device{}()}},
                     is...};
  }
  //----------------------------------------------------------------------------
  template <integral_range Size, typename RandEng = std::mt19937_64>
  static auto randn(value_type const mean, value_type const stddev, Size&& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::normal<value_type, RandEng>{mean, stddev, std::forward<RandEng>(eng)},
        std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size, typename RandEng = std::mt19937_64>
  static auto randn(Size&& size, value_type const mean, value_type const stddev,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::normal<value_type, RandEng>{mean, stddev, std::forward<RandEng>(eng)},
        std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randn(integral auto const... is) {
    return this_type{random::uniform<value_type, RandEng>{
                         value_type(1), value_type(1), RandEng{std::random_device{}()}},
                     is...};
  }
  //----------------------------------------------------------------------------
  template <integral_range Size, random_number_generator Rand>
  static auto rand(Rand&& rand, Size&& size) {
    return this_type{std::forward<Rand>(rand), std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <random_number_generator Rand>
  static auto rand(Rand&& rand, integral auto const... size) {
    return this_type{std::forward<Rand>(rand), size...};
  }
  //============================================================================
  // ctors
  //============================================================================
  dynamic_multidim_array()                                        = default;
  dynamic_multidim_array(dynamic_multidim_array const& other)     = default;
  dynamic_multidim_array(dynamic_multidim_array&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                  =(dynamic_multidim_array const& other)
      -> dynamic_multidim_array& = default;
  auto operator                  =(dynamic_multidim_array&& other) noexcept
      -> dynamic_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  explicit constexpr dynamic_multidim_array(
      dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      : parent_type{other},
        m_data_container(begin(other.internal_container()),
                         end(other.internal_container())) {}
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing>
  auto operator=(dynamic_multidim_array<OtherT, OtherIndexing> const& other)
      -> dynamic_multidim_array& {
    parent_type::operator=(other);
    m_data_container     = std::vector<value_type>(begin(other.internal_container()),
                         end(other.internal_container()));
    return *this;
  }
  //============================================================================
  explicit dynamic_multidim_array(integral auto const... size)
      : parent_type{size...}, m_data_container(num_components(), value_type{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit dynamic_multidim_array(tag::fill<S> const& f,
                                  integral auto const... size)
      : parent_type{size...}, m_data_container(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::zeros_t const& /*z*/,
                                  integral auto const... size)
      : parent_type{size...}, m_data_container(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(tag::ones_t const& /*o*/,
                                  integral auto const... size)
      : parent_type{size...}, m_data_container(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<value_type> const& data,
                                  integral auto const... size)
      : parent_type{size...}, m_data_container(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit dynamic_multidim_array(std::vector<value_type>&& data,
                                  integral auto const... size)
      : parent_type{size...}, m_data_container(std::move(data)) {}
  //----------------------------------------------------------------------------
  template <integral_range Size>
  explicit dynamic_multidim_array(Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(num_components(), value_type{}) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S, integral_range Size>
  dynamic_multidim_array(tag::fill<S> const& f, Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(num_components(), f.value) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  dynamic_multidim_array(tag::zeros_t const& /*z*/, Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(num_components(), 0) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  dynamic_multidim_array(tag::ones_t const& /*o*/, Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(num_components(), 1) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  dynamic_multidim_array(std::vector<value_type> const& data, Size&& size)
      : parent_type{std::forward<Size>(size)}, m_data_container(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size>
  dynamic_multidim_array(std::vector<value_type>&& data, Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral_range Size, random_number_generator Rand>
  requires arithmetic<value_type> dynamic_multidim_array(Rand&& rand, Size&& size)
      : parent_type{std::forward<Size>(size)},
        m_data_container(num_components()) {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<value_type>(rand.get()); });
  }
  //----------------------------------------------------------------------------
  template <random_number_generator Rand>
  requires arithmetic<value_type>
  explicit dynamic_multidim_array(Rand&& rand, integral auto const... size)
      : parent_type{size...}, m_data_container(num_components()) {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<value_type>(rand()); });
  }
  //============================================================================
  // methods
  //============================================================================
  template<integral... Int>
  auto at(Int const... is) -> auto& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Int>
  auto at(Int const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral_range auto const& indices) -> auto& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data_container[plain_index(indices)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral_range auto const& indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data_container[plain_index(indices)];
  }
  //------------------------------------------------------------------------------
  auto operator()(integral_range auto const& indices) -> auto& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral_range auto const& indices) const -> auto const& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Int>
  auto operator()( Int const... is) -> auto& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Int>
  auto operator()( Int const... is) const -> auto const& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto operator[](std::size_t i) const -> auto const& {
    return m_data_container[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](std::size_t i) -> auto& { return m_data_container[i]; }
  //----------------------------------------------------------------------------
  auto resize(integral auto const... size) -> void {
    parent_type::resize(size...);
    m_data_container.resize(num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral_range auto const& res,
              value_type const            value = value_type{}) -> void {
    parent_type::resize(res);
    m_data_container.resize(num_components(), value);
  }
  //----------------------------------------------------------------------------
  constexpr auto internal_container() -> auto& { return m_data_container; }
  constexpr auto internal_container() const -> auto const& {
    return m_data_container;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data(std::size_t const i) -> auto& {
    return m_data_container[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto data(std::size_t const i) const -> auto const& {
    return m_data_container[i];
  }
  //----------------------------------------------------------------------------
  constexpr auto data() { return m_data_container.data(); }
  constexpr auto data() const { return m_data_container.data(); }
  //============================================================================
  template <typename F>
  void unary_operation(F&& f) {
    for (auto is : indices()) {
      at(is) = f(at(is));
    }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing>
  constexpr void binary_operation(
      F&& f, dynamic_multidim_array<OtherT, OtherIndexing> const& other) {
    assert(parent_type::operator==(other));
    for (auto const& is : indices()) {
      at(is) = f(at(is), other(is));
    }
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename ValueType, typename IndexOrder>
dynamic_multidim_array(dynamic_multidim_array<ValueType, IndexOrder> const&)
    -> dynamic_multidim_array<ValueType, IndexOrder>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType, typename IndexOrder>
dynamic_multidim_array(dynamic_multidim_array<ValueType, IndexOrder>&&)
    -> dynamic_multidim_array<ValueType, IndexOrder>;
//----------------------------------------------------------------------------
template <typename ValueType, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, ValueType const& initial)
    -> dynamic_multidim_array<ValueType, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, std::vector<ValueType> const&)
    -> dynamic_multidim_array<ValueType, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType, typename UInt>
dynamic_multidim_array(std::vector<UInt> const&, std::vector<ValueType>&&)
    -> dynamic_multidim_array<ValueType, x_fastest>;
//----------------------------------------------------------------------------
template <typename ValueType, typename UInt, std::size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, ValueType const& initial)
    -> dynamic_multidim_array<ValueType, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType, typename UInt, std::size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, std::vector<ValueType> const&)
    -> dynamic_multidim_array<ValueType, x_fastest>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType, typename UInt, std::size_t N>
dynamic_multidim_array(std::array<UInt, N> const&, std::vector<ValueType>&&)
    -> dynamic_multidim_array<ValueType, x_fastest>;
//==============================================================================
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename FReal>
auto interpolate(dynamic_multidim_array<T0, Indexing0> const& arr0,
                 dynamic_multidim_array<T1, Indexing1> const& arr1,
                 FReal                                        factor) {
  if (factor == 0) {
    return arr0;
  }
  if (factor == 1) {
    return arr1;
  }
  assert(arr0.dyn_size() == arr1.dyn_size());
  auto interpolated =
      dynamic_multidim_array<common_type<T0, T1>, IndexingOut>{arr0};

  for (auto is : interpolated.indices()) {
    interpolated(is) = interpolated(is) * (1 - factor) + arr1(is) * factor;
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename IndexingOut = x_fastest, typename T0, typename T1,
          typename Indexing0, typename Indexing1, typename LinReal,
          typename TReal>
auto interpolate(dynamic_multidim_array<T0, Indexing0> const& arr0,
                 dynamic_multidim_array<T1, Indexing1> const& arr1,
                 linspace<LinReal> const& ts, TReal t) {
  return interpolate<IndexingOut>(arr0, arr1,
                                  (t - ts.front()) / (ts.back() - ts.front()));
}
////------------------------------------------------------------------------------
// template <typename ValueType, typename IndexOrder>
// void write_vtk(dynamic_multidim_array<ValueType, IndexOrder> const& arr,
//               std::string const& filepath, vec<double, 3> const& origin,
//               vec<double, 3> const& spacing,
//               std::string const&    data_name = "tatooine data") {
//  vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
//  if (writer.is_open()) {
//    writer.set_title("tatooine");
//    writer.write_header();
//
//    auto const res = arr.size();
//    writer.write_dimensions(res[0], res[1], res[2]);
//    writer.write_origin(origin(0), origin(1), origin(2));
//    writer.write_spacing(spacing(0), spacing(1), spacing(2));
//    writer.write_point_data(arr.num_components());
//
//    writer.write_scalars(data_name, arr.data());
//    writer.close();
//  }
//}
//
#if TATOOINE_PNG_AVAILABLE
template <arithmetic Real>
void write_png(dynamic_multidim_array<Real> const& arr,
               std::string const&                  filepath) {
  if (arr.num_dimensions() != 2) {
    throw std::runtime_error{
        "multidim array needs 2 dimensions for writing as png."};
  }

  png::image<png::rgb_pixel> image(arr.size(0), arr.size(1));
  for (unsigned int y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      unsigned int idx = x + arr.size(0) * y;

      image[image.get_height() - 1 - y][x].red =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
      image[image.get_height() - 1 - y][x].green =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
      image[image.get_height() - 1 - y][x].blue =
          std::max<Real>(0, std::min<Real>(1, arr[idx])) * 255;
    }
  }
  image.write(filepath);
}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename Real>
// void write_png(dynamic_multidim_array<vec<Real, 2> const>& arr,
//               std::string const&                          filepath) {
//  if (arr.num_dimensions() != 2) {
//    throw std::runtime_error{
//        "multidim array needs 2 dimensions for writing as png."};
//  }
//
//  png::image<png::rgb_pixel> image(dimension(0).size(), dimension(1).size());
//  for (unsigned int y = 0; y < image.get_height(); ++y) {
//    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
//      unsigned int idx = x + dimension(0).size() * y;
//
//      image[image.get_height() - 1 - y][x].red =
//          std::max<Real>(0, std::min<Real>(1, m_data_container[idx * 4 + 0]))
//          * 255;
//      image[image.get_height() - 1 - y][x].green =
//          std::max<Real>(0, std::min<Real>(1, m_data_container[idx * 4 + 1]))
//          * 255;
//      image[image.get_height() - 1 - y][x].blue =
//          std::max<Real>(0, std::min<Real>(1, m_data_container[idx * 4 + 2]))
//          * 255;
//      image[image.get_height() - 1 - y][x].alpha =
//          std::max<Real>(0, std::min<Real>(1, m_data_container[idx * 4 + 3]))
//          * 255;
//    }
//  }
//  image.write(filepath);
//}
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
