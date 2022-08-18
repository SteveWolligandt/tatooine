#ifndef TATOOINE_STATIC_MULTIDIM_ARRAY_H
#define TATOOINE_STATIC_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/index_order.h>
#include <tatooine/linspace.h>
#include <tatooine/make_array.h>
#include <tatooine/multidim.h>
#include <tatooine/multidim_size.h>
#include <tatooine/png.h>
#include <tatooine/random.h>
#include <tatooine/reflection.h>
#include <tatooine/tags.h>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename ValueType, typename IndexOrder, typename MemLoc,
          std::size_t... Resolution>
class static_multidim_array
    : public static_multidim_size<IndexOrder, Resolution...> {
  //============================================================================
  // assertions
  //============================================================================
  static_assert(
      std::is_same<MemLoc, tag::heap>::value ||
          std::is_same<MemLoc, tag::stack>::value,
      "MemLoc must either be tatooine::tag::heap or tatooine::tag::stack");
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type = ValueType;
  using this_type =
      static_multidim_array<ValueType, IndexOrder, MemLoc, Resolution...>;
  using parent_type = static_multidim_size<IndexOrder, Resolution...>;
  using parent_type::in_range;
  using parent_type::indices;
  using parent_type::num_components;
  using parent_type::num_dimensions;
  using parent_type::plain_index;
  using parent_type::size;
  using container_t =
      std::conditional_t<std::is_same<MemLoc, tag::stack>::value,
                         std::array<ValueType, num_components()>,
                         std::vector<ValueType>>;

  //============================================================================
  // static methods
  //============================================================================
 private:
  static constexpr auto init_data(ValueType const init = ValueType{}) {
    if constexpr (std::is_same<MemLoc, tag::stack>::value) {
      return make_array<num_components()>(init);
    } else {
      return std::vector(num_components(), init);
    }
  }
  //============================================================================
  // factories
  //============================================================================
 public:
  static constexpr auto zeros() { return this_type{tag::zeros}; }
  //------------------------------------------------------------------------------
  static constexpr auto ones() { return this_type{tag::ones}; }
  //------------------------------------------------------------------------------
  template <typename S>
  static constexpr auto fill(S&& s) {
    return this_type{tag::fill<std::decay_t<S>>{std::forward<S>(s)}};
  }
  //------------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto randu(ValueType min = 0, ValueType max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) requires
      arithmetic<ValueType> {
    return this_type{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto randn(ValueType mean = 0, ValueType stddev = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) requires
      arithmetic<ValueType> {
    return this_type{random::normal{mean, stddev, std::forward<RandEng>(eng)}};
  }
  //============================================================================
  // members
  //============================================================================
 private:
  container_t m_data_container;
  //============================================================================
  // ctors
  //============================================================================
 public:
  constexpr static_multidim_array(static_multidim_array const& other) = default;
  constexpr static_multidim_array(static_multidim_array&& other) noexcept =
      default;
  //----------------------------------------------------------------------------
  constexpr auto operator       =(static_multidim_array const& other)
      -> static_multidim_array& = default;
  constexpr auto operator       =(static_multidim_array&& other) noexcept
      -> static_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~static_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  explicit constexpr static_multidim_array(
      static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                            Resolution...> const& other)
      : m_data_container(init_data()) {
    for (auto is : static_multidim(Resolution...)) {
      at(is) = other(is);
    }
  }
  //----------------------------------------------------------------------------
  template <typename OtherT, typename OtherIndexing, typename OtherMemLoc>
  constexpr auto operator=(
      static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                            Resolution...> const& other)
      -> static_multidim_array& {
    for (auto is : tatooine::static_multidim{Resolution...}) {
      at(is) = other(is);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  explicit constexpr static_multidim_array(
      convertible_to<ValueType> auto&&... ts)
      : m_data_container{static_cast<ValueType>(ts)...} {
    static_assert(sizeof...(ts) == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array()
      : m_data_container(init_data(ValueType{})) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit constexpr static_multidim_array(tag::fill<S> const& f)
      : m_data_container(init_data(static_cast<ValueType>(f.value))) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(tag::zeros_t /*z*/) requires
      arithmetic<ValueType> : m_data_container(init_data(0)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(tag::ones_t /*o*/) requires
      arithmetic<ValueType> : m_data_container(init_data(1)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit static_multidim_array(std::vector<ValueType> const& data)
      : m_data_container(begin(data), end(data)) {
    assert(data.size() == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      std::array<ValueType, num_components()> const& data)
      : m_data_container(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      std::array<ValueType, num_components()>&& data) requires
      is_same<MemLoc, tag::stack> : m_data_container(std::move(data)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      std::vector<ValueType>&& data) requires is_same<MemLoc, tag::heap>
      : m_data_container(std::move(data)) {
    assert(num_components() == data.size());
  }
  //----------------------------------------------------------------------------
  template <typename RandomReal, typename Engine>
  explicit constexpr static_multidim_array(
      random::uniform<RandomReal, Engine>& rand) requires arithmetic<ValueType>
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<ValueType>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  explicit constexpr static_multidim_array(
      random::uniform<RandomReal, Engine>&& rand) requires arithmetic<ValueType>
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<ValueType>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  explicit constexpr static_multidim_array(
      random::normal<RandomReal, Engine>&& rand) requires arithmetic<ValueType>
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<ValueType>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  explicit constexpr static_multidim_array(
      random::normal<RandomReal, Engine>& rand) requires arithmetic<ValueType>
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<ValueType>(rand()); });
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  [[nodiscard]] constexpr auto at(integral auto const... is) const -> const
      auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral_range auto const& indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    return m_data_container[plain_index(indices)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral_range auto const& indices) -> auto& {
    assert(indices.size() == num_dimensions());
    return m_data_container[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator()(integral auto const... is) const
      -> auto const& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data_container[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral_range auto const& indices) const -> const
      auto& {
    assert(indices.size() == num_dimensions());
    return m_data_container[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral_range auto const& indices) -> auto& {
    assert(indices.size() == num_dimensions());
    return m_data_container[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator[](std::size_t i) -> auto& {
    assert(i < num_components());
    return m_data_container[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto operator[](std::size_t i) const -> auto const& {
    assert(i < num_components());
    return m_data_container[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto internal_container() -> auto& {
    return m_data_container;
  }
  [[nodiscard]] constexpr auto internal_container() const -> auto const& {
    return m_data_container;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data() -> ValueType* {
    return m_data_container.data();
  }
  [[nodiscard]] constexpr auto data() const -> ValueType const* {
    return m_data_container.data();
  }
  //============================================================================
  template <typename F>
  constexpr void unary_operation(F&& f) {
    for (auto i : indices()) {
      at(i) = f(at(i));
    }
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherT, typename OtherIndexing,
            typename OtherMemLoc>
  constexpr void binary_operation(
      F&& f, static_multidim_array<OtherT, OtherIndexing, OtherMemLoc,
                                   Resolution...> const& other) {
    for (auto const& i : indices()) {
      at(i) = f(at(i), other(i));
    }
  }
};
//==============================================================================
namespace reflection {
template <typename ValueType, typename IndexOrder, typename MemLoc,
          std::size_t... Resolution>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (static_multidim_array<ValueType, IndexOrder, MemLoc, Resolution...>),
    TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename FReal,
          std::size_t... Resolution>
auto interpolate(
    static_multidim_array<T0, Indexing0, MemLoc0, Resolution...> const& arr0,
    static_multidim_array<T1, Indexing1, MemLoc1, Resolution...> const& arr1,
    FReal factor) {
  static_multidim_array<common_type<T0, T1>, IndexingOut, MemLocOut,
                        Resolution...>
      interpolated{arr0};

  if constexpr (sizeof...(Resolution) == 2) {
#ifndef NDEBUG
#pragma omp parallel for collapse(2)
#endif
    for (std::size_t iy = 0; iy < interpolated.size(1); ++iy) {
      for (std::size_t ix = 0; ix < interpolated.size(0); ++ix) {
        interpolated(ix, iy) =
            interpolated.data(ix, iy) * (1 - factor) + arr1(ix, iy) * factor;
      }
    }
  } else if constexpr (sizeof...(Resolution) == 3) {
#ifndef NDEBUG
#pragma omp parallel for collapse(3)
#endif
    for (std::size_t iz = 0; iz < interpolated.size(2); ++iz) {
      for (std::size_t iy = 0; iy < interpolated.size(1); ++iy) {
        for (std::size_t ix = 0; ix < interpolated.size(0); ++ix) {
          interpolated(ix, iy, iz) = interpolated(ix, iy, iz) * (1 - factor) +
                                     arr1(ix, iy, iz) * factor;
        }
      }
    }
  } else {
    for (std::size_t is : interpolated.indices()) {
      interpolated(is) = interpolated(is) * (1 - factor) + arr1(is) * factor;
    }
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename LinReal, typename TReal,
          std::size_t... Resolution>
auto interpolate(
    static_multidim_array<T0, Indexing0, MemLoc0, Resolution...> const& arr0,
    static_multidim_array<T1, Indexing1, MemLoc1, Resolution...> const& arr1,
    linspace<LinReal> const& ts, TReal t) {
  return interpolate<MemLocOut, IndexingOut>(
      arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
//#include "vtk_legacy.h"
// template <typename ValueType, typename IndexOrder, typename MemLoc,
// std::size_t... Resolution> void write_vtk(
//    static_multidim_array<ValueType, IndexOrder, MemLoc, Resolution...> const&
//    arr, std::string const& filepath, vec<double, 3> const& origin,
//    vec<double, 3> const& spacing,
//    std::string const&    data_name = "tatooine data") {
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
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
