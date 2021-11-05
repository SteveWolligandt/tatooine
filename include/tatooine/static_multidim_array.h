#ifndef TATOOINE_STATIC_MULTIDIM_ARRAY_H
#define TATOOINE_STATIC_MULTIDIM_ARRAY_H
//==============================================================================
#ifdef __cpp_concepts
#include <tatooine/concepts.h>
#endif
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
template <typename T, typename IndexOrder, typename MemLoc,
          size_t... Resolution>
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
  using value_type = T;
  using this_t   = static_multidim_array<T, IndexOrder, MemLoc, Resolution...>;
  using parent_t = static_multidim_size<IndexOrder, Resolution...>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_components;
  using parent_t::num_dimensions;
  using parent_t::plain_index;
  using parent_t::size;
  using container_t =
      std::conditional_t<std::is_same<MemLoc, tag::stack>::value,
                         std::array<T, num_components()>, std::vector<T>>;

  //============================================================================
  // static methods
  //============================================================================
 private:
  static constexpr auto init_data(T const init = T{}) {
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
  static constexpr auto zeros() { return this_t{tag::zeros}; }
  //------------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::ones}; }
  //------------------------------------------------------------------------------
  template <typename S>
  static constexpr auto fill(S&& s) {
    return this_t{tag::fill<std::decay_t<S>>{std::forward<S>(s)}};
  }
  //------------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename RandEng = std::mt19937_64>
  requires arithmetic<T>
#else
  template <typename RandEng = std::mt19937_64, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      static auto randu(T min = 0, T max = 1,
                        RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename RandEng = std::mt19937_64>
  requires arithmetic<T>
#else
  template <typename RandEng = std::mt19937_64, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      static auto randn(T mean = 0, T stddev = 1,
                        RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::normal{mean, stddev, std::forward<RandEng>(eng)}};
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
      : m_data(init_data()) {
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
#ifdef __cpp_concepts
  template <convertible_to<T>... Ts>
#else
  template <typename... Ts, enable_if<(is_convertible<Ts, T> && ...)> = true>
#endif
  explicit constexpr static_multidim_array(Ts&&... ts)
      : m_data{static_cast<T>(ts)...} {
    static_assert(sizeof...(ts) == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr static_multidim_array() : m_data(init_data(T{})) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename S>
  explicit constexpr static_multidim_array(tag::fill<S> const& f)
      : m_data(init_data(static_cast<T>(f.value))) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires arithmetic<T>
#else
  template <typename T_ = T, enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(tag::zeros_t /*z*/)
      : m_data(init_data(0)) {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires arithmetic<T>
#else
  template <typename T_ = T, enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(tag::ones_t /*o*/)
      : m_data(init_data(1)) {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit static_multidim_array(std::vector<T> const& data)
      : m_data(begin(data), end(data)) {
    assert(data.size() == num_components());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit constexpr static_multidim_array(
      std::array<T, num_components()> const& data)
      : m_data(data) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires std::is_same_v<MemLoc, tag::stack>
#else
  template <typename MemLoc_                        = MemLoc,
            enable_if<is_same<MemLoc_, tag::stack>> = true>
#endif
      explicit constexpr static_multidim_array(
          std::array<T, num_components()>&& data)
      : m_data(std::move(data)) {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires std::is_same_v<MemLoc, tag::heap>
#else
  template <typename MemLoc_                       = MemLoc,
            enable_if<is_same<MemLoc_, tag::heap>> = true>
#endif
      explicit constexpr static_multidim_array(std::vector<T>&& data)
      : m_data(std::move(data)) {
    assert(num_components() == data.size());
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename RandomReal, typename Engine>
  requires arithmetic<T>
#else
  template <typename RandomReal, typename Engine, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(
          random::uniform<RandomReal, Engine>& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandomReal, typename Engine>
  requires arithmetic<T>
#else
  template <typename RandomReal, typename Engine, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(
          random::uniform<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandomReal, typename Engine>
  requires arithmetic<T>
#else
  template <typename RandomReal, typename Engine, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(
          random::normal<RandomReal, Engine>&& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandomReal, typename Engine>
  requires arithmetic<T>
#else
  template <typename RandomReal, typename Engine, typename T_ = T,
            enable_if_arithmetic<T_> = true>
#endif
      explicit constexpr static_multidim_array(
          random::normal<RandomReal, Engine>& rand)
      : static_multidim_array{} {
    this->unary_operation(
        [&](auto const& /*c*/) { return static_cast<T>(rand()); });
  }
  //============================================================================
  // methods
  //============================================================================
 public:
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  [[nodiscard]] constexpr auto at(Is const... is) const -> const auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  constexpr auto at(Is const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto at(Indices const& indices) const -> auto const& {
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    return m_data[plain_index(indices)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto at(Indices const& indices) -> auto& {
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    return m_data[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  [[nodiscard]] constexpr auto operator()(Is const... is) const -> auto const& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  constexpr auto operator()(Is const... is) -> auto& {
    static_assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto operator()(Indices const& indices) const -> const auto& {
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    return m_data[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  constexpr auto operator()(Indices const& indices) -> auto& {
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    return m_data[plain_index(indices)];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator[](size_t i) -> auto& {
    assert(i < num_components());
    return m_data[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  [[nodiscard]] constexpr auto operator[](size_t i) const -> auto const& {
    assert(i < num_components());
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data() -> auto& { return m_data; }
  [[nodiscard]] constexpr auto data() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data(size_t const i) -> auto& {
    return m_data[i];
  }
  [[nodiscard]] constexpr auto data(size_t const i) const -> auto const& {
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto data_ptr() -> T* { return m_data.data(); }
  [[nodiscard]] constexpr auto data_ptr() const -> T const* {
    return m_data.data();
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
template <typename T, typename IndexOrder, typename MemLoc,
          size_t... Resolution>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (static_multidim_array<T, IndexOrder, MemLoc, Resolution...>),
    TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename FReal,
          size_t... Resolution>
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
    for (size_t iy = 0; iy < interpolated.size(1); ++iy) {
      for (size_t ix = 0; ix < interpolated.size(0); ++ix) {
        interpolated(ix, iy) =
            interpolated.data(ix, iy) * (1 - factor) + arr1(ix, iy) * factor;
      }
    }
  } else if constexpr (sizeof...(Resolution) == 3) {
#ifndef NDEBUG
#pragma omp parallel for collapse(3)
#endif
    for (size_t iz = 0; iz < interpolated.size(2); ++iz) {
      for (size_t iy = 0; iy < interpolated.size(1); ++iy) {
        for (size_t ix = 0; ix < interpolated.size(0); ++ix) {
          interpolated(ix, iy, iz) = interpolated(ix, iy, iz) * (1 - factor) +
                                     arr1(ix, iy, iz) * factor;
        }
      }
    }
  } else {
    for (size_t is : interpolated.indices()) {
      interpolated(is) = interpolated(is) * (1 - factor) + arr1(is) * factor;
    }
  }
  return interpolated;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename MemLocOut = tag::stack, typename IndexingOut = x_fastest,
          typename T0, typename T1, typename Indexing0, typename Indexing1,
          typename MemLoc0, typename MemLoc1, typename LinReal, typename TReal,
          size_t... Resolution>
auto interpolate(
    static_multidim_array<T0, Indexing0, MemLoc0, Resolution...> const& arr0,
    static_multidim_array<T1, Indexing1, MemLoc1, Resolution...> const& arr1,
    linspace<LinReal> const& ts, TReal t) {
  return interpolate<MemLocOut, IndexingOut>(
      arr0, arr1, (t - ts.front()) / (ts.back() - ts.front()));
}
//#include "vtk_legacy.h"
// template <typename T, typename IndexOrder, typename MemLoc, size_t...
// Resolution> void write_vtk(
//    static_multidim_array<T, IndexOrder, MemLoc, Resolution...> const& arr,
//    std::string const& filepath, vec<double, 3> const& origin,
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