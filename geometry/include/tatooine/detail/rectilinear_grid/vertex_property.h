#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_PROPERTY_PROPERTY_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_PROPERTY_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/detail/rectilinear_grid/inverse_distance_weighting_vertex_property_sampler.h>
#include <tatooine/detail/rectilinear_grid/vertex_property_sampler.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/interpolation.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/type_traits.h>
#include <tatooine/write_png.h>
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename DefaultInterpolationKernel,
          std::size_t N, template <typename> typename... InterpolationKernels>
struct repeated_interpolation_kernel_for_vertex_property_impl {
  using type = typename repeated_interpolation_kernel_for_vertex_property_impl<
      GridVertexProperty, DefaultInterpolationKernel, N - 1,
      InterpolationKernels..., DefaultInterpolationKernel>::type;
};
//----------------------------------------------------------------------------
template <typename GridVertexProperty,
          template <typename> typename DefaultInterpolationKernel,
          template <typename> typename... InterpolationKernels>
struct repeated_interpolation_kernel_for_vertex_property_impl<
    GridVertexProperty, DefaultInterpolationKernel, 0,
    InterpolationKernels...> {
  using type =
      vertex_property_sampler<GridVertexProperty, InterpolationKernels...>;
};
template <typename GridVertexProperty,
          template <typename> typename DefaultInterpolationKernel>
using repeated_interpolation_kernel_for_vertex_property =
    typename repeated_interpolation_kernel_for_vertex_property_impl<
        GridVertexProperty, DefaultInterpolationKernel,
        GridVertexProperty::num_dimensions()>::type;
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
struct typed_vertex_property_interface;
//==============================================================================
template <typename Grid>
struct vertex_property {
  //============================================================================
  using this_type     = vertex_property<Grid>;
  using real_type     = typename Grid::real_type;
  using vertex_handle = typename Grid::vertex_handle;
  //============================================================================
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
  //============================================================================
 private:
  Grid const* m_grid;
  //============================================================================
 public:
  vertex_property(Grid const& g) : m_grid{&g} {}
  vertex_property(vertex_property const& other)     = default;
  vertex_property(vertex_property&& other) noexcept = default;
  //----------------------------------------------------------------------------
  virtual ~vertex_property() = default;
  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual auto type() const -> std::type_info const&           = 0;
  virtual auto container_type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_type> = 0;
  //----------------------------------------------------------------------------
  auto grid() -> auto& { return *m_grid; }
  auto grid() const -> auto const& { return *m_grid; }
  auto set_grid(Grid const& g) { m_grid = &g; }
  //----------------------------------------------------------------------------
  template <typename T, bool HasNonConstReference = false>
  auto cast_to_typed() -> auto& {
    return *static_cast<
        typed_vertex_property_interface<Grid, T, HasNonConstReference>*>(this);
  }
  //----------------------------------------------------------------------------
  template <typename T, bool HasNonConstReference = false>
  auto cast_to_typed() const -> auto const& {
    return *static_cast<
        typed_vertex_property_interface<Grid, T, HasNonConstReference> const*>(
        this);
  }
};
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
struct typed_vertex_property_interface : vertex_property<Grid> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_type =
      typed_vertex_property_interface<Grid, ValueType, HasNonConstReference>;
  using parent_type     = vertex_property<Grid>;
  using value_type      = ValueType;
  using const_reference = ValueType const&;
  using reference =
      std::conditional_t<HasNonConstReference, ValueType&, const_reference>;
  using grid_type = Grid;
  using real_type = typename Grid::real_type;
  using parent_type::grid;
  using parent_type::num_dimensions;
  using typename parent_type::vertex_handle;
  using inverse_distance_weighting_sampler_type = detail::rectilinear_grid::
      inverse_distance_weighting_vertex_property_sampler<Grid, this_type>;

  //============================================================================
  // ctors
  //============================================================================
  explicit typed_vertex_property_interface(Grid const& g) : parent_type{g} {}
  typed_vertex_property_interface(typed_vertex_property_interface const&) =
      default;
  typed_vertex_property_interface(typed_vertex_property_interface&&) noexcept =
      default;
  //----------------------------------------------------------------------------
  virtual ~typed_vertex_property_interface() = default;
  //============================================================================
  // methods
  //============================================================================
  auto type() const -> std::type_info const& override {
    return typeid(value_type);
  }
  //----------------------------------------------------------------------------
 private:
  template <template <typename> typename InterpolationKernel>
  auto sampler_() const {
    using sampler_t =
        repeated_interpolation_kernel_for_vertex_property<this_type,
                                                          InterpolationKernel>;
    grid().update_diff_stencil_coefficients();
    return sampler_t{*this};
  }
  //----------------------------------------------------------------------------
 public:
  template <template <typename> typename... InterpolationKernels>
  auto sampler() const {
    if (!grid().diff_stencil_coefficients_created_once()) {
      grid().update_diff_stencil_coefficients();
    }
    static_assert(
        sizeof...(InterpolationKernels) == 0 ||
            sizeof...(InterpolationKernels) == 1 ||
            sizeof...(InterpolationKernels) == num_dimensions(),
        "Number of interpolation kernels does not match number of dimensions.");

    if constexpr (sizeof...(InterpolationKernels) == 0) {
      using sampler_t = repeated_interpolation_kernel_for_vertex_property<
          this_type, interpolation::cubic>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    } else if constexpr (sizeof...(InterpolationKernels) == 1) {
      return sampler_<InterpolationKernels...>();
    } else {
      using sampler_t =
          vertex_property_sampler<this_type, InterpolationKernels...>;
      if (!grid().diff_stencil_coefficients_created_once()) {
        grid().update_diff_stencil_coefficients();
      }
      return sampler_t{*this};
    }
  }
  //----------------------------------------------------------------------------
  auto linear_sampler() const -> decltype(auto) {
    return sampler<interpolation::linear>();
  }
  //----------------------------------------------------------------------------
  auto cubic_sampler() const -> decltype(auto) {
    return sampler<interpolation::cubic>();
  }
  //----------------------------------------------------------------------------
  auto inverse_distance_weighting_sampler(real_type const radius) const {
    return inverse_distance_weighting_sampler_type{this->grid(), *this, radius};
  }
  //----------------------------------------------------------------------------
  // data access
  //----------------------------------------------------------------------------
  constexpr auto operator[](vertex_handle const& h) const -> decltype(auto) {
    return at(h);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator[](vertex_handle const& h) -> decltype(auto) {
    return at(h);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(std::array<std::size_t, num_dimensions()> const& is)
      const -> decltype(auto) {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(std::array<std::size_t, num_dimensions()> const& is)
      -> decltype(auto) {
    return at(is);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == num_dimensions()) {
    return at(std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == num_dimensions()) {
    return at(std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  constexpr auto at(vertex_handle const& h) const -> decltype(auto) {
    return at(h, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  constexpr auto at(vertex_handle const& h) -> decltype(auto) {
    return at(h, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto at(vertex_handle const& h,
                    std::index_sequence<Is...> /*seq*/) const
      -> decltype(auto) {
    return at(h.index(Is)...);
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  constexpr auto at(vertex_handle const& h, std::index_sequence<Is...> /*seq*/)
      -> decltype(auto) {
    return at(h.index(Is)...);
  }
  //----------------------------------------------------------------------------
 public:
  auto at(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == num_dimensions()) {
    return at(std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == num_dimensions()) {
    return at(std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  virtual auto at(std::array<std::size_t, num_dimensions()> const& is) const
      -> const_reference = 0;
  //----------------------------------------------------------------------------
  virtual auto at(std::array<std::size_t, num_dimensions()> const& is)
      -> reference = 0;
  //----------------------------------------------------------------------------
  virtual auto plain_at(std::size_t) const -> const_reference = 0;
  //----------------------------------------------------------------------------
  virtual auto plain_at(std::size_t) -> reference = 0;
  //----------------------------------------------------------------------------
  auto resize(integral auto const... size)
      -> decltype(auto) requires(sizeof...(size) == num_dimensions()) {
    return resize(std::array{static_cast<std::size_t>(size)...});
  }
  //----------------------------------------------------------------------------
  virtual auto resize(std::array<std::size_t, num_dimensions()> const& size)
      -> void = 0;
  //----------------------------------------------------------------------------
#if TATOOINE_PNG_AVAILABLE
  template <invocable<ValueType> T>
      auto write_png(filesystem::path const& path, T&& f, auto&& color_scale,
                     tensor_value_type<ValueType> const min = 0,
                     tensor_value_type<ValueType> const max = 1) const
      -> void requires(num_dimensions() == 2) &&
      (static_vec<ValueType>)&&(arithmetic<invoke_result<T, ValueType>>) {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = f(at(x, y));
        if (std::isnan(d)) {
          d = 0;
        } else {
          d = std::max<tensor_value_type<ValueType>>(
              min, std::min<tensor_value_type<ValueType>>(max, d));
          d -= min;
          d /= max - min;
        }
        auto const col                             = color_scale(d);
        image[image.get_height() - 1 - y][x].red   = col.x() * 255;
        image[image.get_height() - 1 - y][x].green = col.y() * 255;
        image[image.get_height() - 1 - y][x].blue  = col.z() * 255;
      }
    }
    image.write(path.string());
  }
  //----------------------------------------------------------------------------
  auto write_png(filesystem::path const&            path,
                 tensor_value_type<ValueType>       min = 0,
                 tensor_value_type<ValueType> const max = 1) const
      -> void requires(num_dimensions() == 2) &&
      ((static_vec<ValueType>) || (arithmetic<ValueType>)) {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if constexpr (is_floating_point<ValueType>) {
          if (std::isnan(d)) {
            d = 0;
          } else {
            d = std::max<ValueType>(min, std::min<ValueType>(max, d));
            d -= min;
            d /= max - min;
          }
          image[image.get_height() - 1 - y][x].red =
              image[image.get_height() - 1 - y][x].green =
                  image[image.get_height() - 1 - y][x].blue = d * 255;
        } else if constexpr (static_vec<ValueType>) {
          if (std::isnan(d(0))) {
            for (auto& c : d) {
              c = 0;
            }
          } else {
            for (auto& c : d) {
              c = std::max<typename ValueType::value_type>(
                  min, std::min<typename ValueType::value_type>(max, c));
              c -= min;
              c /= max - min;
            }
          }
          image[image.get_height() - 1 - y][x].red   = d(0) * 255;
          image[image.get_height() - 1 - y][x].green = d(1) * 255;
          image[image.get_height() - 1 - y][x].blue  = d(2) * 255;
        }
      }
    }
    image.write(path.string());
  }
  //----------------------------------------------------------------------------
  auto write_png(filesystem::path const& path, auto&& color_scale,
                 ValueType const min = 0, ValueType const max = 1) const
      -> void requires(num_dimensions() == 2) &&
      (is_floating_point<ValueType>) {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if (std::isnan(d)) {
          d = 0;
        } else {
          d = std::max<ValueType>(min, std::min<ValueType>(max, d));
          d -= min;
          d /= max - min;
        }
        auto const col                             = color_scale(d) * 255;
        image[image.get_height() - 1 - y][x].red   = col(0);
        image[image.get_height() - 1 - y][x].green = col(1);
        image[image.get_height() - 1 - y][x].blue  = col(2);
      }
    }
    image.write(path.string());
  }
#endif
};
//==============================================================================
#if TATOOINE_PNG_AVAILABLE
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(typed_vertex_property_interface<
                   Grid, ValueType, HasNonConstReference> const& prop,
               filesystem::path const&                           path) -> void {
  prop.write_png(path);
}
//------------------------------------------------------------------------------
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(filesystem::path const& path,
               typed_vertex_property_interface<
                   Grid, ValueType, HasNonConstReference> const& prop) -> void {
  prop.write_png(path);
}
#endif
//==============================================================================
template <typename Grid, typename ValueType, typename Container>
struct typed_vertex_property
    : typed_vertex_property_interface<
          Grid, ValueType,
          std::is_convertible_v<
              decltype(std::declval<Container&>().at(
                  std::declval<
                      std::array<std::size_t, Grid::num_dimensions()>>())),
              ValueType&>>,
      Container {
  static_assert(std::is_same_v<ValueType, typename Container::value_type>);
  //============================================================================
  // typedefs
  //============================================================================
  using this_type = typed_vertex_property<Grid, ValueType, Container>;
  static constexpr bool has_non_const_reference = std::is_convertible_v<
      decltype(std::declval<Container&>().at(
          std::declval<std::array<std::size_t, Grid::num_dimensions()>>())),
      ValueType&>;
  using prop_parent_type =
      typed_vertex_property_interface<Grid, ValueType, has_non_const_reference>;
  using cont_parent_type = Container;
  using value_type       = typename prop_parent_type::value_type;
  using reference        = typename prop_parent_type::reference;
  using const_reference  = typename prop_parent_type::const_reference;
  using grid_type        = Grid;
  using real_type        = typename Grid::real_type;
  using prop_parent_type::num_dimensions;
  //============================================================================
  // ctors
  //============================================================================
  template <typename... Args>
  explicit typed_vertex_property(Grid const& g, Args&&... args)
      : prop_parent_type{g}, cont_parent_type{std::forward<Args>(args)...} {}
  typed_vertex_property(typed_vertex_property const&)     = default;
  typed_vertex_property(typed_vertex_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  virtual ~typed_vertex_property() = default;
  //============================================================================
  // methods
  //============================================================================
  auto clone() const -> std::unique_ptr<vertex_property<Grid>> override {
    return std::unique_ptr<this_type>(new this_type{*this});
  }
  //----------------------------------------------------------------------------
  auto container_type() const -> std::type_info const& override {
    return typeid(Container);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == Grid::num_dimensions()) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == Grid::num_dimensions()) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == Grid::num_dimensions()) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == Grid::num_dimensions()) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  auto at(std::array<std::size_t, num_dimensions()> const& size,
          std::index_sequence<Is...> /*seq*/) const -> const_reference {
    return Container::at(size[Is]...);
  }
  //----------------------------------------------------------------------------
  auto at(std::array<std::size_t, num_dimensions()> const& size) const
      -> const_reference override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  auto at(std::array<std::size_t, num_dimensions()> const& size,
          std::index_sequence<Is...> /*seq*/) -> reference {
    return Container::at(size[Is]...);
  }
  //----------------------------------------------------------------------------
  auto at(std::array<std::size_t, num_dimensions()> const& size)
      -> reference override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto plain_at(std::size_t const i) const -> const_reference override {
    return Container::operator[](i);
  }
  //----------------------------------------------------------------------------
  auto plain_at(std::size_t const i) -> reference override {
    return Container::operator[](i);
  }
  //----------------------------------------------------------------------------
  auto resize(std::array<std::size_t, num_dimensions()> const& size)
      -> void override {
    Container::resize(size);
  }
};
//==============================================================================
template <std::size_t DiffOrder, typename Grid, typename ValueType>
struct vertex_property_differentiated_type_impl;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <std::size_t DiffOrder, typename Grid, typename ValueType>
using vertex_property_differentiated_type =
    typename vertex_property_differentiated_type_impl<DiffOrder, Grid,
                                                      ValueType>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <floating_point T, typename Grid>
struct vertex_property_differentiated_type_impl<1, Grid, T> {
  using type = vec<T, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <floating_point T, typename Grid>
struct vertex_property_differentiated_type_impl<2, Grid, T> {
  using type = mat<T, Grid::num_dimensions(), Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <floating_point T, typename Grid>
struct vertex_property_differentiated_type_impl<3, Grid, T> {
  using type = tensor<T, Grid::num_dimensions(), Grid::num_dimensions(),
                      Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, floating_point T, std::size_t N>
struct vertex_property_differentiated_type_impl<1, Grid, vec<T, N>> {
  using type = mat<T, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, floating_point T, std::size_t N>
struct vertex_property_differentiated_type_impl<2, Grid, vec<T, N>> {
  using type = tensor<T, N, Grid::num_dimensions(), Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, floating_point T, std::size_t M, std::size_t N>
struct vertex_property_differentiated_type_impl<1, Grid, mat<T, M, N>> {
  using type = tensor<T, M, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, floating_point T, std::size_t... Dims>
struct vertex_property_differentiated_type_impl<1, Grid, tensor<T, Dims...>> {
  using type = tensor<T, Dims..., Grid::num_dimensions()>;
};
//==============================================================================
template <std::size_t DiffOrder, typename Grid, typename PropValueType,
          bool        PropHasNonConstReference, typename Impl>
struct differentiated_typed_vertex_property_interface : crtp<Impl> {
  using this_type = differentiated_typed_vertex_property_interface<
      DiffOrder, Grid, PropValueType, PropHasNonConstReference, Impl>;
  using prop_type  = typed_vertex_property_interface<Grid, PropValueType,
                                                    PropHasNonConstReference>;
  using crtp_type  = crtp<Impl>;
  using value_type = vertex_property_differentiated_type<DiffOrder, Grid, PropValueType>;
  using grid_type  = Grid;
  using real_type  = typename Grid::real_type;
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  prop_type const&                                     m_prop;
  std::array<std::array<std::vector<real_type>, num_dimensions()>, DiffOrder>
      m_diff_coeffs_per_order_per_dimension;
  std::size_t                                          m_stencil_size;
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  differentiated_typed_vertex_property_interface(prop_type const&  prop,
                                       std::size_t       stencil_size,
                                       std::index_sequence<Is...> /*seq*/)
      : m_prop{prop},
        m_stencil_size{stencil_size} {
    (generate_coefficients<Is>(), ...);
  }
  //----------------------------------------------------------------------------
 public:
  differentiated_typed_vertex_property_interface(prop_type const&  prop,
                                       std::size_t       stencil_size = 5)
      : differentiated_typed_vertex_property_interface{
            prop, stencil_size,
            std::make_index_sequence<num_dimensions()>{}} {}

 private:
  //----------------------------------------------------------------------------
  /// Generates diffentiation coefficients for dimension DimIndex.
  template <std::size_t DimIndex>
  auto generate_coefficients() {
    for (std::size_t i_order = 0; i_order < differentiation_order();
         ++i_order) {
      auto const& dim      = grid().template dimension<DimIndex>();
      auto&       stencils = m_diff_coeffs_per_order_per_dimension[i_order][DimIndex];
      stencils.reserve(dim.size() * m_stencil_size);

      using dim_type  = std::decay_t<decltype(dim)>;
      using real_type = typename dim_type::value_type;

      // local positions relative to current position on dimension. relative
      // position of current point will be 0.
      auto local_positions = std::vector<real_type>(m_stencil_size, 0);

      auto const half_stencil_size =
          static_cast<std::size_t>(m_stencil_size / 2);
      for (std::size_t i_x = 0; i_x < dim.size(); ++i_x) {
        auto i_left = std::min<long>(
            (dim.size() - 1) - (stencil_size() - 1),
            std::max<long>(0, static_cast<long>(i_x) - half_stencil_size));
        for (std::size_t i_local = 0; i_local < m_stencil_size; ++i_local) {
          local_positions[i_local] = dim[i_left + i_local] - dim[i_x];
        }
        for (auto const c :
             finite_differences_coefficients(DiffOrder, local_positions)) {
          stencils.push_back(c);
        }
      }
    }
  }

 public:
  //----------------------------------------------------------------------------
  auto property() const -> auto const& { return m_prop; }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return property().grid(); }
  //----------------------------------------------------------------------------
  auto stencil_size() const  { return m_stencil_size; }
  //----------------------------------------------------------------------------
  static constexpr auto differentiation_order() { return DiffOrder; }
  //----------------------------------------------------------------------------
  auto differentiation_coefficients(std::size_t const diff_order,
                                    std::size_t const dimension) const
      -> auto const& {
    return m_diff_coeffs_per_order_per_dimension[diff_order - 1][dimension];
  }
  //----------------------------------------------------------------------------
  // data access
  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const -> value_type
      requires(sizeof...(is) == Grid::num_dimensions()) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) const -> value_type
  requires (sizeof...(is) == Grid::num_dimensions()) {
    return this->as_derived().at(is...);
  }
};
//==============================================================================
template <std::size_t DiffOrder, typename Grid, typename PropValueType,
          bool        H>
struct differentiated_typed_vertex_property;
//==============================================================================
/// Specializes first derivative of scalars
template <typename Grid, floating_point PropValueType, bool H>
struct differentiated_typed_vertex_property<1, Grid, PropValueType, H>
    : differentiated_typed_vertex_property_interface<
          1, Grid, PropValueType, H,
          differentiated_typed_vertex_property<1, Grid, PropValueType, H>> {
  using this_type =
      differentiated_typed_vertex_property<1, Grid, PropValueType, H>;
  using parent_type =
      differentiated_typed_vertex_property_interface<1, Grid, PropValueType, H,
                                                     this_type>;
  using parent_type::differentiation_coefficients;
  using parent_type::num_dimensions;
  using parent_type::property;
  using parent_type::stencil_size;
  using typename parent_type::prop_type;
  using typename parent_type::value_type;
  //----------------------------------------------------------------------------
  differentiated_typed_vertex_property(prop_type const& prop,
                                       std::size_t      stencil_size = 5)
      : parent_type{prop, stencil_size} {}
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) const -> value_type {
    auto       d                 = value_type{};
    auto const half_stencil_size = long(stencil_size() / 2);

    auto const indices = std::array{static_cast<std::size_t>(is)...};

    for (std::size_t i_dim = 0; i_dim < num_dimensions(); ++i_dim) {
      auto const i_x             = indices[i_dim];
      auto       running_indices = indices;
      running_indices[i_dim]     = static_cast<std::size_t>(std::min<long>(
          (this->grid().size(i_dim) - 1) - (stencil_size() - 1),
          std::max<long>(0, static_cast<long>(i_x) - half_stencil_size)));

      auto stencil_it = next(begin(differentiation_coefficients(1, i_dim)),
                             running_indices[i_dim] * stencil_size());

      for (std::size_t i = 0; i < stencil_size();
           ++i, ++running_indices[i_dim], ++stencil_it) {
        d(i_dim) += *stencil_it * property().at(running_indices);
      }
    }
    return d;
  }
};
//==============================================================================
/// Specializes second derivative of scalars
template <typename Grid, floating_point PropValueType, bool H>
struct differentiated_typed_vertex_property<2, Grid, PropValueType, H>
    : differentiated_typed_vertex_property_interface<
          2, Grid, PropValueType, H,
          differentiated_typed_vertex_property<2, Grid, PropValueType, H>> {
  using this_type =
      differentiated_typed_vertex_property<2, Grid, PropValueType, H>;
  using parent_type =
      differentiated_typed_vertex_property_interface<2, Grid, PropValueType, H,
                                                     this_type>;
  using parent_type::differentiation_coefficients;
  using parent_type::num_dimensions;
  using parent_type::property;
  using parent_type::stencil_size;
  using typename parent_type::prop_type;
  using typename parent_type::value_type;
  //----------------------------------------------------------------------------
  differentiated_typed_vertex_property(prop_type const& prop,
                                       std::size_t      stencil_size = 5)
      : parent_type{prop, stencil_size} {}
  //----------------------------------------------------------------------------
  auto at(integral auto const... is) const -> value_type {
    auto       d                 = value_type::nans();
    auto const half_stencil_size = long(stencil_size() / 2);

    auto const indices = std::array{static_cast<std::size_t>(is)...};

    for (std::size_t i_dim = 0; i_dim < num_dimensions(); ++i_dim) {
      auto const i_x             = indices[i_dim];
      auto       running_indices = indices;
      running_indices[i_dim]     = static_cast<std::size_t>(std::min<long>(
          (this->grid().size(i_dim) - 1) - (stencil_size() - 1),
          std::max<long>(0, static_cast<long>(i_x) - half_stencil_size)));

      auto stencil_it = next(begin(differentiation_coefficients(1, i_dim)),
                             running_indices[i_dim] * stencil_size());

      for (std::size_t i = 0; i < stencil_size();
           ++i, ++running_indices[i_dim], ++stencil_it) {
        d(i_dim, i_dim) += *stencil_it * property().at(running_indices);
      }
    }
    return d;
  }
};
//==============================================================================
template <std::size_t DiffOrder = 1, typename Grid, typename ValueType,
          bool        HasNonConstReference>
auto diff(typed_vertex_property_interface<Grid, ValueType,
                                          HasNonConstReference> const& prop,
          std::size_t const stencil_size = 5) {
  return differentiated_typed_vertex_property<DiffOrder, Grid, ValueType,
                                              HasNonConstReference>{
      prop, stencil_size};
}
//==============================================================================
template <std::size_t DiffOrder = 1, std::size_t CurDiffOrder, typename Grid,
          typename ValueType, bool               HasNonConstReference>
auto diff(differentiated_typed_vertex_property<CurDiffOrder, Grid, ValueType,
                                               HasNonConstReference> const& d,
          std::size_t const stencil_size = 5) {
  return differentiated_typed_vertex_property<DiffOrder + CurDiffOrder, Grid,
                                              ValueType, HasNonConstReference>{
      d.property(), stencil_size};
}
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
