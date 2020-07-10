#ifndef GRID_VERTEX_PROPERTY_H
#define GRID_VERTEX_PROPERTY_H
//==============================================================================
#include <tatooine/sampler.h>
#include <tatooine/multidim_property.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container, typename... InterpolationKernels>
struct grid_vertex_property
    : typed_multidim_property<Grid, typename Container::value_type>,
      sampler<grid_vertex_property<Grid, Container, InterpolationKernels...>,
              Container, InterpolationKernels...> {
  using value_type = typename Container::value_type;
  using this_t = grid_vertex_property<Grid, Container, InterpolationKernels...>;
  using prop_parent_t    = typed_multidim_property<Grid, value_type>;
  using sampler_parent_t = sampler<this_t,
              Container, InterpolationKernels...>;
  //==============================================================================
  using prop_parent_t::grid;
  static constexpr auto num_dimensions() {return Grid::num_dimensions();}
  //==============================================================================
  grid_vertex_property(grid_vertex_property const&)     = default;
  grid_vertex_property(grid_vertex_property&&) noexcept = default;
  //------------------------------------------------------------------------------
  auto operator=(grid_vertex_property const&)
      -> grid_vertex_property& = default;
  auto operator                =(grid_vertex_property&&) noexcept
      -> grid_vertex_property& = default;
  //------------------------------------------------------------------------------
  template <typename... Args>
  grid_vertex_property(Grid const& grid, Args&&... args)
      : prop_parent_t{grid}, sampler_parent_t{std::forward<Args>(args)...} {}
  //==============================================================================
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of "
                  "dimensions.");
    return prop_parent_t::vertex_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is)
      -> decltype(auto) {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return prop_parent_t::template diff_at<DimIndex, StencilSize>(num_diffs, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is) const
      -> decltype(auto) {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return prop_parent_t::template diff_at<DimIndex, StencilSize>(num_diffs, is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return grid().template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto data_at(std::array<size_t, Grid::num_dimensions()> const& is)
      ->value_type& override {
    return sampler_parent_t::data_at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(std::array<size_t, Grid::num_dimensions()> const& is) const
      -> value_type const& override {
    return sampler_parent_t::data_at(is);
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto sample(typename Grid::pos_t const& x, std::index_sequence<Is...>) const
      -> value_type {
        static_assert(Grid::pos_t::num_dimensions() == sizeof...(Is));
    return sampler_parent_t::sample(x(Is)...);
  }
  //----------------------------------------------------------------------------
  auto sample(typename Grid::pos_t const& x) const -> value_type override {
    return sample(
        x, std::make_index_sequence<Grid::pos_t::num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto stencil_coefficients(size_t const i,
                            unsigned int const num_diffs) const {
    return prop_parent_t::template stencil_coefficients<DimIndex, StencilSize>(
        i, num_diffs);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
