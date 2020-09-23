#ifndef GRID_VERTEX_PROPERTY_H
#define GRID_VERTEX_PROPERTY_H
//==============================================================================
#include <tatooine/multidim_property.h>
#include <tatooine/sampler.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container,
          template <typename> typename... InterpolationKernels>
struct grid_vertex_property
    : typed_multidim_property<Grid, typename Container::value_type>,
      sampler<Grid, Container, InterpolationKernels...> {
  using value_type = typename Container::value_type;
  using this_t = grid_vertex_property<Grid, Container, InterpolationKernels...>;
  using prop_parent_t    = typed_multidim_property<Grid, value_type>;
  using sampler_parent_t = sampler<Grid, Container, InterpolationKernels...>;
  //==============================================================================
  using prop_parent_t::grid;
  using prop_parent_t::out_of_domain_value;
  using prop_parent_t::data_at;
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
  using sampler_parent_t::current_dimension_index;
  //==============================================================================
  Grid const* m_grid;
  //==============================================================================
  grid_vertex_property(grid_vertex_property const&)     = default;
  grid_vertex_property(grid_vertex_property&&) noexcept = default;
  //------------------------------------------------------------------------------
  auto operator=(grid_vertex_property const&)
    -> grid_vertex_property& = default;
  auto operator=(grid_vertex_property&&) noexcept
    -> grid_vertex_property& = default;
  //------------------------------------------------------------------------------
  template <typename... Args>
  grid_vertex_property(Grid const& grid, Args&&... args)
      : prop_parent_t{grid},
        sampler_parent_t{grid, std::forward<Args>(args)...},
        m_grid{&grid} {}
  //==============================================================================
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of "
                  "dimensions.");
    return m_grid->position_at(is...);
  }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto data_at(std::array<size_t, Grid::num_dimensions()> const& is) const
      -> value_type const& override {
    return sampler_parent_t::data_at(is);
  }
  //----------------------------------------------------------------------------
  void set_data_at(std::array<size_t, Grid::num_dimensions()> const& is,
                   value_type const& data) override {
    invoke_unpacked(
        [this](auto const& data, auto const... is) {
          sampler_parent_t::set_data_at(data, is...);
        },
        data, unpack(is));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void set_data_at(value_type const& data, integral auto const... is) {
    sampler_parent_t::set_data_at(data, is...);
  }
  //----------------------------------------------------------------------------
  auto container() -> auto& {
    return sampler_parent_t::container();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& {
    return sampler_parent_t::container();
  }
  //----------------------------------------------------------------------------
  auto resize(std::vector<size_t> const& size) {
    assert(size.size() == num_dimensions());
    return container().resize(size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(std::array<size_t, num_dimensions()> const& size) {
    return container().resize(size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral auto... ss) {
    static_assert(sizeof...(ss) == num_dimensions());
    return container().resize(ss...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto sample(typename Grid::pos_t const& x, std::index_sequence<Is...>) const
      -> value_type {
    static_assert(num_dimensions() == sizeof...(Is));
    return sampler_parent_t::sample(x(Is)...);
  }
  //----------------------------------------------------------------------------
  auto sample(typename Grid::pos_t const& x) const -> value_type override {
    return sample(x, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto sample(real_number auto... xs) const -> value_type {
    static_assert(num_dimensions() == sizeof...(xs));
    return sampler_parent_t::sample(xs...);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
