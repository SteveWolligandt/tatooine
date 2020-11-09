#ifndef GRID_VERTEX_PROPERTY_H
#define GRID_VERTEX_PROPERTY_H
//==============================================================================
#include <tatooine/multidim_property.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/write_png.h>
#include <tatooine/sampler.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container>
struct grid_vertex_property
    : typed_multidim_property<Grid, Container> {
  using this_t     = grid_vertex_property<Grid, Container>;
  using parent_t   = typed_multidim_property<Grid, value_type>;
  using value_type = typename parent_t::value_type;
  //==============================================================================
  using parent_t::grid;
  using parent_t::data_at;
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
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
      : parent_t{grid},
        m_grid{&grid} {}
  //==============================================================================
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of "
                  "dimensions.");
    return m_grid->position_at(is...);
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return *m_grid; }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
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
  template <size_t N = num_dimensions()>
  requires(N == 1) || (N == 2) || (N == 3)
  void write_vtk(std::string const& file_path,
                 std::string const& description = "tatooine grid") const {
    auto writer = [this, &file_path, &description] {
      vtk::legacy_file_writer writer{file_path, vtk::RECTILINEAR_GRID};
      writer.set_title(description);
      writer.write_header();
      if constexpr (Grid::is_regular) {
        if constexpr (num_dimensions() == 1) {
          writer.write_dimensions(grid().template size<0>(), 1, 1);
          writer.write_origin(grid().template front<0>(), 0, 0);
          writer.write_spacing(grid().template dimension<0>().spacing(), 0, 0);
        } else if constexpr (num_dimensions() == 2) {
          writer.write_dimensions(grid().template size<0>(),
                                  grid().template size<1>(), 1);
          writer.write_origin(grid().template front<0>(),
                              grid().template front<1>(), 0);
          writer.write_spacing(grid().template dimension<0>().spacing(),
                               grid().template dimension<1>().spacing(), 0);
        } else if constexpr (num_dimensions() == 3) {
          writer.write_dimensions(grid().template size<0>(),
                                  grid().template size<1>(),
                                  grid().template size<2>());
          writer.write_origin(grid().template front<0>(),
                              grid().template front<1>(),
                              grid().template front<2>());
          writer.write_spacing(grid().template dimension<0>().spacing(),
                               grid().template dimension<1>().spacing(),
                               grid().template dimension<2>().spacing());
        }
        return writer;
      } else {
        if constexpr (num_dimensions() == 1) {
          writer.write_dimensions(grid().template size<0>(), 1, 1);
          writer.write_x_coordinates(
              std::vector<double>(begin(grid().template dimension<0>()),
                                  end(grid().template dimension<0>())));
          writer.write_y_coordinates(std::vector<double>{0});
          writer.write_z_coordinates(std::vector<double>{0});
        } else if constexpr (num_dimensions() == 2) {
          writer.write_dimensions(grid().template size<0>(),
                                  grid().template size<1>(), 1);
          writer.write_x_coordinates(
              std::vector<double>(begin(grid().template dimension<0>()),
                                  end(grid().template dimension<0>())));
          writer.write_y_coordinates(
              std::vector<double>(begin(grid().template dimension<1>()),
                                  end(grid().template dimension<1>())));
          writer.write_z_coordinates(std::vector<double>{0});
        } else if constexpr (num_dimensions() == 3) {
          writer.write_dimensions(grid().template size<0>(),
                                  grid().template size<1>(),
                                  grid().template size<2>());
          writer.write_x_coordinates(
              std::vector<double>(begin(grid().template dimension<0>()),
                                  end(grid().template dimension<0>())));
          writer.write_y_coordinates(
              std::vector<double>(begin(grid().template dimension<1>()),
                                  end(grid().template dimension<1>())));
          writer.write_z_coordinates(
              std::vector<double>(begin(grid().template dimension<2>()),
                                  end(grid().template dimension<2>())));
        }
        return writer;
      }
    }();
    // write vertex data
    writer.write_point_data(grid().num_vertices());
    std::vector<typename Container::value_type> data;
    grid().loop_over_vertex_indices(
        [&](auto const... is) { data.push_back(data_at(is...)); });
    writer.write_scalars("data", data);
  }
  //----------------------------------------------------------------------------
  template <size_t N = num_dimensions()>
  requires (N == 2) && (std::is_floating_point_v<typename Container::value_type>)
  void write_png(std::string const& file_path) const {
    std::vector<typename Container::value_type> data;
    grid().loop_over_vertex_indices(
        [&](auto const... is) { data.push_back(data_at(is...)); });
    tatooine::write_png(file_path, data,
                        grid().template size<0>(),
                        grid().template size<1>());
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
