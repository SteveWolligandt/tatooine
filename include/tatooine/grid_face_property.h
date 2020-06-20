#ifndef TATOOINE_GRID_FACE_PROPERTY_H
#define TATOOINE_GRID_FACE_PROPERTY_H
//==============================================================================
#include <tatooine/sampler.h>
#include <tatooine/multidim_property.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container, size_t FaceDimensionIndex,
          typename... InterpolationKernels>
struct grid_face_property
    : typed_multidim_property<Grid, typename Container::value_type>,
      sampler<grid_face_property<Grid, Container, FaceDimensionIndex,
                                 InterpolationKernels...>,
              Container, InterpolationKernels...> {
  using value_type = typename Container::value_type;
  using this_t = grid_face_property<Grid, Container, FaceDimensionIndex,
                                    InterpolationKernels...>;
  using prop_parent_t    = typed_multidim_property<Grid, value_type>;
  using sampler_parent_t = sampler<this_t, Container, InterpolationKernels...>;

  using prop_parent_t::grid;
  //==============================================================================
  template <typename... Args>
  grid_face_property(Grid const& grid, Args&&... args)
      : prop_parent_t{grid}, sampler_parent_t{std::forward<Args>(args)...} {}
  //==============================================================================
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of "
                  "dimensions.");
    return grid().template face_center_at<FaceDimensionIndex>(
        std::array{static_cast<size_t>(is)...});
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
  template <size_t DimensionIndex, floating_point Real>
  auto cell_index(Real const x) const -> std::pair<size_t, Real> {
    auto dim = [this]() -> decltype(auto) {
      auto const& actual_dim = grid().template dimension<DimensionIndex>();
      if constexpr (DimensionIndex == FaceDimensionIndex) {
        return actual_dim;
      } else {
        std::vector<typename std::decay_t<decltype(actual_dim)>::value_type> halfpos;
        halfpos.reserve(actual_dim.size() - 1);
        for (size_t i = 0; i < actual_dim.size()-1; ++i) {
          halfpos.push_back((actual_dim[i] + actual_dim[i + 1]) / 2);
        }
        return halfpos;
      }
    };
    if constexpr (is_linspace_v<std::decay_t<decltype(dim())>>) {
      auto const& actual_dim = dim();
      // calculate
      auto const pos = (x - actual_dim.front()) /
                       (actual_dim.back() - actual_dim.front()) * (actual_dim.size() - 1);
      auto const quantized_pos = static_cast<size_t>(std::floor(pos));
      return {quantized_pos, pos - quantized_pos};
    } else {
      auto const halfpos_dim = dim();
      // binary search
      size_t left  = 0;
      size_t right = halfpos_dim.size() - 1;
      while (right - left > 1) {
        auto const center = (right + left) / 2;
        if (x < halfpos_dim[center]) {
          right = center;
        } else {
          left = center;
        }
      }
      return {left, (x - halfpos_dim[left]) /
                        (halfpos_dim[left + 1] - halfpos_dim[left])};
    }
  }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto data_at(std::array<size_t, Grid::num_dimensions()> const& is)
      -> value_type& override {
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
    return sampler_parent_t::sample(x(Is)...);
  }
  //----------------------------------------------------------------------------
  auto sample(typename Grid::pos_t const& x) const -> value_type override {
    return sample(x, std::make_index_sequence<sampler_parent_t::num_dimensions()>{});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
