#ifndef TATOOINE_SAMPLED_GRID_PROPERTY_FIELD_H
#define TATOOINE_SAMPLED_GRID_PROPERTY_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>

#include <memory>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, real_number Real, size_t N, size_t... TensorDims>
struct sampled_grid_property_field
    : field<sampled_grid_property_field<Grid, Real, N, TensorDims...>, Real, N,
            TensorDims...> {
  using this_t   = sampled_grid_property_field<Grid, Real, N, TensorDims...>;
  using grid_t   = Grid;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using tensor_prop_t = typename grid_t::template typed_property_t<tensor_t>;
  using real_prop_t   = typename grid_t::template typed_property_t<real_t>;
  using parent_t::num_dimensions;
  static_assert(num_dimensions() == grid_t::num_dimensions() ||
                num_dimensions() == grid_t::num_dimensions() - 1);
  static constexpr bool is_time_dependent =
      num_dimensions() == grid_t::num_dimensions() - 1;
  //============================================================================
  std::shared_ptr<grid_t>   m_grid;
  tensor_prop_t*            m_tensor_prop = nullptr;
  std::vector<real_prop_t*> m_real_props;
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return *m_grid; }
  auto grid() -> auto& { return *m_grid; }
  //============================================================================
  sampled_grid_property_field() = default;
  //----------------------------------------------------------------------------
  explicit sampled_grid_property_field(std::filesystem::path const& path)
      : m_grid{std::make_shared<grid_t>(path)} {
    for (auto const& [name, prop] : m_grid->vertex_properties()) {
      if (prop->type() == typeid(tensor_t)) {
        m_tensor_prop = &m_grid->template vertex_property<tensor_t>(name);
        break;
      }
    }
    if (m_tensor_prop == nullptr) {
      throw std::runtime_error{"could not find any matching property"};
    }
  }
  //----------------------------------------------------------------------------
  explicit sampled_grid_property_field(std::filesystem::path const& path,
                                       std::string const& name) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    m_tensor_prop = &m_grid->template vertex_property<tensor_t>(name);
  }
  //----------------------------------------------------------------------------
  template <std::convertible_to<std::string>... CompNames>
  requires(sizeof...(CompNames) == num_dimensions())
  sampled_grid_property_field(std::filesystem::path const& path,
                              CompNames const&... comp_names) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    ((m_real_props.push_back(
         &m_grid->template vertex_property<real_t>(comp_names))),
     ...);
  }
  //----------------------------------------------------------------------------
  auto read(std::filesystem::path const& path) -> void {
    m_grid = std::make_shared<grid_t>();
    m_grid->read(path);
    for (auto const& [name, prop] : m_grid->vertex_properties()) {
      if (prop->type() == typeid(tensor_t)) {
        m_tensor_prop = &m_grid->template vertex_property<tensor_t>(name);
        break;
      }
    }
    if (m_tensor_prop == nullptr) {
      throw std::runtime_error{"could not find any matching property"};
    }
  }
  //----------------------------------------------------------------------------
  template <std::convertible_to<std::string>... CompNames>
  requires(sizeof...(CompNames) == num_dimensions())
  auto read(std::filesystem::path const& path, CompNames const&... comp_names)
      -> void {
    m_grid = std::make_shared<grid_t>();
    m_grid->read(path);
    ((m_real_props.push_back(
         &m_grid->template vertex_property<real_t>(comp_names))),
     ...);
  }
  //----------------------------------------------------------------------------
  sampled_grid_property_field(sampled_grid_property_field const&)     = default;
  sampled_grid_property_field(sampled_grid_property_field&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                       =(sampled_grid_property_field const&)
      -> sampled_grid_property_field& = default;
  auto operator                       =(sampled_grid_property_field&&) noexcept
      -> sampled_grid_property_field& = default;
  //----------------------------------------------------------------------------
  ~sampled_grid_property_field() override = default;
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto evaluate(pos_t const& x, [[maybe_unused]] real_t const t,
                std::index_sequence<Is...> /*seq*/) const -> tensor_t {
    if (m_tensor_prop != nullptr) {
      auto sampler = m_tensor_prop->template sampler<interpolation::linear>();
      if constexpr (is_time_dependent) {
        return sampler(x(Is)..., t);
      } else {
        return sampler(x(Is)...);
      }
    } else if (!m_real_props.empty()) {
      if constexpr (std::is_arithmetic_v<tensor_t>) {
        auto sampler =
            m_real_props.front()->template sampler<interpolation::linear>();
        if constexpr (is_time_dependent) {
          return sampler(x(Is)..., t);
        } else {
          return sampler(x(Is)...);
        }
      } else {
        tensor_t v;
        size_t   i = 0;
        for (auto prop : m_real_props) {
          auto sampler = prop->template sampler<interpolation::linear>();
          if constexpr (is_time_dependent) {
            v(i++) = sampler(x(Is)..., t);
          } else {
            v(i++) = sampler(x(Is)...);
          }
        }
        return v;
      }
    }
    if constexpr (std::is_arithmetic_v<tensor_t>) {
      return 0.0 / 0.0;
    } else {
      return tensor_t{tag::fill{0.0 / 0.0}};
    }
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    return evaluate(x, t, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  [[nodiscard]] auto in_domain(pos_t const& x, [[maybe_unused]] real_t const t,
                               std::index_sequence<Is...> /*seq*/) const
      -> bool {
    if (m_grid == nullptr) {
      return false;
    }
    if constexpr (is_time_dependent) {
      return m_grid->is_inside(x(Is)..., t);
    } else {
      return m_grid->is_inside(x(Is)...);
    }
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t t) const -> bool final {
    return in_domain(x, t, std::make_index_sequence<num_dimensions()>{});
  }
};
//==============================================================================
template <real_number Real, size_t N, bool is_time_dependent,
          size_t... TensorDims>
struct sampled_grid_property_field_creator;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, bool is_time_dependent,
          size_t... TensorDims>
using sampled_grid_property_field_creator_t =
    typename sampled_grid_property_field_creator<Real, N, is_time_dependent,
                                                 TensorDims...>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
struct sampled_grid_property_field_creator<Real, N, false, TensorDims...> {
  using type = sampled_grid_property_field<non_uniform_grid<Real, N>, Real, N,
                                           TensorDims...>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
struct sampled_grid_property_field_creator<Real, N, true, TensorDims...> {
  using type = sampled_grid_property_field<non_uniform_grid<Real, N + 1>, Real,
                                           N, TensorDims...>;
};
//==============================================================================
template <real_number Real, size_t N, size_t... TensorDims>
using time_dependent_sampled_grid_property_field =
    sampled_grid_property_field_creator_t<Real, N, true, TensorDims...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
using unsteady_sampled_grid_property_field =
    time_dependent_sampled_grid_property_field<Real, N, TensorDims...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
using time_independent_sampled_grid_property_field =
    sampled_grid_property_field_creator_t<Real, N, false, TensorDims...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
using steady_sampled_grid_property_field =
    time_independent_sampled_grid_property_field<Real, N, TensorDims...>;
//------------------------------------------------------------------------------
template <real_number Real, size_t N, size_t VecDim = N>
using time_dependent_sampled_grid_property_vectorfield =
    time_dependent_sampled_grid_property_field<Real, N, VecDim>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t VecDim = N>
using unsteady_sampled_grid_property_vectorfield =
    time_dependent_sampled_grid_property_vectorfield<Real, N, VecDim>;
//------------------------------------------------------------------------------
template <real_number Real, size_t N, size_t VecDim = N>
using time_independent_sampled_grid_property_vectorfield =
    time_independent_sampled_grid_property_field<Real, N, VecDim>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t VecDim = N>
using steady_sampled_grid_property_vectorfield =
    time_independent_sampled_grid_property_vectorfield<Real, N, VecDim>;
//------------------------------------------------------------------------------
template <real_number Real, size_t N>
using time_dependent_sampled_grid_property_scalarfield =
    time_dependent_sampled_grid_property_field<Real, N>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N>
using unsteady_sampled_grid_property_scalarfield =
    time_dependent_sampled_grid_property_scalarfield<Real, N>;
//------------------------------------------------------------------------------
template <real_number Real, size_t N>
using time_independent_sampled_grid_property_scalarfield =
    time_independent_sampled_grid_property_field<Real, N>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N>
using steady_sampled_grid_property_scalarfield =
    time_independent_sampled_grid_property_scalarfield<Real, N>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
