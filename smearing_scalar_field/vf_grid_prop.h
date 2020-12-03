#ifndef TATOOINE_SMEARING_VF_FROM_GRID_H
#define TATOOINE_SMEARING_VF_FROM_GRID_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>

#include <memory>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
template <indexable_space... Dimensions>
struct vf_grid_time_dependent_prop
    : vectorfield<vf_grid_time_dependent_prop<Dimensions...>, double,
                  sizeof...(Dimensions) - 1> {
  using this_t   = vf_grid_time_dependent_prop<Dimensions...>;
  using parent_t = vectorfield<this_t, double, sizeof...(Dimensions) - 1>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using grid_t = grid<Dimensions...>;
  using prop_t = typename grid_t::template typed_property_t<tensor_t>;
  using parent_t::num_dimensions;
  //============================================================================
  std::shared_ptr<grid_t> m_grid;
  prop_t*                 m_prop = nullptr;
  auto                    grid() const -> auto const& { return *m_grid; }
  auto                    grid() -> auto& { return *m_grid; }
  //============================================================================
  explicit constexpr vf_grid_time_dependent_prop(
      std::filesystem::path const& path) 
      : m_grid{std::make_shared<grid_t>(path)} {
    for (auto const& [name, prop] : m_grid->vertex_properties()) {
      if (prop->type() == typeid(tensor_t)) {
        m_prop = &m_grid->template vertex_property<tensor_t>(name);
        break;
      }
    }

    if (m_prop == nullptr) {
      throw std::runtime_error{"could not find any matching property"};
    }
  }
  explicit constexpr vf_grid_time_dependent_prop(
      std::filesystem::path const& path, std::string const& name) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    m_prop = &m_grid->template vertex_property<tensor_t>(name);
  }
  //------------------------------------------------------------------------------
  constexpr vf_grid_time_dependent_prop(vf_grid_time_dependent_prop const&) =
      default;
  constexpr vf_grid_time_dependent_prop(
      vf_grid_time_dependent_prop&&) noexcept = default;
  //------------------------------------------------------------------------------
  constexpr auto operator             =(vf_grid_time_dependent_prop const&)
      -> vf_grid_time_dependent_prop& = default;
  constexpr auto operator             =(vf_grid_time_dependent_prop&&) noexcept
      -> vf_grid_time_dependent_prop& = default;
  //------------------------------------------------------------------------------
  ~vf_grid_time_dependent_prop() override = default;
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto evaluate(pos_t const& x, double const t,
                          std::index_sequence<Is...> /*seq*/) const
      -> tensor_t {
    auto sampler = m_prop->template sampler<interpolation::linear>();
    return sampler(x(Is)..., t);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& x, double const t) const
      -> tensor_t final {
    return evaluate(x, t, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  [[nodiscard]] constexpr auto in_domain(
      pos_t const& x, double t, std::index_sequence<Is...> /*seq*/) const
      -> bool {
    return m_grid->is_inside(x(Is)..., t);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_t const& x, double t) const
      -> bool final {
    return in_domain(x, t, std::make_index_sequence<num_dimensions()>{});
  }
};
//==============================================================================
template <indexable_space... Dimensions>
struct vf_grid_prop
    : vectorfield<vf_grid_prop<Dimensions...>, double, sizeof...(Dimensions)> {
  using this_t   = vf_grid_prop<Dimensions...>;
  using parent_t = vectorfield<this_t, double, sizeof...(Dimensions)>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using grid_t = grid<Dimensions...>;
  using prop_t = typename grid_t::template typed_property_t<tensor_t>;
  using parent_t::num_dimensions;
  //============================================================================
  std::shared_ptr<grid_t> m_grid;
  prop_t*                 m_prop = nullptr;
  //============================================================================
  explicit constexpr vf_grid_prop(std::filesystem::path const& path) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    for (auto const& [name, prop] : m_grid->vertex_properties()) {
      if (prop.type() == typeid(tensor_t)) {
        m_prop = &m_grid->template vertex_property<tensor_t>(name);
        break;
      }
    }

    if (m_prop == nullptr) {
      throw std::runtime_error{"could not find any matching property"};
    }
  }
  explicit constexpr vf_grid_prop(std::filesystem::path const& path,
                                  std::string const&           name) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    m_prop = &m_grid->template vertex_property<tensor_t>(name);
  }
  //------------------------------------------------------------------------------
  constexpr vf_grid_prop(vf_grid_prop const&)     = default;
  constexpr vf_grid_prop(vf_grid_prop&&) noexcept = default;
  //------------------------------------------------------------------------------
  constexpr auto operator=(vf_grid_prop const&) -> vf_grid_prop& = default;
  constexpr auto operator=(vf_grid_prop&&) noexcept -> vf_grid_prop& = default;
  //------------------------------------------------------------------------------
  ~vf_grid_prop() override = default;
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto evaluate(pos_t const& x,
                          std::index_sequence<Is...> /*seq*/) const
      -> tensor_t {
    auto sampler = m_prop->template sampler<interpolation::linear>();
    return sampler(x(Is)...);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& x, double /*t*/) const
      -> tensor_t final {
    return evaluate(x, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(pos_t const& x, double /*t*/) const
      -> bool final {
    return m_grid->is_inside(x);
  }
};
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
