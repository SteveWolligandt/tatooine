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
    : vectorfield<vf_grid_time_dependent_prop<Dimensions...>,
                  typename grid<Dimensions...>::real_t,
                  sizeof...(Dimensions) - 1> {
  using this_t   = vf_grid_time_dependent_prop<Dimensions...>;
  using grid_t = grid<Dimensions...>;
  using parent_t =
      vectorfield<this_t, typename grid_t::real_t, sizeof...(Dimensions) - 1>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using tensor_prop_t = typename grid_t::template typed_property_t<tensor_t>;
  using real_prop_t = typename grid_t::template typed_property_t<real_t>;
  using parent_t::num_dimensions;
  //============================================================================
  std::shared_ptr<grid_t>   m_grid;
  tensor_prop_t*            m_tensor_prop = nullptr;
  std::vector<real_prop_t*> m_real_props;
  //----------------------------------------------------------------------------
  auto                    grid() const -> auto const& { return *m_grid; }
  auto                    grid() -> auto& { return *m_grid; }
  //============================================================================
  explicit vf_grid_time_dependent_prop(
      std::filesystem::path const& path) 
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
  explicit vf_grid_time_dependent_prop(
      std::filesystem::path const& path, std::string const& name) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    m_tensor_prop = &m_grid->template vertex_property<tensor_t>(name);
  }
  //----------------------------------------------------------------------------
  template <std::convertible_to<std::string>... CompNames>
  requires(sizeof...(CompNames) == num_dimensions())
  vf_grid_time_dependent_prop(std::filesystem::path const& path,
                              CompNames const&... comp_names) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    ((m_real_props.push_back(
         &m_grid->template vertex_property<real_t>(comp_names))),
     ...);
  }
  //----------------------------------------------------------------------------
  vf_grid_time_dependent_prop(vf_grid_time_dependent_prop const&) =
      default;
  vf_grid_time_dependent_prop(
      vf_grid_time_dependent_prop&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator             =(vf_grid_time_dependent_prop const&)
      -> vf_grid_time_dependent_prop& = default;
  auto operator             =(vf_grid_time_dependent_prop&&) noexcept
      -> vf_grid_time_dependent_prop& = default;
  //----------------------------------------------------------------------------
  ~vf_grid_time_dependent_prop() override = default;
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto evaluate(pos_t const& x, real_t const t,
                std::index_sequence<Is...> /*seq*/) const -> tensor_t {
    if (m_tensor_prop != nullptr) {
      auto sampler = m_tensor_prop->template sampler<interpolation::linear>();
      return sampler(x(Is)..., t);
    } else {
      tensor_t v;
      size_t i = 0;
      for (auto prop : m_real_props) {
        auto sampler = prop->template sampler<interpolation::linear>();
        v(i++)       = sampler(x(Is)..., t);
      }
      return v;
    }
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    return evaluate(x, t, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  [[nodiscard]] auto in_domain(
      pos_t const& x, real_t t, std::index_sequence<Is...> /*seq*/) const
      -> bool {
    return m_grid->is_inside(x(Is)..., t);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t t) const
      -> bool final {
    return in_domain(x, t, std::make_index_sequence<num_dimensions()>{});
  }
};
//==============================================================================
template <indexable_space... Dimensions>
struct vf_grid_prop
    : vectorfield<vf_grid_prop<Dimensions...>,
                  typename grid<Dimensions...>::real_t, sizeof...(Dimensions)> {
  using this_t = vf_grid_prop<Dimensions...>;
  using grid_t = grid<Dimensions...>;
  using parent_t =
      vectorfield<this_t, typename grid_t::real_t, sizeof...(Dimensions)>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using tensor_prop_t = typename grid_t::template typed_property_t<tensor_t>;
  using real_prop_t = typename grid_t::template typed_property_t<real_t>;
  using parent_t::num_dimensions;
  //============================================================================
  std::shared_ptr<grid_t>   m_grid;
  tensor_prop_t*            m_tensor_prop = nullptr;
  std::vector<real_prop_t*> m_real_props;
  //----------------------------------------------------------------------------
  auto                    grid() const -> auto const& { return *m_grid; }
  auto                    grid() -> auto& { return *m_grid; }
  //============================================================================
  explicit vf_grid_prop(
      std::filesystem::path const& path) 
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
  explicit vf_grid_prop(
      std::filesystem::path const& path, std::string const& name) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    m_tensor_prop = &m_grid->template vertex_property<tensor_t>(name);
  }
  //----------------------------------------------------------------------------
  template <std::convertible_to<std::string>... CompNames>
  requires(sizeof...(CompNames) == num_dimensions())
  vf_grid_prop(std::filesystem::path const& path,
                              CompNames const&... comp_names) noexcept
      : m_grid{std::make_shared<grid_t>(path)} {
    ((m_real_props.push_back(
         &m_grid->template vertex_property<real_t>(comp_names))),
     ...);
  }
  //----------------------------------------------------------------------------
  vf_grid_prop(vf_grid_prop const&) =
      default;
  vf_grid_prop(
      vf_grid_prop&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator             =(vf_grid_prop const&)
      -> vf_grid_prop& = default;
  auto operator             =(vf_grid_prop&&) noexcept
      -> vf_grid_prop& = default;
  //----------------------------------------------------------------------------
  ~vf_grid_prop() override = default;
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto evaluate(pos_t const& x, std::index_sequence<Is...> /*seq*/) const
      -> tensor_t {
    if (m_tensor_prop != nullptr) {
      auto sampler = m_tensor_prop->template sampler<interpolation::linear>();
      return sampler(x(Is)...);
    } else {
      tensor_t v;
      size_t   i = 0;
      for (auto prop : m_real_props) {
        auto sampler = prop->template sampler<interpolation::linear>();
        v(i++)       = sampler(x(Is)...);
      }
      return v;
    }
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const /*t*/) const
      -> tensor_t final {
    return evaluate(x, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  [[nodiscard]] auto in_domain(pos_t const& x,
                               std::index_sequence<Is...> /*seq*/) const
      -> bool {
    return m_grid->is_inside(x(Is)...);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t /*t*/) const
      -> bool final {
    return in_domain(x, std::make_index_sequence<num_dimensions()>{});
  }
};
//==============================================================================
template <typename V>
struct vf_split
    : vectorfield<vf_split<V>, typename V::real_t, V::num_dimensions() - 1> {
  using this_t = vf_split<V>;
  using parent_t =
      vectorfield<this_t, typename V::real_t, V::num_dimensions() - 1>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using parent_t::num_dimensions;
  //============================================================================
  V const& m_v;
  //============================================================================
  template <typename VReal, size_t N>
  explicit vf_split(vectorfield<V, VReal, N> const& v) : m_v{v.as_derived()} {}
  //----------------------------------------------------------------------------
  vf_split(vf_split const&)     = default;
  vf_split(vf_split&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~vf_split() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    auto const vt        = m_v(pt, t);
    tensor_t   v;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      v(i) = vt(i);
    }
    return v;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, real_t t) const
      -> bool final {
    vec<real_t, num_dimensions() + 1> pt;
    for (size_t i = 0; i < num_dimensions(); ++i) {
      pt(i) = x(i);
    }
    pt(num_dimensions()) = t;
    return m_v.in_domain(pt, t);
  }
};
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
