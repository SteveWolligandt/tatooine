#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H
//==============================================================================
#include <tatooine/demangling.h>
#include <tatooine/detail/line/vertex_container.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/handle.h>
#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>
#include <tatooine/property.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
#include <tatooine/vtk/xml.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/algorithm_ext/iota.hpp>
#include <cassert>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
//==============================================================================
namespace tatooine {
//============================================================================
namespace detail::line {
//============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          template <typename> typename InterpolationKernel>
struct vertex_property_sampler;
//============================================================================
}  // namespace detail::line
//============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct line {
  //============================================================================
  using this_type       = line<Real, NumDimensions>;
  using real_type       = Real;
  using vec_type        = vec<Real, NumDimensions>;
  using pos_type        = vec_type;
  using pos_container_t = std::deque<pos_type>;
  using value_type      = pos_type;
  //============================================================================
  // Handles
  //============================================================================
  struct vertex_handle : handle<vertex_handle> {
    using handle<vertex_handle>::handle;
  };

  using vertex_container_type =
      detail::line::vertex_container<Real, NumDimensions, vertex_handle>;
  friend struct detail::line::vertex_container<Real, NumDimensions,
                                               vertex_handle>;

  using vertex_property_type = deque_property<vertex_handle>;
  template <typename T>
  using typed_vertex_property_type = typed_deque_property<vertex_handle, T>;
  using vertex_property_container_type =
      std::map<std::string, std::unique_ptr<vertex_property_type>>;

  using parameterization_property_type = typed_vertex_property_type<Real>;
  using tangent_property_type = typed_vertex_property_type<vec<Real, NumDimensions>>;

  template <typename T, template <typename> typename InterpolationKernel>
  using vertex_property_sampler_type =
      detail::line::vertex_property_sampler<Real, NumDimensions, T,
                                            InterpolationKernel>;
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_dimensions() -> std::size_t noexcept { return NumDimensions; }

  //============================================================================
  // members
  //============================================================================
 private:
  pos_container_t m_vertices;
  bool            m_is_closed = false;

 protected:
  vertex_property_container_type  m_vertex_properties;
  parameterization_property_type* m_parameterization_property = nullptr;
  tangent_property_type*          m_tangent_property          = nullptr;

  //============================================================================
 public:
  line() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(line const& other)
      : m_vertices{other.m_vertices}, m_is_closed{other.m_is_closed} {
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
    if (other.m_parameterization_property) {
      m_parameterization_property = &vertex_property<Real>("parameterization");
    }
    if (other.m_tangent_property) {
      m_tangent_property =
          &vertex_property<vec<Real, NumDimensions>>("tangents");
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line(line&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line& operator=(line const& other) {
    m_vertices  = other.m_vertices;
    m_is_closed = other.m_is_closed;
    m_vertex_properties.clear();
    for (auto& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties[name] = prop->clone();
    }
    if (other.m_parameterization_property) {
      m_parameterization_property = &vertex_property<Real>("parameterization");
    }
    if (other.m_tangent_property) {
      m_tangent_property =
          &vertex_property<vec<Real, NumDimensions>>("tangents");
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line& operator=(line&& other) noexcept = default;
  //----------------------------------------------------------------------------
  line(pos_container_t const& data, bool is_closed = false)
      : m_vertices{data}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(pos_container_t&& data, bool is_closed = false)
      : m_vertices{std::move(data)}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(fixed_size_real_vec<NumDimensions> auto&&... vs)
      : m_vertices{pos_type{std::forward<decltype(vs)>(vs)}...},
        m_is_closed{false} {}
  //----------------------------------------------------------------------------
  auto copy_without_properties() {

  }
  //----------------------------------------------------------------------------
  auto empty() const { return m_vertices.empty(); }
  //----------------------------------------------------------------------------
  auto clear() { return m_vertices.clear(); }
  //============================================================================
  // vertex
  //============================================================================
  auto vertex_at(std::size_t const i) const -> auto const& {
    return m_vertices[i];
  }
  auto vertex_at(std::size_t const i) -> auto& { return m_vertices[i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertex_at(vertex_handle const i) const -> auto const& {
    return m_vertices[i.index()];
  }
  auto vertex_at(vertex_handle const i) -> auto& {
    return m_vertices[i.index()];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(vertex_handle const i) const -> auto const& {
    return m_vertices[i.index()];
  }
  auto at(vertex_handle const i) -> auto& { return m_vertices[i.index()]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](vertex_handle const i) const -> auto const& {
    return m_vertices[i.index()];
  }
  auto operator[](vertex_handle const i) -> auto& {
    return m_vertices[i.index()];
  }
  //----------------------------------------------------------------------------
  auto front_vertex() const -> auto const& { return m_vertices.front(); }
  auto front_vertex() -> auto& { return m_vertices.front(); }
  //----------------------------------------------------------------------------
  auto back_vertex() const -> auto const& { return m_vertices.back(); }
  auto back_vertex() -> auto& { return m_vertices.back(); }
  //----------------------------------------------------------------------------
  auto push_back(arithmetic auto const... components) requires(
      sizeof...(components) == NumDimensions) {
    m_vertices.push_back(pos_type{static_cast<Real>(components)...});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(pos_type const& p) {
    m_vertices.push_back(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(pos_type&& p) {
    m_vertices.emplace_back(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal>
  auto push_back(vec<OtherReal, NumDimensions> const& p) {
    m_vertices.push_back(pos_type{p});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto pop_back() { m_vertices.pop_back(); }
  //----------------------------------------------------------------------------
  auto push_front(arithmetic auto const... components) requires(
      sizeof...(components) == NumDimensions) {
    m_vertices.push_front(pos_type{static_cast<Real>(components)...});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_front(pos_type const& p) {
    m_vertices.push_front(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_front(pos_type&& p) {
    m_vertices.emplace_front(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal>
  auto push_front(vec<OtherReal, NumDimensions> const& p) {
    m_vertices.push_front(pos_type{p});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto pop_front() { m_vertices.pop_front(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertices() const { return vertex_container_type{*this}; }
  //----------------------------------------------------------------------------
  template <template <typename>
            typename InterpolationKernel = interpolation::cubic>
  auto sampler() const {
    return vertex_property_sampler_type<this_type, InterpolationKernel>{*this,
                                                                        *this};
  }
  //----------------------------------------------------------------------------
  auto linear_sampler() const { return sampler<interpolation::linear>(); }
  //----------------------------------------------------------------------------
  auto cubic_sampler() const { return sampler<interpolation::cubic>(); }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename T>
  auto sampler(typed_vertex_property_type<T> const& prop) const {
    return vertex_property_sampler_type<typed_vertex_property_type<T>,
                                        InterpolationKernel>{*this, prop};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_sampler(typed_vertex_property_type<T> const& prop) const {
    return sampler<interpolation::linear>(prop);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cubic_sampler(typed_vertex_property_type<T> const& prop) const {
    return sampler<interpolation::cubic>(prop);
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename T>
  auto vertex_property_sampler(std::string const& name) const {
    return sampler<InterpolationKernel>(vertex_property<T>(name));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(std::string const& name) const {
    return vertex_property_sampler<interpolation::linear, T>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cubic_vertex_property_sampler(std::string const& name) const {
    return vertex_property_sampler<interpolation::cubic, T>(name);
  }
  //============================================================================
  auto arc_length() const {
    Real len = 0;
    for (std::size_t i = 0; i < vertices().size() - 1; ++i) {
      len += euclidean_distance(vertex_at(i), vertex_at(i + 1));
    }
    return len;
  }
  //----------------------------------------------------------------------------
  auto is_closed() const { return m_is_closed; }
  auto set_closed(bool const is_closed = true) { m_is_closed = is_closed; }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) -> auto& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      return insert_vertex_property<T>(name);
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_vertex_property_type<T>*>(
          m_vertex_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) const -> const auto& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_vertex_property_type<T>*>(
          m_vertex_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<tatooine::real_number>(name);
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) -> auto& {
    return vertex_property<tatooine::real_number>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec2>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec3>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto vec4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat2>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat3>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  auto mat4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat4>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_vertex_property(std::string const& name, T const& value = T{})
      -> auto& {
    auto [it, suc] = m_vertex_properties.insert(
        std::pair{name, std::make_unique<typed_vertex_property_type<T>>(value)});
    auto prop = dynamic_cast<typed_vertex_property_type<T>*>(it->second.get());
    prop->resize(m_vertices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  auto insert_scalar_vertex_property(
      std::string const&          name,
      tatooine::real_number const value = tatooine::real_number{}) -> auto& {
    return insert_vertex_property<tatooine::real_number>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec2_vertex_property(
      std::string const& name, tatooine::vec2 const value = tatooine::vec2{})
      -> auto& {
    return insert_vertex_property<vec2>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec3_vertex_property(
      std::string const& name, tatooine::vec3 const value = tatooine::vec3{})
      -> auto& {
    return insert_vertex_property<vec3>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_vec4_vertex_property(
      std::string const& name, tatooine::vec4 const value = tatooine::vec4{})
      -> auto& {
    return insert_vertex_property<vec4>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat2_vertex_property(
      std::string const& name, tatooine::mat2 const value = tatooine::mat2{})
      -> auto& {
    return insert_vertex_property<mat2>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat3_vertex_property(
      std::string const& name, tatooine::mat3 const value = tatooine::mat3{})
      -> auto& {
    return insert_vertex_property<mat3>(name, value);
  }
  //----------------------------------------------------------------------------
  auto insert_mat4_vertex_property(
      std::string const& name, tatooine::mat4 const value = tatooine::mat4{})
      -> auto& {
    return insert_vertex_property<mat4>(name, value);
  }
  //----------------------------------------------------------------------------
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  //----------------------------------------------------------------------------
  auto has_vertex_property(const std::string& name) const -> bool {
    return m_vertex_properties.find(name) != end(m_vertex_properties);
  }
  //----------------------------------------------------------------------------
  auto tangents() -> auto& {
    if (!m_tangent_property) {
      m_tangent_property =
          &vertex_property<vec<Real, NumDimensions>>("tangents");
    }
    return *m_tangent_property;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto tangents() const -> auto const& {
    if (!m_tangent_property) {
      throw std::runtime_error{"no tangent property present"};
    }
    return *m_tangent_property;
  }
  //----------------------------------------------------------------------------
  auto compute_tangents(std::size_t const stencil_size = 3) {
    auto&      t    = parameterization();
    auto&      tang = tangents();
    auto const half = stencil_size / 2;

    for (auto const v : vertices()) {
      auto       lv         = half > v.index() ? vertex_handle{0} : v - half;
      auto const rv         = lv.index() + stencil_size - 1 >= vertices().size()
                                  ? vertex_handle{vertices().size() - 1}
                                  : lv + stencil_size - 1;
      auto const rpotential = stencil_size - (rv.index() - lv.index() + 1);
      lv = rpotential > lv.index() ? vertex_handle{0} : lv - rpotential;

      std::vector<real_type> ts(stencil_size);
      std::size_t         i = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        ts[i] = t[vi] - t[v];
      }
      auto coeffs = finite_differences_coefficients(1, ts);
      tang[v]     = vec<Real, NumDimensions>::zeros();
      i           = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        tang[v] += vertex_at(vi) * coeffs[i];
      }
    }
  }
  //----------------------------------------------------------------------------
  auto has_parameterization() const {
    return m_parameterization_property != nullptr;
  }
  //----------------------------------------------------------------------------
  auto parameterization() -> auto& {
    if (!has_parameterization()) {
      m_parameterization_property = &vertex_property<Real>("parameterization");
    }
    return *m_parameterization_property;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto parameterization() const -> auto const& {
    if (!has_parameterization()) {
      throw std::runtime_error{
          "Cannot create parameterization property on const line."};
    }
    return *m_parameterization_property;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_uniform_parameterization(Real const t0 = 0) -> void {
    auto& t               = parameterization();
    t.front() = t0;
    for (std::size_t i = 1; i < vertices().size(); ++i) {
      t[i] = t[i - 1] + 1;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_uniform_parameterization(Real const t0 = 0) -> void {
    compute_uniform_parameterization(t0);
    normalize_parameterization();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_chordal_parameterization(Real const t0 = 0) -> void {
    auto& t               = parameterization();
    t.front() = t0;
    for (std::size_t i = 1; i < vertices().size(); ++i) {
      t[i] =
          t[i - 1] + euclidean_distance(vertex_at(i), vertex_at(i - 1));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_chordal_parameterization(Real const t0 = 0) -> void {
    compute_chordal_parameterization(t0);
    normalize_parameterization();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_centripetal_parameterization(Real const t0 = 0) -> void {
    auto& t               = parameterization();
    t.front() = t0;
    for (std::size_t i = 1; i < vertices().size(); ++i) {
      t[i] = t[i - 1] +
             std::sqrt(euclidean_distance(vertex_at(i), vertex_at(i - 1)));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_centripetal_parameterization(Real const t0 = 0) -> void {
    compute_centripetal_parameterization(t0);
    normalize_parameterization();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_parameterization(Real const t0 = 0) -> void {
    compute_centripetal_parameterization(t0);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_parameterization(Real const t0 = 0) -> void {
    compute_parameterization(t0);
    normalize_parameterization();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto normalize_parameterization() -> void {
    auto&      t    = parameterization();
    auto const min  = t[vertices().front()];
    auto const max  = t[vertices().back()];
    auto const norm = 1 / (max - min);
    for (auto const v : vertices()) {
      t[v] = (t[v] - min) * norm;
    }
  }
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& path) -> void {
    auto const ext = path.extension();
    if constexpr (NumDimensions == 2 || NumDimensions == 3) {
      if (ext == ".vtk") {
        write_vtk(path);
        return;
      } else if (ext == ".vtp") {
        write_vtp(path);
        return;
      }
    }
    throw std::runtime_error(
        "Could not write line. Unknown file extension: \"" + ext.string() +
        "\".");
  }
  //----------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path,
                 std::string const& title = "tatooine line") const -> void {
    auto writer = vtk::legacy_file_writer{path, vtk::dataset_type::polydata};
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      auto ps = std::vector<std::array<Real, 3>>{};
      ps.reserve(vertices().size());
      for (auto const& v : vertices()) {
        auto const& p = at(v);
        if constexpr (NumDimensions == 3) {
          ps.push_back({p(0), p(1), p(2)});
        } else {
          ps.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(ps);

      // write lines
      auto line_seq = std::vector<std::vector<std::size_t>>(
          1, std::vector<std::size_t>(vertices().size()));
      boost::iota(line_seq.front(), 0);
      if (this->is_closed()) {
        line_seq.front().push_back(0);
      }
      writer.write_lines(line_seq);

      writer.write_point_data(vertices().size());

      // write properties
      for (auto& [name, prop] : m_vertex_properties) {
        auto const& type = prop->type();
        if (type == typeid(float)) {
          write_prop_to_vtk<float>(writer, name, prop);
        } else if (type == typeid(vec<float, 2>)) {
          write_prop_to_vtk<vec<float, 2>>(writer, name, prop);
        } else if (type == typeid(vec<float, 3>)) {
          write_prop_to_vtk<vec<float, 3>>(writer, name, prop);
        } else if (type == typeid(vec<float, 4>)) {
          write_prop_to_vtk<vec<float, 4>>(writer, name, prop);

        } else if (type == typeid(double)) {
          write_prop_to_vtk<double>(writer, name, prop);
        } else if (type == typeid(vec<double, 2>)) {
          write_prop_to_vtk<vec<double, 2>>(writer, name, prop);
        } else if (type == typeid(vec<double, 3>)) {
          write_prop_to_vtk<vec<double, 3>>(writer, name, prop);
        } else if (type == typeid(vec<double, 4>)) {
          write_prop_to_vtk<vec<double, 4>>(writer, name, prop);
        }
      }
      writer.close();
    }
  }
  //----------------------------------------------------------------------------
  auto write_vtp(filesystem::path const& path) const -> void {
    auto file = std::ofstream{path, std::ios::binary};
    if (!file.is_open()) {
      throw std::runtime_error{"Could not write " + path.string()};
    }
    auto offset              = std::size_t{};
    using header_type        = std::uint32_t;
    using connectivity_int_t = std::int32_t;
    using offset_int_t       = connectivity_int_t;
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\" "
            "byte_order=\"LittleEndian\""
         << " header_type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<header_type>())
         << "\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece"
         << " NumberOfPoints=\"" << vertices().size() << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"" << (vertices().size() - (is_closed() ? 0 : 1)) << "\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<real_type>())
         << "\" NumberOfComponents=\"3\"/>\n";
    auto const num_bytes_points =
        header_type(sizeof(real_type) * 3 * vertices().size());
    offset += num_bytes_points + sizeof(header_type);
    file << "      </Points>\n";

    // Lines
    file << "      <Lines>\n";
    // Lines - connectivity
    file << "        <DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_connectivity =
        (vertices().size() - (is_closed() ? 0 : 1)) * 2 * sizeof(connectivity_int_t);
    offset += num_bytes_connectivity + sizeof(header_type);
    // Lines - offsets
    file << "        <DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_offsets =
        sizeof(offset_int_t) * (vertices().size() - (is_closed() ? 0 : 1));
    offset += num_bytes_offsets + sizeof(header_type);
    file << "      </Lines>\n";
    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "  <AppendedData encoding=\"raw\">\n    _";
    // Writing vertex data to appended data section
    auto arr_size = header_type{};
    arr_size      = header_type(sizeof(real_type) * 3 * vertices().size());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    auto zero = real_type(0);
    for (auto const v : vertices()) {
      if constexpr (num_dimensions() == 2) {
        file.write(reinterpret_cast<char const*>(at(v).data()),
                   sizeof(real_type) * 2);
        file.write(reinterpret_cast<char const*>(&zero), sizeof(real_type));
      } else if constexpr (num_dimensions() == 3) {
        file.write(reinterpret_cast<char const*>(at(v).data()),
                   sizeof(real_type) * 3);
      }
    }

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<connectivity_int_t>{};
      connectivity_data.reserve((vertices().size() - (is_closed() ? 0 : 1)) * 2);
      for (std::size_t i = 0; i < vertices().size() - 1; ++i) {
        connectivity_data.push_back(i);
        connectivity_data.push_back(i + 1);
      }
      if (is_closed()) {
        connectivity_data.push_back(vertices().size() - 1);
        connectivity_data.push_back(0);
      }
      arr_size = connectivity_data.size() * sizeof(connectivity_int_t);
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 arr_size);
    }

    // Writing lines offsets to appended data section
    {
      auto offsets =
          std::vector<offset_int_t>(vertices().size() - (is_closed() ? 0 : 1), 2);
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      arr_size =
          sizeof(offset_int_t) * (vertices().size() - (is_closed() ? 0 : 1));
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);
    }
    file << "\n  </AppendedData>\n";
    file << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
  template <typename T>
  static auto write_prop_to_vtk(
      vtk::legacy_file_writer& writer, std::string const& name,
      std::unique_ptr<vertex_property_type> const& prop) -> void {
    auto const& deque =
        dynamic_cast<typed_vertex_property_type<T>*>(prop.get())->internal_container();

    writer.write_scalars(name, std::vector<T>(begin(deque), end(deque)));
  }
  //----------------------------------------------------------------------------
  static auto read_vtk(std::string const& filepath) requires(num_dimensions() ==
                                                             3) {
    struct reader : vtk::legacy_file_listener {
      std::vector<std::array<Real, 3>> points;
      std::vector<int>                 lines;

      void on_points(std::vector<std::array<Real, 3>> const& points_) override {
        points = points_;
      }
      void on_lines(std::vector<int> const& lines_) override { lines = lines_; }
    } listener;

    vtk::legacy_file file{filepath};
    file.add_listener(listener);
    file.read();

    std::vector<line<Real, 3>> lines;
    auto const&                vs = listener.points;
    std::size_t                i  = 0;
    while (i < listener.lines.size()) {
      auto const size = static_cast<std::size_t>(listener.lines[i++]);
      auto&      l    = lines.emplace_back();
      for (; i < size; ++i) {
        l.push_back({vs[i][0], vs[i][1], vs[i][2]});
      }
    }
    return lines;
  }
  //----------------------------------------------------------------------------
  template <typename Pred>
  std::vector<line<Real, NumDimensions>> filter(Pred&& pred) const;
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename T,
            floating_point ResampleSpaceReal>
  auto resample_vertex_property(
      this_type& resampled_line, std::string const& name,
      typed_vertex_property_type<T> const& prop,
      linspace<ResampleSpaceReal> const&   resample_space) {
    auto&      resampled_prop = resampled_line.vertex_property<T>(name);
    auto const prop_sampler   = sampler<InterpolationKernel>(prop);
    auto       v              = resampled_line.vertices().front();
    for (auto const t : resample_space) {
      resampled_prop[v] = prop_sampler(t);
      ++v;
    }
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename... Ts,
            floating_point ResampleSpaceReal>
  auto resample_vertex_property(
      this_type& resampled_line, std::string const& name,
      vertex_property_type const&        prop,
      linspace<ResampleSpaceReal> const& resample_space) {
    (
        [&] {
          if (prop.type() == typeid(Ts)) {
            resample_vertex_property<InterpolationKernel>(
                resampled_line, name,
                *dynamic_cast<typed_vertex_property_type<Ts> const*>(&prop),
                resample_space);
          }
        }(),
        ...);
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel,
            floating_point ResampleSpaceReal>
  auto resample(linspace<ResampleSpaceReal> const& resample_space) {
    this_type  resampled_line;
    auto&      p         = resampled_line.parameterization();
    auto const positions = sampler<InterpolationKernel>();

    for (auto const t : resample_space) {
      auto const v = resampled_line.push_back(positions(t));
      p[v]         = t;
    }
    for (auto const& [name, prop] : m_vertex_properties) {
      resample_vertex_property<
          InterpolationKernel, long double, double, float, vec<long double, 2>,
          vec<double, 2>, vec<float, 2>, vec<long double, 3>, vec<double, 3>,
          vec<float, 3>, vec<long double, 4>, vec<double, 4>, vec<float, 4>>(
          resampled_line, name, *prop, resample_space);
    }
    return resampled_line;
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename... Tensors, typename... Reals, std::size_t NumDimensions>
line(base_tensor<Tensors, Reals, NumDimensions>&&... vertices)
    -> line<common_type<Reals...>, NumDimensions>;
//==============================================================================
// typedefs
//==============================================================================
template <typename T>
using Line2 = line<T, 2>;
template <typename T>
using Line3 = line<T, 3>;
template <typename T>
using Line4 = line<T, 4>;
template <typename T>
using Line5 = line<T, 5>;
template <std::size_t NumDimensions>
using Line  = line<real_number, NumDimensions>;
using line2 = Line<2>;
using line3 = Line<3>;
using line4 = Line<4>;
using line5 = Line<5>;
//==============================================================================
// implementations
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
template <typename Pred>
std::vector<line<Real, NumDimensions>> line<Real, NumDimensions>::filter(
    Pred&& pred) const {
  std::vector<line<Real, NumDimensions>> filtered_lines;
  bool                                   need_new_strip = true;

  std::size_t i      = 0;
  bool        closed = is_closed();
  for (auto const x : vertices()) {
    if (pred(x, i)) {
      if (need_new_strip) {
        filtered_lines.emplace_back();
        need_new_strip = false;
      }
      filtered_lines.back().push_back(x);
    } else {
      closed         = false;
      need_new_strip = true;
      if (!filtered_lines.empty() &&
          filtered_lines.back().vertices().size() <= 1)
        filtered_lines.pop_back();
    }
    i++;
  }

  if (!filtered_lines.empty() && filtered_lines.back().vertices().size() <= 1) {
    filtered_lines.pop_back();
  }
  if (filtered_lines.size() == 1) {
    filtered_lines.front().set_closed(closed);
  }
  return filtered_lines;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/detail/line/operations.h>
#include <tatooine/detail/line/vertex_property_sampler.h>
#endif
