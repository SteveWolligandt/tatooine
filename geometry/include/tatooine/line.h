#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H
//==============================================================================
#include <tatooine/demangling.h>
#include <tatooine/detail/line/vertex_container.h>
#include <tatooine/detail/line/vtk_writer.h>
#include <tatooine/detail/line/vtp_writer.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/functional.h>
#include <tatooine/handle.h>
#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>
#include <tatooine/property.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>

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
  using this_type          = line<Real, NumDimensions>;
  using real_type          = Real;
  using vec_type           = vec<Real, NumDimensions>;
  using pos_type           = vec_type;
  using pos_container_type = std::deque<pos_type>;
  using value_type         = pos_type;
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
  using tangent_property_type =
      typed_vertex_property_type<vec<Real, NumDimensions>>;

  template <typename T, template <typename> typename InterpolationKernel>
  using vertex_property_sampler_type =
      detail::line::vertex_property_sampler<Real, NumDimensions, T,
                                            InterpolationKernel>;
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_dimensions() -> std::size_t {
    return NumDimensions;
  }
  //============================================================================
  // members
  //============================================================================
 private:
  pos_container_type m_vertices;
  bool               m_is_closed = false;

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
  auto operator=(line const& other) -> line& {
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
  auto operator=(line&& other) noexcept -> line& = default;
  //----------------------------------------------------------------------------
  line(pos_container_type const& data, bool is_closed = false)
      : m_vertices{data}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(pos_container_type&& data, bool is_closed = false)
      : m_vertices{std::move(data)}, m_is_closed{is_closed} {}
  //----------------------------------------------------------------------------
  line(fixed_size_real_vec<NumDimensions> auto&&... vs)
      : m_vertices{pos_type{std::forward<decltype(vs)>(vs)}...},
        m_is_closed{false} {}
  //----------------------------------------------------------------------------
  auto copy_without_properties() {}
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
  auto num_vertices() const { return m_vertices.size(); }
  auto num_line_segments() const {
    return (num_vertices() - (is_closed() ? 0 : 1));
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel>
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
    auto len = Real{};
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
    auto [it, suc] = m_vertex_properties.insert(std::pair{
        name, std::make_unique<typed_vertex_property_type<T>>(value)});
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
      std::size_t            i = 0;
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
    auto& t   = parameterization();
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
    auto& t   = parameterization();
    t.front() = t0;
    for (std::size_t i = 1; i < vertices().size(); ++i) {
      t[i] = t[i - 1] + euclidean_distance(vertex_at(i), vertex_at(i - 1));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_chordal_parameterization(Real const t0 = 0) -> void {
    compute_chordal_parameterization(t0);
    normalize_parameterization();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_centripetal_parameterization(Real const t0 = 0) -> void {
    auto& t   = parameterization();
    t.front() = t0;
    for (std::size_t i = 1; i < vertices().size(); ++i) {
      t[i] = t[i - 1] +
             std::sqrt(euclidean_distance(vertex_at(i), vertex_at(i - 1)));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_normalized_centripetal_parameterization(Real const t0 = 0)
      -> void {
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
    detail::line::vtk_writer{*this}.write(path, title);
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral HeaderType      = std::uint64_t,
            integral          ConnectivityInt = std::int64_t,
            integral          OffsetInt       = std::int64_t>
  auto write_vtp(filesystem::path const& path) const -> void {
    detail::line::vtp_writer<this_type, HeaderType, ConnectivityInt, OffsetInt>{
        *this}
        .write(path);
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
  static auto read_vtp(std::string const& filepath)
  requires(num_dimensions() == 3) {
    auto reader = vtk::xml::reader{filepath};
    if (reader.type() != vtk::xml::vtk_type::poly_data) {
      throw std::runtime_error{"[line::read_vtp] can only read from poly_data"};
    }
    return read(reader.poly_data()->pieces.front());
  }
  //----------------------------------------------------------------------------
  /// Reads data_array as vertex property if the number of components is equal
  /// to the template parameter N.
  template <std::size_t N>
  auto read_vtp_prop(std::string const          &name,
                     vtk::xml::data_array const &data_array)
  requires (num_dimensions() == 2) || (num_dimensions() == 3) {
    if (data_array.num_components() != N) {
      return;
    }
    auto data_type_getter = [&]<typename value_t>(value_t /*val*/) {
      using prop_t = std::conditional_t<N == 1, value_t, vec<value_t, N>>;
      auto &prop   = insert_vertex_property<prop_t>(name);
      auto  prop_data_setter = [&prop, i = std::size_t{},
                               this](std::vector<value_t> const &data) mutable {
        for (auto const v : vertices()) {
          auto &p = prop[v];
          if constexpr (N == 1) {
            p = data[i++];
          } else {
            for (std::size_t j = 0; j < N; ++j) {
              p(j) = data[i++];
            }
          }
        };
      };
      data_array.visit_data(prop_data_setter);
    };
    vtk::xml::visit(data_array.type(), data_type_getter);
  }
  //----------------------------------------------------------------------------
  /// Calls read_vtp_prop<N> with N = 1..10
  auto read_vtp_prop(std::string const&          name,
                     vtk::xml::data_array const& data_array)
  requires (num_dimensions() == 2) || (num_dimensions() == 3) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (std::invoke(&this_type::read_vtp_prop<Is + 1>, this, name, data_array),
       ...);
    } (std::make_index_sequence<10>{});
  }
  //----------------------------------------------------------------------------
  auto read_vtp_positions(vtk::xml::data_array const& points) 
  requires (num_dimensions() == 2) || (num_dimensions() == 3) {
    points.visit_data([&](auto&& point_data) {
      // always 3 components in vtk data array 
      for (std::size_t i = 0; i < point_data.size(); i += 3) { 
        if constexpr (num_dimensions() == 2) {
          // just omit third component when reading to a 3d line
          push_back(point_data[i], point_data[i + 1]);
        } else if constexpr (num_dimensions() == 3) {
          push_back(point_data[i], point_data[i + 1], point_data[i + 2]);
        }
      }
    });
  }
  //----------------------------------------------------------------------------
  /// TODO actually read connectivy data array from the lines tag
  static auto read(vtk::xml::piece const& p)
  requires (num_dimensions() == 2) || (num_dimensions() == 3) {
    auto l = this_type{};
    l.read_vtp_positions(p.points);
    for (auto const &[name, data_array] : p.point_data) {
      l.read_vtp_prop(name, data_array);
    }
    // p.lines.at("connectivity").visit_data([](auto&& connectivity) {
    //   auto i     = std::size_t{};
    //   auto left  = connectivity[i++];
    //   auto right = connectivity[i++];
    //   l.push_back(positions[left * 3], positions[left * 3 + 1],
    //               positions[left * 3 + 2]);
    //   l.push_back(positions[right * 3], positions[right * 3 + 1],
    //               positions[right * 3 + 2]);
    //   auto num_processed = std::size_t(1);
    //
    //   while (num_processed *2 != connectivity.size()) {
    //     for (std::size_t i = 1; i < connectivity.size(); ++i) {
    //
    //     }
    //   }
    // });
    return l;
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
      resampled_prop[v++] = prop_sampler(t);
    }
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename... Ts,
            floating_point ResampleSpaceReal>
  auto resample_vertex_property(
      this_type& resampled_line, std::string const& name,
      vertex_property_type const&        prop,
      linspace<ResampleSpaceReal> const& resample_space) {
    invoke([&] {
      if (prop.type() == typeid(Ts)) {
        resample_vertex_property<InterpolationKernel>(
            resampled_line, name,
            *dynamic_cast<typed_vertex_property_type<Ts> const*>(&prop),
            resample_space);
      }
    }...);
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
// type traits
//==============================================================================
/// All types that are no lines.
template <typename T>
struct is_line_impl : std::false_type {};
//------------------------------------------------------------------------------
/// All types are no lines.
template <typename Real, std::size_t N>
struct is_line_impl<line<Real, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
static auto constexpr is_line = is_line_impl<T>::value;
//==============================================================================
// concepts
//==============================================================================
template <typename T>
concept range_of_lines = range<T> && is_line<std::ranges::range_value_t<T>>;
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
auto line<Real, NumDimensions>::filter(Pred&& pred) const
    -> std::vector<line<Real, NumDimensions>> {
  auto filtered_lines = std::vector<line<Real, NumDimensions>>{};
  auto need_new_strip = true;

  auto i      = std::size_t{};
  auto closed = is_closed();
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
    ++i;
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
template <floating_point Real = real_number>
auto read_lines(filesystem::path const& filepath) {
  auto ls     = std::vector<Line3<Real>>{};
  auto reader = vtk::xml::reader{filepath};
  if (reader.type() != vtk::xml::vtk_type::poly_data) {
    throw std::runtime_error{"[read_lines] can only read from poly_data"};
  }
  for (auto& piece : reader.poly_data()->pieces) {
    ls.push_back(Line3<Real>::read(piece));
  }
  return ls;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/detail/line/operations.h>
#include <tatooine/detail/line/vertex_property_sampler.h>
//==============================================================================
#endif
