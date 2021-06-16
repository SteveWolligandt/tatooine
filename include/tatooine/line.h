#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H
//==============================================================================
#include <tatooine/demangling.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/handle.h>
#include <tatooine/interpolation.h>
#include <tatooine/line_vertex_container.h>
#include <tatooine/linspace.h>
#include <tatooine/property.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
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
template <typename Real, size_t N, typename T,
          template <typename> typename InterpolationKernel>
struct line_vertex_property_sampler;
template <typename Real, size_t N>
struct line {
  struct empty_exception : std::exception {};
  //============================================================================
  using this_t          = line<Real, N>;
  using real_t          = Real;
  using vec_t           = vec<Real, N>;
  using pos_t           = vec_t;
  using pos_container_t = std::deque<pos_t>;
  using value_type      = pos_t;

  //============================================================================
  // Handles
  //============================================================================
  struct vertex_handle : handle<vertex_handle> {
    using handle<vertex_handle>::handle;
  };

  using vertex_container_t =
      line_vertex_container<Real, N, vertex_handle>;
  friend struct line_vertex_container<Real, N, vertex_handle>;

  template <typename T>
  using vertex_property_t = deque_property_impl<vertex_handle, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<deque_property<vertex_handle>>>;

  using parameterization_property_t = vertex_property_t<Real>;
  using tangent_property_t = vertex_property_t<vec<Real, N>>;

  template <typename T, template <typename> typename InterpolationKernel>
  using vertex_property_sampler_t =
      line_vertex_property_sampler<Real, N, T, InterpolationKernel>;
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto num_dimensions() noexcept { return N; }

  //============================================================================
  // members
  //============================================================================
 private:
  pos_container_t m_vertices;
  bool            m_is_closed = false;

 protected:
  vertex_property_container_t  m_vertex_properties;
  parameterization_property_t* m_parameterization_property = nullptr;
  tangent_property_t*          m_tangent_property          = nullptr;

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
      m_parameterization_property =
          &vertex_property<Real>("parameterization");
    }
    if (other.m_tangent_property) {
      m_tangent_property =
          &vertex_property<vec<Real, N>>("tangents");
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
          &vertex_property<vec<Real, N>>("tangents");
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
#ifdef __cpp_concepts
  template <typename... Vertices>
  requires((is_vec<std::decay_t<Vertices>> &&
            std::is_arithmetic_v<typename std::decay_t<Vertices>::value_type> &&
            std::decay_t<Vertices>::num_components() == N) &&
           ...)
#else
  template <
      typename... Vertices,
      enable_if<((is_vec<std::decay_t<Vertices>> &&
                  is_arithmetic<typename std::decay_t<Vertices>::value_type> &&
                  std::decay_t<Vertices>::num_components() == N) &&
                 ...)> = true>

#endif
      line(Vertices&&... vertices)
      : m_vertices{pos_t{std::forward<Vertices>(vertices)}...},
        m_is_closed{false} {
  }
  //----------------------------------------------------------------------------
  auto num_vertices() const { return m_vertices.size(); }
  //----------------------------------------------------------------------------
  auto empty() const { return m_vertices.empty(); }
  //----------------------------------------------------------------------------
  auto clear() { return m_vertices.clear(); }
  //============================================================================
  // vertex
  //============================================================================
  auto vertex_at(size_t const i) const -> auto const& { return m_vertices[i]; }
  auto vertex_at(size_t const i) -> auto& { return m_vertices[i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertex_at(vertex_handle const i) const -> auto const& {
    return m_vertices[i.i];
  }
  auto vertex_at(vertex_handle const i) -> auto& { return m_vertices[i.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(vertex_handle const i) const -> auto const& {
    return m_vertices[i.i];
  }
  auto at(vertex_handle const i) -> auto& { return m_vertices[i.i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](vertex_handle const i) const -> auto const& {
    return m_vertices[i.i];
  }
  auto operator[](vertex_handle const i) -> auto& { return m_vertices[i.i]; }
  //----------------------------------------------------------------------------
  auto front_vertex() const -> auto const& { return m_vertices.front(); }
  auto front_vertex() -> auto& { return m_vertices.front(); }
  //----------------------------------------------------------------------------
  auto back_vertex() const -> auto const& { return m_vertices.back(); }
  auto back_vertex() -> auto& { return m_vertices.back(); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <arithmetic... Components>
  requires(sizeof...(Components) == N)
#else
  template <typename... Components,
            enable_if<is_arithmetic<Components...>> = true,
            enable_if<(sizeof...(Components) == N)> = true>
#endif
      auto push_back(Components... comps) {
    m_vertices.push_back(pos_t{static_cast<Real>(comps)...});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(pos_t const& p) {
    m_vertices.push_back(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_back(pos_t&& p) {
    m_vertices.emplace_back(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal>
  auto push_back(vec<OtherReal, N> const& p) {
    m_vertices.push_back(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_back();
    }
    return vertex_handle{m_vertices.size() - 1};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto pop_back() { m_vertices.pop_back(); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <arithmetic... Components>
  requires(sizeof...(Components) == N)
#else
  template <typename... Components,
            enable_if<is_arithmetic<Components...>> = true,
            enable_if<(sizeof...(Components) == N)> = true>
#endif
      auto push_front(Components... comps) {
    m_vertices.push_front(pos_t{static_cast<Real>(comps)...});
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_front(pos_t const& p) {
    m_vertices.push_front(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto push_front(pos_t&& p) {
    m_vertices.emplace_front(std::move(p));
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal>
  auto push_front(vec<OtherReal, N> const& p) {
    m_vertices.push_front(p);
    for (auto& [name, prop] : m_vertex_properties) {
      prop->push_front();
    }
    return vertex_handle{0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto pop_front() { m_vertices.pop_front(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertices() const { return vertex_container_t{*this}; }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel>
  auto sampler() const {
    return vertex_property_sampler_t<this_t, InterpolationKernel>{*this, *this};
  }
  //----------------------------------------------------------------------------
  auto linear_sampler() const { return sampler<interpolation::linear>(); }
  //----------------------------------------------------------------------------
  auto cubic_sampler() const { return sampler<interpolation::cubic>(); }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel, typename T>
  auto sampler(vertex_property_t<T> const& prop) const {
    return vertex_property_sampler_t<vertex_property_t<T>, InterpolationKernel>{
        *this, prop};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_sampler(vertex_property_t<T> const& prop) const {
    return sampler<interpolation::linear>(prop);
  }
  //----------------------------------------------------------------------------
  template < typename T>
  auto cubic_sampler(vertex_property_t<T> const& prop) const {
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
    for (size_t i = 0; i < this->num_vertices() - 1; ++i) {
      len += distance(vertex_at(i), vertex_at(i + 1));
    }
    return len;
  }
  //----------------------------------------------------------------------------
  bool is_closed() const { return m_is_closed; }
  void set_closed(bool is_closed) { m_is_closed = is_closed; }

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
      return *dynamic_cast<vertex_property_t<T>*>(
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
      return *dynamic_cast<vertex_property_t<T>*>(
          m_vertex_properties.at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<tatooine::real_t>(name);
  }
  //----------------------------------------------------------------------------
  auto scalar_vertex_property(std::string const& name) -> auto& {
    return vertex_property<tatooine::real_t>(name);
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
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(m_vertices.size());
    return *prop;
  }
  //----------------------------------------------------------------------------
  auto insert_scalar_vertex_property(
      std::string const&     name,
      tatooine::real_t const value = tatooine::real_t{}) -> auto& {
    return insert_vertex_property<tatooine::real_t>(name, value);
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
          &insert_vertex_property<vec<Real, N>>("tangents");
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
  auto compute_tangents(size_t const stencil_size = 3) {
    auto &t = parameterization();
    auto &tang = tangents();
    auto const half = stencil_size / 2;

    for (auto const v : vertices()) {
      auto       lv         = half > v.i ? vertex_handle{0} : v - half;
      auto const rv         = lv.i + stencil_size - 1 >= num_vertices()
                                  ? vertex_handle{num_vertices() - 1}
                                  : lv + stencil_size - 1;
      auto const rpotential = stencil_size - (rv.i - lv.i + 1);
      lv = rpotential > lv.i ? vertex_handle{0} : lv - rpotential;

      std::vector<real_t> ts(stencil_size);
      size_t              i = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        ts[i] = t[vi] - t[v];
      }
      auto coeffs = finite_differences_coefficients(1, ts);
      tang[v]     = vec<Real, N>::zeros();
      i           = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        tang[v] += vertex_at(vi) * coeffs[i];
      }
    }
  }
  //----------------------------------------------------------------------------
  auto parameterization() -> auto& {
    if (!m_parameterization_property) {
      m_parameterization_property =
          &insert_vertex_property<Real>("parameterization");
    }
    return *m_parameterization_property;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto parameterization() const -> auto const& {
    if (!m_parameterization_property) {
      throw std::runtime_error{"no parameterization property present"};
    }
    return *m_parameterization_property;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_uniform_parameterization(Real const t0 = 0) -> void {
    auto& t = parameterization();
    t[vertices().front()]    = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      t[vertex_handle{i}] = t[vertex_handle{i - 1}] + 1;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_chordal_parameterization(Real const t0 = 0) -> void {
    auto& t               = parameterization();
    t[vertices().front()] = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      t[vertex_handle{i}] =
          t[vertex_handle{i - 1}] + distance(vertex_at(i), vertex_at(i - 1));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_centripetal_parameterization(Real const t0 = 0) -> void {
    auto& t               = parameterization();
    t[vertices().front()] = t0;
    for (size_t i = 1; i < this->num_vertices(); ++i) {
      t[vertex_handle{i}] = t[vertex_handle{i - 1}] +
                            std::sqrt(distance(vertex_at(i), vertex_at(i - 1)));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto compute_parameterization(Real const t0 = 0) -> void {
    compute_centripetal_parameterization(t0);
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
  auto write(std::string const& file) -> void;
  //----------------------------------------------------------------------------
  static auto write(std::vector<line<Real, N>> const& line_set,
                    std::string const&                file) -> void;
  //----------------------------------------------------------------------------
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine line") const -> void {
    vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      std::vector<std::array<Real, 3>> ps;
      ps.reserve(this->num_vertices());
      for (auto const& v : vertices()) {
        auto const& p = at(v);
        if constexpr (N == 3) {
          ps.push_back({p(0), p(1), p(2)});
        } else {
          ps.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(ps);

      // write lines
      std::vector<std::vector<size_t>> line_seq(
          1, std::vector<size_t>(this->num_vertices()));
      boost::iota(line_seq.front(), 0);
      // if (this->is_closed()) {
      //  line_seq.front().push_back(0);
      //}
      writer.write_lines(line_seq);

      writer.write_point_data(this->num_vertices());

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
  template <typename T>
  static auto write_prop_to_vtk(
      vtk::legacy_file_writer& writer, std::string const& name,
      std::unique_ptr<deque_property<vertex_handle>> const& prop) -> void {
    auto const& deque =
        dynamic_cast<vertex_property_t<T>*>(prop.get())->container();

    writer.write_scalars(name, std::vector<T>(begin(deque), end(deque)));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename = void>
  requires(num_dimensions() == 3)
#else
  template <size_t _N = num_dimensions(), enable_if<(_N == 3)> = true>
#endif
  static auto read_vtk(std::string const& filepath) {
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
    size_t                     i  = 0;
    while (i < listener.lines.size()) {
      auto const size = static_cast<size_t>(listener.lines[i++]);
      auto&      l    = lines.emplace_back();
      for (; i < size; ++i) {
        l.push_back({vs[i][0], vs[i][1], vs[i][2]});
      }
    }
    return lines;
  }
  template <typename Pred>
  std::vector<line<Real, N>> filter(Pred&& pred) const;
};

template <typename... Tensors, typename... Reals, size_t N>
line(base_tensor<Tensors, Reals, N>&&... vertices)
    -> line<common_type<Reals...>, N>;

template <size_t N>
using Line  = line<real_t, N>;
using line2 = Line<2>;
using line3 = Line<3>;
using line4 = Line<4>;
using line5 = Line<5>;
//==============================================================================
template <typename Real, size_t N>
template <typename Pred>
std::vector<line<Real, N>> line<Real, N>::filter(Pred&& pred) const {
  std::vector<line<Real, N>> filtered_lines;
  bool                       need_new_strip = true;

  size_t i      = 0;
  bool   closed = is_closed();
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
      if (!filtered_lines.empty() && filtered_lines.back().num_vertices() <= 1)
        filtered_lines.pop_back();
    }
    i++;
  }

  if (!filtered_lines.empty() && filtered_lines.back().num_vertices() <= 1) {
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
#include <tatooine/line_operations.h>
#include <tatooine/line_vertex_property_sampler.h>
#endif
