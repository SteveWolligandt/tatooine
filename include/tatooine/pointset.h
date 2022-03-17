#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H
//==============================================================================
#include <tatooine/available_libraries.h>
#include <tatooine/thin_plate_spline.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/iterator_facade.h>

#include <boost/range/algorithm/find.hpp>
#if TATOOINE_FLANN_AVAILABLE
#include <flann/flann.hpp>
#endif
#include <tatooine/concepts.h>
#include <tatooine/field.h>
#include <tatooine/handle.h>
#include <tatooine/polynomial.h>
#include <tatooine/property.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/vtk/xml/data_array.h>
#include <tatooine/vtk_legacy.h>

#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail::pointset {
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> F>
struct moving_least_squares_sampler;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T>
struct inverse_distance_weighting_sampler;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> F>
struct radial_basis_functions_sampler;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T,
          invocable<Real> F>
struct radial_basis_functions_sampler_with_polynomial;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct vertex_container;
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct const_vertex_container;
}  // namespace detail::pointset
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct pointset {
  // static constexpr std::size_t triangle_dims = 2;
  // static constexpr std::size_t tetgen_dims = 3;
  static constexpr auto num_dimensions() { return NumDimensions; }
  using real_type = Real;
  using this_type = pointset<Real, NumDimensions>;
  using vec_type     = vec<Real, NumDimensions>;
  using pos_type  = vec_type;
#if TATOOINE_FLANN_AVAILABLE || defined(TATOOINE_DOC_ONLY)
  using flann_index_type = flann::Index<flann::L2<Real>>;
#endif
  //----------------------------------------------------------------------------
  struct vertex_handle : handle<vertex_handle> {
    using handle<vertex_handle>::handle;
    using handle<vertex_handle>::operator=;
  };
  //----------------------------------------------------------------------------
  using vertex_container =
      detail::pointset::vertex_container<Real, NumDimensions>;
  //----------------------------------------------------------------------------
  using const_vertex_container =
      detail::pointset::const_vertex_container<Real, NumDimensions>;
  //----------------------------------------------------------------------------
  using vertex_property_type = vector_property<vertex_handle>;
  template <typename T>
  using typed_vertex_property_type = typed_vector_property<vertex_handle, T>;
  using vertex_property_container_type =
      std::map<std::string, std::unique_ptr<vertex_property_type>>;
  template <typename T>
  using inverse_distance_weighting_sampler_type =
      detail::pointset::inverse_distance_weighting_sampler<Real, NumDimensions,
                                                           T>;
  //============================================================================
 private:
  std::vector<pos_type>          m_vertex_position_data;
  std::set<vertex_handle>        m_invalid_vertices;
  vertex_property_container_type m_vertex_properties;
#if TATOOINE_FLANN_AVAILABLE || defined(TATOOINE_DOC_ONLY)
  mutable std::unique_ptr<flann_index_type> m_kd_tree;
  mutable std::mutex                     m_flann_mutex;
#endif
  //============================================================================
 public:
  pointset()  = default;
  ~pointset() = default;
  //----------------------------------------------------------------------------
  pointset(std::initializer_list<pos_type>&& vertices)
      : m_vertex_position_data(std::move(vertices)) {}
  //----------------------------------------------------------------------------
  // #ifdef USE_TRIANGLE
  //   pointset(const triangle::io& io) {
  //     for (int i = 0; i < io.numberofpoints; ++i)
  //       insert_vertex(io.pointlist[i * 2], io.pointlist[i * 2 + 1]);
  //   }
  // #endif

  // template <typename = void>
  // requires (NumDimensions == 3)
  // pointset(const tetgenio& io) {
  //   for (int i = 0; i < io.numberofpoints; ++i)
  //     insert_vertex(io.pointlist[i * 3], io.pointlist[i * 3 + 1],
  //                   io.pointlist[i * 3 + 2]);
  // }
  //----------------------------------------------------------------------------
  pointset(pointset const& other)
      : m_vertex_position_data(other.m_vertex_position_data),
        m_invalid_vertices(other.m_invalid_vertices) {
    vertex_properties().clear();
    for (auto const& [name, prop] : other.vertex_properties())
      vertex_properties().insert(std::pair{name, prop->clone()});
  }
  //----------------------------------------------------------------------------
  pointset(pointset&& other) noexcept
      : m_vertex_position_data(std::move(other.m_vertex_position_data)),
        m_invalid_vertices(std::move(other.m_invalid_vertices)),
        m_vertex_properties(std::move(other.m_vertex_properties)) {}
  //----------------------------------------------------------------------------
  pointset(std::vector<pos_type> const& vertices)
      : m_vertex_position_data(vertices) {}
  //----------------------------------------------------------------------------
  pointset(std::vector<pos_type>&& vertices)
      : m_vertex_position_data(std::move(vertices)) {}
  //----------------------------------------------------------------------------
  auto operator=(pointset const& other) -> pointset& {
    if (&other == this) {
      return *this;
    }
    vertex_properties().clear();
    m_vertex_position_data = other.m_vertex_position_data;
    m_invalid_vertices     = other.m_invalid_vertices;
    for (auto const& [name, prop] : other.vertex_properties()) {
      vertex_properties().emplace(name, prop->clone());
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(pointset&& other) noexcept -> pointset& = default;
  //----------------------------------------------------------------------------
  auto axis_aligned_bounding_box() const {
    auto aabb = tatooine::axis_aligned_bounding_box<Real, NumDimensions>{};
    for (auto v : vertices()) {
      aabb += at(v);
    }
    return aabb;
  }
  //----------------------------------------------------------------------------
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  auto vertex_properties() -> auto& { return m_vertex_properties; }
  //----------------------------------------------------------------------------
  auto at(vertex_handle const v) -> auto& {
    return vertex_position_data()[v.index()];
  }
  auto at(vertex_handle const v) const -> auto const& {
    return vertex_position_data()[v.index()];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(vertex_handle const v) -> auto& {
    assert(v.index() < vertex_position_data().size());
    return vertex_position_data()[v.index()];
  }
  auto vertex_at(vertex_handle const v) const -> auto const& {
    assert(v.index() < vertex_position_data().size());
    return vertex_position_data()[v.index()];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(std::size_t const i) -> auto& {
    assert(i < vertex_position_data().size());
    return vertex_position_data()[i];
  }
  auto vertex_at(std::size_t const i) const -> auto const& {
    assert(i < vertex_position_data().size());
    return vertex_position_data()[i];
  }
  //----------------------------------------------------------------------------
  auto operator[](vertex_handle const v) -> auto& { return at(v); }
  auto operator[](vertex_handle const v) const -> auto const& { return at(v); }
  //----------------------------------------------------------------------------
  auto vertices() const { return const_vertex_container{this}; }
  auto vertices() { return vertex_container{this}; }
  //----------------------------------------------------------------------------
 protected:
  auto vertex_position_data() -> auto& { return m_vertex_position_data; }
  auto vertex_position_data() const -> auto const& {
    return m_vertex_position_data;
  }
  //----------------------------------------------------------------------------
  auto invalid_vertices() -> auto& { return m_invalid_vertices; }
  auto invalid_vertices() const -> auto const& { return m_invalid_vertices; }
  //----------------------------------------------------------------------------
 public:
  ///\{
  auto insert_vertex(arithmetic auto const... ts) requires(sizeof...(ts) ==
                                                           NumDimensions) {
    vertex_position_data().push_back(pos_type{static_cast<Real>(ts)...});
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type const& v) {
    vertex_position_data().push_back(v);
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type&& v) {
    vertex_position_data().emplace_back(std::move(v));
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
  ///\}
  //----------------------------------------------------------------------------
  /// tidies up invalid vertices
  auto tidy_up() {
    for (auto const v : invalid_vertices()) {
      vertex_position_data().erase(begin(vertex_position_data()) + v.index());
      for (auto const& [key, prop] : vertex_properties()) {
        prop->erase(v.index());
      }
    }
    invalid_vertices().clear();
  }
  //----------------------------------------------------------------------------
  auto remove(vertex_handle const v) {
    if (is_valid(v) &&
        std::ranges::find(invalid_vertices(), v) == end(invalid_vertices())) {
      invalid_vertices().insert(v);
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto is_valid(vertex_handle const v) const -> bool {
    return v.is_valid() &&
           std::ranges::find(invalid_vertices(), v) == end(invalid_vertices());
  }

  //----------------------------------------------------------------------------
  auto clear_vertices() {
    vertex_position_data().clear();
    vertex_position_data().shrink_to_fit();
    invalid_vertices().clear();
    for (auto& [key, val] : vertex_properties())
      val->clear();
  }
  auto clear() { clear_vertices(); }

  //----------------------------------------------------------------------------
  auto join(this_type const& other) {
    for (auto v : other.vertices()) {
      insert_vertex(other[v]);
    }
  }
  //----------------------------------------------------------------------------
  auto find_duplicates(Real eps = 1e-6) {
    std::vector<std::pair<vertex_handle, vertex_handle>> duplicates;
    for (auto v0 = vertices().begin(); v0 != vertices().end(); ++v0)
      for (auto v1 = next(v0); v1 != vertices().end(); ++v1)
        if (approx_equal(at(v0), at(v1), eps)) duplicates.emplace_back(v0, v1);

    return duplicates;
  }
  //#ifdef USE_TRIANGLE
  //----------------------------------------------------------------------------
  //  template <typename = void>
  //  requires (NumDimensions == triangle_dims>>
  //  auto to_triangle_io() const {
  //    triangle::io in;
  //    std::size_t       i    = 0;
  //    in.pointlist      = new triangle::Real[in.numberofpoints *
  //    triangle_dims]; for (auto v : vertices()) {
  //      for (std::size_t j = 0; j < triangle_dims; ++j) {
  //        in.pointlist[i++] = at(v)(j);
  //      }
  //    }
  //
  //    return in;
  //  }
  //
  //--------------------------------------------------------------------------
  //  template <typename vertex_cont_t>
  //  requires (NumDimensions == triangle_dims)
  //  auto to_triangle_io(vertex_cont_t const& vertices) const {
  //    triangle::io in;
  //    std::size_t       i    = 0;
  //    in.numberofpoints = vertices().size();
  //    in.pointlist      = new triangle::Real[in.numberofpoints *
  //    triangle_dims]; for (auto v : vertices()) {
  //      for (std::size_t j = 0; j < triangle_dims; ++j) {
  //        in.pointlist[i++] = at(v)(j);
  //      }
  //    }
  //
  //    return in;
  //  }
  //#endif

  //----------------------------------------------------------------------------
  // template <typename = void>
  // requires (NumDimensions == tetgen_dims)
  // void to_tetgen_io(tetgenio& in) const {
  //   std::size_t i           = 0;
  //   in.numberofpoints  = vertices().size();
  //   in.pointlist       = new tetgen::Real[in.numberofpoints * tetgen_dims];
  //   in.pointmarkerlist = new int[in.numberofpoints];
  //   in.numberofpointattributes = 1;
  //   in.pointattributelist =
  //       new tetgen::Real[in.numberofpoints * in.numberofpointattributes];
  //   for (auto v : vertices()) {
  //     for (std::size_t j = 0; j < tetgen_dims; ++j) {
  //       in.pointlist[i * 3 + j] = at(v)(j);
  //     }
  //     in.pointmarkerlist[i]    = i;
  //     in.pointattributelist[i] = v.index();
  //     ++i;
  //   }
  // }

  //----------------------------------------------------------------------------
  /// using specified vertices of point_set
  // template <typename vertex_cont_t>
  // auto to_tetgen_io(vertex_cont_t const& vertices) const {
  //   tetgenio io;
  //   std::size_t       i    = 0;
  //   io.numberofpoints = vertices.size();
  //   io.pointlist      = new tetgen_real_type[io.numberofpoints * 3];
  //   for (auto v : vertices) {
  //     auto const& x       = at(v);
  //     io.pointlist[i]     = x(0);
  //     io.pointlist[i + 1] = x(1);
  //     io.pointlist[i + 2] = x(2);
  //     i += 2;
  //   }
  //
  //   return io;
  // }
  //----------------------------------------------------------------------------
  /// \{
  template <typename T>
  auto vertex_property(std::string const& name) -> auto& {
    if (auto it = vertex_properties().find(name);
        it == end(vertex_properties())) {
      return insert_vertex_property<T>(name);
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_vertex_property_type<T>*>(
          vertex_properties().at(name).get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) const -> const auto& {
    if (auto it = vertex_properties().find(name);
        it == end(vertex_properties())) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_vertex_property_type<T>*>(
          vertex_properties().at(name).get());
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
    auto [it, suc] = vertex_properties().insert(std::pair{
        name, std::make_unique<typed_vertex_property_type<T>>(value)});
    auto prop = dynamic_cast<typed_vertex_property_type<T>*>(it->second.get());
    prop->resize(vertex_position_data().size());
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
  /// \}
  //----------------------------------------------------------------------------
  /// \{
  auto write(filesystem::path const& path) const {
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
        "Could not write pointset. Unknown file extension: \"" + ext.string() +
        "\".");
  }
  //----------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path,
                 std::string const&      title = "Tatooine pointset") const
      -> void
  requires(NumDimensions == 3 || NumDimensions == 2)
  {
    auto writer = vtk::legacy_file_writer {path, vtk::dataset_type::polydata};
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      write_vertices_vtk(writer);
      write_prop_vtk<int, float, double, vec2f, vec3f, vec4f, vec2d, vec3d,
                     vec4d>(writer);
      writer.close();
    }
  }
  //----------------------------------------------------------------------------
 private:
  auto write_vertices_vtk(vtk::legacy_file_writer& writer) const {
    using namespace std::ranges;
    if constexpr (NumDimensions == 2) {
      auto three_dims = [](vec<Real, 2> const& v2) {
        return vec<Real, 3>{v2(0), v2(1), 0};
      };
      auto v3s               = std::vector<vec<Real, 3>>(vertices().size());
      auto three_dimensional = views::transform(three_dims);
      copy(vertex_position_data() | three_dimensional, begin(v3s));
      writer.write_points(v3s);
    } else if constexpr (NumDimensions == 3) {
      writer.write_points(vertex_position_data());
    }

    auto vertex_indices = std::vector<std::vector<std::size_t>>(
        1, std::vector<std::size_t>(vertices().size()));
    copy(views::iota(std::size_t(0), vertices().size()),
         vertex_indices.front().begin());
    writer.write_vertices(vertex_indices);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto write_prop_vtk(
      vtk::legacy_file_writer& writer, std::string const& name,
      typed_vertex_property_type<T> const& prop)
      const -> void {
    writer.write_scalars(name, prop.container());
  }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_prop_vtk(vtk::legacy_file_writer& writer) const -> void {
    if (!vertex_properties().empty()) {
      writer.write_point_data(vertices().size());
    }
    for (const auto& [name, prop] : this->m_vertex_properties) {
      ([&] {
        if (prop->type() == typeid(Ts)) {
          write_prop_vtk(writer, name, prop->template cast_to_typed<Ts>());
        }
      }(), ...);
    }
  }
  //----------------------------------------------------------------------------
  auto write_vtp(filesystem::path const& path) const {
    auto file = std::ofstream{path, std::ios::binary};
    if (!file.is_open()) {
      throw std::runtime_error{"Could not write " + path.string()};
    }
    // tidy_up();
    auto offset                    = std::size_t{};
    using header_type              = std::uint64_t;
    using verts_connectivity_int_type = std::int64_t;
    using verts_offset_int_type       = verts_connectivity_int_type;
    auto const num_bytes_points =
        header_type(sizeof(Real) * 3 * vertices().size());
    auto const num_bytes_verts_connectivity =
        vertices().size() * sizeof(verts_connectivity_int_type);
    auto const num_bytes_verts_offsets =
        sizeof(verts_offset_int_type) * vertices().size();
    file << "<VTKFile"
         << " type=\"PolyData\""
         << " version=\"1.0\""
         << " byte_order=\"LittleEndian\""
         << " header_type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<header_type>())
         << "\">\n"
         << "<PolyData>\n"
         << "<Piece"
         << " NumberOfPoints=\"" << vertices().size() << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"" << vertices().size() << "\""
         << " NumberOfLines=\"0\""
         << " NumberOfStrips=\"0\""
         << ">\n"
         // Points
         << "<Points>"
         << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<Real>())
         << "\" NumberOfComponents=\"3\"/>"
         << "</Points>\n";
    offset += num_bytes_points + sizeof(header_type);
    // Verts
    file << "<Verts>\n"
         // Verts - connectivity
         << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<verts_connectivity_int_type>())
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes_verts_connectivity + sizeof(header_type);
    // Verts - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<verts_offset_int_type>())
         << "\" Name=\"offsets\"/>\n";
    offset += num_bytes_verts_offsets + sizeof(header_type);
    file << "</Verts>\n";

    {
      // Writing vertex data to appended data section
      file << "<PointData>\n";
      for (auto const& [name, prop] : vertex_properties()) {
        offset += write_vertex_property_data_array_vtp<float, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec2f, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec3f, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec4f, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<double, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec2d, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec3d, header_type>(
            name, prop, file, offset);
        offset += write_vertex_property_data_array_vtp<vec4d, header_type>(
            name, prop, file, offset);
      }
      file << "</PointData>\n";
    }
    file << "</Piece>\n"
         << "</PolyData>\n"
         << "<AppendedData encoding=\"raw\">\n_";

    using namespace std::ranges;
    {
      file.write(reinterpret_cast<char const*>(&num_bytes_points),
                 sizeof(header_type));
      if constexpr (NumDimensions == 2) {
        auto point_data      = std::vector<vec<Real, 3>>(vertices().size());
        auto position        = [this](auto const v) -> auto& { return at(v); };
        constexpr auto to_3d = [](auto const& p) {
          return vec{p.x(), p.y(), Real(0)};
        };
        copy(vertex_position_data() | views::transform(to_3d),
             begin(point_data));
        file.write(reinterpret_cast<char const*>(point_data.data()),
                   num_bytes_points);
      } else if constexpr (NumDimensions == 3) {
        file.write(reinterpret_cast<char const*>(vertices().data()),
                   num_bytes_points);
      }
    }

    // Writing verts connectivity data to appended data section
    {
      auto connectivity_data =
          std::vector<verts_connectivity_int_type>(vertices().size());
      copy(views::iota(verts_connectivity_int_type(0),
                       verts_connectivity_int_type(vertices().size())),
           begin(connectivity_data));
      file.write(reinterpret_cast<char const*>(&num_bytes_verts_connectivity),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 num_bytes_verts_connectivity);
    }

    // Writing verts offsets to appended data section
    {
      auto offsets = std::vector<verts_offset_int_type>(vertices().size(), 1);
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      };
      file.write(reinterpret_cast<char const*>(&num_bytes_verts_offsets),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()),
                 num_bytes_verts_offsets);
    }
    for (auto const& [name, prop] : vertex_properties()) {
      write_vertex_property_appended_data_vtp<float, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec2f, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec3f, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec4f, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<double, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec2d, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec3d, header_type>(prop, file);
      write_vertex_property_appended_data_vtp<vec4d, header_type>(prop, file);
    }
    file << "\n</AppendedData>\n"
         << "</VTKFile>";
  }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  template <typename T, typename header_type>
  auto write_vertex_property_data_array_vtp(auto const& name, auto const& prop,
                                            auto& file, auto offset) const
      -> std::size_t {
    if (prop->type() == typeid(T)) {
      file << "<DataArray"
           << " Name=\"" << name << "\""
           << " format=\"appended\""
           << " offset=\"" << offset << "\""
           << " type=\""
           << vtk::xml::data_array::to_string(
                  vtk::xml::data_array::to_type<tensor_value_type<T>>())
           << "\" NumberOfComponents=\""
           << tensor_num_components<T> << "\"/>\n";
      return vertices().size() * sizeof(T) + sizeof(header_type);
    }
    return 0;
  }
  //----------------------------------------------------------------------------
  template <typename T, typename header_type>
  auto write_vertex_property_appended_data_vtp(auto const& prop,
                                               auto&       file) const {
    if (prop->type() == typeid(T)) {
      auto const num_bytes =
          header_type(sizeof(tensor_value_type<T>) * tensor_num_components<T> *
                      vertices().size());
      file.write(reinterpret_cast<char const*>(&num_bytes),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(
                     prop->template cast_to_typed<T>().data()),
                 num_bytes);
    }
  }
  /// \}
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
#if TATOOINE_FLANN_AVAILABLE || defined(TATOOINE_DOC_ONLY)
  /// \{
  auto rebuild_kd_tree() {
    invalidate_kd_tree();
    build_kd_tree();
  }
  //----------------------------------------------------------------------------
 private:
  auto build_kd_tree() const -> auto& {
    auto lock = std::scoped_lock{m_flann_mutex};
    if (m_kd_tree == nullptr && vertices().size() > 0) {
      flann::Matrix<Real> dataset{
          const_cast<Real*>(vertex_position_data().front().data_ptr()),
          vertices().size(), num_dimensions()};
      m_kd_tree = std::make_unique<flann_index_type>(
          dataset, flann::KDTreeSingleIndexParams{});
      m_kd_tree->buildIndex();
    }
    return m_kd_tree;
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  auto invalidate_kd_tree() const {
    auto lock = std::scoped_lock{m_flann_mutex};
    m_kd_tree.reset();
  }
  //----------------------------------------------------------------------------
  auto nearest_neighbor(pos_type const& x) const {
    auto& h = build_kd_tree();
    if (h == nullptr) {
      return std::pair{vertex_handle::invalid(), Real(1) / Real(0)};
    }
    auto qm        = flann::Matrix<Real>{const_cast<Real*>(x.data_ptr()), 1,
                                  num_dimensions()};
    auto indices   = std::vector<std::vector<int>>{};
    auto distances = std::vector<std::vector<Real>>{};
    auto params    = flann::SearchParams{};
    h->knnSearch(qm, indices, distances, 1, params);
    return std::pair{
        vertex_handle{static_cast<std::size_t>(indices.front().front())},
        distances.front().front()};
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_raw(pos_type const&           x,
                             std::size_t const         num_nearest_neighbors,
                             flann::SearchParams const params = {}) const {
    auto& h = build_kd_tree();
    if (h == nullptr) {
      return std::pair{std::vector<int>{}, std::vector<Real>{}};
    }
    auto qm        = flann::Matrix<Real>{const_cast<Real*>(x.data_ptr()), 1,
                                  num_dimensions()};
    auto indices   = std::vector<std::vector<int>>{};
    auto distances = std::vector<std::vector<Real>>{};
    h->knnSearch(qm, indices, distances, num_nearest_neighbors, params);
    return std::pair{std::move(indices.front()), std::move(distances.front())};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors(pos_type const&   x,
                         std::size_t const num_nearest_neighbors) const {
    auto [indices, distances] = nearest_neighbors_raw(x, num_nearest_neighbors);
    auto handles = std::pair{std::vector<vertex_handle>(size(indices)),
                             std::move(distances)};
    std::ranges::copy(indices | std::views::transform([](auto const i) {
                        return vertex_handle{i};
                      }),
                      begin(handles.first));
    return handles;
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_radius_raw(pos_type const& x, Real const radius,
                                    flann::SearchParams const params = {}) const
      -> std::pair<std::vector<int>, std::vector<Real>> {
    auto& h = build_kd_tree();
    if (h == nullptr) {
      return std::pair{std::vector<int>{}, std::vector<Real>{}};
    }
    flann::Matrix<Real>           qm{const_cast<Real*>(x.data_ptr()),  // NOLINT
                           1, num_dimensions()};
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<Real>> distances;
    {
      // auto lock = std::scoped_lock{m_flann_mutex};
      h->radiusSearch(qm, indices, distances, radius, params);
    }
    return {std::move(indices.front()), std::move(distances.front())};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors_radius(pos_type const& x, Real const radius) const {
    auto const [indices, distances] = nearest_neighbors_radius_raw(x, radius);
    auto handles = std::pair{std::vector<vertex_handle>(size(indices)),
                             std::move(distances)};
    std::ranges::copy(indices | std::views::transform([](auto const i) {
                        return vertex_handle{i};
                      }),
                      begin(handles.first));
    return handles;
  }
  /// \}
#endif
  //============================================================================
  /// \{
  template <typename T>
  auto inverse_distance_weighting_sampler(
      typed_vertex_property_type<T> const& prop, Real const radius = 1) const {
    return inverse_distance_weighting_sampler_type<T>{*this, prop, radius};
  }
  /// \}
  //============================================================================
  /// \{
  /// \brief Moving Least Squares Sampler.
  ///
  /// Creates a field that interpolates scattered data with moving least squares.
  ///
  /// \param prop Some vertex property
  /// \param radius Radius of local support
  /// \param weighting Callable that gets as parameter the normalized distance.
  ///                  1 means point is distant radius. 0 means point is exactly
  ///                  at currently queried point.
  template <typename T>
  auto moving_least_squares_sampler(typed_vertex_property_type<T> const& prop,
                                    Real const                           radius,
                                    invocable<real_type> auto&& weighting) const
      requires(NumDimensions == 3 || NumDimensions == 2) {
    return detail::pointset::moving_least_squares_sampler<
        real_type, num_dimensions(), T, std::decay_t<decltype(weighting)>>{
        *this, prop, radius, std::forward<decltype(weighting)>(weighting)};
  }
  //----------------------------------------------------------------------------
  /// \brief Moving Least Squares Sampler.
  ///
  /// Creates a field that interpolates scattered data with moving least squares
  /// with predefind weighting function 
  ///
  /// \f$w(d) = 1 - 6\cdot d^2 + 8\cdot d^3 + 3\cdot d^4\f$,
  ///
  /// where d is the normalized distance.
  ///
  /// \param prop Some vertex property
  /// \param radius Radius of local support
  template <typename T>
  auto moving_least_squares_sampler(typed_vertex_property_type<T> const& prop,
                                    Real const radius) const
      requires(NumDimensions == 3 || NumDimensions == 2) {
    return moving_least_squares_sampler(
        prop, radius,

        [](auto const d) {
          return 1 - 6 * d * d + 8 * d * d * d - 3 * d * d * d * d;
        }

        //[](auto const d) {
        //  return std::exp(-d * d);
        //}
    );
  }
  ///\}
  //============================================================================
  /// \{
  /// \addtogroup radial_basis_functions Radial Basis Functions Interpolation
  /// \{

  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with with polynomial
  /// constraint kernel function:
  ///
  /// \f$k(d) = d\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_polynomial_and_linear_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler_with_polynomial(
        prop, [](auto const sqr_dist) { return gcem::sqrt(sqr_dist); });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with polynomial
  /// constraint and kernel function:
  ///
  /// \f$k(d) = d^3\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_polynomial_and_cubic_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler_with_polynomial(
        prop,
        [](auto const sqr_dist) { return sqr_dist * gcem::sqrt(sqr_dist); });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with polynomial
  /// constraint kernel function:
  ///
  /// \f$k(d) = e^{-(\epsilon d)^2}\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_polynomial_and_gaussian_kernel(
      typed_vertex_property_type<T> const& prop, Real const epsilon) const {
    return radial_basis_functions_sampler_with_polynomial(
        prop, [epsilon](auto const sqr_dist) {
          return gcem::exp(-epsilon * epsilon * sqr_dist);
        });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with polynomial
  /// constraint and a thin plate spline
  /// kernel function:
  ///
  /// \f$k(d) = d^2 \cdot \log(d)\f$
  template <typename T>
  auto
  radial_basis_functions_sampler_with_polynomial_and_thin_plate_spline_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler_with_polynomial(prop,
                                                          thin_plate_spline_from_squared);
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with polynomial
  /// constraint and a user-defined kernel function.
  template <typename T>
  auto radial_basis_functions_sampler_with_polynomial(
      typed_vertex_property_type<T> const& prop, auto&& f) const {
    return detail::pointset::radial_basis_functions_sampler_with_polynomial{
        *this, prop, std::forward<decltype(f)>(f)};
  }
  //============================================================================
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with kernel function:
  ///
  /// \f$k(d) = d\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_linear_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler(
        prop, [](auto const sqr_dist) { return gcem::sqrt(sqr_dist); });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with kernel function:
  ///
  /// \f$k(d) = d^3\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_cubic_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler(prop, [](auto const sqr_dist) {
      return sqr_dist * gcem::sqrt(sqr_dist);
    });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with kernel function:
  ///
  /// \f$k(d) = e^{-(\epsilon d)^2}\f$
  ///
  /// \param epsilon Shape parameter
  template <typename T>
  auto radial_basis_functions_sampler_with_gaussian_kernel(
      typed_vertex_property_type<T> const& prop, Real const epsilon) const {
    return radial_basis_functions_sampler(prop, [epsilon](auto const sqr_dist) {
      return std::exp(-(epsilon * epsilon * sqr_dist));
    });
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with a thin plate spline
  /// kernel function:
  ///
  /// \f$k(d) = d^2 \cdot \log(d)\f$
  template <typename T>
  auto radial_basis_functions_sampler_with_thin_plate_spline_kernel(
      typed_vertex_property_type<T> const& prop) const {
    return radial_basis_functions_sampler(prop, thin_plate_spline_from_squared);
  }
  //----------------------------------------------------------------------------
  /// \brief Constructs a radial basis functions interpolator.
  ///
  /// Constructs a radial basis functions interpolator with a user-defined
  /// kernel function.
  template <typename T>
  auto radial_basis_functions_sampler(typed_vertex_property_type<T> const& prop,
                                      auto&& f) const {
    return detail::pointset::radial_basis_functions_sampler{
        *this, prop, std::forward<decltype(f)>(f)};
  }
  ///\}
  ///\}
  //----------------------------------------------------------------------------
  friend struct detail::pointset::vertex_container<Real, NumDimensions>;
  friend struct detail::pointset::const_vertex_container<Real, NumDimensions>;
};
//==============================================================================
template <std::size_t NumDimensions>
using Pointset  = pointset<real_number, NumDimensions>;
using pointset2 = Pointset<2>;
using pointset3 = Pointset<3>;
using pointset4 = Pointset<4>;
using pointset5 = Pointset<5>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/detail/pointset/inverse_distance_weighting_sampler.h>
#include <tatooine/detail/pointset/moving_least_squares_sampler.h>
#include <tatooine/detail/pointset/radial_basis_functions_sampler.h>
#include <tatooine/detail/pointset/radial_basis_functions_sampler_with_polynomial.h>
#include <tatooine/detail/pointset/vertex_container.h>
//==============================================================================
#endif
