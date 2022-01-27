#ifndef TATOOINE_POINTSET_H
#define TATOOINE_POINTSET_H
//==============================================================================
#include <tatooine/available_libraries.h>
#include <tatooine/iterator_facade.h>

#include <boost/range/algorithm/find.hpp>
#if TATOOINE_FLANN_AVAILABLE
#include <flann/flann.hpp>
#endif
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
template <typename Real, std::size_t NumDimensions, typename T>
struct moving_least_squares_sampler;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename T>
struct inverse_distance_weighting_sampler;
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct vertex_container;
}  // namespace detail::pointset
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct pointset {
  // static constexpr std::size_t triangle_dims = 2;
  // static constexpr std::size_t tetgen_dims = 3;
  static constexpr auto num_dimensions() { return NumDimensions; }
  using real_t = Real;
  using this_t = pointset<Real, NumDimensions>;
  using vec_t  = vec<Real, NumDimensions>;
  using pos_t  = vec_t;
#if TATOOINE_FLANN_AVAILABLE
  using flann_index_t = flann::Index<flann::L2<Real>>;
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
  template <typename T>
  using vertex_property_t = vector_property_impl<vertex_handle, T>;
  using vertex_property_container_t =
      std::map<std::string, std::unique_ptr<vector_property<vertex_handle>>>;
  //============================================================================
 private:
  std::vector<pos_t>          m_vertex_position_data;
  std::set<vertex_handle>     m_invalid_vertices;
  vertex_property_container_t m_vertex_properties;
#if TATOOINE_FLANN_AVAILABLE
  mutable std::unique_ptr<flann_index_t> m_kd_tree;
#endif
  //============================================================================
 public:
  pointset()  = default;
  ~pointset() = default;
  //----------------------------------------------------------------------------
  pointset(std::initializer_list<pos_t>&& vertices)
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
  pointset(std::vector<pos_t> const& vertices)
      : m_vertex_position_data(vertices) {}
  //----------------------------------------------------------------------------
  pointset(std::vector<pos_t>&& vertices)
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
 protected:
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  auto vertex_properties() -> auto& { return m_vertex_properties; }
  //----------------------------------------------------------------------------
 public:
  auto at(vertex_handle const v) -> auto& {
    return vertex_position_data()[v.index()];
  }
  auto at(vertex_handle const v) const -> auto const& {
    return vertex_position_data()[v.index()];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(vertex_handle const v) -> auto& {
    return vertex_position_data()[v.index()];
  }
  auto vertex_at(vertex_handle const v) const -> auto const& {
    return vertex_position_data()[v.index()];
  }
  //----------------------------------------------------------------------------
  auto vertex_at(std::size_t const i) -> auto& { return vertex_position_data()[i]; }
  auto vertex_at(std::size_t const i) const -> auto const& {
    return vertex_position_data()[i];
  }
  //----------------------------------------------------------------------------
  auto operator[](vertex_handle const v) -> auto& { return at(v); }
  auto operator[](vertex_handle const v) const -> auto const& { return at(v); }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{this}; }
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
  auto insert_vertex(arithmetic auto const... ts) requires(sizeof...(ts) ==
                                                           NumDimensions) {
    vertex_position_data().push_back(pos_t{static_cast<Real>(ts)...});
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
  //----------------------------------------------------------------------------
  template <typename Vec, typename OtherReal>
  auto insert_vertex(base_tensor<Vec, OtherReal, NumDimensions> const& v) {
    vertex_position_data().push_back(pos_t{v});
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& v) {
    vertex_position_data().emplace_back(std::move(v));
    for (auto& [key, prop] : vertex_properties()) {
      prop->push_back();
    }
    return vertex_handle{size(vertex_position_data()) - 1};
  }
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
    return std::ranges::find(invalid_vertices(), v) == end(invalid_vertices());
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
  auto join(this_t const& other) {
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
  //----------------------------------------------------------------------------
  auto resize(std::size_t const s) {
    vertex_position_data().resize(s);
    for (auto& [key, prop] : vertex_properties()) {
      prop->resize(s);
    }
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
  //   io.pointlist      = new tetgen_real_t[io.numberofpoints * 3];
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
      return *dynamic_cast<vertex_property_t<T>*>(
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
      return *dynamic_cast<vertex_property_t<T>*>(
          vertex_properties().at(name).get());
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
    auto [it, suc] = vertex_properties().insert(
        std::pair{name, std::make_unique<vertex_property_t<T>>(value)});
    auto prop = dynamic_cast<vertex_property_t<T>*>(it->second.get());
    prop->resize(vertex_position_data().size());
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
      -> void requires(NumDimensions == 3 || NumDimensions == 2) {
    using namespace std::ranges;
    vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      if constexpr (NumDimensions == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(vertices().size());
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
      if (!vertex_properties().empty()) {
        writer.write_point_data(vertices().size());
      }

      for (auto const& [name, prop] : vertex_properties()) {
        std::vector<std::vector<Real>> data;
        data.reserve(vertex_position_data().size());

        if (prop->type() == typeid(vec<Real, 4>)) {
          for (auto const& v4 :
               *dynamic_cast<vertex_property_t<vec<Real, 4>> const*>(
                   prop.get())) {
            data.push_back({v4(0), v4(1), v4(2), v4(3)});
          }
        } else if (prop->type() == typeid(vec<Real, 3>)) {
          for (auto const& v3 :
               *dynamic_cast<vertex_property_t<vec<Real, 3>> const*>(
                   prop.get())) {
            data.push_back({v3(0), v3(1), v3(2)});
          }
        } else if (prop->type() == typeid(vec<Real, 2>)) {
          for (auto const& v2 :
               *dynamic_cast<vertex_property_t<vec<Real, 2>> const*>(
                   prop.get())) {
            data.push_back({v2(0), v2(1)});
          }
        } else if (prop->type() == typeid(Real)) {
          for (auto const& scalar :
               *dynamic_cast<vertex_property_t<Real> const*>(prop.get())) {
            data.push_back({scalar});
          }
        }
        writer.write_scalars(name, data);
      }
      writer.close();
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
    using verts_connectivity_int_t = std::int64_t;
    using verts_offset_int_t       = verts_connectivity_int_t;
    auto const num_bytes_points =
        header_type(sizeof(Real) * 3 * vertices().size());
    auto const num_bytes_verts_connectivity = vertices().size() *
                                              sizeof(verts_connectivity_int_t);
    auto const num_bytes_verts_offsets =
        sizeof(verts_offset_int_t) * vertices().size();
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
                vtk::xml::data_array::to_type<verts_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    offset += num_bytes_verts_connectivity + sizeof(header_type);
    // Verts - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<verts_offset_int_t>())
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
        copy(vertices() | views::transform(position) | views::transform(to_3d),
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
      auto connectivity_data = std::vector<verts_connectivity_int_t>(
          vertices().size());
      copy(views::iota(verts_connectivity_int_t(0), verts_connectivity_int_t(vertices().size())),
           begin(connectivity_data));
      file.write(reinterpret_cast<char const*>(&num_bytes_verts_connectivity),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 num_bytes_verts_connectivity);
    }

    // Writing verts offsets to appended data section
    {
      auto offsets = std::vector<verts_offset_int_t>(
          vertices().size(), 1);
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
                   vtk::xml::data_array::to_type<internal_value_type<T>>())
            << "\" NumberOfComponents=\""<<num_components<T><<"\"/>\n";
       return vertices().size() * sizeof(T) + sizeof(header_type);
     }
     return 0;
   }
   //----------------------------------------------------------------------------
   template <typename T, typename header_type>
   auto write_vertex_property_appended_data_vtp(auto const& prop,
                                                auto&       file) const {
     if (prop->type() == typeid(T)) {
       auto const num_bytes = header_type(
           sizeof(internal_value_type<T>) * num_components<T> * vertices().size());
       file.write(reinterpret_cast<char const*>(&num_bytes),
                  sizeof(header_type));
       file.write(
           reinterpret_cast<char const*>(
               dynamic_cast<vertex_property_t<T>*>(prop.get())->data().data()),
           num_bytes);
     }
   }
   //----------------------------------------------------------------------------
  public:
  //----------------------------------------------------------------------------
#if TATOOINE_FLANN_AVAILABLE
  auto rebuild_kd_tree() {
    m_kd_tree.reset();
    kd_tree();
  }
  //----------------------------------------------------------------------------
 private:
  auto kd_tree() const -> auto& {
    if (m_kd_tree == nullptr) {
      flann::Matrix<Real> dataset{
          const_cast<Real*>(vertex_position_data().front().data_ptr()),
          vertices().size(), num_dimensions()};
      m_kd_tree = std::make_unique<flann_index_t>(
          dataset, flann::KDTreeSingleIndexParams{});
      m_kd_tree->buildIndex();
    }
    return *m_kd_tree;
  }
  //----------------------------------------------------------------------------
 public:
  auto nearest_neighbor(pos_t const& x) const {
    flann::Matrix<Real> qm{const_cast<Real*>(x.data_ptr()), 1,  // NOLINT
                           num_dimensions()};
    std::vector<std::vector<int>>  indices;
    std::vector<std::vector<Real>> distances;
    flann::SearchParams            params;
    kd_tree().knnSearch(qm, indices, distances, 1, params);
    return vertex_handle{static_cast<std::size_t>(indices.front().front())};
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_raw(pos_t const& x, std::size_t const num_nearest_neighbors,
                             flann::SearchParams const params = {}) const {
    auto qm  = flann::Matrix<Real>{const_cast<Real*>(x.data_ptr()), 1,
                                  num_dimensions()};
    auto ret = std::pair{std::vector<std::vector<std::size_t>>{},
                         std::vector<std::vector<Real>>{}};
    auto& [indices, distances] = ret;
    kd_tree().knnSearch(qm, indices, distances, num_nearest_neighbors, params);
    return ret;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors(pos_t const& x,
                         std::size_t const num_nearest_neighbors) const {
    auto const [indices, distances] =
        nearest_neighbors_raw(x, num_nearest_neighbors);
    auto handles = std::vector<std::vector<vertex_handle>>{};
    handles.reserve(size(indices));
    // TODO make it work
    // for (auto const& i : indices) {
    //
    //  handles.emplace_back(static_cast<std::size_t>(i));
    //}
    return handles;
  }
  //----------------------------------------------------------------------------
  /// Takes the raw output indices of flann without converting them into vertex
  /// handles.
  auto nearest_neighbors_radius_raw(pos_t const& x, Real const radius,
                                    flann::SearchParams const params = {}) const
      -> std::pair<std::vector<int>, std::vector<Real>> {
    flann::Matrix<Real>           qm{const_cast<Real*>(x.data_ptr()),  // NOLINT
                           1, num_dimensions()};
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<Real>> distances;
    kd_tree().radiusSearch(qm, indices, distances, radius, params);
    return {std::move(indices.front()), std::move(distances.front())};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto nearest_neighbors_radius(pos_t const& x, Real const radius) const {
    auto const [indices, distances] = nearest_neighbors_radius_raw(x, radius);
    std::vector<vertex_handle> handles;
    handles.reserve(size(indices));
    for (auto const i : indices) {
      handles.emplace_back(static_cast<std::size_t>(i));
    }
    return handles;
  }
#endif
  //============================================================================
  template <typename T>
  auto inverse_distance_weighting_sampler(vertex_property_t<T> const& prop,
                                          Real const radius = 1) const {
    return detail::pointset::inverse_distance_weighting_sampler<
        Real, NumDimensions, T>{*this, prop, radius};
  }
  //============================================================================
  template <typename T>
  auto moving_least_squares_sampler(vertex_property_t<T> const& prop,
                                    Real const radius = 1) const
      requires(NumDimensions == 3 || NumDimensions == 2) {
    return detail::pointset::moving_least_squares_sampler<Real, NumDimensions,
                                                          T>{*this, prop,
                                                             radius};
  }
  friend struct detail::pointset::vertex_container<Real, NumDimensions>;
};
template <typename Real, std::size_t NumDimensions>
auto vertices(pointset<Real, NumDimensions> const& ps) {
  return ps.vertices();
}
//==============================================================================
template <std::size_t NumDimensions>
using Pointset  = pointset<real_t, NumDimensions>;
using pointset2 = Pointset<2>;
using pointset3 = Pointset<3>;
using pointset4 = Pointset<4>;
using pointset5 = Pointset<5>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/detail/pointset/vertex_container.h>
#include <tatooine/detail/pointset/inverse_distance_weighting_sampler.h>
#include <tatooine/detail/pointset/moving_least_squares_sampler.h>
#endif
