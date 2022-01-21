#ifndef TATOOINE_EDGESET_H
#define TATOOINE_EDGESET_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct edgeset : unstructured_simplicial_grid<Real, NumDimensions, 1> {
  using this_t   = edgeset<Real, NumDimensions>;
  using parent_t = unstructured_simplicial_grid<Real, NumDimensions, 1>;
  using parent_t::parent_t;
  using typename parent_t::vertex_handle;
  using edge_handle = typename parent_t::simplex_handle;
  template <typename... Handles>
  auto insert_edge(vertex_handle const v0, vertex_handle const v1)  {
    return this->insert_simplex(v0, v1);
  }
  auto edge_at(edge_handle const h) { return this->simplex_at(h); }
  auto edge_at(edge_handle const h) const { return this->simplex_at(h); }
  auto edges() const { return this->simplices(); }
  auto are_connected(vertex_handle const v0, vertex_handle const v1) const {
    for (auto e : edges()) {
      auto [ev0, ev1] = this->at(e);
      if ((ev0 == v0 && ev1 == v1) || (ev0 == v1 && ev1 == v0)) {
        return true;
      }
    }
    return false;
  }
};
template <std::size_t NumDimensions>
using Edgeset = edgeset<real_t, NumDimensions>;
template <typename Real>
using Edgeset2 = edgeset<Real, 2>;
template <typename Real>
using Edgeset3 = edgeset<Real, 3>;
template <typename Real>
using Edgeset4 = edgeset<Real, 4>;
template <typename Real>
using Edgeset5 = edgeset<Real, 5>;
using edgeset2 = Edgeset<2>;
using edgeset3 = Edgeset<3>;
using edgeset4 = Edgeset<4>;
using edgeset5 = Edgeset<5>;
//==============================================================================
template <typename T>
struct is_edgeset_impl : std::false_type {};
template <typename Real, std::size_t NumDimensions>
struct is_edgeset_impl<edgeset<Real, NumDimensions>> : std::true_type {};
template <typename T>
static constexpr auto is_edgeset = is_edgeset_impl<T>::value;
//==============================================================================
namespace detail {
//==============================================================================
template <typename MeshCont>
auto write_edgeset_container_to_vtk(
    MeshCont const& grids, std::filesystem::path const& path,
    std::string const& title = "tatooine edgeset") {
  vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
  if (writer.is_open()) {
    std::size_t num_pts   = 0;
    std::size_t cur_first = 0;
    for (auto const& m : grids) {
      num_pts += m.vertices().size();
    }
    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
    std::vector<std::vector<std::size_t>>                             faces;
    points.reserve(num_pts);
    faces.reserve(grids.size());

    for (auto const& m : grids) {
      // add points
      for (auto const& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add faces
      for (auto c : m.cells()) {
        faces.emplace_back();
        auto [v0, v1] = m[c];
        faces.back().push_back(cur_first + v0.index());
        faces.back().push_back(cur_first + v1.index());
      }
      cur_first += m.vertices().size();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(faces);
    // writer.write_point_data(num_pts);
    writer.close();
  }
}
//==============================================================================
auto write_edgeset_container_to_vtp(range auto const&            grids,
                                    std::filesystem::path const& path) {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset                    = std::size_t{};
  using header_type              = std::uint64_t;
  using polys_connectivity_int_t = std::int32_t;
  using polys_offset_int_t       = polys_connectivity_int_t;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<header_type>())
       << "\">";
  file << "<PolyData>\n";
  for (auto const& g : grids) {
    using real_t = typename std::decay_t<decltype(g)>::real_t;
    file << "<Piece"
         << " NumberOfPoints=\"" << g.vertices().size() << "\""
         << " NumberOfPolys=\"" << g.cells().size() << "\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"0\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<real_t>())
         << "\" NumberOfComponents=\"" << g.num_dimensions() << "\"/>";
    auto const num_bytes_points =
        header_type(sizeof(real_t) * g.num_dimensions() *
                    g.vertices().data_container().size());
    offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Polys
    file << "<Polys>\n";
    // Polys - connectivity
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<polys_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_polys_connectivity = g.cells().size() *
                                              g.num_vertices_per_simplex() *
                                              sizeof(polys_connectivity_int_t);
    offset += num_bytes_polys_connectivity + sizeof(header_type);

    // Polys - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<polys_offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_polys_offsets =
        sizeof(polys_offset_int_t) * g.cells().size();
    offset += num_bytes_polys_offsets + sizeof(header_type);
    file << "</Polys>\n";
    file << "</Piece>\n\n";
  }
  file << "</PolyData>\n";

  file << "<AppendedData encoding=\"raw\">_";
  // Writing vertex data to appended data section
  auto arr_size = header_type{};

  for (auto const& g : grids) {
    using real_t = typename std::decay_t<decltype(g)>::real_t;
    arr_size     = header_type(sizeof(real_t) * g.num_dimensions() *
                               g.vertices().data_container().size());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    file.write(reinterpret_cast<char const*>(g.vertices().data()), arr_size);

    // Writing polys connectivity data to appended data section
    {
      auto connectivity_data = std::vector<polys_connectivity_int_t>(
          g.cells().size() * g.num_vertices_per_simplex());
      std::ranges::copy(g.cells().data_container() |
                            std::views::transform(
                                [](auto const x) -> polys_connectivity_int_t {
                                  return x.index();
                                }),
                        begin(connectivity_data));
      arr_size = g.cells().size() * g.num_vertices_per_simplex() *
                 sizeof(polys_connectivity_int_t);
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 arr_size);
    }

    // Writing polys offsets to appended data section
    {
      auto offsets = std::vector<polys_offset_int_t>(
          g.cells().size(), g.num_vertices_per_simplex());
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      arr_size = sizeof(polys_offset_int_t) * g.cells().size();
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);
    }
  }
  file << "</AppendedData>";
  file << "</VTKFile>";
}
}  // namespace detail
//==============================================================================
auto write_vtk(range auto const& grids, std::filesystem::path const& path,
               std::string const& title = "tatooine grids") requires
    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
  detail::write_edgeset_container_to_vtk(grids, path, title);
}
//------------------------------------------------------------------------------
auto write_vtp(range auto const&            grids,
               std::filesystem::path const& path) requires
    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
  detail::write_edgeset_container_to_vtp(grids, path);
}
//------------------------------------------------------------------------------
auto write(range auto const& grids, std::filesystem::path const& path) requires
    is_edgeset<typename std::decay_t<decltype(grids)>::value_type> {
  auto const ext = path.extension();
  if (ext == ".vtp") {
    detail::write_edgeset_container_to_vtp(grids, path);
  } else if (ext == ".vtk") {
    detail::write_edgeset_container_to_vtk(grids, path);
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
