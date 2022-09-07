#include <tatooine/edgeset.h>
//==============================================================================
namespace detail::edgeset {
//==============================================================================
template <typename MeshCont>
auto write_container_to_vtk(
    MeshCont const& edgesets, std::filesystem::path const& path,
    std::string const& title = "tatooine edgeset") {
  vtk::legacy_file_writer writer(path, vtk::dataset_type::polydata);
  if (writer.is_open()) {
    std::size_t num_pts   = 0;
    std::size_t cur_first = 0;
    for (auto const& m : edgesets) {
      num_pts += m.vertices().size();
    }
    std::vector<std::array<typename MeshCont::value_type::real_type, 3>> points;
    std::vector<std::vector<std::size_t>>                             edges;
    points.reserve(num_pts);
    edges.reserve(edgesets.size());

    for (auto const& m : edgesets) {
      // add points
      for (auto const& v : m.vertices()) {
        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
      }

      // add edges
      for (auto s : m.simplices()) {
        edges.emplace_back();
        auto [v0, v1] = m[s];
        edges.back().push_back(cur_first + v0.index());
        edges.back().push_back(cur_first + v1.index());
      }
      cur_first += m.vertices().size();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_polygons(edges);
    writer.close();
  }
}
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto write_to_vtp(edgeset<Real, NumDimensions> const& es,
                  std::filesystem::path const&        path) {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset                    = std::size_t{};
  using header_type              = std::uint64_t;
  using lines_connectivity_int_t = std::int32_t;
  using lines_offset_int_t       = lines_connectivity_int_t;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<header_type>())
       << "\">";
  file << "<PolyData>\n";
  using real_type = typename std::decay_t<decltype(es)>::real_type;
  file << "<Piece"
       << " NumberOfPoints=\"" << es.vertices().size() << "\""
       << " NumberOfPolys=\"0\""
       << " NumberOfVerts=\"0\""
       << " NumberOfLines=\"" << es.simplices().size() << "\""
       << " NumberOfStrips=\"0\""
       << ">\n";

  // Points
  file << "<Points>";
  file << "<DataArray"
       << " format=\"appended\""
       << " offset=\"" << offset << "\""
       << " type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<real_type>())
       << "\" NumberOfComponents=\"3\"/>";
  auto const num_bytes_points =
      header_type(sizeof(real_type) * es.num_dimensions() *
                  es.vertices().data_container().size());
  offset += num_bytes_points + sizeof(header_type);
  file << "</Points>\n";

  // Lines
  file << "<Lines>\n";
  // Lines - connectivity
  file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<lines_connectivity_int_t>())
       << "\" Name=\"connectivity\"/>\n";
  auto const num_bytes_lines_connectivity = es.simplices().size() *
                                            es.num_vertices_per_simplex() *
                                            sizeof(lines_connectivity_int_t);
  offset += num_bytes_lines_connectivity + sizeof(header_type);

  // Lines - offsets
  file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<lines_offset_int_t>())
       << "\" Name=\"offsets\"/>\n";
  auto const num_bytes_lines_offsets =
      sizeof(lines_offset_int_t) * es.simplices().size();
  offset += num_bytes_lines_offsets + sizeof(header_type);
  file << "</Lines>\n";
  file << "</Piece>\n\n";
  file << "</PolyData>\n";

  file << "<AppendedData encoding=\"raw\">_";
  using namespace std::ranges;
  // Writing vertex data to appended data section
  auto const num_bytes =
      header_type(sizeof(real_type) * 3 * es.vertices().size());
  file.write(reinterpret_cast<char const*>(&num_bytes), sizeof(header_type));
  if constexpr (NumDimensions == 2) {
    auto point_data = std::vector<vec<real_type, 3>>{};
    point_data.reserve(es.vertices().size());
    auto           position = [this](auto const v) -> auto& { return at(v); };
    constexpr auto to_3d    = [](auto const& p) {
      return vec{p.x(), p.y(), real_type(0)};
    };
    for (auto const v : es.vertices()) {
      point_data.push_back(to_3d(es[v]));
    }

    file.write(reinterpret_cast<char const*>(point_data.data()), num_bytes);
  } else if constexpr (NumDimensions == 3) {
    file.write(reinterpret_cast<char const*>(es.vertices().data()), num_bytes);
  }
  // Writing lines connectivity data to appended data section
  {
    auto connectivity_data = std::vector<lines_connectivity_int_t>(
        es.simplices().size() * es.num_vertices_per_simplex());
    copy(
        es.simplices().data_container() |
            std::views::transform([](auto const x) -> lines_connectivity_int_t {
              return x.index();
            }),
        begin(connectivity_data));
    arr_size = es.simplices().size() * es.num_vertices_per_simplex() *
               sizeof(lines_connectivity_int_t);
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    file.write(reinterpret_cast<char const*>(connectivity_data.data()),
               arr_size);
  }

  // Writing lines offsets to appended data section
  {
    auto offsets = std::vector<lines_offset_int_t>(
        es.simplices().size(), es.num_vertices_per_simplex());
    for (std::size_t i = 1; i < size(offsets); ++i) {
      offsets[i] += offsets[i - 1];
    }
    arr_size = sizeof(lines_offset_int_t) * es.simplices().size();
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);
  }
  file << "</AppendedData>";
  file << "</VTKFile>";
}
//==============================================================================
auto write_container_to_vtp(range auto const&            edgesets,
                            std::filesystem::path const& path) {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset                    = std::size_t{};
  using header_type              = std::uint64_t;
  using lines_connectivity_int_t = std::int32_t;
  using lines_offset_int_t       = lines_connectivity_int_t;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::data_array::to_string(
              vtk::xml::data_array::to_type<header_type>())
       << "\">";
  file << "<PolyData>\n";
  for (auto const& g : edgesets) {
    using real_type = typename std::decay_t<decltype(g)>::real_type;
    file << "<Piece"
         << " NumberOfPoints=\"" << g.vertices().size() << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"" << es.simplices().size() << "\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<real_type>())
         << "\" NumberOfComponents=\"" << g.num_dimensions() << "\"/>";
    auto const num_bytes_points =
        header_type(sizeof(real_type) * g.num_dimensions() *
                    g.vertices().data_container().size());
    offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Lines
    file << "<Lines>\n";
    // Lines - connectivity
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<lines_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_lines_connectivity = g.simplices().size() *
                                              g.num_vertices_per_simplex() *
                                              sizeof(lines_connectivity_int_t);
    offset += num_bytes_lines_connectivity + sizeof(header_type);

    // Lines - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::data_array::to_string(
                vtk::xml::data_array::to_type<lines_offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_lines_offsets =
        sizeof(lines_offset_int_t) * g.simplices().size();
    offset += num_bytes_lines_offsets + sizeof(header_type);
    file << "</Lines>\n";
    file << "</Piece>\n\n";
  }
  file << "</PolyData>\n";

  file << "<AppendedData encoding=\"raw\">_";
  // Writing vertex data to appended data section
  auto arr_size = header_type{};

  for (auto const& g : edgesets) {
    using real_type = typename std::decay_t<decltype(g)>::real_type;
    arr_size     = header_type(sizeof(real_type) * g.num_dimensions() *
                               g.vertices().data_container().size());
    file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
    file.write(reinterpret_cast<char const*>(g.vertices().data()), arr_size);

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<lines_connectivity_int_t>(
          g.simplices().size() * g.num_vertices_per_simplex());
      std::ranges::copy(g.simplices().data_container() |
                            std::views::transform(
                                [](auto const x) -> lines_connectivity_int_t {
                                  return x.index();
                                }),
                        begin(connectivity_data));
      arr_size = g.simplices().size() * g.num_vertices_per_simplex() *
                 sizeof(lines_connectivity_int_t);
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 arr_size);
    }

    // Writing lines offsets to appended data section
    {
      auto offsets = std::vector<lines_offset_int_t>(
          g.simplices().size(), g.num_vertices_per_simplex());
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      arr_size = sizeof(lines_offset_int_t) * g.simplices().size();
      file.write(reinterpret_cast<char const*>(&arr_size), sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()), arr_size);
    }
  }
  file << "</AppendedData>";
  file << "</VTKFile>";
}
//==============================================================================
}  // namespace detail::edgeset
//==============================================================================
