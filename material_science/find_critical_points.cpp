#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <tatooine/critical_points.h>
#include <tatooine/vtk_legacy.h>
//==============================================================================
static const std::array<std::string_view, 4> filenames{
    "texture_case1.vtk", "texture_case2.vtk", "texture_case3.vtk",
    "texture_case4.vtk"};
//==============================================================================
void write_points(std::vector<tatooine::vec<double, 2>>& points,
                  const std::string&           path) {
  using namespace tatooine;
  using namespace boost;
  using namespace adaptors;
  std::vector<vec<double, 3>> points3(points.size());
  copy(points | transformed([](const auto& p2) {
         return vec{p2(0), p2(1), 0};
       }),
       points3.begin());
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  std::vector<std::vector<size_t>> verts(1, std::vector<size_t>(points.size()));
  for (size_t i = 0; i < points.size(); ++i) { verts.back()[i] = i; }
  if (writer.is_open()) {
    writer.set_title("critical points");
    writer.write_header();
    writer.write_points(points3);
    writer.write_vertices(verts);
    writer.close();
  }
}
//------------------------------------------------------------------------------
int main() {
  using namespace tatooine;
  using namespace interpolation;
  for (const auto& filename : filenames) {
    std::cerr << "[" << filename << "]\n";
    grid_sampler<double, 2, vec<double, 2>, linear, linear> sampler;
    sampler.read_vtk_scalars(std::string{filename}, "vector_field");
    auto cps = find_critical_points(sampler);
    for (auto cp : cps) { std::cerr << "  " << cp << '\n'; }
    std::string out_filename{filename};
    auto        dotpos = out_filename.find_last_of(".");
    out_filename.replace(dotpos, 4, "_critical_points.vtk");
    write_points(cps, out_filename);
  }
}
