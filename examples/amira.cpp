#include <tatooine/amira/read.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/streamsurface.h>

#include <boost/program_options.hpp>
using namespace tatooine;
auto main(int argc, char** argv) -> int {
  namespace po = boost::program_options;
  auto desc = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "file", po::value<std::string>(), "file to read");

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return 0;
  }
  auto filepath = filesystem::path{};
  if (variables_map.count("file") > 0) {
    filepath = filesystem::path{variables_map["file"].as<std::string>()};
    auto [data, dims, aabb, num_components] =
        tatooine::amira::read(filepath);

    std::cout << "dims: " << dims[0] << ", " << dims[1] << ", " << dims[2]
              << '\n';
    std::cout << "num_components: " << num_components << '\n';
    std::cout << "aabb:\n" << aabb << '\n';
    std::cout << "data:\n";
    std::cout << data[0];
    for (std::size_t i = 1; i < 10; ++i) {
      std::cout << ", " << data[i];
    }
    std::cout << ", ...\n";


    if (dims[2] == 1) {
      auto grid = uniform_rectilinear_grid2{filepath};
      grid.write_vtr(filepath.replace_extension(".vtr"));
    } else if (dims[2] > 1) {
      auto grid = uniform_rectilinear_grid3{filepath};

      if (num_components == 3) {
        auto field = grid.vec3_vertex_property(filepath.filename().string()).linear_sampler();
        auto seedcurve = line3{};
        auto & param = seedcurve.parameterization();
        auto const v0 = seedcurve.push_back(grid.bounding_box().random_point());
        auto const v1 = seedcurve.push_back(grid.bounding_box().random_point());
        param[v0] = 0;
        param[v1] = 1;
        auto surf = streamsurface{flowmap(field), std::move(seedcurve)};
        surf.discretize(20, 0.01, -1, 1).write("streamsurface.vtp");
      }
      grid.write_vtr(filepath.replace_extension(".vtr"));


    }
  }
}
