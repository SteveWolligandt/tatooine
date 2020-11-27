#ifndef TATOOINE_SMEARING_PARSE_ARGUMENTS_H
#define TATOOINE_SMEARING_PARSE_ARGUMENTS_H
//==============================================================================
#include <tatooine/geometry/sphere.h>

#include <filesystem>
#include <optional>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
struct arguments {
  std::filesystem::path input_file_path, output_file_path;
  geometry::sphere2    sphere;
  double                inner_radius;
  vec2                  end_point;
  double                temporal_range, t0;
  vec2                  dir;
  size_t                num_steps;
  bool                  write_vtk;
  std::vector<std::string> fields;
};
//==============================================================================
auto parse_arguments(int argc, char const** argv) -> std::optional<arguments>;
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
