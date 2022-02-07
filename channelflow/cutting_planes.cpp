#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/field.h>
#include <tatooine/hdf5.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct chengs_color_scale_t {
  using real_t  = double;
  using this_type  = chengs_color_scale_t;
  using color_t = vec<real_t, 3>;
  //==============================================================================
  std::unique_ptr<real_t[]> m_data;
  //==============================================================================
  chengs_color_scale_t()
      : m_data{new real_t[]{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            0.0, 0.0, 0.0, 0.0, 0.0}} {}
  //----------------------------------------------------------------------------
  auto sample(real_t t) const {
    if (t <= 0) {
      return color_t{m_data[0], m_data[1], m_data[2]};
    }
    if (t >= 1) {
      return color_t{m_data[4 * 3], m_data[4 * 3 + 1], m_data[4 * 3 + 2]};
    }
    t *= 4;
    auto const i = static_cast<size_t>(std::floor(t));
    t            = t - i;
    return color_t{m_data[i * 3] * (1 - t) + m_data[(i + 1) * 3] * t,
                   m_data[i * 3 + 1] * (1 - t) + m_data[(i + 1) * 3 + 1] * t,
                   m_data[i * 3 + 2] * (1 - t) + m_data[(i + 1) * 3 + 2] * t};
  }
  auto operator()(real_t const t) const { return sample(t); }
};
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto const filepath = [&] {
    if (argc > 1) {
      return filesystem::path{argv[1]};
    } else {
      return filesystem::path{"/home/steve/channelflow/dino_res_186200.h5"};
    }
  }();
  auto const out_path = filepath.filename().replace_extension("");
  if (filesystem::exists(out_path)) {
    filesystem::remove_all(out_path);
  }
  filesystem::create_directory(out_path);
  filesystem::create_directory(out_path / "viridis");
  filesystem::create_directory(out_path / "cheng");
  hdf5::file channelflow_file{filepath};
  auto const axis0 =
      channelflow_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      channelflow_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      channelflow_file.dataset<double>("CartGrid/axis2").read_as_vector();

  std::cerr << "creating grids ...\n";
  // rectilinear_grid full_domain{axis0, axis1, axis2};
  std::cerr << "[ " << axis0[0] << ", " << axis0[1] << ", ..., " << axis0.back()
            << " ], " << size(axis0) << "\n";
  std::cerr << "[ " << axis1[0] << ", " << axis1[1] << ", ..., " << axis1.back()
            << " ], " << size(axis1) << "\n";
  std::cerr << "[ " << axis2[0] << ", " << axis2[1] << ", ..., " << axis2.back()
            << " ], " << size(axis2) << "\n";
  // full_domain.set_chunk_size_for_lazy_properties(4);
  // full_domain.max_num_chunks_loaded(1000);
  std::cerr << "done!\n";

  // std::cerr << "loading data ...";
  // auto& velocity_y = full_domain.insert_vertex_property(
  //    channelflow_file.dataset<double>("velocity/yvel"), "velocity_y");
  // std::cerr << "done!\n";

  // std::cerr << "creating samplers ...";
  // auto velocity_y_sampler = velocity_y.linear_sampler();
  // std::cerr << "done!\n";

  // auto basis   = MatD<3, 2>{};
  // basis.col(0) = vec{0, axis1.back(), 0};
  // basis.col(1) = vec{0, 0, axis2.back()};
  // std::cerr << "basis:\n" <<basis << '\n';

  auto const chengs_color_scale  = chengs_color_scale_t{};
  auto const viridis_color_scale = color_scales::viridis{};
  auto const plane_positions     = std::vector{512, 256, 128};
  auto       cutting_plane_grid  = rectilinear_grid{axis1, axis2};
  for (auto const plane_position : plane_positions) {
    auto str          = std::to_string(plane_position);
    auto filename_png = filesystem::path{"cutting_plane_" + str + ".png"};
    auto filename_vtk = filesystem::path{"cutting_plane_" + str + ".vtk"};
    std::cerr << "discretizing plane at " << str << "...\n";

    auto& cutting_plane =
        cutting_plane_grid.insert_contiguous_vertex_property<double>(str);

    std::cerr << "  reading slice...\n";
    channelflow_file.dataset<double>("velocity/yvel")
        .read_chunk(std::vector<size_t>{size_t(plane_position - 1), 0, 0},
                    std::vector<size_t>{1, size(axis1), size(axis2)},
                    cutting_plane);
    std::cerr << "  reading slice... done!\n";

    //auto cutting_plane =
    //    discretize(velocity_y_sampler, vec3{axis0[plane_position - 1], 0, 0},
    //               basis, vec2{1, 1}, size(axis1), size(axis2), str, 0);
    cutting_plane.write_png(out_path / "cheng" / filename_png,
                            chengs_color_scale, 13, 27);
    cutting_plane.write_png(out_path / "viridis" / filename_png,
                            viridis_color_scale, 13, 27);
    std::cerr << "discretizing plane at " << str << "... done!\n";
  }
  cutting_plane_grid.write_vtk(out_path / "planes.vtk");
}
