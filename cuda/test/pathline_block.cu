#include <catch2/catch.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <tatooine/cuda/pathline_block.cuh>
#include <tatooine/doublegyre.h>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================

TEST_CASE("cuda_pathline_block1", "[dg]") {
  numerical::doublegyre<double> v;
  grid<double, 3> block{linspace<double>{0, 2, 21},
                        linspace<double>{0, 1, 11},
                        linspace<double>{0, 10, 5}};

  auto d_v = upload<float>(
      v,
      grid<double, 2>{linspace<double>{0, 2, 101}, linspace<double>{0, 1, 51}},
      linspace<double>{0, 10, 51});

  auto d_pathlines = pathline_block(d_v, block, 20);

  auto pathlines = d_pathlines.download();
  free(d_v, d_pathlines);

  vtk::legacy_file_writer writer("cuda_pathlines_doublegyre.vtk", vtk::POLYDATA);
  if (writer.is_open()) {
    size_t num_pts = num_pathline_samples * block.num_vertices();
    std::vector<std::array<float, 3>> points;
    std::vector<std::vector<size_t>> line_seqs;
    points.reserve(num_pts);
    line_seqs.reserve(block.num_vertices());

    size_t cur_first = 0;
    for (size_t i = 0; i < block.num_vertices(); ++i) {
      // add points
      for (size_t j = 0; j < num_pathline_samples; ++j) {
        size_t idx = i * num_pathline_samples*3;
        points.push_back(
            {pathlines[idx], pathlines[idx + 1], pathlines[idx + 1]});
      }

      // add lines
      boost::iota(line_seqs.emplace_back(l.num_vertices()), cur_first);
      if (l.is_closed()) { line_seqs.back().push_back(cur_first); }
      cur_first += num_pathline_samples;
    }

    // write
    writer.set_title("foo");
    writer.write_header();
    writer.write_points(points);
    writer.write_lines(line_seqs);
    writer.write_point_data(num_pts);
    writer.close();
  }
}

//==============================================================================
}}}
//==============================================================================
