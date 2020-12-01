#include "io.h"

#include <tatooine/linspace.h>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto read_ascii(std::filesystem::path const& file_path, int& res_x, int& res_y,
                int& res_t, double& min_x, double& min_y, double& min_t,
                double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_grid_2d<double>> {
  std::ifstream file{file_path};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not open file."};
  }
  file >> res_x >> res_y >> res_t >> min_x >> min_y >> min_t >> extent_x >>
      extent_y >> extent_t;
  std::vector<uniform_grid_2d<double>> grids(
      res_t,
      uniform_grid_2d<double>{
          linspace{min_x, min_x + extent_x, static_cast<size_t>(res_x)},
          linspace{min_y, min_y + extent_y, static_cast<size_t>(res_y)}});
  double val;
  for (int i = 0; i < res_t; ++i) {
    auto& a = grids[i].add_vertex_property<double>("a");
    grids[i].loop_over_vertex_indices([&](auto x, auto y) {
      file >> val;
      a(x, y) = val;
    });
    grids[i].loop_over_vertex_indices([&](auto x, auto y) {
      if (std::isnan(a(x, y))) {
        double mean = 0.0;
        size_t cnt  = 0;
        if (x > 0) {
          mean += a(x - 1, y);
          ++cnt;
        }
        if (y > 0) {
          mean += a(x, y - 1);
          ++cnt;
        }
        if (x < grids[i].size(0) - 1) {
          mean += a(x + 1, y);
          ++cnt;
        }
        if (y < grids[i].size(1) - 1) {
          mean += a(x, y + 1);
          ++cnt;
        }
        a(x, y) = mean / cnt;
      }
    });
  }
  for (int i = 0; i < res_t; ++i) {
    auto& b = grids[i].add_vertex_property<double>("b");
    grids[i].loop_over_vertex_indices([&](auto const... is) {
      file >> val;
      b(is...) = val;
    });
    grids[i].loop_over_vertex_indices([&](auto x, auto y) {
      if (std::isnan(b(x, y))) {
        double mean = 0.0;
        size_t cnt  = 0;
        if (x > 0) {
          mean += b(x - 1, y);
          ++cnt;
        }
        if (y > 0) {
          mean += b(x, y - 1);
          ++cnt;
        }
        if (x < grids[i].size(0) - 1) {
          mean += b(x + 1, y);
          ++cnt;
        }
        if (y < grids[i].size(1) - 1) {
          mean += b(x, y + 1);
          ++cnt;
        }
        b(x, y) = mean / cnt;
      }
    });
  }
  return grids;
}
//------------------------------------------------------------------------------
auto read_binary(std::filesystem::path const& file_path, int& res_x, int& res_y,
                 int& res_t, double& min_x, double& min_y, double& min_t,
                 double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_grid_2d<double>> {
  std::ifstream file{file_path};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not open file."};
  }
  file.read(reinterpret_cast<char*>(&res_x), sizeof(int));
  file.read(reinterpret_cast<char*>(&res_y), sizeof(int));
  file.read(reinterpret_cast<char*>(&res_t), sizeof(int));
  file.read(reinterpret_cast<char*>(&min_x), sizeof(double));
  file.read(reinterpret_cast<char*>(&min_y), sizeof(double));
  file.read(reinterpret_cast<char*>(&min_t), sizeof(double));
  file.read(reinterpret_cast<char*>(&extent_x), sizeof(double));
  file.read(reinterpret_cast<char*>(&extent_y), sizeof(double));
  file.read(reinterpret_cast<char*>(&extent_t), sizeof(double));
  std::vector<uniform_grid_2d<double>> grids(
      res_t,
      uniform_grid_2d<double>{
          linspace{min_x, min_x + extent_x, static_cast<size_t>(res_x)},
          linspace{min_y, min_y + extent_y, static_cast<size_t>(res_y)}});
  for (int i = 0; i < res_t; ++i) {
    auto& a = grids[i].add_vertex_property<double>("a");
    grids[i].loop_over_vertex_indices([&](auto const... is) {
      file.read(reinterpret_cast<char*>(&a(is...)), sizeof(double));
    });
    grids[i].loop_over_vertex_indices([&](auto x, auto y) {
      if (std::isnan(a(x, y))) {
        double mean = 0.0;
        size_t cnt  = 0;
        if (x > 0) {
          mean += a(x - 1, y);
          ++cnt;
        }
        if (y > 0) {
          mean += a(x, y - 1);
          ++cnt;
        }
        if (x < grids[i].size(0) - 1) {
          mean += a(x + 1, y);
          ++cnt;
        }
        if (y < grids[i].size(1) - 1) {
          mean += a(x, y + 1);
          ++cnt;
        }
        a(x, y) = mean / cnt;
      }
    });
  }
  for (int i = 0; i < res_t; ++i) {
    auto& b = grids[i].add_vertex_property<double>("b");
    grids[i].loop_over_vertex_indices([&](auto const... is) {
      file.read(reinterpret_cast<char*>(&b(is...)), sizeof(double));
    });
    grids[i].loop_over_vertex_indices([&](auto x, auto y) {
      if (std::isnan(b(x, y))) {
        double mean = 0.0;
        size_t cnt  = 0;
        if (x > 0) {
          mean += b(x - 1, y);
          ++cnt;
        }
        if (y > 0) {
          mean += b(x, y - 1);
          ++cnt;
        }
        if (x < grids[i].size(0) - 1) {
          mean += b(x + 1, y);
          ++cnt;
        }
        if (y < grids[i].size(1) - 1) {
          mean += b(x, y + 1);
          ++cnt;
        }
        b(x, y) = mean / cnt;
      }
    });
  }
  return grids;
}
//------------------------------------------------------------------------------
auto write_ascii(std::filesystem::path const&                file_path,
                 std::vector<uniform_grid_2d<double>> const& grids,
                 int const res_x, int const res_y, int const res_t,
                 double const min_x, double const min_y, double const min_t,
                 double const extent_x, double const extent_y,
                 double const extent_t) -> void {
  std::ofstream file{file_path};
  if (file.is_open()) {
    file << res_x << " " << res_y << " " << res_t << '\n';
    file << min_x << " " << min_y << " " << min_t << '\n';
    file << extent_x << " " << extent_y << " " << extent_t << '\n';
  }
  for (auto const& g : grids) {
    auto& a = g.vertex_property<double>("a");
    g.loop_over_vertex_indices(
        [&](auto const... is) { file << a(is...) << ' '; });
  }
  file << '\n';
  for (auto const& g : grids) {
    auto& b = g.vertex_property<double>("b");
    g.loop_over_vertex_indices(
        [&](auto const... is) { file << b(is...) << ' '; });
  }
}
//------------------------------------------------------------------------------
auto write_binary(std::filesystem::path const&                file_path,
                  std::vector<uniform_grid_2d<double>> const& grids,
                  int const res_x, int const res_y, int const res_t,
                  double const min_x, double const min_y, double const min_t,
                  double const extent_x, double const extent_y,
                  double const extent_t) -> void {
  std::ofstream file{file_path, std::ios::binary};
  if (file.is_open()) {
    file.write(reinterpret_cast<char const*>(&res_x), sizeof(int));
    file.write(reinterpret_cast<char const*>(&res_y), sizeof(int));
    file.write(reinterpret_cast<char const*>(&res_t), sizeof(int));
    file.write(reinterpret_cast<char const*>(&min_x), sizeof(double));
    file.write(reinterpret_cast<char const*>(&min_y), sizeof(double));
    file.write(reinterpret_cast<char const*>(&min_t), sizeof(double));
    file.write(reinterpret_cast<char const*>(&extent_x), sizeof(double));
    file.write(reinterpret_cast<char const*>(&extent_y), sizeof(double));
    file.write(reinterpret_cast<char const*>(&extent_t), sizeof(double));
  }
  for (auto const& g : grids) {
    auto& a = g.vertex_property<double>("a");
    g.loop_over_vertex_indices([&](auto const... is) {
      file.write(reinterpret_cast<char const*>(&a(is...)), sizeof(double));
    });
  }
  for (auto const& g : grids) {
    auto& b = g.vertex_property<double>("b");
    g.loop_over_vertex_indices([&](auto const... is) {
      file.write(reinterpret_cast<char const*>(&b(is...)), sizeof(double));
    });
  }
}
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
