#ifndef TATOOINE_AUTONOMOUS_PARTICLES_WRITE_ELLIPSES_H
#define TATOOINE_AUTONOMOUS_PARTICLES_WRITE_ELLIPSES_H
//==============================================================================
#include <tatooine/autonomous_particle.h>

#include <filesystem>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <flowmap_c Flowmap>
auto write_x0(std::vector<autonomous_particle<Flowmap>> const& particles,
              std::filesystem::path const& path) -> void {
  std::vector<size_t> const cnt{1, 2, 3};
  std::vector<size_t>       indices{0, 0, 0};
  netcdf::file              file{path.string(),
                    netCDF::NcFile::replace};
  auto                      var = file.add_variable<float>(
      "transformations",
      {file.add_dimension("index"), file.add_dimension("row", 2),
       file.add_dimension("column", 3)});
  for (auto const& p : particles) {
    auto sqrS =
        inv(p.nabla_phi1()) * p.S() * p.S() * inv(transposed(p.nabla_phi1()));
    auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
    eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
    transposed(eig_vecs);

    auto   Sback = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
    mat23f T{{Sback(0, 0), Sback(0, 1), p.x0()(0)},
             {Sback(1, 0), Sback(1, 1), p.x0()(1)}};
    {
      var.write(indices, cnt, T.data_ptr());
      ++indices.front();
    }
  }
}
//------------------------------------------------------------------------------
template <flowmap_c Flowmap>
auto write_x1(std::vector<autonomous_particle<Flowmap>> const& particles,
              std::filesystem::path const& path) -> void {
  std::vector<size_t> const cnt{1, 2, 3};
  std::vector<size_t>       indices{0, 0, 0};
  netcdf::file ellipsis_file{path.string(), netCDF::NcFile::replace};
  auto         var = ellipsis_file.add_variable<float>(
      "transformations", {ellipsis_file.add_dimension("index"),
                          ellipsis_file.add_dimension("row", 2),
                          ellipsis_file.add_dimension("column", 3)});

  for (auto const& ap : particles) {
    mat23f T{{ap.S()(0, 0), ap.S()(0, 1), ap.x1()(0)},
             {ap.S()(1, 0), ap.S()(1, 1), ap.x1()(1)}};
    var.write(indices, cnt, T.data_ptr());
    ++indices.front();
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
