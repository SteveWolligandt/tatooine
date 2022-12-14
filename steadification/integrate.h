#ifndef TATOOINE_STEADIFICATION_INTEGRATE_H
#define TATOOINE_STEADIFICATION_INTEGRATE_H
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/streamsurface.h>

#include "pathsurface.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename Real>
auto integrate(
    const vectorfield<V, Real, 2>& v, const std::string& dataset_name,
    const std::set<std::tuple<size_t, grid_edge_iterator<Real, 2>, Real>>&
                         edges,
    const grid<Real, 2>& domain, const Real min_t, const Real max_t,
    const Real min_btau, const Real max_ftau, const size_t seed_res,
    const Real stepsize) {
  if (!filesystem::exists("pathsurfaces")) {
    filesystem::create_directory("pathsurfaces");
  }

  const auto pathsurface_dir = +"pathsurfaces/" + dataset_name + "/";
  if (!filesystem::exists(pathsurface_dir)) {
    filesystem::create_directory(pathsurface_dir);
  }

  std::atomic_size_t progress_counter = 0;
  std::thread        t{[&] {
    float     progress  = 0.0;
    const int bar_width = 10;
    std::cerr << "integrating pathsurfaces... ";
    while (progress < 1.0) {
      progress = float(progress_counter) / (edges.size() );

      int pos = bar_width * progress;
      for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
          std::cerr << "\u2588";
        else if (i == pos)
          std::cerr << "\u2592";
        else
          std::cerr << "\u2591";
      }
      std::cerr << " " << int(progress * 100.0) << " %" << '\r';
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    for (int i = 0; i < bar_width; ++i) { std::cerr << "\u2588"; }
    std::cerr << "integrating pathsurfaces, done!       \n";
  }};

  for (const auto& [edge_idx, edge_it, t0] : edges) {
    auto filename_vtk = pathsurface_dir;
    for (size_t i = 0; i < 2; ++i) {
      filename_vtk += std::to_string(domain.size(i)) + "_";
    }
    const auto ftau = min(max_ftau, max_t - t0);
    const auto btau = max(min_btau, min_t - t0);
    filename_vtk += std::to_string(t0) + "_" + std::to_string(btau) + "_" +
                    std::to_string(ftau) + "_" + std::to_string(seed_res) +
                    "_" + std::to_string(stepsize) + "_" +
                    std::to_string(edge_idx) + ".vtk";
    if (!filesystem::exists(filename_vtk)) {
      simple_tri_mesh<Real, 2> psf =
          pathsurface(v, *edge_it, t0, btau, ftau, seed_res, stepsize)
              .first;
      psf.write_vtk(filename_vtk);
    }
    progress_counter++;
  }
  t.join();
  return pathsurface_dir;
}
template <typename V, typename Real>
auto integrate(const vectorfield<V, Real, 2>& v, const std::string& dataset_name,
               const std::set<std::pair<size_t, grid_edge_iterator<Real, 3>>>&
                                      edges,
               const grid<Real, 3>& domain, const Real btau,
               const Real ftau, const size_t seed_res,
               const Real stepsize) {
  if (!filesystem::exists("pathsurfaces")) {
    filesystem::create_directory("pathsurfaces");
  }

  const auto pathsurface_dir = +"pathsurfaces/" + dataset_name + "/";
  if (!filesystem::exists(pathsurface_dir)) {
    filesystem::create_directory(pathsurface_dir);
  }

  std::string        filename_vtk;
  std::atomic_size_t progress_counter = 0;
  std::thread        t{[&] {
    float     progress  = 0.0;
    const int bar_width = 10;
    while (progress < 1.0) {
      progress = float(progress_counter) / (edges.size());

      int pos = bar_width * progress;
      std::cerr << "integrating pathsurfaces ";
      for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
          std::cerr << "\u2588";
        else if (i == pos)
          std::cerr << "\u2592";
        else
          std::cerr << "\u2591";
      }
      std::cerr << " " << int(progress * 100.0) << " %" << '\r';
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    for (int i = 0; i < bar_width; ++i) { std::cerr << "\u2588"; }
    std::cerr << "integrating pathsurfaces done!                  \n";
  }};

  for (const auto& [edge_idx, edge_it] : edges) {
    filename_vtk = pathsurface_dir;
    for (size_t i = 0; i < 3; ++i) {
      filename_vtk += std::to_string(domain.size(i)) + "_";
    }
    const auto min_t0 = domain.front(2);
    const auto max_t0 = domain.back(2);
    filename_vtk += std::to_string(min_t0) + "_" + std::to_string(max_t0) +
                    "_" + std::to_string(btau) + "_" + std::to_string(ftau) +
                    "_" + std::to_string(seed_res) + "_" +
                    std::to_string(stepsize) + "_" + std::to_string(edge_idx) +
                    ".vtk";
    if (!filesystem::exists(filename_vtk)) {
      simple_tri_mesh<Real, 2> psf =
          pathsurface(v, *edge_it, btau, ftau, seed_res, stepsize).first;
      psf.write_vtk(filename_vtk);
    }
    progress_counter++;
  }
  t.join();
  return pathsurface_dir;
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
