#ifndef TATOOINE_MISC_BIFURCATIONLINE_PROGRESS_H
#define TATOOINE_MISC_BIFURCATIONLINE_PROGRESS_H

#include <tatooine/line.h>
#include "write_step.h"
#include <tatooine/curve_to_streamline.h>
#include <filesystem>
#include <string>

//==============================================================================
namespace tatooine::misc::bifurcationline {
//==============================================================================

template <typename V, typename Lines>
void progress(V&& v, Lines&& lines, const std::string& name, size_t vertex_inc) {
  size_t cnt = 0;
  for (auto& l_ : lines) {
    line<double, 3> l;
    for (size_t i = 0; i < l_.size(); i+= vertex_inc){
      l.push_back(l_.vertex_at(i));
    }
    curve_to_streamline c2s;
    const double        initial_stepsize = 0.01;
    const double        delta            = 0.999;
    const size_t        n                = 10000;
    const double        t0               = 0;
    auto                current_stepsize = initial_stepsize;

    if (!std::filesystem::exists(name)) {
      std::filesystem::create_directory(name);
    }
    for (size_t i = 0; i < n; ++i) {
      auto new_l = c2s(v, t0, l, current_stepsize, delta, 1);
      write_step(v, t0, l, c2s,
                 name + "/" + name + "_" + std::to_string(cnt) + "__" +
                     std::to_string(i) + ".vtk");
      l = std::move(new_l);
      current_stepsize *= delta;
    }
    ++cnt;
  }
}

//==============================================================================
}  // namespace tatooine::misc::curve_to_streamline
//==============================================================================

#endif
