#ifndef TATOOINE_LINE_RENDERER_H
#define TATOOINE_LINE_RENDERER_H
//==============================================================================
#include <tatooine/line.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::gpu {
//==============================================================================
template <typename Real, size_t N, template <typename> typename Interpolator>
auto upload(const parameterized_line<Real, N, Interpolator>& l) {
  using gpu_t = yavin::indexeddata<yavin::vec<float, N>,
                                   yavin::vec<float, N>,
                                   yavin::Scalar<float>>;
  typename gpu_t::vbo_data_vec vbo_data;
  vbo_data.reserve(l.num_vertices());
  typename gpu_t::ibo_data_vec ibo_data;
  ibo_data.reserve(l.num_vertices() * 2 - 2);

  for (size_t i = 0; i < l.num_vertices(); ++i) {
    const auto& pos = l.vertex_at(i);
    const auto tan = l.tangent_at(i);
    const auto t = l.parameterization_at(i);
    yavin::vec<float, N> ypos, ytan;
    for (size_t j = 0; j < N; ++j) {
      ypos(j) = static_cast<float>(pos(j));
      ytan(j) = static_cast<float>(tan(j));
    }
    ypos(0) *= -1;
    vbo_data.push_back({ypos, ytan, static_cast<float>(t)});
  }
  for (size_t i = 0; i < l.num_vertices() - 1; ++i) {
    ibo_data.push_back(i);
    ibo_data.push_back(i + 1);
  }
  return gpu_t{vbo_data, ibo_data};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, template <typename> typename Interpolator>
auto upload(const std::vector<parameterized_line<Real, N, Interpolator>>& ls) {
  using gpu_t = yavin::indexeddata<yavin::vec<float, N>,
                                   yavin::vec<float, N>,
                                   yavin::Scalar<float>>;
  std::vector<gpu_t> uploads;
  uploads.reserve(ls.size());
  for (const auto& l : ls) { uploads.push_back(upload(l)); }
  return uploads;
}
//==============================================================================
}
//==============================================================================
#endif
