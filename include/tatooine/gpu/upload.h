#ifndef TATOOINE_GPU_FIELD_TO_GPU_H
#define TATOOINE_GPU_FIELD_TO_GPU_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <yavin/texture.h>
//==============================================================================
namespace tatooine::gpu {
//==============================================================================
template <typename GPUReal = float>
auto download(const yavin::texture<2, GPUReal, yavin::R>& tex) {
  dynamic_multidim_array<float> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal = float>
auto download(const yavin::texture<2, GPUReal, yavin::RG>& tex) {
  dynamic_multidim_array<vec<float, 2>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal = float>
auto download(const yavin::texture<2, GPUReal, yavin::RGB>& tex) {
  dynamic_multidim_array<vec<float, 3>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal = float>
auto download(const yavin::texture<2, GPUReal, yavin::RGBA>& tex) {
  dynamic_multidim_array<vec<float, 4>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain>
auto upload(const typed_multidim_property<grid<XDomain>, Real>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<1, GPUReal, R>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain>
auto upload(
    const typed_multidim_property<grid<XDomain, YDomain>, Real>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<2, GPUReal, R>(gpu_data, sampler.grid().template size<0>(),
                                sampler.grid().template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain, ZDomain>,
                                          Real>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<3, GPUReal, R>(gpu_data, sampler.grid().template size<0>(),
                                sampler.grid().template size<1>(),
                                sampler.template size<2>());
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain>
auto upload(
    const typed_multidim_property<grid<XDomain>, vec<Real, 2>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices() * 2);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  return texture<1, GPUReal, RG>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 2>>&
                sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 2);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  return texture<2, GPUReal, RG>(gpu_data, sampler.grid().template size<0>(),
                                 sampler.grid().template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain, ZDomain>,
                                          vec<Real, 2>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 2);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  return texture<3, GPUReal, RG>(gpu_data, sampler.grid().template size<0>(),
                                 sampler.grid().template size<1>(),
                                 sampler.grid().template size<2>());
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain>
auto upload(
    const typed_multidim_property<grid<XDomain>, vec<Real, 3>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  return texture<1, GPUReal, RGB>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 3>>&
                sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  return texture<2, GPUReal, RGB>(gpu_data, sampler.grid().template size<0>(),
                                  sampler.template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain, ZDomain>,
                                          vec<Real, 3>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  return texture<3, GPUReal, RGB>(gpu_data, sampler.grid().template size<0>(),
                                  sampler.grid().template size<1>(),
                                  sampler.grid().template size<2>());
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain>
auto upload(
    const typed_multidim_property<grid<XDomain>, vec<Real, 4>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 4);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  return texture<1, GPUReal, RGBA>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 4>>&
                sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 4);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  return texture<2, GPUReal, RGBA>(gpu_data, sampler.grid().template size<0>(),
                                   sampler.template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain>
auto upload(const typed_multidim_property<grid<XDomain, YDomain, ZDomain>,
                                          vec<Real, 4>>& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  sampler.reserve(gpu_data.grid().num_vertices() * 4);
  sampler.grid().loop_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  return texture<3, GPUReal, RGBA>(gpu_data, sampler.grid().template size<0>(),
                                   sampler.template size<1>(),
                                   sampler.template size<2>());
}
//==============================================================================
}  // namespace tatooine::gpu
//==============================================================================

#endif
