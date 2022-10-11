#if TATOOINE_GL_AVAILABLE
#ifndef TATOOINE_GPU_UPLOAD_H
#define TATOOINE_GPU_UPLOAD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/gl/texture.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rank.h>
//==============================================================================
namespace tatooine::gpu {
//==============================================================================
template <typename GPUReal>
auto download(gl::texture<2, GPUReal, gl::R> const& tex) {
  auto data = dynamic_multidim_array<GPUReal>{tex.width(), tex.height()};
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(gl::texture<2, GPUReal, gl::RG> const& tex) {
  dynamic_multidim_array<vec<GPUReal, 2>> data(tex.width(), tex.height());
  tex.download_data(reinterpret_cast<GPUReal*>(data.data().data()));
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(gl::texture<2, GPUReal, gl::RGB> const& tex) {
  dynamic_multidim_array<vec<GPUReal, 3>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(gl::texture<2, GPUReal, gl::RGBA> const& tex) {
  dynamic_multidim_array<vec<GPUReal, 4>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
//==============================================================================
template <floating_point GPUReal = float, typename Tensor>
auto upload_tex(std::vector<Tensor> const& data,
                integral auto const... res) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  for (auto const& d : data) {
    if constexpr (std::is_floating_point_v<Tensor>) {
      gpu_data.push_back(d);
    } else {
      for (size_t i = 0; i < Tensor::size(0); ++i) {
        gpu_data.push_back(d(i));
      }
    }
  }
  if constexpr (rank<Tensor>() == 0) {
    return texture<sizeof...(res), GPUReal, R>(gpu_data, res...);
  } else if constexpr (rank<Tensor>() == 1) {
    return texture<sizeof...(res), GPUReal, RG>(gpu_data, res...);
  } else if constexpr (rank<Tensor>() == 2) {
    return texture<sizeof...(res), GPUReal, RGB>(gpu_data, res...);
  } else if constexpr (rank<Tensor>() == 3) {
    return texture<sizeof...(res), GPUReal, RGBA>(gpu_data, res...);
  }
}
//------------------------------------------------------------------------------
template <size_t Dimensions, typename TexComps, floating_point GPUReal = float,
          typename Tensor, size_t... Is>
auto upload_tex(dynamic_multidim_array<Tensor> const& data,
                std::index_sequence<Is...> /*seq*/) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  return upload_tex<TexComps, GPUReal>(data.data(), data.size(Is)...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t Dimensions, typename TexComps, typename GPUReal = float,
          typename Tensor>
auto upload_tex(dynamic_multidim_array<Tensor> const& data) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  static_assert(Dimensions >= 1 && Dimensions <= 3);
  return upload_tex<Dimensions, TexComps, GPUReal>(
      data, std::make_index_sequence<Dimensions>{});
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename Tensor>
auto upload_tex1d(dynamic_multidim_array<Tensor> const& data) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  using namespace gl;
  if constexpr (rank<Tensor>() == 0) {
    return upload_tex<1, R, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 1) {
    return upload_tex<1, RG, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 2) {
    return upload_tex<1, RGB, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 3) {
    return upload_tex<1, RGBA, GPUReal>(data);
  }
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename Tensor>
auto upload_tex2d(const dynamic_multidim_array<Tensor>& data) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  using namespace gl;
  if constexpr (rank<Tensor>() == 0) {
    return upload_tex<2, R, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 1) {
    return upload_tex<2, RG, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 2) {
    return upload_tex<2, RGB, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 3) {
    return upload_tex<2, RGBA, GPUReal>(data);
  }
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename Tensor>
auto upload_tex3d(const dynamic_multidim_array<Tensor>& data) requires
    std::is_floating_point_v<Tensor> || static_vec<Tensor> {
  using namespace gl;
  if constexpr (rank<Tensor>() == 0) {
    return upload_tex<3, R, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 1) {
    return upload_tex<3, RG, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 2) {
    return upload_tex<3, RGB, GPUReal>(data);
  } else if constexpr (rank<Tensor>() == 3) {
    return upload_tex<3, RGBA, GPUReal>(data);
  }
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<rectilinear_grid<XDomain>, Real,
                                                 HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size());
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(grid_vertex_property(is...)));
  });

  auto tex =  texture<1, GPUReal, R>(gpu_data,
                                grid_vertex_property.grid().template size<0>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<rectilinear_grid<XDomain, YDomain>, Real,
                                                 HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size());
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(grid_vertex_property(is...)));
  });

  auto tex =  texture<2, GPUReal, R>(gpu_data,
                                grid_vertex_property.grid().template size<0>(),
                                grid_vertex_property.grid().template size<1>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          floating_point_range ZDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain, ZDomain>, Real, HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size());
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(grid_vertex_property(is...)));
  });

  auto tex =  texture<3, GPUReal, R>(gpu_data,
                                grid_vertex_property.grid().template size<0>(),
                                grid_vertex_property.grid().template size<1>(),
                                grid_vertex_property.grid().template size<2>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<rectilinear_grid<XDomain>, vec<Real, 2>,
                                                 HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 2);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  auto tex =  texture<1, GPUReal, RG>(
      gpu_data, grid_vertex_property.grid().template size<0>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          bool HasNonConstReference>
auto upload(
    typed_grid_vertex_property_interface<rectilinear_grid<XDomain, YDomain>, vec<Real, 2>,
                                         HasNonConstReference> const& data) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  // gpu_data.reserve(data.grid().vertices().size() * 2);
  data.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = data(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  auto tex = texture<2, GPUReal, RG>(gpu_data, data.grid().template size<0>(),
                                     data.grid().template size<1>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          floating_point_range ZDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain, ZDomain>, vec<Real, 2>,
            HasNonConstReference> const& grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 2);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  auto tex =  texture<3, GPUReal, RG>(
      gpu_data, grid_vertex_property.grid().template size<0>(),
      grid_vertex_property.grid().template size<1>(),
      grid_vertex_property.grid().template size<2>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<rectilinear_grid<XDomain>, vec<Real, 3>,
                                                 HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 3);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  auto tex =  texture<1, GPUReal, RGB>(
      gpu_data, grid_vertex_property.grid().template size<0>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain>, vec<Real, 3>, HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 3);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  auto tex = texture<2, GPUReal, RGB>(
      gpu_data, grid_vertex_property.grid().template size<0>(),
      grid_vertex_property.grid().template size<1>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          floating_point_range ZDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain, ZDomain>, vec<Real, 3>,
            HasNonConstReference> const& grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 3);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  auto tex =  texture<3, GPUReal, RGB>(
      gpu_data, grid_vertex_property.grid().template size<0>(),
      grid_vertex_property.grid().template size<1>(),
      grid_vertex_property.grid().template size<2>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<rectilinear_grid<XDomain>, vec<Real, 4>,
                                                 HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 4);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  auto tex =  texture<1, GPUReal, RGBA>(
      gpu_data, grid_vertex_property.grid().template size<0>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain>, vec<Real, 4>, HasNonConstReference> const&
                grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(grid_vertex_property.grid().vertices().size() * 4);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  auto tex =  texture<2, GPUReal, RGBA>(
      gpu_data, grid_vertex_property.grid().template size<0>(),
      grid_vertex_property.grid().template size<1>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          floating_point_range XDomain, floating_point_range YDomain,
          floating_point_range ZDomain, bool HasNonConstReference>
auto upload(typed_grid_vertex_property_interface<
            rectilinear_grid<XDomain, YDomain, ZDomain>, vec<Real, 4>,
            HasNonConstReference> const& grid_vertex_property) {
  using namespace gl;
  std::vector<GPUReal> gpu_data;
  grid_vertex_property.reserve(gpu_data.grid().vertices().size() * 4);
  grid_vertex_property.grid().vertices().iterate_indices([&](auto const... is) {
    auto const& v = grid_vertex_property(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
    gpu_data.push_back(static_cast<GPUReal>(v(3)));
  });

  auto tex =  texture<3, GPUReal, RGBA>(
      gpu_data, grid_vertex_property.grid().template size<0>(),
      grid_vertex_property.grid().template size<1>(),
      grid_vertex_property.grid().template size<2>());
  tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
  return tex;
}
//==============================================================================
}  // namespace tatooine::gpu
//==============================================================================
#endif
#endif
