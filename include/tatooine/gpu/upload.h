#if TATOOINE_YAVIN_AVAILABLE
#ifndef TATOOINE_GPU_UPLOAD_H
#define TATOOINE_GPU_UPLOAD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/rank.h>
#include <yavin/texture.h>
//==============================================================================
namespace tatooine::gpu {
//==============================================================================
template <typename GPUReal>
auto download(const yavin::texture<2, GPUReal, yavin::R>& tex) {
  dynamic_multidim_array<GPUReal> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(const yavin::texture<2, GPUReal, yavin::RG>& tex) {
  dynamic_multidim_array<vec<GPUReal, 2>> data(tex.width(), tex.height());
  tex.download_data(reinterpret_cast<GPUReal*>(data.data().data()));
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(const yavin::texture<2, GPUReal, yavin::RGB>& tex) {
  dynamic_multidim_array<vec<GPUReal, 3>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GPUReal>
auto download(const yavin::texture<2, GPUReal, yavin::RGBA>& tex) {
  dynamic_multidim_array<vec<GPUReal, 4>> data(tex.width(), tex.height());
  tex.download_data(data.data());
  return data;
}
//==============================================================================
template <floating_point GPUReal = float, typename Tensor>
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex(std::vector<Tensor> const& data,
                                   integral auto... res) {
  using namespace yavin;
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
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex(const dynamic_multidim_array<Tensor>& data,
                                   std::index_sequence<Is...> /*seq*/) {
  return upload_tex<TexComps, GPUReal>(data.data(), data.size(Is)...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t Dimensions, typename TexComps, typename GPUReal = float,
          typename Tensor>
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex(const dynamic_multidim_array<Tensor>& data) {
  static_assert(Dimensions >= 1 && Dimensions <= 3);
  return upload_tex<Dimensions, TexComps, GPUReal>(
      data, std::make_index_sequence<Dimensions>{});
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename Tensor>
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex1d(
        const dynamic_multidim_array<Tensor>& data) {
  using namespace yavin;
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
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex2d(
        const dynamic_multidim_array<Tensor>& data) {
  using namespace yavin;
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
    requires std::is_floating_point_v<Tensor> ||
    is_vec<Tensor> auto upload_tex3d(
        const dynamic_multidim_array<Tensor>& data) {
  using namespace yavin;
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
          indexable_space XDomain, bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain>, Real,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<1, GPUReal, R>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain, YDomain>, Real,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<2, GPUReal, R>(gpu_data, sampler.grid().template size<0>(),
                                sampler.grid().template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain, bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain, YDomain, ZDomain>, Real,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices());
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
    gpu_data.push_back(static_cast<GPUReal>(sampler.data_at(is...)));
  });

  return texture<3, GPUReal, R>(gpu_data, sampler.grid().template size<0>(),
                                sampler.grid().template size<1>(),
                                sampler.template size<2>());
}
//==============================================================================
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain>, vec<Real, 2>,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid.num_vertices() * 2);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  return texture<1, GPUReal, RG>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 2>,
                                    HasNonConstReference> const& data) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  // gpu_data.reserve(data.grid().num_vertices() * 2);
  data.grid().iterate_over_vertex_indices([&](auto const... is) {
    auto const& v = data.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
  });

  return texture<2, GPUReal, RG>(gpu_data, data.grid().template size<0>(),
                                 data.grid().template size<1>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          indexable_space ZDomain, bool HasNonConstReference>
auto upload(
    typed_multidim_property<grid<XDomain, YDomain, ZDomain>, vec<Real, 2>,
                            HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 2);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
          indexable_space XDomain, bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain>, vec<Real, 3>,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
    auto const& v = sampler.data_at(is...);
    gpu_data.push_back(static_cast<GPUReal>(v(0)));
    gpu_data.push_back(static_cast<GPUReal>(v(1)));
    gpu_data.push_back(static_cast<GPUReal>(v(2)));
  });

  return texture<1, GPUReal, RGB>(gpu_data, sampler.grid().template size<0>());
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, floating_point Real,
          indexable_space XDomain, indexable_space YDomain,
          bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 3>,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
          indexable_space ZDomain, bool HasNonConstReference>
auto upload(
    typed_multidim_property<grid<XDomain, YDomain, ZDomain>, vec<Real, 3>,
                            HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 3);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
          indexable_space XDomain, bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain>, vec<Real, 4>,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 4);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
          indexable_space XDomain, indexable_space YDomain,
          bool HasNonConstReference>
auto upload(typed_multidim_property<grid<XDomain, YDomain>, vec<Real, 4>,
                                    HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  gpu_data.reserve(sampler.grid().num_vertices() * 4);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
          indexable_space ZDomain, bool HasNonConstReference>
auto upload(
    typed_multidim_property<grid<XDomain, YDomain, ZDomain>, vec<Real, 4>,
                            HasNonConstReference> const& sampler) {
  using namespace yavin;
  std::vector<GPUReal> gpu_data;
  sampler.reserve(gpu_data.grid().num_vertices() * 4);
  sampler.grid().iterate_over_vertex_indices([&](auto const... is) {
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
#endif
