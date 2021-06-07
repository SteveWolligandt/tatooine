#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/direct_isosurface_rendering.h>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/hdf5.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tat = tatooine;
//==============================================================================
template <typename SamplerX, typename SamplerY, typename SamplerZ>
struct vectorfield
    : tat::vectorfield<vectorfield<SamplerX, SamplerY, SamplerZ>, double, 3> {
  using this_t   = vectorfield<SamplerX, SamplerY, SamplerZ>;
  using parent_t = tat::vectorfield<this_t, double, 3>;

  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;

  SamplerX m_sampler_x;
  SamplerY m_sampler_y;
  SamplerZ m_sampler_z;
  //============================================================================
  template <typename _SamplerX, typename _SamplerY, typename _SamplerZ>
  vectorfield(_SamplerX sampler_x, _SamplerY sampler_y, _SamplerZ sampler_z)
      : m_sampler_x{sampler_x},
        m_sampler_y{sampler_y},
        m_sampler_z{sampler_z} {}
  //----------------------------------------------------------------------------
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t {
    return {m_sampler_x(x), m_sampler_y(x), m_sampler_z(x)};
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool {
    return m_sampler_x.grid().in_domain(x(0), x(1), x(2));
  }
};
//==============================================================================
template <typename SamplerX, typename SamplerY, typename SamplerZ>
vectorfield(SamplerX, SamplerY, SamplerZ)
    -> vectorfield<std::decay_t<SamplerX>, std::decay_t<SamplerY>,
                   std::decay_t<SamplerZ>>;
//==============================================================================
auto main() -> int {
  // read full domain axes
  tat::hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5"};
  tat::hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5"};
  tat::hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const axis0 =
      axis0_file.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file.dataset<double>("CartGrid/axis2").read_as_vector();

  tat::grid full_domain{axis0, axis1, axis2};
  full_domain.set_chunk_size_for_lazy_properties(256);
  std::cerr << "full_domain:\n" << full_domain << '\n';

  auto axis0_Q = axis0;
  axis0_Q.pop_back();
  tat::grid full_domain_Q{axis0_Q, axis1, axis2};
  full_domain_Q.set_chunk_size_for_lazy_properties(256);
  std::cerr << "full_domain_Q:\n" << full_domain_Q << '\n';

  tat::hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};
  //auto velx_122_full_dataset =
  //    channelflow_122_full_file.dataset<double>("velocity/xvel");
  //auto vely_122_full_dataset =
  //    channelflow_122_full_file.dataset<double>("velocity/yvel");
  //auto velz_122_full_dataset =
  //    channelflow_122_full_file.dataset<double>("velocity/zvel");
  auto Q_122_full_dataset =
      channelflow_122_full_file.dataset<double>("Q_pnorm");
  auto vel_mag_122_full_dataset =
      channelflow_122_full_file.dataset<double>("vel_mag");

  //auto& velx_122_full =
  //    full_domain.insert_lazy_vertex_property(velx_122_full_dataset, "Vx_122");
  //auto& vely_122_full =
  //    full_domain.insert_lazy_vertex_property(vely_122_full_dataset, "Vy_122");
  //auto& velz_122_full =
  //    full_domain.insert_lazy_vertex_property(velz_122_full_dataset, "Vz_122");
  auto& Q_122_full =
      full_domain_Q.insert_vertex_property(Q_122_full_dataset, "Q_122");
  auto& vel_mag_122_full =
      full_domain_Q.insert_vertex_property(vel_mag_122_full_dataset, "vel_mag_122");
  //velx_122_full.limit_num_chunks_loaded();
  //vely_122_full.limit_num_chunks_loaded();
  //velz_122_full.limit_num_chunks_loaded();
  //Q_122_full.limit_num_chunks_loaded();
  //velx_122_full.set_max_num_chunks_loaded(30);
  //vely_122_full.set_max_num_chunks_loaded(30);
  //velz_122_full.set_max_num_chunks_loaded(30);
  //Q_122_full.set_max_num_chunks_loaded(30);

  //auto velx_122_full_sampler = velx_122_full.linear_sampler();
  //auto vely_122_full_sampler = vely_122_full.linear_sampler();
  //auto velz_122_full_sampler = velz_122_full.linear_sampler();
  //auto vel_122_field = vectorfield{velx_122_full_sampler, vely_122_full_sampler,
  //                                 velz_122_full_sampler};
  auto Q_122_full_sampler       = Q_122_full.linear_sampler();
  auto vel_mag_122_full_sampler = vel_mag_122_full.linear_sampler();

  tat::color_scales::viridis color_scale;

  size_t const width = 20000, height = 10000;

  auto const full_domain_eye =
      tat::vec3{0.7940901239835871, 0.04097490152128994, 0.5004262802265552};
  auto const full_domain_lookat =
      tat::vec3{-0.7384532106212904, 0.7745404345929863, -0.4576538576946477};
  auto const full_domain_up =
      tat::vec3{-0.35221800146747856, 0.3807796045093859, 0.8549557720911246};
  auto full_domain_cam =
      tat::rendering::perspective_camera<double>{full_domain_eye,
                                                 full_domain_lookat,
                                                 full_domain_up,
                                                 60,
                                                 0.01,
                                                 1000,
                                                 width,
                                                 height};
  tat::real_t const min      = 13;
  tat::real_t const max      = 27;
  tat::real_t const isovalue = 5e6;

  auto shader =
      [&](auto const& x_iso, auto const& gradient, auto const& view_dir) {
        auto const normal      = normalize(gradient);
        auto const diffuse     = std::abs(dot(view_dir, normal));
        auto const reflect_dir = reflect(-view_dir, normal);
        auto const spec_dot =
            std::max<tat::real_t>(std::abs(dot(reflect_dir, view_dir)), 0);
        auto const specular = std::pow(spec_dot, 100);
        auto const scalar   = std::clamp<tat::real_t>(
            (vel_mag_122_full_sampler(x_iso) - min) / (max - min), 0, 1);
        auto const albedo = color_scale(scalar);
        auto const col    = albedo * diffuse;
        return tat::vec{col(0), col(1), col(2),
                        scalar * scalar * scalar * scalar * scalar};
      };
  auto const rendering_grid = direct_isosurface_rendering(
      full_domain_cam, Q_122_full_sampler, isovalue, shader);
  write_png("channel_flow_Q_5e6_with_velocity_magnitude.png",
            rendering_grid.vec3_vertex_property("rendered_isosurface"));
}
