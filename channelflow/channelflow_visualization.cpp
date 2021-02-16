#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/direct_volume_rendering.h>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/hdf5.h>
#include <tatooine/isosurface.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tat = tatooine;
template <typename Sampler>
struct scalarfield : tat::scalarfield<scalarfield<Sampler>, double, 3> {
  using this_t   = scalarfield<Sampler>;
  using parent_t = tat::scalarfield<this_t, double, 3>;

  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;

  Sampler m_sampler;
  //============================================================================
  template <typename _Sampler>
  scalarfield(_Sampler sampler) : m_sampler{sampler} {}
  //----------------------------------------------------------------------------
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t final {
    return m_sampler(x(0), x(1), x(2));
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool final {
    return m_sampler.grid().in_domain(x(0), x(1), x(2));
  }
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Sampler>
scalarfield(Sampler) -> scalarfield<std::decay_t<Sampler>>;
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
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t final {
    return {m_sampler_x(x(0), x(1), x(2)), m_sampler_y(x(0), x(1), x(2)),
            m_sampler_z(x(0), x(1), x(2))};
  }
  //----------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool final {
    return m_sampler_x.grid().in_domain(x(0), x(1), x(2));
  }
};
//==============================================================================
template <typename SamplerX, typename SamplerY, typename SamplerZ>
vectorfield(SamplerX, SamplerY, SamplerZ)
    -> vectorfield<std::decay_t<SamplerX>, std::decay_t<SamplerY>,
                   std::decay_t<SamplerZ>>;
//==============================================================================
auto add_scalar_prop(auto const& threedpart_domain, auto& channelflow_file,
                     auto&& creator, std::string const& name) -> auto& {
  auto arr = tat::dynamic_multidim_array<tat::vec3>(512, 4096, 256);
  threedpart_domain.loop_over_vertex_indices(
      [&](auto const... is) { arr(is...) = creator(is...); });
  auto& prop = channelflow_file.group("variables")
                   .template add_dataset<double>(name, 512, 4096, 256);
  prop.write(arr.data());
  return prop;
}
//------------------------------------------------------------------------------
auto add_vec3_prop(auto const& threedpart_domain, auto& channelflow_file, auto&& creator,
                   std::string const& name) {
  auto arr_x = tat::dynamic_multidim_array<double>(512, 4096, 256);
  auto arr_y = tat::dynamic_multidim_array<double>(512, 4096, 256);
  auto arr_z = tat::dynamic_multidim_array<double>(512, 4096, 256);
  threedpart_domain.loop_over_vertex_indices([&](auto const... is) {
    auto const p = creator(is...);
    arr_x(is...) = p.x();
    arr_y(is...) = p.y();
    arr_z(is...) = p.z();
  });
  channelflow_file.group("variables")
      .template add_dataset<double>(name + "_x", 512, 4096, 256)
      .write(arr_x.data());

  channelflow_file.group("variables")
      .template add_dataset<double>(name + "_y", 512, 4096, 256)
      .write(arr_y.data());

  channelflow_file.group("variables")
      .template add_dataset<double>(name + "_z", 512, 4096, 256)
      .write(arr_z.data());
}
//------------------------------------------------------------------------------
auto add_acceleration(auto const& full_domain, auto const& threedpart_domain,
                      auto& channelflow_file, auto const& velx,
                      auto const& vely, auto const& velz,
                      auto const& temporal_diff_x, auto const& temporal_diff_y,
                      auto const& temporal_diff_z) {
  auto calc_acceleration = [&](auto ix, auto iy, auto iz) {
    tat::mat3  J;
    auto const ixpos = ix == 511 ? ix : ix + 1;
    auto const ixneg = ix == 0 ? ix : ix - 1;
    auto const iypos = iy == 4095 ? iy : iy + 1;
    auto const iyneg = iy == 0 ? iy : iy - 1;
    auto const izpos = iz == 255 ? iz : iz + 1;
    auto const izneg = iz == 0 ? iz : iz - 1;

    auto const dx = full_domain.template dimension<0>()[ixpos] -
                    full_domain.template dimension<0>()[ixneg];
    auto const dy = full_domain.template dimension<1>()[iypos] -
                    full_domain.template dimension<1>()[iyneg];
    auto const dz = full_domain.template dimension<2>()[izpos] -
                    full_domain.template dimension<2>()[izneg];

    J.col(0) = tat::vec3{(velx(ixpos, iy, iz) - velx(ixneg, iy, iz)) / dx,
                         (vely(ixpos, iy, iz) - vely(ixneg, iy, iz)) / dx,
                         (velz(ixpos, iy, iz) - velz(ixneg, iy, iz)) / dx};
    J.col(1) = tat::vec3{(velx(ix, iypos, iz) - velx(ix, iyneg, iz)) / dy,
                         (vely(ix, iypos, iz) - vely(ix, iyneg, iz)) / dy,
                         (velz(ix, iypos, iz) - velz(ix, iyneg, iz)) / dy};
    J.col(2) = tat::vec3{(velx(ix, iy, izpos) - velx(ix, iy, izneg)) / dz,
                         (vely(ix, iy, izpos) - vely(ix, iy, izneg)) / dz,
                         (velz(ix, iy, izpos) - velz(ix, iy, izneg)) / dz};

    return J * tat::vec3{velx(ix, iy, iz), vely(ix, iy, iz), velz(ix, iy, iz)} +
           tat::vec3{
               temporal_diff_x(ix, iy, iz),
               temporal_diff_y(ix, iy, iz),
               temporal_diff_y(ix, iy, iz),
           };
  };
  return add_vec3_prop(threedpart_domain, channelflow_file, calc_acceleration,
                       "acceleration");
}
//------------------------------------------------------------------------------
auto add_temporal_derivative(auto const& full_domain,
                             auto const& threedpart_domain,
                             auto& channelflow_121_file, auto const& velx_121,
                             auto const& vely_121, auto const& velz_121,
                             auto& channelflow_122_file,
                             auto& channelflow_123_file, auto const& velx_123,
                             auto const& vely_123, auto const& velz_123) {
  auto arr_x = tat::dynamic_multidim_array<double>(512, 4096, 256);
  auto arr_y = tat::dynamic_multidim_array<double>(512, 4096, 256);
  auto arr_z = tat::dynamic_multidim_array<double>(512, 4096, 256);

  auto calc = [&](auto const ix, auto const iy, auto const iz) {
    auto const ixpos = ix == 511 ? ix : ix + 1;
    auto const ixneg = ix == 0 ? ix : ix - 1;
    auto const iypos = iy == 4095 ? iy : iy + 1;
    auto const iyneg = iy == 0 ? iy : iy - 1;
    auto const izpos = iz == 255 ? iz : iz + 1;
    auto const izneg = iz == 0 ? iz : iz - 1;

    auto const dt = 0.5;  // 1 / ms

    return tat::vec3{velx_123(ix, iy, iz) - velx_121(ix, iy, iz),
                     vely_123(ix, iy, iz) - vely_121(ix, iy, iz),
                     velz_123(ix, iy, iz) - velz_121(ix, iy, iz)} *
           dt;
  };

  add_vec3_prop(threedpart_domain, channelflow_122_file, calc, "dvdt");
}
//------------------------------------------------------------------------------
auto add_Q_steve(auto const& full_domain, auto const& threedpart_domain,
                 auto& channelflow_file, auto const& velx, auto const& vely,
                 auto const& velz, auto&& creator) -> auto& {
  auto calc_Q = [&](auto ix, auto iy, auto iz) {
    tat::mat3  J;
    auto const ixpos = ix == 511 ? ix : ix + 1;
    auto const ixneg = ix == 0 ? ix : ix - 1;
    auto const iypos = iy == 4095 ? iy : iy + 1;
    auto const iyneg = iy == 0 ? iy : iy - 1;
    auto const izpos = iz == 255 ? iz : iz + 1;
    auto const izneg = iz == 0 ? iz : iz - 1;

    auto const dx = 1 / (full_domain.template dimension<0>()[ixpos] -
                         full_domain.template dimension<0>()[ixneg]);
    auto const dy = 1 / (full_domain.template dimension<1>()[iypos] -
                         full_domain.template dimension<1>()[iyneg]);
    auto const dz = 1 / (full_domain.template dimension<2>()[izpos] -
                         full_domain.template dimension<2>()[izneg]);

    J.col(0) = tat::vec3{(velx(ixpos, iy, iz) - velx(ixneg, iy, iz)),
                         (vely(ixpos, iy, iz) - vely(ixneg, iy, iz)),
                         (velz(ixpos, iy, iz) - velz(ixneg, iy, iz))} *
               dx;
    J.col(1) = tat::vec3{(velx(ix, iypos, iz) - velx(ix, iyneg, iz)),
                         (vely(ix, iypos, iz) - vely(ix, iyneg, iz)),
                         (velz(ix, iypos, iz) - velz(ix, iyneg, iz))} *
               dy;
    J.col(2) = tat::vec3{(velx(ix, iy, izpos) - velx(ix, iy, izneg)),
                         (vely(ix, iy, izpos) - vely(ix, iy, izneg)),
                         (velz(ix, iy, izpos) - velz(ix, iy, izneg))} *
               dz;

    auto S     = (J + transposed(J)) / 2;
    auto Omega = (J - transposed(J)) / 2;
    return (sqr_norm(Omega) - sqr_norm(S)) / 2;
  };
  return add_scalar_prop(threedpart_domain, channelflow_file, calc_Q, "Q_steve");
}
//==============================================================================
auto main() -> int {
  // read full domain axes
  tat::hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5",
                             H5F_ACC_RDONLY};
  tat::hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5",
                             H5F_ACC_RDONLY};
  tat::hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5",
                             H5F_ACC_RDONLY};
  auto const      full_domain_z =
      axis0_file.group("CartGrid").dataset<double>("axis0").read_as_vector();
  auto const full_domain_y =
      axis1_file.group("CartGrid").dataset<double>("axis1").read_as_vector();
  auto const full_domain_x =
      axis2_file.group("CartGrid").dataset<double>("axis2").read_as_vector();
  tat::grid full_domain{full_domain_x, full_domain_y, full_domain_z};
  std::cerr << "full_domain:\n" << full_domain << '\n';

  // generate a small part from domain
  std::vector<double> small_domain_x(begin(full_domain_z),
                                     begin(full_domain_z) + 64);
  std::vector<double> small_domain_y(begin(full_domain_y),
                                     begin(full_domain_y) + 64);
  std::vector<double> small_domain_z(begin(full_domain_x),
                                     begin(full_domain_x) + 64);
  tat::grid small_domain{small_domain_x, full_domain_y, small_domain_z};
  std::cerr << "small_domain:\n" << small_domain << '\n';

  // generate the 3dpart domain
  std::vector<double> threedpart_domain_x(begin(full_domain_z),
                                          begin(full_domain_z) + 512);
  std::vector<double> threedpart_domain_z(begin(full_domain_x),
                                          begin(full_domain_x) + 256);
  tat::grid           threedpart_domain{threedpart_domain_x, full_domain_y,
                              threedpart_domain_z};
  std::cerr << "3dpart_domain:\n" << threedpart_domain << '\n';

  // generate the pod domain
  std::vector<double> pod_domain_y(begin(full_domain_y),
                                   begin(full_domain_y) + 1024);
  tat::grid pod_domain{threedpart_domain_x, pod_domain_y, threedpart_domain_z};
  std::cerr << "pod_domain:\n" << pod_domain << '\n';

  // open hdf5 files
  tat::hdf5::file channelflow_121_file{
      "/home/vcuser/channel_flow/dino_res_121000.h5", H5F_ACC_RDWR};
  tat::hdf5::file channelflow_122_file{
      "/home/vcuser/channel_flow/dino_res_122000.h5", H5F_ACC_RDWR};
  tat::hdf5::file channelflow_123_file{
      "/home/vcuser/channel_flow/dino_res_123000.h5", H5F_ACC_RDWR};
  tat::hdf5::file pod0_file{"/home/vcuser/channel_flow/pod_0.h5",
                            H5F_ACC_RDONLY};

  // create grid properties of pod
  auto& pod0_velx = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vx"));
  auto& pod0_vely = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vy"));
  auto& pod0_velz = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vz"));

  // create grid properties of 121000 time step
  auto& velx_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vx"), "Vx_121");
  auto& vely_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vy"), "Vy_121");
  auto& velz_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vz"), "Vz_121");
  // auto& Q_121 = threedpart_domain.add_lazy_vertex_property<double>(
  //    channelflow_121_file.group("variables").dataset<double>("Q"), "Q_121");
  // scalarfield Q_121_field{Q_121.sampler<tat::interpolation::linear>()};

  // create grid properties of 122000 time step
  auto& velx_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vx"), "Vx_122");
  auto& vely_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vy"), "Vy_122");
  auto& velz_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vz"), "Vz_122");
  auto& Q_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Q_steve"),
      "Q_122");
  auto& temporal_diff_x_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("dvdt_x"),
      "temporal_diff_x_122");
  auto& temporal_diff_y_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("dvdt_y"),
      "temporal_diff_y_122");
  auto& temporal_diff_z_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("dvdt_z"),
      "temporal_diff_z_122");
  auto& accx_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("acceleration_x"),
      "accx_122");
  auto& accy_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("acceleration_y"),
      "accy_122");
  auto& accz_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("acceleration_z"),
      "accz_122");
  scalarfield Q_122_field{Q_122.sampler<tat::interpolation::linear>()};
  scalarfield vely_122_field{vely_122.sampler<tat::interpolation::linear>()};
  vectorfield vel_122_field{velx_122.sampler<tat::interpolation::linear>(),
                            vely_122.sampler<tat::interpolation::linear>(),
                            velz_122.sampler<tat::interpolation::linear>()};

  // create grid properties of 123000 time step
  auto& velx_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vx"), "Vx_123");
  auto& vely_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vy"), "Vy_123");
  auto& velz_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vz"), "Vz_123");
  // auto& Q_123 = threedpart_domain.add_lazy_vertex_property<double>(
  //    channelflow_123_file.group("variables").dataset<double>("Q"), "Q_123");
  // scalarfield Q_123_field{Q_123.sampler<tat::interpolation::linear>()};

  //add_temporal_derivative(full_domain, threedpart_domain, channelflow_121_file,
  //                        velx_121, vely_121, velz_121, channelflow_122_file,
  //                        channelflow_123_file, velx_123, vely_123, velz_123);
  //add_acceleration(full_domain, threedpart_domain, channelflow_122_file,
  //                 velx_122, vely_122, velz_122, temporal_diff_x_122,
  //                 temporal_diff_y_122, temporal_diff_z_122);

  //auto Q_iso_mesh =
  //    isosurface([&](auto const ix, auto const iy, auto const iz,
  //                   auto const& [>p<]) { return Q_122(ix, iy, iz); },
  //               threedpart_domain, 1e2);
  // for (auto v : Q_iso_mesh.vertices()) {
  //  std::swap(Q_iso_mesh[v](2), Q_iso_mesh[v](0));
  //}
  //Q_iso_mesh.write_vtk("Q_pod.vtk");
  // isosurface(
  //    [&](auto ix, auto iy, auto iz, auto const & [>p<]) -> auto const& {
  //      return Q_122(ix, iy, iz);
  //    },
  //    threedpart_domain, 0)
  //    .write_vtk("Q_122_0.vtk");

  tat::color_scales::viridis color_scale;

  auto         boundingbox = threedpart_domain.bounding_box();
  size_t const width = 8000, height = 4000;
  // auto const eye = tat::vec3{-threedpart_domain.dimension<0>().back() * 2,
  //                           -threedpart_domain.dimension<1>().back() / 2,
  //                           threedpart_domain.dimension<2>().back() * 2};
  auto const eye =
      tat::vec3{0.6545555350748051, -0.09376604401454308, 0.4996597917002379};
  auto const lookat =
      tat::vec3{0.1398584798389628, 0.3012404329452348, 0.11518570840278948};
  auto const up =
      tat::vec3{-0.40265911328979515, 0.3319599250288133, 0.8530347276984336};
  tat::rendering::perspective_camera<double> cam{eye,   lookat, up,    60,
                                                 0.01, 1000,   width, height};
  auto alpha = [](auto const t) -> double {
    auto const min    = 0.0;
    auto const max    = 0.2;
    if (t < 0) {
      return min;
    } else if (t > 1) {
      return max + min;
    } else {
      return t * t * max + min;
    }
  };
  ;

  //write_png("direct_volume_channelflow_Q.png",
  //          tat::direct_volume_rendering(cam, boundingbox, Q_122_field, 0, 1e1,
  //                                       1e3, 0.0001, color_scale, alpha,
  //                                       tat::vec3::ones())
  //              .vertex_property<tat::vec3>("rendering"));

  //write_png("direct_volume_channelflow_velmag.png",
  //          tat::direct_volume_rendering(cam, boundingbox, length(vel_122_field),
  //                                       0, 20, 30, 0.0001, color_scale,
  //                                       alpha, tat::vec3::ones())
  //              .vertex_property<tat::vec3>("rendering"));
  //
  //write_png("direct_volume_channelflow_vely.png",
  //          tat::direct_volume_rendering(cam, boundingbox, vely_122_field,
  //                                       0, 20, 30, 0.0001, color_scale,
  //                                       alpha, tat::vec3::ones())
  //              .vertex_property<tat::vec3>("rendering"));

  auto       J_122_field  = diff(vel_122_field, 1e-7);
  write_vtk(tat::detail::calc_parallel_vectors<double>(
                [&](auto ix, auto iy, auto iz, auto const& /*p*/) {
                  return tat::vec3{velx_122(ix, iy, iz), vely_122(ix, iy, iz),
                                   velz_122(ix, iy, iz)};
                },
                [&](auto ix, auto iy, auto iz, auto const& /*p*/) {
                  return tat::vec3{accx_122(ix, iy, iz), accy_122(ix, iy, iz),
                                   accz_122(ix, iy, iz)};
                },
                threedpart_domain,
                [&](auto const& x) {
                  auto const eig = eigenvalues(J_122_field(x, 0));
                  return std::abs(eig(0).imag()) > 0 ||
                         std::abs(eig(1).imag()) > 0 ||
                         std::abs(eig(2).imag()) > 0;
                }),
            "pv_122.vtk");
}
