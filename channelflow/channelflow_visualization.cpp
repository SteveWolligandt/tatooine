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
  scalarfield(Sampler sampler) : m_sampler{sampler} {}
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
// template <typename Sampler>
// scalarfield(Sampler) -> scalarfield<std::decay_t<Sampler>>;
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
//------------------------------------------------------------------------------
template <typename FullDomain, typename ThreeDPartDomain,
          typename File, typename Vx, typename Vy, typename Vz>
auto add_Q_steve(FullDomain const& full_domain,
                 ThreeDPartDomain const& threedpart_domain,
                 File& channelflow_file,
                 Vx const& velx,
                 Vy const& vely,
                 Vz const& velz) -> auto& {
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
//------------------------------------------------------------------------------
template <typename DomainGrid, typename Axis0, typename Axis1, typename Axis2,
          typename VelX, typename VelY, typename VelZ, typename QField,
          typename POD0VelY, typename Vel122Y>
auto calc_pv(DomainGrid const& domain_grid, Axis0 const& axis0,
             Axis1 const& axis1, Axis2 const& axis2, VelX const& velx,
             VelY const& vely, VelZ const& velz, QField const& Q,
             POD0VelY const& vely_pod0, Vel122Y const& vely_122,
             std::string const& pathout) {
  auto       velx_sampler = velx.linear_sampler();
  auto       vely_sampler = vely.linear_sampler();
  auto       velz_sampler = velz.linear_sampler();
  auto       vely_field   = scalarfield{vely_sampler};
  auto       vel_field = vectorfield{velx_sampler, vely_sampler, velz_sampler};
  auto const diff_velx = diff(velx);
  auto const diff_velx_sampler = diff_velx.sampler();
  auto const diff_vely         = diff(vely);
  auto const diff_vely_sampler = diff_vely.sampler();
  auto const diff_velz         = diff(velz);
  auto const diff_velz_sampler = diff_velz.sampler();

  auto const Q_sampler = Q.linear_sampler();
  auto const vely_122_sampler = vely_122.linear_sampler();
  auto const vely_pod0_sampler = vely_pod0.linear_sampler();

  constexpr std::array<size_t, 3>   partial_size{512, 16, 16};
  std::vector<tat::line<double, 3>> vortex_core_lines;
  std::array<size_t, 3>             offset{0, 0, 0};
   std::array<size_t, 3>           counts{
      (size_t)std::ceil(double(domain_grid.size(0) - 1) /
                        (partial_size[0] - 1)),
      (size_t)std::ceil(double(domain_grid.size(1) - 1) /
                        (partial_size[1] - 1)),
      (size_t)std::ceil(double(domain_grid.size(2) - 1) /
                        (partial_size[2] - 1))};
  std::cout << "calulating...\n";
  std::cout << "counts = [" << counts[0] << ", " << counts[1] << ", "
            << counts[2] << "]\n";
    for (size_t iy = 0; iy < counts[1]; ++iy) {
  for (size_t iz = 0; iz < counts[2]; ++iz) {
      for (size_t ix = 0; ix < counts[0]; ++ix) {
        offset[0] = ix * (partial_size[0] - 1);
        offset[2] = iz * (partial_size[2] - 1);
        offset[1] = iy * (partial_size[1] - 1);
        std::array<size_t, 3> cur_size{
            std::min(partial_size[0], domain_grid.size(0) - offset[0]),
            std::min(partial_size[1], domain_grid.size(1) - offset[1]),
            std::min(partial_size[2], domain_grid.size(2) - offset[2])};
        std::vector<double> cur_domain_x(
            begin(axis0) + offset[0],
            begin(axis0) + offset[0] + cur_size[0]);
        std::vector<double> cur_domain_y(
            begin(axis1) + offset[1],
            begin(axis1) + offset[1] + cur_size[1]);
        std::vector<double> cur_domain_z(
            begin(axis2) + offset[2],
            begin(axis2) + offset[2] + cur_size[2]);
        tat::grid cur_domain{cur_domain_x, cur_domain_y, cur_domain_z};

        std::cout << "indices = [" << ix << ", " << iy << ", " << iz << "]\n";
        std::cout << "offset = [" << offset[0] << ", " << offset[1] << ", "
                  << offset[2] << "]\n";
        std::cout << "cur_size = [" << cur_size[0] << ", " << cur_size[1] << ", "
                  << cur_size[2] << "]\n";
        std::cout << "cur_domain:\n" << cur_domain;
        auto new_vortex_core_lines = tat::detail::calc_parallel_vectors<double>(
            [&](auto vix, auto viy, auto viz, auto const& /*p*/) {
              vix += offset[0];
              viy += offset[1];
              viz += offset[2];
              return tat::vec3{velx(vix, viy, viz),
                               vely(vix, viy, viz),
                               velz(vix, viy, viz)};
            },
            [&](auto wix, auto wiy, auto wiz, auto const& /*p*/) {
              wix += offset[0];
              wiy += offset[1];
              wiz += offset[2];
              auto const vel =
                  tat::vec3{velx(wix, wiy, wiz),
                            vely(wix, wiy, wiz),
                            velz(wix, wiy, wiz)};
              return tat::vec3{dot(diff_velx(wix, wiy, wiz), vel),
                               dot(diff_vely(wix, wiy, wiz), vel),
                               dot(diff_velz(wix, wiy, wiz), vel)};
            },
            cur_domain
            //, [&](auto const& x) {
            //  auto J         = tat::mat3::zeros();
            //  J.row(0)       = diff_velx_sampler(x);
            //  J.row(1)       = diff_vely_sampler(x);
            //  J.row(2)       = diff_velz_sampler(x);
            //  auto const eig = eigenvalues(J);
            //  return std::abs(eig(0).imag()) > 0 ||
            //         std::abs(eig(1).imag()) > 0 ||
            //         std::abs(eig(2).imag()) > 0;
            //}
        );
        std::cout << size(vortex_core_lines) << '\n';
        std::cout << size(new_vortex_core_lines) << '\n';
        for (auto& core : new_vortex_core_lines) {
          using core_t        = std::decay_t<decltype(core)>;
          using vertex_handle = typename core_t::vertex_idx;
          auto& line_Q        = core.template add_vertex_property<double>("Q");
          auto& line_vely_pod0 =
              core.template add_vertex_property<double>("vely_pod0");
          auto& line_vely_122 =
              core.template add_vertex_property<double>("vely_122");
          for (size_t i = 0; i < core.num_vertices(); ++i) {
            vertex_handle v{i};
            line_Q[v]         = Q_sampler(core[v]);
            line_vely_pod0[v] = vely_pod0_sampler(core[v]);
            line_vely_122[v]  = vely_122_sampler(core[v]);
          }
        }
        std::move(begin(new_vortex_core_lines), end(new_vortex_core_lines),
                  std::back_inserter(vortex_core_lines));
        std::cout << size(vortex_core_lines) << '\n';
        std::cout << "writing...\n";
        write_vtk(vortex_core_lines, pathout);
      }
    }
  }
}
//==============================================================================
auto main() -> int {
  // read full domain axes
  tat::hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5"};
  tat::hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5"};
  tat::hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5"};
  auto const      axis0 =
      axis0_file/*.group("CartGrid")*/.dataset<double>("CartGrid/axis0").read_as_vector();
  auto const axis1 =
      axis1_file/*.group("CartGrid")*/.dataset<double>("CartGrid/axis1").read_as_vector();
  auto const axis2 =
      axis2_file/*.group("CartGrid")*/.dataset<double>("CartGrid/axis2").read_as_vector();
  tat::grid full_domain{axis2, axis1, axis0};
  std::cerr << "full_domain:\n" << full_domain << '\n';


  // generate the 3dpart domain
  std::vector<double> threedpart_domain_x(begin(axis0),
                                          begin(axis0) + 512);
  std::vector<double> threedpart_domain_z(begin(axis2),
                                          begin(axis2) + 256);
  tat::grid           threedpart_domain{threedpart_domain_x, axis1,
                              threedpart_domain_z};
  std::cerr << "3dpart_domain:\n" << threedpart_domain << '\n';

  // generate the pod domain
  std::vector<double> pod0_domain_y(begin(axis1),
                                   begin(axis1) + 1024);
  tat::grid pod0_domain{threedpart_domain_x, pod0_domain_y, threedpart_domain_z};
  std::cerr << "pod0_domain:\n" << pod0_domain << '\n';

  // open hdf5 files
  tat::hdf5::file channelflow_121_file{
      "/home/vcuser/channel_flow/dino_res_121000.h5"};
  tat::hdf5::file channelflow_122_file{
      "/home/vcuser/channel_flow/dino_res_122000.h5"};
  tat::hdf5::file channelflow_123_file{
      "/home/vcuser/channel_flow/dino_res_123000.h5"};
  tat::hdf5::file pod0_file{"/home/vcuser/channel_flow/pod_0.h5"};

  // create grid properties of pod
  auto& velx_pod0 = pod0_domain.add_lazy_vertex_property(
      pod0_file/*.group("variables")*/.dataset<double>("variables/Vx"));
  auto& vely_pod0 = pod0_domain.add_lazy_vertex_property(
      pod0_file/*.group("variables")*/.dataset<double>("variables/Vy"));
  auto& velz_pod0 = pod0_domain.add_lazy_vertex_property(
      pod0_file/*.group("variables")*/.dataset<double>("variables/Vz"));
  auto& pod0_Q = pod0_domain.add_lazy_vertex_property(
      pod0_file/*.group("variables")*/.dataset<double>("variables/Q"));

   // create grid properties of 121000 time step
   auto& velx_121 = threedpart_domain.add_lazy_vertex_property(
       channelflow_121_file/*.group("variables")*/.dataset<double>("variables/Vx"), "Vx_121");
   auto& vely_121 = threedpart_domain.add_lazy_vertex_property(
       channelflow_121_file/*.group("variables")*/.dataset<double>("variables/Vy"), "Vy_121");
   auto& velz_121 = threedpart_domain.add_lazy_vertex_property(
       channelflow_121_file/*.group("variables")*/.dataset<double>("variables/Vz"), "Vz_121");
   // auto& Q_121 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_121_file.group("variables").dataset<double>("Q"), "Q_121");
   // scalarfield Q_121_field{Q_121.sampler<tat::interpolation::linear>()};

   // create grid properties of 122000 time step
   auto& velx_122 = threedpart_domain.add_lazy_vertex_property(
       channelflow_122_file/*.group("variables")*/.dataset<double>("variables/Vx"), "Vx_122");
   auto& vely_122 = threedpart_domain.add_lazy_vertex_property(
       channelflow_122_file/*.group("variables")*/.dataset<double>("variables/Vy"), "Vy_122");
   auto& velz_122 = threedpart_domain.add_lazy_vertex_property(
       channelflow_122_file/*.group("variables")*/.dataset<double>("variables/Vz"), "Vz_122");

   auto velx_122_sampler = velx_122.linear_sampler();
   auto vely_122_sampler = vely_122.linear_sampler();
   auto velz_122_sampler = velz_122.linear_sampler();
   auto vely_122_field   = scalarfield{vely_122_sampler};
   auto vel_122_field =
       vectorfield{velx_122_sampler, vely_122_sampler, velz_122_sampler};
   auto const diff_velx_122 = diff(velx_122);
   static_assert(tat::is_vec<decltype(diff_velx_122)::value_type>);
   auto const diff_velx_122_sampler = diff_velx_122.sampler();
   auto const diff_vely_122         = diff(vely_122);
   auto const diff_vely_122_sampler = diff_vely_122.sampler();
   auto const diff_velz_122         = diff(velz_122);
   auto const diff_velz_122_sampler = diff_velz_122.sampler();

   auto& Q_122 = threedpart_domain.add_lazy_vertex_property(
       channelflow_122_file/*.group("variables")*/.dataset<double>("variables/Q"), "Q_122");
   auto Q_122_sampler = Q_122.linear_sampler();
   auto Q_122_field   = scalarfield{Q_122_sampler};

   // create grid properties of 123000 time step
   // auto& velx_123 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_123_file.group("variables").dataset<double>("Vx"),
   //    "Vx_123");
   // auto& vely_123 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_123_file.group("variables").dataset<double>("Vy"),
   //    "Vy_123");
   // auto& velz_123 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_123_file.group("variables").dataset<double>("Vz"),
   //    "Vz_123");
   // auto& Q_123 = threedpart_domain.add_lazy_vertex_property<double>(
   //    channelflow_123_file.group("variables").dataset<double>("Q"), "Q_123");
   // scalarfield Q_123_field{Q_123.sampler<tat::interpolation::linear>()};

   // add_temporal_derivative(full_domain, threedpart_domain,
   // channelflow_121_file,
   //                        velx_121, vely_121, velz_121, channelflow_122_file,
   //                        channelflow_123_file, velx_123, vely_123,
   //                        velz_123);
   // add_acceleration(full_domain, threedpart_domain, channelflow_122_file,
   //                 velx_122, vely_122, velz_122, temporal_diff_x_122,
   //                 temporal_diff_y_122, temporal_diff_z_122);

   // auto Q_iso_mesh =
   //    isosurface([&](auto const ix, auto const iy, auto const iz,
   //                   auto const& [>p<]) { return Q_122(ix, iy, iz); },
   //               threedpart_domain, 1e2);
   // for (auto v : Q_iso_mesh.vertices()) {
   //  std::swap(Q_iso_mesh[v](2), Q_iso_mesh[v](0));
   //}
   // Q_iso_mesh.write_vtk("Q_pod.vtk");
   // isosurface(
   //    [&](auto ix, auto iy, auto iz, auto const & [>p<]) -> auto const& {
   //      return Q_122(ix, iy, iz);
   //    },
   //    threedpart_domain, 0)
   //    .write_vtk("Q_122_0.vtk");

   tat::color_scales::viridis color_scale;

   auto         pod0_boundingbox        = pod0_domain.bounding_box();
   auto         threedpart_boundingbox = threedpart_domain.bounding_box();
   size_t const width = 1000, height = 500;
   // auto const eye = tat::vec3{-threedpart_domain.dimension<0>().back() * 2,
   //                           -threedpart_domain.dimension<1>().back() / 2,
   //                           threedpart_domain.dimension<2>().back() * 2};
   auto const threedpart_eye =
       tat::vec3{0.6545555350748051, -0.09376604401454308, 0.4996597917002379};
   auto const threedpart_lookat =
       tat::vec3{0.1398584798389628, 0.3012404329452348, 0.11518570840278948};
   auto const threedpart_up =
       tat::vec3{-0.40265911328979515, 0.3319599250288133, 0.8530347276984336};
   tat::rendering::perspective_camera<double> threedpart_cam{
       threedpart_eye, threedpart_lookat, threedpart_up, 60, 0.01, 1000, width,
       height};
   // auto const pod0_eye = tat::vec3{0.17436402903670775, -0.029368613711112865,
   //                               0.11376422220314832};
   auto const pod0_eye = tat::vec3{1, -1, 1} * 0.1;
   // auto const pod0_lookat =
   //    tat::vec3{0.03328671241261789, 0.0723694723172821,
   //    0.033031680721043566};
   auto const pod0_lookat = pod0_domain.center();
   // auto const pod0_up =
   //    tat::vec3{-0.35434985513228934, 0.2282347045784469,
   //    0.9068324540915563};
   auto const                                 pod0_up = tat::vec3{0, 0, 1};
   tat::rendering::perspective_camera<double> pod0_cam{
       pod0_eye, pod0_lookat, pod0_up, 60, 0.01, 1000, width, height};
   auto alpha = [](auto const t) -> double {
     auto const min = 0.0;
     auto const max = 0.2;
     if (t < 0) {
       return min;
     } else if (t > 1) {
       return max + min;
     } else {
       return t * t * (max - min) + min;
     }
   };
   ;

   // write_png("direct_volume_channelflow_pod0_y.png",
   //          tat::direct_volume_rendering(
   //              pod0_cam, pod0_boundingbox, vely_pod0_field, 0,
   //              15, 20, 0.0001, color_scale, alpha, tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_Q_pod0_combined.png",
   //          tat::direct_volume_rendering(
   //              pod0_cam, pod0_boundingbox, Q_122_field * vely_pod0_field, 0,
   //              15e2, 60e3, 0.0001, color_scale, alpha, tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_Q_pod0_domain.png",
   //          tat::direct_volume_rendering(pod0_cam, pod0_boundingbox,
   //          Q_122_field, 0, 1e2,
   //                                       3e3, 0.0001, color_scale, alpha,
   //                                       tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_Q.png",
   //          tat::direct_volume_rendering(threedpart_cam,
   //          threedpart_boundingbox, Q_122_field, 0, 1e1,
   //                                       1e3, 0.0001, color_scale, alpha,
   //                                       tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_velmag_pod0_domain.png",
   //          tat::direct_volume_rendering(pod0_cam, pod0_boundingbox,
   //          length(vel_122_field),
   //                                       0, 20, 30, 0.0001, color_scale,
   //                                       alpha, tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_velmag.png",
   //          tat::direct_volume_rendering(threedpart_cam,
   //          threedpart_boundingbox, length(vel_122_field),
   //                                       0, 20, 30, 0.0001, color_scale,
   //                                       alpha, tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // write_png("direct_volume_channelflow_vely.png",
   //          tat::direct_volume_rendering(threedpart_cam,
   //          threedpart_boundingbox, vely_122_field,
   //                                       0, 20, 30, 0.0001, color_scale,
   //                                       alpha, tat::vec3::ones())
   //              .vertex_property<tat::vec3>("rendering"));

   // std::array<size_t, 3>           counts{1, 2, 2};
   calc_pv(pod0_domain, axis0, axis1, axis2, velx_122, vely_122, velz_122,
           Q_122, vely_pod0, vely_122, "vortex_core_lines_122.vtk");
   calc_pv(pod0_domain, axis0, axis1, axis2, velx_pod0, vely_pod0, velz_pod0,
           pod0_Q, vely_pod0, vely_122, "vortex_core_lines_pod.vtk");
}
