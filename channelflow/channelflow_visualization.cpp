#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/direct_volume_rendering.h>
#include <tatooine/direct_volume_iso.h>
#include <tatooine/isosurface.h>
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
template <typename Domain, typename File, typename Vx, typename Vy, typename Vz>
auto add_Q_cheng(Domain const& domain, File& channelflow_file, Vx const& velx,
                 Vy const& vely, Vz const& velz) {
  auto calc_Q = [&](auto ix, auto iy, auto iz) {
    tat::mat3  J;
    auto const ixpos = ix == domain.size(0) - 1 ? ix : ix + 1;
    auto const ixneg = ix == 0 ? ix : ix - 1;
    auto const iypos = iy == domain.size(1) - 1 ? iy : iy + 1;
    auto const iyneg = iy == 0 ? iy : iy - 1;
    auto const izpos = iz == domain.size(2) - 1 ? iz : iz + 1;
    auto const izneg = iz == 0 ? iz : iz - 1;

    auto const dx = 1 / (domain.template dimension<0>()[ixpos] -
                         domain.template dimension<0>()[ixneg]);
    auto const dy = 1 / (domain.template dimension<1>()[iypos] -
                         domain.template dimension<1>()[iyneg]);
    auto const dz = 1 / (domain.template dimension<2>()[izpos] -
                         domain.template dimension<2>()[izneg]);

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

    return 0.5 * (J(0, 0) + J(1, 1) + J(2, 2)) * (J(0, 0) + J(1, 1) + J(2, 2)) -
           0.5 * (J(0, 0) * J(0, 0) + J(1, 1) * J(1, 1) + J(2, 2) * J(2, 2)) -
           J(0, 1) * J(1, 0) - J(0, 2) * J(2, 0) - J(1, 2) * J(2, 1);
  };
  std::atomic_size_t cnt  = 0;
  size_t const       max  = domain.num_vertices();
  bool               stop = false;
  tat::dynamic_multidim_array<double> Q_data{domain.size(0) - 1, domain.size(1),
                                             domain.size(2)};
  std::thread            watcher{[&cnt, &stop, max] {
    while (cnt < max && !stop) {
      std::cerr << std::setprecision(2) << double(cnt) / max * 100
                << "%                \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
  }};
  domain.parallel_loop_over_vertex_indices(
      [&](auto const ix, auto const iy, auto const iz) {
        if (ix < domain.size(0) - 2) {
          Q_data(ix, iy, iz) = calc_Q(ix, iy, iz);
        }
        ++cnt;
      });
  stop = true;
  watcher.join();
  auto Q = channelflow_file.template add_dataset<double>(
      "Q_cheng", domain.size(0) - 1, domain.size(1), domain.size(2));
  Q.write(Q_data);
}
//------------------------------------------------------------------------------
template <typename DomainGrid, typename Axis0, typename Axis1, typename Axis2,
          typename ScalarField, typename QField, typename POD0VelY,
          typename Vel122Y>
auto calc_iso_surface(DomainGrid const& domain_grid, Axis0 const& axis0,
                      Axis1 const& axis1, Axis2 const& axis2,
                      ScalarField const& s, QField const& Q,
                      POD0VelY const& vely_pod0, Vel122Y const& vely_122_threedpart,
                      std::string const& pathout) {
  auto const Q_sampler = Q.linear_sampler();
  auto const vely_122_sampler = vely_122_threedpart.linear_sampler();
  auto const vely_pod0_sampler = vely_pod0.linear_sampler();

  auto isosurface = tat::isosurface(
      [&](auto const vix, auto const viy, auto const viz, auto const& /*p*/) {
        return s(vix, viy, viz);
      },
      domain_grid, 5e6);
  auto& Q_prop = isosurface.template add_vertex_property<double>("Q");
  auto& vely_pod0_prop =
      isosurface.template add_vertex_property<double>("vely_pod0");
  auto& vely_122_prop = isosurface.template add_vertex_property<double>("vely_122");
  for (auto v : isosurface.vertices()) {
    Q_prop[v]         = Q_sampler(isosurface[v]);
    if (vely_pod0.grid().is_inside(isosurface[v])) {
      vely_pod0_prop[v] = vely_pod0_sampler(isosurface[v]);
    } else {
      vely_pod0_prop[v] = 0.0 / 0.0;
    }
    vely_122_prop[v]  = vely_122_sampler(isosurface[v]);
  }

  std::cout << "writing...\n";
  isosurface.write_vtk(pathout);
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
      }
    }
  }
  std::cout << "writing...\n";
  write_vtk(vortex_core_lines, pathout);
}
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
  full_domain.set_chunk_size_for_lazy_properties(2);
  std::cerr << "full_domain:\n" << full_domain << '\n';

  auto axis0_Q = axis0;
  axis0_Q.pop_back();
  tat::grid full_domain_Q{axis0_Q, axis1, axis2};
  full_domain_Q.set_chunk_size_for_lazy_properties(2);
  std::cerr << "full_domain_Q:\n" << full_domain_Q << '\n';


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
  tat::hdf5::file channelflow_121_threedpart_file{
      "/home/vcuser/channel_flow/dino_res_121000.h5"};
  tat::hdf5::file channelflow_122_threedpart_file{
      "/home/vcuser/channel_flow/dino_res_122000.h5"};
  tat::hdf5::file channelflow_122_full_file{
      "/home/vcuser/channel_flow/dino_res_122000_full.h5"};
  tat::hdf5::file channelflow_123_threedpart_file{
      "/home/vcuser/channel_flow/dino_res_123000.h5"};
  tat::hdf5::file pod0_file{"/home/vcuser/channel_flow/pod_0.h5"};

  // create grid properties of pod
  //auto& velx_pod0 = pod0_domain.add_lazy_vertex_property(
  //    pod0_file.dataset<double>("variables/Vx"));
  //auto& vely_pod0 = pod0_domain.add_lazy_vertex_property(
  //    pod0_file.dataset<double>("variables/Vy"));
  //auto& velz_pod0 = pod0_domain.add_lazy_vertex_property(
  //    pod0_file.dataset<double>("variables/Vz"));
  //auto& pod0_Q = pod0_domain.add_lazy_vertex_property(
  //    pod0_file.dataset<double>("variables/Q"));

   // create grid properties of 121000 time step
   //auto& velx_121_threedpart = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_121_threedpart_file.dataset<double>("variables/Vx"), "Vx_121");
   //auto& vely_121 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_121_threedpart_file.dataset<double>("variables/Vy"), "Vy_121");
   //auto& velz_121_threedpart = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_121_threedpart_file.dataset<double>("variables/Vz"), "Vz_121");
   // auto& Q_121 = threedpart_domain.add_lazy_vertex_property(
   //    channelflow_121_threedpart_file.dataset<double>("variables/Q"), "Q_121");
   // scalarfield Q_121_field{Q_121.sampler<tat::interpolation::linear>()};

   //auto ds_velx_122_threedpart =
   //    channelflow_122_threedpart_file.dataset<double>("variables/Vx");
   //auto& velx_122_threedpart = threedpart_domain.add_lazy_vertex_property(
   //    ds_velx_122_threedpart, "Vx_122");
   //velx_122_threedpart.limit_num_chunks_loaded();
   //auto ds_vely_122_threedpart =
   //    channelflow_122_threedpart_file.dataset<double>("variables/Vy");
   //auto& vely_122_threedpart = threedpart_domain.add_lazy_vertex_property(
   //    ds_vely_122_threedpart, "Vy_122");
   //vely_122_threedpart.limit_num_chunks_loaded();
   //auto ds_velz_122_threedpart =
   //    channelflow_122_threedpart_file.dataset<double>("variables/Vz");
   //auto& velz_122_threedpart = threedpart_domain.add_lazy_vertex_property(
   //    ds_velz_122_threedpart, "Vz_122");
   //velz_122_threedpart.limit_num_chunks_loaded();
   //auto velx_122_threedpart_dataset =
   //    channelflow_122_threedpart_file.dataset<double>("variables/Vx");

   //auto ds_velx_122_full =
   //    channelflow_122_full_file.dataset<double>("velocity/xvel");
   //auto& velx_122_full = full_domain.add_lazy_vertex_property<tat::x_fastest>(
   //    ds_velx_122_full, "Vx_122");
   //velx_122_full.limit_num_chunks_loaded();
   //velx_122_full.set_max_num_chunks_loaded(128);
   //
   //auto ds_vely_122_full =
   //    channelflow_122_full_file.dataset<double>("velocity/yvel");
   //auto& vely_122_full = full_domain.add_lazy_vertex_property<tat::x_fastest>(
   //    ds_vely_122_full, "Vy_122");
   //vely_122_full.limit_num_chunks_loaded();
   //vely_122_full.set_max_num_chunks_loaded(128);
   //
   //auto ds_velz_122_full =
   //    channelflow_122_full_file.dataset<double>("velocity/zvel");
   //auto& velz_122_full = full_domain.add_lazy_vertex_property<tat::x_fastest>(
   //    ds_velz_122_full, "Vz_122");
   //velz_122_full.limit_num_chunks_loaded();
   //velz_122_full.set_max_num_chunks_loaded(128);

  // if (ds_velx_122_threedpart.read(19, 20, 1) != ds_velx_122_full.read(19, 20,
  // 1) ||
  //    ds_velx_122_threedpart.read(511, 4095, 255) !=
  //    ds_velx_122_full.read(511, 4095, 255)) {
  //  std::stringstream ss;
  //  ss << "Velocities do not match: \n  "
  //     << ds_velx_122_threedpart.read(19, 20, 1)
  //     << "\n  " << ds_velx_122_full.read(19, 20, 1) << "\n  "
  //     << ds_velx_122_full.read(1, 20, 19) << "\n\n  "
  //     << ds_velx_122_threedpart.read(511, 4095, 255) << "\n  "
  //     << ds_velx_122_full.read(511, 4095, 255) << "\n  "
  //     << ds_velx_122_full.read(255, 4095, 511)
  //     << "\n";
  //  throw std::runtime_error{ss.str()};
  //} else {
  //  std::cout << "it works!\n";
  //}
  // if (velx_122_threedpart(19, 20, 1) != velx_122_full(19, 20, 1) ||
  //    velx_122_threedpart(511, 4095, 255) != velx_122_full(511, 4095, 255)) {
  //  std::stringstream ss;
  //  ss << "Velocities do not match: \n  "
  //     << velx_122_threedpart(19, 20, 1)
  //     << "\n  " << velx_122_full(19, 20, 1) << "\n  "
  //     << velx_122_full(1, 20, 19) << "\n\n  "
  //     << velx_122_threedpart(511, 4095, 255) << "\n  "
  //     << velx_122_full(511, 4095, 255) << "\n  "
  //     << velx_122_full(255, 4095, 511)
  //     << "\n";
  //  throw std::runtime_error{ss.str()};
  //} else {
  //  std::cout << "it works!\n";
  //}
  auto velx_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/xvel");
  auto vely_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/yvel");
  auto velz_122_full_dataset =
      channelflow_122_full_file.dataset<double>("velocity/zvel");
  auto Q_122_full_dataset =
      channelflow_122_full_file.dataset<double>("Q_cheng");

  auto& velx_122_full = full_domain.add_lazy_vertex_property<tat::x_slowest>(
      velx_122_full_dataset, "Vx_122");
  auto& vely_122_full = full_domain.add_lazy_vertex_property<tat::x_slowest>(
      vely_122_full_dataset, "Vy_122");
  auto& velz_122_full = full_domain.add_lazy_vertex_property<tat::x_slowest>(
      velz_122_full_dataset, "Vz_122");
  auto& Q_122_full =
      full_domain_Q.add_lazy_vertex_property(Q_122_full_dataset, "Q_122");
  velx_122_full.limit_num_chunks_loaded();
  velx_122_full.set_max_num_chunks_loaded(4);
  vely_122_full.limit_num_chunks_loaded();
  vely_122_full.set_max_num_chunks_loaded(4);
  velz_122_full.limit_num_chunks_loaded();
  velz_122_full.set_max_num_chunks_loaded(4);
  Q_122_full.limit_num_chunks_loaded();
  Q_122_full.set_max_num_chunks_loaded(4);

  auto velx_122_full_sampler = velx_122_full.linear_sampler();
  auto vely_122_full_sampler = vely_122_full.linear_sampler();
  auto velz_122_full_sampler = velz_122_full.linear_sampler();
  auto Q_122_full_sampler    = Q_122_full.linear_sampler();

  // create grid properties of 122000 time step

  // auto velx_122_sampler = velx_122_threedpart.linear_sampler();
  // auto vely_122_sampler = vely_122_threedpart.linear_sampler();
  // auto velz_122_sampler = velz_122_threedpart.linear_sampler();
  // auto const diff_velx_122 = tat::diff(velx_122_threedpart);
  // static_assert(tat::is_vec<decltype(diff_velx_122)::value_type>);
  // auto const diff_velx_122_sampler = diff_velx_122.sampler();
  // auto const diff_vely_122         = diff(vely_122_threedpart);
  // auto const diff_vely_122_sampler = diff_vely_122.sampler();
  // auto const diff_velz_122         = diff(velz_122_threedpart);
  // auto const diff_velz_122_sampler = diff_velz_122.sampler();

  // create grid properties of 123000 time step
  // auto& velx_123_threedpart = threedpart_domain.add_lazy_vertex_property(
  //    channelflow_123_threedpart_file.dataset<double>("variables/Vx"),
  //    "Vx_123");
  // auto& vely_123_threedpart = threedpart_domain.add_lazy_vertex_property(
  //    channelflow_123_threedpart_file.dataset<double>("variables/Vy"),
  //    "Vy_123");
  // auto& velz_123_threedpart = threedpart_domain.add_lazy_vertex_property(
  //    channelflow_123_threedpart_file.dataset<double>("variables/Vz"),
  //    "Vz_123");
  // auto& Q_123 = threedpart_domain.add_lazy_vertex_property<double>(
  //    channelflow_123_threedpart_file.dataset<double>("variables/Q"),
  //    "Q_123");
  // scalarfield Q_123_field0Q_123.sampler<tat::interpolation::linear>()};

  // add_temporal_derivative(full_domain, threedpart_domain,
  // channelflow_121_threedpart_file,
  //                        velx_121_threedpart, vely_121_threedpart,
  //                        velz_121_threedpart,
  //                        channelflow_122_threedpart_file,
  //                        channelflow_123_threedpart_file,
  //                        velx_123_threedpart, vely_123_threedpart,
  //                        velz_123_threedpart);
  // add_acceleration(full_domain, threedpart_domain,
  // channelflow_122_threedpart_file,
  //                 velx_122_threedpart, vely_122, velz_122_threedpart,
  //                 temporal_diff_x_122, temporal_diff_y_122,
  //                 temporal_diff_z_122);

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

  auto         pod0_boundingbox       = pod0_domain.bounding_box();
  auto         threedpart_boundingbox = threedpart_domain.bounding_box();
  size_t const width = 10000, height = 5000;

  auto const full_domain_eye =
      tat::vec3{0.7940901239835871, 0.04097490152128994, 0.5004262802265552};
  auto const full_domain_lookat =
      tat::vec3{-0.7384532106212904, 0.7745404345929863, -0.4576538576946477};
  auto const full_domain_up =
      tat::vec3{-0.35221800146747856, 0.3807796045093859, 0.8549557720911246};
  tat::rendering::perspective_camera<double> full_domain_cam{full_domain_eye,
                                                             full_domain_lookat,
                                                             full_domain_up,
                                                             60,
                                                             0.01,
                                                             1000,
                                                             width,
                                                             height};

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

  auto vel_122_field = vectorfield{velx_122_full_sampler, vely_122_full_sampler,
                                   velz_122_full_sampler};
  // write_png("direct_volume_channelflow_velmag_pod0_domain.png",
  //         tat::direct_volume_rendering(pod0_cam, pod0_boundingbox,
  //         length(vel_122_field),
  //                                      0, 20, 30, 0.0001, color_scale,
  //                                      alpha, tat::vec3::ones())
  //             .vertex_property<tat::vec3>("rendering"));

  // write_png("direct_volume_channelflow_velmag.png",
  //          tat::direct_volume_rendering(threedpart_cam,
  //          threedpart_boundingbox, length(vel_122_field),
  //                                       0, 20, 30, 0.0001, color_scale,
  //                                       alpha, tat::vec3::ones())
  //              .vertex_property<tat::vec3>("rendering"));

  // auto vely_122_field   = scalarfield{vely_122_sampler};
  // write_png("direct_volume_channelflow_vely.png",
  //          tat::direct_volume_rendering(threedpart_cam,
  //          threedpart_boundingbox, vely_122_field,
  //                                       0, 20, 30, 0.0001, color_scale,
  //                                       alpha, tat::vec3::ones())
  //              .vertex_property<tat::vec3>("rendering"));

  // std::cout << "adding Q...\n";
  // add_Q_cheng(full_domain, channelflow_122_full_file, velx_122_full,
  //            vely_122_full, velz_122_full);

  //
  std::cerr << "calculating iso surface Q...\n";
  write_png("direct_volume_iso_channelflow_Q_with_velmag.png",
            tat::direct_volume_iso(
                threedpart_cam, threedpart_domain.bounding_box(),
                // pod0_cam, pod0_domain.bounding_box(),
                [&](auto const& x) -> decltype(auto) {
                  return Q_122_full_sampler(x);
                },
                5e6,
                [&](auto const& x) -> decltype(auto) {
                  // return Q_122_full_sampler(x);
                  return length(tat::vec3{velx_122_full_sampler(x),
                                          vely_122_full_sampler(x),
                                          velz_122_full_sampler(x)});
                },
                13, 27, [](auto const&) { return true; },
                (axis0[1] - axis0[0]) * 0.5, color_scale,
                [](auto const t) -> double { return 1; }, tat::vec3{1, 1, 1})
                .vertex_property<tat::vec3>("rendering"));

  // calc_pv(pod0_domain, axis0, axis1, axis2, velx_122_threedpart,
  // vely_122_threedpart, velz_122_threedpart,
  //        Q_122, vely_pod0, vely_122_threedpart, "vortex_core_lines_122.vtk");
  // calc_pv(pod0_domain, axis0, axis1, axis2, velx_pod0, vely_pod0, velz_pod0,
  //        pod0_Q, vely_pod0, vely_122_threedpart,
  //        "vortex_core_lines_pod.vtk");
}
