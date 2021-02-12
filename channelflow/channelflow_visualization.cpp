#include <tatooine/grid.h>
#include <tatooine/hdf5.h>
#include <tatooine/isosurface.h>
#include <tatooine/field.h>
//==============================================================================
namespace tat = tatooine;
template <typename Sampler>
struct scalarfield : tat::scalarfield<scalarfield<Sampler>, double, 3> {
  using this_t = scalarfield<Sampler>;
  using parent_t = tat::scalarfield<this_t, double, 3>;

  using typename parent_t::real_t;
  using typename parent_t::pos_t;
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
//==============================================================================
template <typename Sampler>
scalarfield(Sampler) -> scalarfield<std::decay_t<Sampler>>;
//==============================================================================
auto main() -> int {
  // read full domain axes
  tat::hdf5::file axis0_file{"/home/vcuser/channel_flow/axis0.h5", H5F_ACC_RDONLY};
  tat::hdf5::file axis1_file{"/home/vcuser/channel_flow/axis1.h5", H5F_ACC_RDONLY};
  tat::hdf5::file axis2_file{"/home/vcuser/channel_flow/axis2.h5", H5F_ACC_RDONLY};
  auto const full_domain_z =
      axis0_file.group("CartGrid").dataset<double>("axis0").read_as_vector();
  auto const full_domain_y =
      axis1_file.group("CartGrid").dataset<double>("axis1").read_as_vector();
  auto const full_domain_x =
      axis2_file.group("CartGrid").dataset<double>("axis2").read_as_vector();
  tat::grid full_domain{full_domain_x, full_domain_y, full_domain_z};

  // generate a small part from domain
  std::vector<double> small_domain_x(begin(full_domain_z), begin(full_domain_z) + 64);
  std::vector<double> small_domain_y(begin(full_domain_y), begin(full_domain_y) + 64);
  std::vector<double> small_domain_z(begin(full_domain_x), begin(full_domain_x) + 64);
  tat::grid small_domain{small_domain_x, full_domain_y, small_domain_z};

  // generate the 3dpart domain
  std::vector<double> threedpart_domain_x(begin(full_domain_z), begin(full_domain_z) + 512);
  std::vector<double> threedpart_domain_z(begin(full_domain_x), begin(full_domain_x) + 256);
  tat::grid threedpart_domain{threedpart_domain_x, full_domain_y, threedpart_domain_z};

  // generate the pod domain
  std::vector<double> pod_domain_y(begin(full_domain_y), begin(full_domain_y) + 1024);
  std::cerr << "size(threedpart_domain_z): " << size(threedpart_domain_z) << '\n';
  std::cerr << "size(pod_domain_y): " << size(pod_domain_y) << '\n';
  tat::grid pod_domain{threedpart_domain_x, pod_domain_y, threedpart_domain_z};

  // open hdf5 files
  tat::hdf5::file channelflow_121_file{
      "/home/vcuser/channel_flow/dino_res_121000.h5", H5F_ACC_RDONLY};
  tat::hdf5::file channelflow_122_file{
      "/home/vcuser/channel_flow/dino_res_122000.h5", H5F_ACC_RDONLY};
  tat::hdf5::file channelflow_123_file{
      "/home/vcuser/channel_flow/dino_res_123000.h5", H5F_ACC_RDONLY};
  tat::hdf5::file pod0_file{
      "/home/vcuser/channel_flow/pod_0.h5", H5F_ACC_RDONLY};

  // create grid properties of pod
  auto& pod0_velx = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vy"));
  auto& pod0_vely = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vy"));
  auto& pod0_velz = pod_domain.add_lazy_vertex_property<double>(
      pod0_file.group("variables").dataset<double>("Vy"));

  // create grid properties of 121000 time step
  auto& velx_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vx"), "Vx_121");
  auto& vely_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vy"), "Vy_121");
  auto& velz_121 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_121_file.group("variables").dataset<double>("Vz"), "Vz_121");
  //auto& Q_121 = threedpart_domain.add_lazy_vertex_property<double>(
  //    channelflow_121_file.group("variables").dataset<double>("Q"), "Q_121");
  //scalarfield Q_121_field{Q_121.sampler<tat::interpolation::linear>()};

  // create grid properties of 122000 time step
  auto& velx_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vx"), "Vx_122");
  auto& vely_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vy"), "Vy_122");
  auto& velz_122 = threedpart_domain.add_lazy_vertex_property(
      channelflow_122_file.group("variables").dataset<double>("Vz"), "Vz_122");
  //auto& Q_122 = threedpart_domain.add_vertex_property(
  //    channelflow_122_file.group("variables").dataset<double>("Q"), "Q_122");
  //scalarfield Q_122_field{Q_122.sampler<tat::interpolation::linear>()};

  // create grid properties of 123000 time step
  auto& velx_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vx"), "Vx_123");
  auto& vely_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vy"), "Vy_123");
  auto& velz_123 = threedpart_domain.add_lazy_vertex_property<double>(
      channelflow_123_file.group("variables").dataset<double>("Vz"), "Vz_123");
  //auto& Q_123 = threedpart_domain.add_lazy_vertex_property<double>(
  //    channelflow_123_file.group("variables").dataset<double>("Q"), "Q_123");
  //scalarfield Q_123_field{Q_123.sampler<tat::interpolation::linear>()};

  // Q-criterion of 122 field
  auto Q_122_calc =
      [&](auto ix, auto iy, auto iz) {
        tat::mat3  J;
        auto const ixpos = ix == 511 ? ix : ix + 1;
        auto const ixneg = ix == 0 ? ix : ix - 1;
        auto const iypos = iy == 4095 ? iy : iy + 1;
        auto const iyneg = iy == 0 ? iy : iy - 1;
        auto const izpos = iz == 255 ? iz : iz + 1;
        auto const izneg = iz == 0 ? iz : iz - 1;

        auto const dx = full_domain.dimension<0>()[ixpos] -
                        full_domain.dimension<0>()[ixneg];
        auto const dy = full_domain.dimension<1>()[iypos] -
                        full_domain.dimension<1>()[iyneg];
        auto const dz = full_domain.dimension<2>()[izpos] -
                        full_domain.dimension<2>()[izneg];

        J.col(0) =
            tat::vec3{(velx_122(ixpos, iy, iz) - velx_122(ixneg, iy, iz)) / dx,
                      (vely_122(ixpos, iy, iz) - vely_122(ixneg, iy, iz)) / dx,
                      (velz_122(ixpos, iy, iz) - velz_122(ixneg, iy, iz)) / dx};
        J.col(1) =
            tat::vec3{(velx_122(ix, iypos, iz) - velx_122(ix, iyneg, iz)) / dy,
                      (vely_122(ix, iypos, iz) - vely_122(ix, iyneg, iz)) / dy,
                      (velz_122(ix, iypos, iz) - velz_122(ix, iyneg, iz)) / dy};
        J.col(2) =
            tat::vec3{(velx_122(ix, iy, izpos) - velx_122(ix, iy, izneg)) / dz,
                      (vely_122(ix, iy, izpos) - vely_122(ix, iy, izneg)) / dz,
                      (velz_122(ix, iy, izpos) - velz_122(ix, iy, izneg)) / dz};

        auto S     = (J + transposed(J)) / 2;
        auto Omega = (J - transposed(J)) / 2;
        return (sqr_norm(Omega) - sqr_norm(S)) / 2;
      };

  auto Q_iso_mesh =
      isosurface([&](auto const ix, auto const iy, auto const iz,
                     auto const& /*p*/) { return Q_122_calc(ix, iy, iz); },
                 threedpart_domain, 10);
  for (auto v : Q_iso_mesh.vertices()) {
    std::swap(Q_iso_mesh[v](2), Q_iso_mesh[v](0));
  }
  Q_iso_mesh.write_vtk("Q_pod.vtk");
  //isosurface(
  //    [&](auto ix, auto iy, auto iz, auto const & [>p<]) -> auto const& {
  //      return Q_122(ix, iy, iz);
  //    },
  //    threedpart_domain, 0)
  //    .write_vtk("Q_122_0.vtk");
}
