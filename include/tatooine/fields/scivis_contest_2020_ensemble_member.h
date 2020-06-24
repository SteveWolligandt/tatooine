#ifndef TATOOINE_SCIVIS_CONTEST_2020_ENSEMBLE_MEMBER_H
#define TATOOINE_SCIVIS_CONTEST_2020_ENSEMBLE_MEMBER_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/netcdf.h>
#include <tatooine/lazy_netcdf_reader.h>
//==============================================================================
namespace tatooine::fields {
//==============================================================================
struct scivis_contest_2020_ensemble_member
    : vectorfield<scivis_contest_2020_ensemble_member, double, 3> {
  using parent_t = vectorfield<scivis_contest_2020_ensemble_member, double, 3>;
  using parent_t::pos_t;
  using parent_t::real_t;
  using parent_t::tensor_t;
  using component_grid_t        = grid<linspace<double>, std::vector<double>,
                                linspace<double>, linspace<double>>;
  using chunked_grid_property_t =
      typed_multidim_property<component_grid_t, double>;
  //============================================================================
  component_grid_t         m_u_grid;
  component_grid_t         m_v_grid;
  component_grid_t         m_w_grid;
  chunked_grid_property_t* m_u;
  chunked_grid_property_t* m_v;
  chunked_grid_property_t* m_w;
  //============================================================================
  scivis_contest_2020_ensemble_member(std::string const& file_path) {
    auto f           = netcdf::file{file_path, netCDF::NcFile::read};
    auto t_ax_var    = f.variable<double>("T_AX");
    auto z_mit40_var = f.variable<double>("Z_MIT40");
    auto xg_var      = f.variable<double>("XG");
    auto xc_var      = f.variable<double>("XC");
    auto yg_var      = f.variable<double>("YG");
    auto yc_var      = f.variable<double>("YC");
    auto u_var       = f.variable<double>("U");
    auto v_var       = f.variable<double>("V");
    auto w_var       = f.variable<double>("W");

    linspace xg_axis{xg_var.read_single(0), xg_var.read_single(499), 500};
    linspace xc_axis{xc_var.read_single(0), xc_var.read_single(499), 500};
    linspace yg_axis{yg_var.read_single(0), yg_var.read_single(499), 500};
    linspace yc_axis{yc_var.read_single(0), yc_var.read_single(499), 500};
    linspace t_axis{t_ax_var.read_single(0), t_ax_var.read_single(59), 60};
    auto     z_axis = z_mit40_var.read_as_vector();
    std::cerr << "XG: " << xg_axis << '\n';
    std::cerr << "XC: " << xc_axis << '\n';
    std::cerr << "YG: " << yg_axis << '\n';
    std::cerr << "YC: " << yc_axis << '\n';
    std::cerr << "T: " << t_axis << '\n';

    m_u_grid.dimension<0>() = t_axis;
    m_v_grid.dimension<0>() = t_axis;
    m_w_grid.dimension<0>() = t_axis;

    m_u_grid.dimension<1>() = z_axis;
    m_v_grid.dimension<1>() = z_axis;
    m_w_grid.dimension<1>() = z_axis;

    m_u_grid.dimension<2>() = yc_axis;
    m_v_grid.dimension<2>() = yg_axis;
    m_w_grid.dimension<2>() = yc_axis;

    m_u_grid.dimension<3>() = xg_axis;
    m_v_grid.dimension<3>() = xc_axis;
    m_w_grid.dimension<3>() = xc_axis;

    m_u = &m_u_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "u", u_var, std::vector<size_t>(4, 2));
    m_v = &m_v_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "v", v_var, std::vector<size_t>(4, 2));
    m_w = &m_w_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "w", w_var, std::vector<size_t>(4, 2));
  }
  //==============================================================================
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t final {
    return tensor_t{m_u->sample(t, x(2), x(1), x(0)),
                    m_v->sample(t, x(2), x(1), x(0)),
                    m_w->sample(t, x(2), x(1), x(0))};
  }
  //------------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool final {
    return m_u_grid.in_domain(t, x(2), x(1), x(0)) &&
           m_v_grid.in_domain(t, x(2), x(1), x(0)) &&
           m_w_grid.in_domain(t, x(2), x(1), x(0));
  }
};
//==============================================================================
}  // namespace tatooine::fields
//==============================================================================
#endif
