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
  using component_grid_t = grid<linspace<double>, linspace<double>,
                                std::vector<double>, linspace<double>>;
  using chunked_grid_property_t =
      typed_multidim_property<component_grid_t, double>;
  //============================================================================
  component_grid_t         m_u_grid;
  component_grid_t         m_v_grid;
  component_grid_t         m_w_grid;
  chunked_grid_property_t* m_u;
  chunked_grid_property_t* m_v;
  chunked_grid_property_t* m_w;
  linspace<double>                 xg_axis;
  linspace<double>                 xc_axis;
  linspace<double>                 yg_axis;
  linspace<double>                 yc_axis;
  linspace<double>                 t_axis;
  std::vector<double>      z_axis;
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

    xg_axis = linspace{xg_var.read_single(0), xg_var.read_single(499), 500};
    xc_axis = linspace{xc_var.read_single(0), xc_var.read_single(499), 500};
    yg_axis = linspace{yg_var.read_single(0), yg_var.read_single(499), 500};
    yc_axis = linspace{yc_var.read_single(0), yc_var.read_single(499), 500};
    t_axis  = linspace{t_ax_var.read_single(0), t_ax_var.read_single(59), 60};
    z_axis  = z_mit40_var.read_as_vector();
    std::cerr << "XG: " << xg_axis << '\n';
    std::cerr << "XC: " << xc_axis << '\n';
    std::cerr << "YG: " << yg_axis << '\n';
    std::cerr << "YC: " << yc_axis << '\n';
    std::cerr << "T: " << t_axis << '\n';

    m_u_grid.dimension<3>() = t_axis;
    m_v_grid.dimension<3>() = t_axis;
    m_w_grid.dimension<3>() = t_axis;

    m_u_grid.dimension<2>() = z_axis;
    m_v_grid.dimension<2>() = z_axis;
    m_w_grid.dimension<2>() = z_axis;

    m_u_grid.dimension<1>() = yc_axis;
    m_v_grid.dimension<1>() = yg_axis;
    m_w_grid.dimension<1>() = yc_axis;

    m_u_grid.dimension<0>() = xg_axis;
    m_v_grid.dimension<0>() = xc_axis;
    m_w_grid.dimension<0>() = xc_axis;

    m_u = &m_u_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "u", u_var, std::vector<size_t>(4, 5));
    m_v = &m_v_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "v", v_var, std::vector<size_t>(4, 5));
    m_w = &m_w_grid.add_vertex_property<netcdf::lazy_reader<double>>(
        "w", w_var, std::vector<size_t>(4, 5));
  }
  //==============================================================================
  auto evaluate(pos_t const& x, real_t const t) const -> tensor_t final {
    return tensor_t{m_u->sample(x(0), x(1), x(2), t),
                    m_v->sample(x(0), x(1), x(2), t),
                    m_w->sample(x(0), x(1), x(2), t)};
  }
  //------------------------------------------------------------------------------
  auto in_domain(pos_t const& x, real_t const t) const -> bool final {
    bool const in_u = m_u_grid.in_domain(x(0), x(1), x(2), t);
    bool const in_v = m_v_grid.in_domain(x(0), x(1), x(2), t);
    bool const in_w = m_w_grid.in_domain(x(0), x(1), x(2), t);
    return in_u && in_v && in_w;
  }
};
//==============================================================================
}  // namespace tatooine::fields
//==============================================================================
#endif
