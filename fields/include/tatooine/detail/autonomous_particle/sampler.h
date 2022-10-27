#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/mat.h>
#include <tatooine/vec.h>

#include <tatooine/linspace.h>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct sampler {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using vec_type     = vec<Real, NumDimensions>;
  using pos_type     = vec_type;
  using mat_type     = mat<Real, NumDimensions, NumDimensions>;
  using ellipse_type = tatooine::geometry::hyper_ellipse<Real, NumDimensions>;

 private:
  //============================================================================
  // MEMBERS
  //============================================================================
  ellipse_type m_ellipse0, m_ellipse1;
  mat_type     m_nabla_phi0, m_nabla_phi1;

 public:
  //============================================================================
  // CTORS
  //============================================================================
  sampler(sampler const&)     = default;
  sampler(sampler&&) noexcept = default;
  //============================================================================
  auto operator=(sampler const&) -> sampler& = default;
  auto operator=(sampler&&) noexcept -> sampler& = default;
  //============================================================================
  sampler()  = default;
  ~sampler() = default;
  //----------------------------------------------------------------------------
  sampler(ellipse_type const& e0, ellipse_type const& e1,
          mat_type const& nabla_phi)
      : m_ellipse0{e0},
        m_ellipse1{e1},
        m_nabla_phi0{nabla_phi},
        m_nabla_phi1{*inv(nabla_phi)} {}
  //============================================================================
  /// \{
  auto ellipse(forward_tag const /*tag*/) const -> auto const& {
    return m_ellipse0;
  }
  //----------------------------------------------------------------------------
  auto ellipse(backward_tag const /*tag*/) const -> auto const& {
    return m_ellipse1;
  }
  /// \}
  //----------------------------------------------------------------------------
  /// \{
  auto nabla_phi(forward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi0;
  }
  //----------------------------------------------------------------------------
  auto nabla_phi(backward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi1;
  }
  /// \}
  //============================================================================
  auto local_pos(pos_type const&                    q,
                 forward_or_backward_tag auto const tag) const {
    return transposed(nabla_phi(tag)) * (q - x0(tag));
  }
  //----------------------------------------------------------------------------
  auto sample(pos_type const& q, forward_or_backward_tag auto const tag) const {
    return phi(tag) + local_pos(q, tag);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const&                    q,
                  forward_or_backward_tag auto const tag) const {
    return sample(q, tag);
  }
  //----------------------------------------------------------------------------
  auto is_inside(pos_type const&                    q,
                 forward_or_backward_tag auto const tag) const {
    return ellipse(tag).is_inside(q);
  }
  //----------------------------------------------------------------------------
  auto x0(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).center();
  }
  //----------------------------------------------------------------------------
  auto phi(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(opposite(tag)).center();
  }
  //----------------------------------------------------------------------------
  auto distance_sqr(pos_type const&                    q,
                    forward_or_backward_tag auto const tag) const {
    return tatooine::euclidean_length(local_pos(q, tag));
  }
  //----------------------------------------------------------------------------
  auto distance(pos_type const& q, auto const tag) const {
    return gcem::sqrt(distance_sqr(q, tag));
  }
  //----------------------------------------------------------------------------
  auto S(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).S();
  }
  //----------------------------------------------------------------------------
  auto discretize(std::size_t const                  n,
                  forward_or_backward_tag auto const tag) const {
    return ellipse(tag).discretize(n);
  }
};
//------------------------------------------------------------------------------
template <floating_point Real>
auto write_vtp(std::vector<sampler<Real, 2>> const& samplers,
               std::size_t const n, filesystem::path const& path,
               forward_or_backward_tag auto const tag) {
  auto file = std::ofstream{path, std::ios::binary};
  if (!file.is_open()) {
    throw std::runtime_error{"Could not write " + path.string()};
  }
  auto offset                    = std::size_t{};
  using header_type              = std::uint64_t;
  using lines_connectivity_int_t = std::int32_t;
  using lines_offset_int_t       = lines_connectivity_int_t;
  file << "<VTKFile"
       << " type=\"PolyData\""
       << " version=\"1.0\" "
          "byte_order=\"LittleEndian\""
       << " header_type=\""
       << vtk::xml::to_string(
              vtk::xml::to_type<header_type>())
       << "\">";
  file << "<PolyData>\n";
  for (std::size_t i = 0 ;i < size(samplers); ++i) {
    file << "<Piece"
         << " NumberOfPoints=\"" << n << "\""
         << " NumberOfPolys=\"0\""
         << " NumberOfVerts=\"0\""
         << " NumberOfLines=\"" << n - 1 << "\""
         << " NumberOfStrips=\"0\""
         << ">\n";

    // Points
    file << "<Points>";
    file << "<DataArray"
         << " format=\"appended\""
         << " offset=\"" << offset << "\""
         << " type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<Real>())
         << "\" NumberOfComponents=\"" << 3 << "\"/>";
    auto const num_bytes_points =
        header_type(sizeof(Real) * 3 * n);
    offset += num_bytes_points + sizeof(header_type);
    file << "</Points>\n";

    // Lines
    file << "<Lines>\n";
    // Lines - connectivity
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<lines_connectivity_int_t>())
         << "\" Name=\"connectivity\"/>\n";
    auto const num_bytes_lines_connectivity =
        (n - 1) * 2 *
        sizeof(lines_connectivity_int_t);
    offset += num_bytes_lines_connectivity + sizeof(header_type);
    // Lines - offsets
    file << "<DataArray format=\"appended\" offset=\"" << offset << "\" type=\""
         << vtk::xml::to_string(
                vtk::xml::to_type<lines_offset_int_t>())
         << "\" Name=\"offsets\"/>\n";
    auto const num_bytes_lines_offsets =
        sizeof(lines_offset_int_t) * (n - 1) * 2;
    offset += num_bytes_lines_offsets + sizeof(header_type);
    file << "</Lines>\n";
    file << "</Piece>\n";
  }
  file << "</PolyData>\n";
  file << "<AppendedData encoding=\"raw\">_";
  // Writing vertex data to appended data section
  for (auto const& sampler : samplers) {
    auto const num_bytes_points =
        header_type(sizeof(Real) * 3 * n);
    using namespace std::ranges;
    auto radial = tatooine::linspace<Real>{0, M_PI * 2, n + 1};
    radial.pop_back();

    auto discretization      = tatooine::line<Real, 3>{};
    auto radian_to_cartesian = [](auto const t) {
      return tatooine::vec{gcem::cos(t), gcem::sin(t), 0};
    };
    auto out_it = std::back_inserter(discretization);
    copy(radial | views::transform(radian_to_cartesian), out_it);
    discretization.set_closed(true);
    for (auto const v : discretization.vertices()) {
      auto v2 = sampler.S(tag) * discretization[v].xy() + sampler.x0(tag);;
      discretization[v].x() = v2.x();
      discretization[v].y() = v2.y();
    }

    // Writing points
    file.write(reinterpret_cast<char const*>(&num_bytes_points), sizeof(header_type));
    for (auto const v : discretization.vertices()) {
      file.write(reinterpret_cast<char const*>(discretization.at(v).data()),
                 sizeof(Real) * 3);
    }

    // Writing lines connectivity data to appended data section
    {
      auto connectivity_data = std::vector<lines_connectivity_int_t>{};
      connectivity_data.reserve((n - 1) * 2);
      for (std::size_t i = 0; i < n - 1; ++i) {
        connectivity_data.push_back(static_cast<lines_connectivity_int_t>(i));
        connectivity_data.push_back(static_cast<lines_connectivity_int_t>(i + 1));
      }

      auto const num_bytes_lines_connectivity =
          header_type((n - 1) * 2 * sizeof(lines_connectivity_int_t));
      file.write(reinterpret_cast<char const*>(&num_bytes_lines_connectivity),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(connectivity_data.data()),
                 static_cast<std::streamsize>(num_bytes_lines_connectivity));
    }

    // Writing lines offsets to appended data section
    {
      auto offsets = std::vector<lines_offset_int_t>(n, 2);
      for (std::size_t i = 1; i < size(offsets); ++i) {
        offsets[i] += offsets[i - 1];
      }
      auto const num_bytes_lines_offsets =
          header_type(sizeof(lines_offset_int_t) * (n - 1) * 2);
      file.write(reinterpret_cast<char const*>(&num_bytes_lines_offsets),
                 sizeof(header_type));
      file.write(reinterpret_cast<char const*>(offsets.data()),
                 static_cast<std::streamsize>(num_bytes_lines_offsets));
    }
  }

  file << "</AppendedData>";
  file << "</VTKFile>";
}
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
