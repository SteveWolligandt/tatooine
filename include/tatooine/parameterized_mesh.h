#ifndef TATOOINE_PARAMETERIZED_SURFACE_H
#define TATOOINE_PARAMETERIZED_SURFACE_H

#include "interpolation.h"
#include "mesh.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N,
          template <typename> typename Interpolator =
              interpolation::linear>
struct parameterized_mesh : public mesh<Real, N> {
  using interpolator_t = Interpolator<Real>;

  using this_t   = parameterized_mesh<Real, N, Interpolator>;
  using parent_t = mesh<Real, N>;

  using parent_t::num_dimensions;
  using typename parent_t::face;
  using typename parent_t::pos_t;
  using typename parent_t::vertex;
  template <typename T>
  using vertex_prop = typename parent_t::template vertex_prop<T>;

  using parent_t::at;
  using parent_t::operator[];
  using parent_t::faces;
  using parent_t::vertices;

  using uv_t      = vec<Real, 2>;
  using uv_prop_t = vertex_prop<uv_t>;

  struct out_of_domain : std::exception {};

  //============================================================================
  uv_prop_t* m_uv;

  //============================================================================
  parameterized_mesh() : m_uv(add_uv_prop()) {}

  //----------------------------------------------------------------------------
  parameterized_mesh(const parameterized_mesh& other)
      : parent_t(other), m_uv(find_uv_prop()) {}

  //----------------------------------------------------------------------------
  parameterized_mesh(parameterized_mesh&& other)
      : parent_t(std::move(other)), m_uv(find_uv_prop()) {}

  //----------------------------------------------------------------------------
  auto& operator=(const parameterized_mesh& other) {
    parent_t::operator=(other);
    m_uv              = find_uv_prop();
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(parameterized_mesh&& other) {
    parent_t::operator=(std::move(other));
    m_uv              = find_uv_prop();
    return *this;
  }

  //============================================================================
  auto insert_vertex(const pos_t& point, const uv_t& uv) {
    auto v      = parent_t::insert_vertex(point);
    m_uv->at(v) = uv;
    return v;
  }

  //----------------------------------------------------------------------------
  auto&       uv(vertex v) { return m_uv->at(v); }
  const auto& uv(vertex v) const { return m_uv->at(v); }

  //----------------------------------------------------------------------------
  auto operator()(Real u, Real v) const { return sample(u, v); }
  auto sample(Real u, Real v) const {
    vec<Real, 3> b{u, v, 1};
    auto           inside_face = face::invalid();
    vec<Real, 3> bary;
    for (auto f : faces()) {
      mat<Real, 3, 3> A{
          {uv(at(f)[0])(0), uv(at(f)[1])(0), uv(at(f)[2])(0)},
          {uv(at(f)[0])(1), uv(at(f)[1])(1), uv(at(f)[2])(1)},
          {1, 1, 1}};

      bary = gesv(A, b);
      bary.for_indices([&](const auto... is) {
        if (bary(is...) >= -1e-6 && bary(is...) < 0) { bary(is...) = 0; }
      });

      if (bary(0) >= 0 && bary(1) >= 0 && bary(2) >= 0) {
        inside_face = f;
        break;
      }
    }
    if (inside_face == face::invalid()) {
      throw out_of_domain{};
    } else {
      return interpolator_t::interpolate(
          at(at(inside_face)[0]), at(at(inside_face)[1]),
          at(at(inside_face)[2]), bary(0), bary(1), bary(2));
    }
  }

  //============================================================================
 private:
  auto find_uv_prop() {
    return dynamic_cast<uv_prop_t*>(
        &this->template vertex_property<uv_t>("uv"));
  }

  //----------------------------------------------------------------------------
  auto add_uv_prop() {
    return dynamic_cast<uv_prop_t*>(
        &this->template add_vertex_property<uv_t>("uv"));
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
