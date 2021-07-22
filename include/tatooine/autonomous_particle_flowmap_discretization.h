#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/simplex_mesh.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct autonomous_particle_flowmap_discretization {
  using vec_t               = vec<Real, N>;
  using pos_t               = vec_t;
  using sampler_t           = autonomous_particle_sampler<Real, N>;
  using sampler_container_t = std::vector<sampler_t>;
  using mesh_t              = simplex_mesh<Real, N>;
  using mesh_prop_t =
      typename mesh_t::template vertex_property_t<sampler_t const*>;
  //============================================================================
 private:
  sampler_container_t m_samplers;
  mesh_t              m_mesh0;
  mesh_t              m_mesh1;
  mesh_prop_t*        m_mesh0_samplers;
  mesh_prop_t*        m_mesh1_samplers;
  //============================================================================
 public:
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(Flowmap&&             flowmap,
                                             arithmetic auto const t0,
                                             arithmetic auto const t1,
                                             arithmetic auto const tau_step,
                                             uniform_grid<Real, N> const& g) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto  initial_particle_distribution = g.copy_without_properties();
    for (size_t i = 0; i < N; ++i) {
      initial_particle_distribution.dimension(i).pop_front();
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    std::deque<autonomous_particle<Real, N>> particles;
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing()/2);
        });
    fill(std::forward<Flowmap>(flowmap), particles, t1, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const                            tau_step,
      std::deque<autonomous_particle<Real, N>> const& initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == N,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t1, tau_step);
  }
  //============================================================================
  auto samplers() const -> auto const& { return m_samplers; }

 public:
  auto mesh0() const -> auto const& { return m_mesh0; }

 public:
  auto mesh1() const -> auto const& { return m_mesh1; }
  //============================================================================
  template <typename Flowmap>
  auto fill(Flowmap&&                                       flowmap,
            std::deque<autonomous_particle<Real, N>> const& initial_particles,
            arithmetic auto const t1, arithmetic auto const tau_step) {
    auto const advected_particles =
        autonomous_particle<Real, N>::advect_with_3_splits(
            std::forward<Flowmap>(flowmap), tau_step, t1, initial_particles);
    m_samplers.reserve(size(advected_particles));
    boost::copy(
        advected_particles | boost::adaptors::transformed(
                                 [](auto const& p) { return p.sampler(); }),
        std::back_inserter(m_samplers));

    m_mesh0_samplers =
        &m_mesh0.template vertex_property<sampler_t const*>("samplers");
    m_mesh1_samplers =
        &m_mesh1.template vertex_property<sampler_t const*>("samplers");

    auto const ts = linspace{0.0, 2 * M_PI, 17};
    for (auto const& sampler : samplers()) {
      {auto const v = m_mesh0.insert_vertex(sampler.ellipse0().center());
      m_mesh0_samplers->at(v) = &sampler;}

      {auto const v = m_mesh1.insert_vertex(sampler.ellipse1().center());
        m_mesh1_samplers->at(v) = &sampler;
      }

      for (auto t_it = begin(ts); t_it != prev(end(ts)); ++t_it) {
        auto const t = *t_it;
        auto const y = vec{std::cos(t), std::sin(t)};
        {
          auto const v = m_mesh0.insert_vertex(sampler.ellipse0().center() +
                                               sampler.ellipse0().S() * y);
          m_mesh0_samplers->at(v) = &sampler;
        }

        {
          auto const v = m_mesh1.insert_vertex(sampler.ellipse1().center() +
                                               sampler.ellipse1().S() * y);
          m_mesh1_samplers->at(v) = &sampler;
        }
      }
    }

    m_mesh0.build_delaunay_mesh();
    m_mesh0.build_hierarchy();

    m_mesh1.build_delaunay_mesh();
    m_mesh1.build_hierarchy();
  }
  //----------------------------------------------------------------------------
 private:
  template <typename Tag, size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_t const& x, mesh_t const& mesh,
                            mesh_prop_t const& vertex_samplers,
                            Tag const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    // try to find ellipse that includes x
    for (auto const& sampler : m_samplers) {
      if (sampler.is_inside(x, tag)) {
        return sampler(x, tag);
      }
    }

    // try to find mesh cell that includes x
    for (auto c : mesh.hierarchy().nearby_cells(x)) {
    //for (auto c : mesh.cells()) {
      auto const            vs = mesh.cell_at(c);
      auto                  A  = mat<Real, N + 1, N + 1>::ones();
      auto                  b  = vec<Real, N + 1>::ones();
      for (size_t r = 0; r < N; ++r) {
        (
            [&]() {
              A(r, VertexSeq) = VertexSeq > 0
                                    ? mesh[std::get<VertexSeq>(vs)](r) -
                                          mesh[std::get<0>(vs)](r)
                                    : 0;
            }(),
            ...);

        b(r) = x(r) - mesh[std::get<0>(vs)](r);
      }

      static real_t constexpr eps               = 1e-10;
      auto const   barycentric_coord = solve(A, b);
      if (((barycentric_coord(VertexSeq) >= -eps) && ...) &&
          ((barycentric_coord(VertexSeq) <= 1 + eps) && ...)) {
        auto const samplers =
            std::array{vertex_samplers[std::get<VertexSeq>(vs)]...};
      //  auto const inner_cell =
      //      std::array{vertex_samplers[std::get<VertexSeq>(vs)]
      //                     ->ellipse(tag)
      //                     .nearest_point_on_boundary(x)...};
      //  auto weights =
      //      vec{distance(inner_cell[VertexSeq],
      //                   samplers[VertexSeq]->ellipse(tag).center())...};
      //
      //  auto inner_A = mat<Real, N + 1, N + 1>::ones();
      //  auto inner_b = vec<Real, N + 1>::ones();
      //  for (size_t r = 0; r < N; ++r) {
      //    (
      //        [&]() {
      //          inner_A(r, VertexSeq) =
      //              VertexSeq > 0 ? inner_cell[VertexSeq](r) - inner_cell[0](r)
      //                            : 0;
      //        }(),
      //        ...);
      //
      //    inner_b(r) = x(r) - inner_cell[0](r);
      //  }
      //
      //  auto inner_barycentric_coords = solve(inner_A, inner_b);
      //  //weights = weights * inner_barycentric_coords;
      //  //weights = weights / sum(weights);
      //  weights = inner_barycentric_coords;
        auto       map                     = pos_t::zeros();
        for (size_t i = 0; i < N + 1; ++i) {
          map += samplers[i]->sample(x, tag) * barycentric_coord[i];
        }
        return map;
      }
    }

    // if point is not included by anything throw exception
    throw std::runtime_error{"out of domain"};
  }

 public:
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(pos_t const& x) const {
    return sample(x, mesh0(), *m_mesh0_samplers, tag::forward,
                  std::make_index_sequence<N+1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_t const& x) const {
    return sample(x, mesh1(), *m_mesh1_samplers, tag::backward,
                  std::make_index_sequence<N+1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
};
template <size_t N>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_t, N>;
using autonomous_particle_flowmap_discretization2 =
    AutonomousParticleFlowmapDiscretization<2>;
using autonomous_particle_flowmap_discretization3 =
    AutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
