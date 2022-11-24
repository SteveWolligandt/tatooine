#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/huber_loss.h>
#include <tatooine/unstructured_triangular_grid.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions,
          typename SplitBehavior = typename autonomous_particle<
              Real, NumDimensions>::split_behaviors::three_splits>
struct autonomous_particle_flowmap_discretization {
  using this_type =
      autonomous_particle_flowmap_discretization<Real, NumDimensions,
                                                 SplitBehavior>;
  using real_type          = Real;
  using vec_type           = vec<real_type, NumDimensions>;
  using pos_type           = vec_type;
  using gradient_type      = mat<real_type, NumDimensions, NumDimensions>;
  using particle_type      = autonomous_particle<real_type, NumDimensions>;
  using particle_list_type = std::vector<particle_type>;

  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  // tatooine::unstructured_triangular_grid<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using flowmap_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<pos_type>;
  using flowmap_gradient_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<
          gradient_type>;

  using cgal_kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using cgal_triangulation_type = std::conditional_t<
      NumDimensions == 2,
      cgal::delaunay_triangulation_with_info<2, vertex_handle, cgal_kernel>,
      std::conditional_t<
          NumDimensions == 3,
          cgal::delaunay_triangulation_with_info<
              3, vertex_handle, cgal_kernel,
              cgal::delaunay_triangulation_simplex_base_with_circumcenter<
                  3, cgal_kernel>>,
          void>>;
  using cgal_point = typename cgal_triangulation_type::Point;

  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  pointset_type                          m_pointset_forward;
  flowmap_vertex_property_type          *m_flowmaps_forward          = nullptr;
  flowmap_gradient_vertex_property_type *m_flowmap_gradients_forward = nullptr;
  cgal_triangulation_type                m_triangulation_forward;

  pointset_type                          m_pointset_backward;
  flowmap_vertex_property_type          *m_flowmaps_backward          = nullptr;
  flowmap_gradient_vertex_property_type *m_flowmap_gradients_backward = nullptr;
  cgal_triangulation_type                m_triangulation_backward;
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  auto pointset(forward_or_backward_tag auto const direction) -> auto & {
    if constexpr (is_forward<decltype(direction)>) {
      return m_pointset_forward;
    } else {
      return m_pointset_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto pointset(forward_or_backward_tag auto const direction) const
      -> auto const & {
    if constexpr (is_forward<decltype(direction)>) {
      return m_pointset_forward;
    } else {
      return m_pointset_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmaps(forward_or_backward_tag auto const direction) -> auto & {
    if constexpr (is_forward<decltype(direction)>) {
      return *m_flowmaps_forward;
    } else {
      return *m_flowmaps_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmaps(forward_or_backward_tag auto const direction) const
      -> auto const & {
    if constexpr (is_forward<decltype(direction)>) {
      return *m_flowmaps_forward;
    } else {
      return *m_flowmaps_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmap_gradients(forward_or_backward_tag auto const direction)
      -> auto & {
    if constexpr (is_forward<decltype(direction)>) {
      return *m_flowmap_gradients_forward;
    } else {
      return *m_flowmap_gradients_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto flowmap_gradients(forward_or_backward_tag auto const direction) const
      -> auto const & {
    if constexpr (is_forward<decltype(direction)>) {
      return *m_flowmap_gradients_forward;
    } else {
      return *m_flowmap_gradients_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto triangulation(forward_or_backward_tag auto const direction) -> auto & {
    if constexpr (is_forward<decltype(direction)>) {
      return m_triangulation_forward;
    } else {
      return m_triangulation_backward;
    }
  }
  //----------------------------------------------------------------------------
  auto triangulation(forward_or_backward_tag auto const direction) const
      -> auto const & {
    if constexpr (is_forward<decltype(direction)>) {
      return m_triangulation_forward;
    } else {
      return m_triangulation_backward;
    }
  }
  //============================================================================
 private:
  autonomous_particle_flowmap_discretization()
      : m_flowmaps_forward{&m_pointset_forward
                                .template vertex_property<pos_type>(
                                    "flowmaps")},
        // m_flowmap_gradients_forward{
        //     &m_pointset_forward.template vertex_property<gradient_type>(
        //         "flowmap_gradients")},

        m_flowmaps_backward{
            &m_pointset_backward.template vertex_property<pos_type>("flowmaps")}

  //,m_flowmap_gradients_backward{
  //    &m_pointset_backward.template vertex_property<gradient_type>(
  //        "flowmap_gradients")}
  {}
  //============================================================================
 public:
  static auto from_advected(
      std::vector<particle_type> const &advected_particles) -> this_type {
    return from_advected(advected_particles,
                         std::make_index_sequence<num_dimensions()>{});
  }
  //============================================================================
 private:
  template <std::size_t... Is>
  static auto from_advected(
      std::vector<particle_type> const &advected_particles,
      std::index_sequence<Is...> /*seq*/) -> this_type {
    auto disc = this_type{};
    using namespace std::ranges;
    using cgal_point_list = std::vector<std::pair<cgal_point, vertex_handle>>;
    auto points_forward   = cgal_point_list{};
    auto points_backward  = cgal_point_list{};
    points_forward.reserve(size(advected_particles));
    points_backward.reserve(size(advected_particles));
    for (auto const &p : advected_particles) {
      {
        // forward
        auto v = disc.m_pointset_forward.insert_vertex(p.x0());
        disc.flowmaps(forward)[v]          = p.x();
        //disc.flowmap_gradients(forward)[v] = p.nabla_phi();
        points_forward.emplace_back(
            cgal_point{disc.m_pointset_forward[v](Is)...}, v);
      }
      {
        // backward
        auto v = disc.m_pointset_backward.insert_vertex(p.x());
        disc.flowmaps(backward)[v]          = p.x0();
        //disc.flowmap_gradients(backward)[v] = *inv(p.nabla_phi());
        points_backward.emplace_back(
            cgal_point{disc.m_pointset_backward[v](Is)...}, v);
      }
    }

    disc.m_triangulation_forward =
        cgal_triangulation_type{begin(points_forward), end(points_forward)};
    if constexpr (NumDimensions == 2) {
      for (auto it = disc.m_triangulation_forward.finite_faces_begin();
           it != disc.m_triangulation_forward.finite_faces_end(); ++it) {
        disc.m_pointset_forward.insert_simplex(
            vertex_handle{it->vertex(0)->info()},
            vertex_handle{it->vertex(1)->info()},
            vertex_handle{it->vertex(2)->info()});
      }
      disc.m_triangulation_backward =
          cgal_triangulation_type{begin(points_backward), end(points_backward)};
      for (auto it = disc.m_triangulation_backward.finite_faces_begin();
           it != disc.m_triangulation_backward.finite_faces_end(); ++it) {
        disc.m_pointset_backward.insert_simplex(
            vertex_handle{it->vertex(0)->info()},
            vertex_handle{it->vertex(1)->info()},
            vertex_handle{it->vertex(2)->info()});
      }
    } else if constexpr (NumDimensions == 3) {
      for (auto it = disc.m_triangulation_forward.finite_cells_begin();
           it != disc.m_triangulation_forward.finite_cells_end(); ++it) {
        disc.m_pointset_forward.insert_simplex(
            vertex_handle{it->vertex(0)->info()},
            vertex_handle{it->vertex(1)->info()},
            vertex_handle{it->vertex(2)->info()},
            vertex_handle{it->vertex(3)->info()});
      }
      disc.m_triangulation_backward =
          cgal_triangulation_type{begin(points_backward), end(points_backward)};
      for (auto it = disc.m_triangulation_backward.finite_cells_begin();
           it != disc.m_triangulation_backward.finite_cells_end(); ++it) {
        disc.m_pointset_backward.insert_simplex(
            vertex_handle{it->vertex(0)->info()},
            vertex_handle{it->vertex(1)->info()},
            vertex_handle{it->vertex(2)->info()},
            vertex_handle{it->vertex(3)->info()});
      }
    }
    return disc;
  }
  //============================================================================
 public:
  autonomous_particle_flowmap_discretization(
      autonomous_particle_flowmap_discretization const &other)
      : m_flowmaps_forward{&m_pointset_forward
                                .template vertex_property<pos_type>(
                                    "flowmaps")},
        // m_flowmap_gradients_forward{
        //     &m_pointset_forward.template vertex_property<gradient_type>(
        //         "flowmap_gradients")},

        m_flowmaps_backward{
            &m_pointset_backward.template vertex_property<pos_type>("flowmaps")}

  //, m_flowmap_gradients_backward{
  //    &m_pointset_backward.template vertex_property<gradient_type>(
  //        "flowmap_gradients")}
  {
    copy(other);
  }
  //----------------------------------------------------------------------------
  autonomous_particle_flowmap_discretization(
      autonomous_particle_flowmap_discretization &&other)
      : m_pointset_forward{std::move(other.m_pointset_forward)},
        m_flowmaps_forward{std::exchange(other.m_flowmaps_forward, nullptr)},
        m_flowmap_gradients_forward{
            std::exchange(other.m_flowmap_gradients_forward, nullptr)},
        m_pointset_backward{std::move(other.m_pointset_backward)},
        m_flowmaps_backward{std::exchange(other.m_flowmaps_backward, nullptr)},
        m_flowmap_gradients_backward{
            std::exchange(other.m_flowmap_gradients_backward, nullptr)},
        m_triangulation_forward{std::move(other.m_triangulation_forward)},
        m_triangulation_backward{std::move(other.m_triangulation_backward)} {}
  //----------------------------------------------------------------------------
  auto operator=(autonomous_particle_flowmap_discretization const &other)
      -> autonomous_particle_flowmap_discretization & {
    m_flowmaps_forward =
        &m_pointset_forward.template vertex_property<pos_type>("flowmaps");
    // m_flowmap_gradients_forward =
    //     &m_pointset_forward.template vertex_property<gradient_type>(
    //         "flowmap_gradients");
    m_flowmaps_backward =
        &m_pointset_backward.template vertex_property<pos_type>("flowmaps");
    // m_flowmap_gradients_backward =
    //     &m_pointset_backward.template vertex_property<gradient_type>(
    //         "flowmap_gradients");
    copy(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(autonomous_particle_flowmap_discretization &&other)
      -> autonomous_particle_flowmap_discretization & {
    m_flowmaps_forward = std::exchange(other.m_flowmaps_forward, nullptr);
    m_flowmap_gradients_forward =
        std::exchange(other.m_flowmap_gradients_forward, nullptr);
    m_flowmaps_backward = std::exchange(other.m_flowmaps_backward, nullptr);
    m_flowmap_gradients_backward =
        std::exchange(other.m_flowmap_gradients_backward, nullptr);
    m_triangulation_forward  = std::move(other.m_triangulation_forward);
    m_triangulation_backward = std::move(other.m_triangulation_backward);
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap &&flowmap, arithmetic auto const t_end,
      arithmetic auto const     tau_step,
      particle_list_type const &initial_particles,
      std::atomic_uint64_t     &uuid_generator)
      : m_flowmaps_forward{&m_pointset_forward
                                .template vertex_property<pos_type>(
                                    "flowmaps")},
        // m_flowmap_gradients_forward{
        //     &m_pointset_forward.template vertex_property<gradient_type>(
        //         "flowmap_gradients")},

        m_flowmaps_backward{
            &m_pointset_backward.template vertex_property<pos_type>("flowmaps")}
  //, m_flowmap_gradients_backward{
  //    &m_pointset_backward.template vertex_property<gradient_type>(
  //        "flowmap_gradients")}
  {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t_end, tau_step,
         uuid_generator);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap &&flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                     tau_step,
      uniform_rectilinear_grid<real_type, NumDimensions> const &g,
      std::uint8_t const max_split_depth =
          particle_type::default_max_split_depth)
      : m_flowmaps_forward{&m_pointset_forward
                                .template vertex_property<pos_type>(
                                    "flowmaps")},
        // m_flowmap_gradients_forward{
        //     &m_pointset_forward.template vertex_property<gradient_type>(
        //         "flowmap_gradients")},

        m_flowmaps_backward{
            &m_pointset_backward.template vertex_property<pos_type>("flowmaps")}
  //, m_flowmap_gradients_backward{
  //    &m_pointset_backward.template vertex_property<gradient_type>(
  //        "flowmap_gradients")}
  {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");

    auto uuid_generator = std::atomic_uint64_t{};
    fill(std::forward<Flowmap>(flowmap),
         particle_type::particles_from_grid_filling_gaps(t0, g, max_split_depth,
                                                         uuid_generator),
         t0 + tau, tau_step, uuid_generator);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap &&flowmap, arithmetic auto const t_end,
      arithmetic auto const tau_step, particle_type const &initial_particle,
      std::atomic_uint64_t &uuid_generator)
      : m_flowmaps_forward{&m_pointset_forward
                                .template vertex_property<pos_type>(
                                    "flowmaps")},
        //m_flowmap_gradients_forward{
        //    &m_pointset_forward.template vertex_property<gradient_type>(
        //        "flowmap_gradients")},

        m_flowmaps_backward{
            &m_pointset_backward.template vertex_property<pos_type>(
                "flowmaps")}
        //, m_flowmap_gradients_backward{
        //    &m_pointset_backward.template vertex_property<gradient_type>(
        //        "flowmap_gradients")}
  {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, t_end,
         tau_step, uuid_generator);
  }

 private:
  auto copy(autonomous_particle_flowmap_discretization const &other) {
    for (auto const v : m_pointset_forward.vertices()) {
      flowmaps(forward)[v]          = other.flowmaps(forward)[v];
      //flowmap_gradients(forward)[v] = other.flowmap_gradients(forward)[v];
    }
    for (auto const v : m_pointset_backward.vertices()) {
      flowmaps(backward)[v]          = other.flowmaps(backward)[v];
      //flowmap_gradients(backward)[v] = other.flowmap_gradients(backward)[v];
    }
    // TODO create triangulations
  }

 public:
  //============================================================================
  auto num_particles() const -> std::size_t {
    return m_pointset_forward.vertices().size();
  }
  auto write(filesystem::path const &p) const {
    auto const filename      = p.filename().replace_extension("");
    auto       forward_path  = p;
    auto       backward_path = p;
    forward_path.replace_filename(filename.string() + "_forward.vtp");
    backward_path.replace_filename(filename.string() + "_backward.vtp");

    m_pointset_forward.write_vtp(forward_path);
    m_pointset_backward.write_vtp(backward_path);
  }
  auto read(filesystem::path const &p) {
    auto const filename      = p.filename().replace_extension("");
    auto       forward_path  = p;
    auto       backward_path = p;
    forward_path.replace_filename(filename.string() + "_forward.vtp");
    backward_path.replace_filename(filename.string() + "_backward.vtp");

    m_pointset_forward.read_vtp(forward_path);
    m_pointset_backward.read_vtp(backward_path);
    m_flowmaps_forward =
        &m_pointset_forward.template vertex_property<pos_type>("flowmaps");
    //m_flowmap_gradients_forward =
    //    &m_pointset_forward.template vertex_property<gradient_type>(
    //        "flowmap_gradients");
    m_flowmaps_backward =
        &m_pointset_backward.template vertex_property<pos_type>("flowmaps");
    //m_flowmap_gradients_backward =
    //    &m_pointset_backward.template vertex_property<gradient_type>(
    //        "flowmap_gradients");
  }
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap &&flowmap, range auto const &initial_particles,
            arithmetic auto const t_end, arithmetic auto const tau_step,
            std::atomic_uint64_t &uuid_generator) {
    fill(std::forward<Flowmap>(flowmap), initial_particles, t_end, tau_step,
         uuid_generator, std::make_index_sequence<NumDimensions>{});
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap, std::size_t... Is>
  auto fill(Flowmap &&flowmap, range auto const &initial_particles,
            arithmetic auto const t_end, arithmetic auto const tau_step,
            std::atomic_uint64_t &uuid_generator,
            std::index_sequence<Is...> /*seq*/) {
    auto [advected_particles, simple_particles, edges] =
        particle_type::template advect<SplitBehavior>(
            std::forward<Flowmap>(flowmap), tau_step, t_end, initial_particles,
            uuid_generator);
    write_vtp(initial_particles, "initial_particles.vtp");
    write_vtp(advected_particles, "advected_particles.vtp");
    auto advected_t0 =
        std::vector<geometry::hyper_ellipse<real_type, NumDimensions>>{};
    std::ranges::copy(
        advected_particles | std::views::transform([](auto const &p) {
          return p.initial_ellipse();
        }),
        std::back_inserter(advected_t0));
    write_vtp(advected_t0, "advected_t0_particles.vtp");
    auto points_forward = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points_forward.reserve(size(advected_particles));
    auto points_backward = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points_backward.reserve(size(advected_particles));
    for (auto const &p : advected_particles) {
      {
        // forward
        auto v               = m_pointset_forward.insert_vertex(p.x0());
        flowmaps(forward)[v] = p.x();
        //flowmap_gradients(forward)[v] = p.nabla_phi();
        points_forward.emplace_back(cgal_point{m_pointset_forward[v](Is)...},
                                    v);
      }
      {
        // backward
        auto v                = m_pointset_backward.insert_vertex(p.x());
        flowmaps(backward)[v] = p.x0();
        //flowmap_gradients(backward)[v] = *inv(p.nabla_phi());
        points_backward.emplace_back(cgal_point{m_pointset_backward[v](Is)...},
                                     v);
      }
    }

    m_triangulation_forward =
        cgal_triangulation_type{begin(points_forward), end(points_forward)};
    m_triangulation_backward =
        cgal_triangulation_type{begin(points_backward), end(points_backward)};
  }
  //----------------------------------------------------------------------------
 public:
  [[nodiscard]] auto sample(
      pos_type const &q, forward_or_backward_tag auto const direction) const {
    return sample(q, direction, std::make_index_sequence<NumDimensions>{});
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  [[nodiscard]] auto sample(pos_type const                    &q,
                            forward_or_backward_tag auto const direction,
                            std::index_sequence<Is...> /*seq*/) const {
    // coordinates computation
    auto const [result, nnc_per_vertex] = cgal::natural_neighbor_coordinates<
        NumDimensions, typename cgal_triangulation_type::Geom_traits,
        typename cgal_triangulation_type::Triangulation_data_structure>(
        triangulation(direction), cgal_point{q(Is)...});
    auto const success = result.third;
    if (!success) {
      return pos_type::fill(nan<real_type>());
    }
    auto const norm = 1 / result.second;

    auto Z0 = [&] {
      auto sum = pos_type{};
      for (auto const &[cgal_handle, coeff] : nnc_per_vertex) {
        auto const v        = cgal_handle->info();
        auto const lambda_i = coeff * norm;
        sum += lambda_i * flowmaps(direction)[v];
      }
      return sum;
    }();
    return Z0;

    //auto xi = [&] {
    //  auto numerator   = pos_type{};
    //  auto denominator = real_type{};
    //  for (auto const &[cgal_handle, coeff] : nnc_per_vertex) {
    //    auto const  v        = cgal_handle->info();
    //    auto const  lambda_i = coeff * norm;
    //    auto const &p_i      = pointset(direction)[v];
    //    auto const &g_i      = flowmap_gradients(direction)[v];
    //    auto const  xi_i     = [&] {
    //      if constexpr (tensor_rank<pos_type> == 0) {
    //        return flowmaps(direction)[v] + dot(g_i, q - p_i);
    //      } else {
    //        return flowmaps(direction)[v] + transposed(g_i) * (q - p_i);
    //      }
    //    }();
    //    auto const w = lambda_i / euclidean_distance(q, p_i);
    //    numerator += w * xi_i;
    //    denominator += w;
    //  }
    //  return numerator / denominator;
    //}();
    //
    //auto alpha = [&] {
    //  auto numerator   = pos_type{};
    //  auto denominator = real_type{};
    //  for (auto const &[cgal_handle, coeff] : nnc_per_vertex) {
    //    auto const  v        = cgal_handle->info();
    //    auto const  lambda_i = coeff * norm;
    //    auto const &p_i      = pointset(direction)[v];
    //    auto const  w        = lambda_i / squared_euclidean_distance(q, p_i);
    //    numerator += lambda_i;
    //    denominator += w;
    //  }
    //  return numerator / denominator;
    //}();
    //
    //auto beta = [&] {
    //  auto sum = real_type{};
    //  for (auto const &[cgal_handle, coeff] : nnc_per_vertex) {
    //    auto const  v        = cgal_handle->info();
    //    auto const  lambda_i = coeff * norm;
    //    auto const &p_i      = pointset(direction)[v];
    //    sum += lambda_i * squared_euclidean_distance(q, p_i);
    //  }
    //  return sum;
    //}();
    // return (alpha * Z0 + beta * xi) / (alpha + beta);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const              &q,
                  forward_or_backward_tag auto direction) const {
    return sample(q, direction);
  }
};
//==============================================================================
template <std::size_t NumDimensions,
          typename SplitBehavior = typename autonomous_particle<
              real_number, NumDimensions>::split_behaviors::three_splits>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_number, NumDimensions>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -
template <typename SplitBehavior = typename autonomous_particle<
              real_number, 2>::split_behaviors::three_splits>
using autonomous_particle_flowmap_discretization2 =
    AutonomousParticleFlowmapDiscretization<2, SplitBehavior>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -
template <typename SplitBehavior = typename autonomous_particle<
              real_number, 3>::split_behaviors::three_splits>
using autonomous_particle_flowmap_discretization3 =
    AutonomousParticleFlowmapDiscretization<3, SplitBehavior>;
//==============================================================================
}  // namespace tatooine
//==============================================================================

//==============================================================================
// Staggered Discretization
#include <tatooine/staggered_flowmap_discretization.h>
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename SplitBehavior = typename autonomous_particle<
              real_number, NumDimensions>::split_behaviors::three_splits>
using staggered_autonomous_particle_flowmap_discretization =
    staggered_flowmap_discretization<autonomous_particle_flowmap_discretization<
        Real, NumDimensions, SplitBehavior>>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -
template <std::size_t NumDimensions,
          typename SplitBehavior = typename autonomous_particle<
              real_number, NumDimensions>::split_behaviors::three_splits>
using StaggeredAutonomousParticleFlowmapDiscretization =
    staggered_autonomous_particle_flowmap_discretization<
        real_number, NumDimensions, SplitBehavior>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -
template <typename SplitBehavior = typename autonomous_particle<
              real_number, 2>::split_behaviors::three_splits>
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2, SplitBehavior>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -
template <typename SplitBehavior = typename autonomous_particle<
              real_number, 3>::split_behaviors::three_splits>
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3, SplitBehavior>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
