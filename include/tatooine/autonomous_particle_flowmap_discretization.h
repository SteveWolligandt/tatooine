#ifndef TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AUTONOMOUS_PARTICLE_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/unstructured_simplicial_grid.h>
#include <tatooine/staggered_flowmap_discretization.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Derived, typename Real, size_t NumDimensions>
struct autonomous_particle_sampler_hierarchy
    : base_uniform_tree_hierarchy<
          Real, NumDimensions,
          autonomous_particle_sampler_hierarchy<Derived, Real, NumDimensions>> {
  using ellipse_t           = autonomous_particle_sampler<Real, NumDimensions>;
  using ellipse_container_t = std::vector<ellipse_t>;
  using this_t =
      autonomous_particle_sampler_hierarchy<Derived, Real, NumDimensions>;
  using parent_t = base_uniform_tree_hierarchy<Real, NumDimensions, this_t>;
  using real_t   = typename parent_t::real_t;
  using parent_t::center;
  using parent_t::children;
  using parent_t::is_at_max_depth;
  using parent_t::is_inside;
  using parent_t::is_simplex_inside;
  using parent_t::is_splitted;
  using parent_t::max;
  using parent_t::min;
  using parent_t::split_and_distribute;
  using typename parent_t::vec_t;
  using pos_t = vec_t;

  //============================================================================
 private:
  ellipse_container_t const*         m_ellipses = nullptr;
  std::vector<size_t>                m_indices;
  //============================================================================
 public:
  autonomous_particle_sampler_hierarchy() = default;
  autonomous_particle_sampler_hierarchy(
      autonomous_particle_sampler_hierarchy const&) = default;
  autonomous_particle_sampler_hierarchy(
      autonomous_particle_sampler_hierarchy&&) noexcept = default;
  auto operator=(autonomous_particle_sampler_hierarchy const&)
      -> autonomous_particle_sampler_hierarchy& = default;
  auto operator=(autonomous_particle_sampler_hierarchy&&) noexcept
      -> autonomous_particle_sampler_hierarchy&    = default;
  virtual ~autonomous_particle_sampler_hierarchy() = default;
  //----------------------------------------------------------------------------
  explicit autonomous_particle_sampler_hierarchy(
      ellipse_container_t const& ellipses,
      size_t const               max_depth = parent_t::default_max_depth)
      : parent_t{pos_t::ones() * -std::numeric_limits<real_t>::max(),
                 pos_t::ones() * std::numeric_limits<real_t>::max(), max_depth},
        m_ellipses{&ellipses} {
    for (size_t i = 0; i < size(ellipses); ++i) {
      insert_ellipse(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  autonomous_particle_sampler_hierarchy(
      vec_t const& min, vec_t const& max, ellipse_container_t const& ellipses,
      size_t const max_depth = parent_t::default_max_depth)
      : parent_t{min, max, 1, max_depth}, m_ellipses{&ellipses} {
    for (size_t i = 0; i < size(ellipses); ++i) {
      insert_ellipse(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  autonomous_particle_sampler_hierarchy(vec_t const& min, vec_t const& max,
                                        size_t const               level,
                                        size_t const               max_depth,
                                        ellipse_container_t const& ellipses)
      : parent_t{min, max, level, max_depth}, m_ellipses{&ellipses} {}
  //============================================================================
 public:
  auto ellipses() const -> auto const& { return *m_ellipses; }
  auto ellipse_indices_in_node() const -> auto const& { return m_indices; }
  auto ellipse(size_t const i) const -> auto const& {
    return static_cast<Derived const*>(this)->ellipse(i);
  }
  auto constexpr holds_ellipses() const { return !m_indices.empty(); }
  //----------------------------------------------------------------------------
  auto insert_ellipse(size_t const i) -> bool {
    auto const& e = ellipse(i);
    if (!this->is_rectangle_inside(e.center() + e.S() * vec_t{-1, -1},
                                   e.center() + e.S() * vec_t{1, -1},
                                   e.center() + e.S() * vec_t{1, 1},
                                   e.center() + e.S() * vec_t{-1, 1})) {
      return false;
    }
    if (holds_ellipses()) {
      if (is_at_max_depth()) {
        m_indices.push_back(i);
      } else {
        split_and_distribute();
        distribute_ellipse(i);
      }
    } else {
      if (is_splitted()) {
        distribute_ellipse(i);
      } else {
        m_indices.push_back(i);
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto distribute() {
    if (!m_indices.empty()) {
      distribute_ellipse(m_indices.front());
      m_indices.clear();
    }
  }
  //------------------------------------------------------------------------------
  auto construct(vec_t const& min, vec_t const& max, size_t const level,
                 size_t const max_depth) const {
    return std::unique_ptr<this_t>{
        new this_t{min, max, level, max_depth, ellipses()}};
  }
  //----------------------------------------------------------------------------
  auto distribute_ellipse(size_t const i) {
    for (auto& child : children()) {
      child->insert_ellipse(i);
    }
  }
  //----------------------------------------------------------------------------
  auto collect_nearby_ellipses(pos_t const&                pos,
                               std::unordered_set<size_t>& cells) const
      -> void {
    if (is_inside(pos)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_nearby_ellipses(pos, cells);
        }
      } else {
        if (!m_indices.empty()) {
          std::copy(begin(m_indices), end(m_indices),
                    std::inserter(cells, end(cells)));
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_ellipses(pos_t const& pos) const {
    std::unordered_set<size_t> indices;
    collect_nearby_ellipses(pos, indices);
    return indices;
  }
};
//==============================================================================
template <typename Real, size_t NumDimensions>
struct forward_autonomous_particle_sampler_hierarchy
    : autonomous_particle_sampler_hierarchy<
          forward_autonomous_particle_sampler_hierarchy<Real, NumDimensions>,
          Real, NumDimensions> {
  using parent_t = autonomous_particle_sampler_hierarchy<
      forward_autonomous_particle_sampler_hierarchy<Real, NumDimensions>, Real,
      NumDimensions>;
  using parent_t::parent_t;
  auto ellipse(size_t const i) const -> auto const& {
    return this->ellipses().at(i).ellipse1();
  }
};
//==============================================================================
template <typename Real, size_t NumDimensions>
struct backward_autonomous_particle_sampler_hierarchy
    : autonomous_particle_sampler_hierarchy<
          backward_autonomous_particle_sampler_hierarchy<Real, NumDimensions>,
          Real, NumDimensions> {
  using parent_t = autonomous_particle_sampler_hierarchy<
      backward_autonomous_particle_sampler_hierarchy<Real, NumDimensions>, Real,
      NumDimensions>;
  using parent_t::parent_t;
  auto ellipse(size_t const i) const -> auto const& {
    return this->ellipses().at(i).ellipse1();
  }
};
//==============================================================================
template <typename Real, size_t NumDimensions>
struct autonomous_particle_flowmap_discretization {
  using real_t              = Real;
  using vec_t               = vec<Real, NumDimensions>;
  using pos_t               = vec_t;
  using particle_t          = autonomous_particle<Real, NumDimensions>;
  using sampler_t           = autonomous_particle_sampler<Real, NumDimensions>;
  using sampler_container_t = std::vector<sampler_t>;
  using mesh_t              = unstructured_simplicial_grid<Real, NumDimensions>;
  using mesh_prop_t =
      typename mesh_t::template vertex_property_t<sampler_t const*>;
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
 private:
  std::optional<filesystem::path> m_path;
  std::vector<sampler_t>          m_samplers;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers0 = nullptr;
  mutable std::mutex m_centers0_mutex;
  mutable std::unique_ptr<pointset<Real, NumDimensions>> m_centers1 = nullptr;
  mutable std::mutex m_centers1_mutex;
  //============================================================================
 public:
  explicit autonomous_particle_flowmap_discretization(
      filesystem::path const& path)
      : m_path{path} {
    auto         file              = hdf5::file{*m_path};
    auto         particles_on_disk = file.dataset<particle_t>("finished");
    size_t const total_num_particles =
        particles_on_disk.dataspace().current_resolution()[0];

    auto ps = std::vector<particle_t>(total_num_particles);
    particles_on_disk.read(ps);
    m_path = std::nullopt;

    m_samplers.resize(total_num_particles);
#pragma omp parallel for
    for (std::size_t i = 0; i < total_num_particles; ++i) {
      m_samplers[i] = ps[i].sampler();
    }
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      std::vector<autonomous_particle<Real, NumDimensions>> const& initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t0 + tau, tau_step);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::vector<autonomous_particle<Real, NumDimensions>> particles;
    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2);
        });
    auto const small_particle_size =
        (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
                   initial_particle_distribution.dimension(0).spacing()) -
         initial_particle_distribution.dimension(0).spacing()) /
        2;

    std::cerr << small_particle_size << '\n';
    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              small_particle_size);
        });
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
  }
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t0, arithmetic auto const tau,
      arithmetic auto const                                tau_step,
      uniform_rectilinear_grid<Real, NumDimensions> const& g,
      filesystem::path const&                              path)
      : m_path{path} {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    auto initial_particle_distribution = g.copy_without_properties();
    std::vector<autonomous_particle<Real, NumDimensions>> particles;
    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2);
        });
    auto const small_particle_size =
        (std::sqrt(2 * initial_particle_distribution.dimension(0).spacing() *
                   initial_particle_distribution.dimension(0).spacing()) -
         initial_particle_distribution.dimension(0).spacing()) /
        2;

    for (size_t i = 0; i < NumDimensions; ++i) {
      auto const spacing = initial_particle_distribution.dimension(i).spacing();
      initial_particle_distribution.dimension(i).pop_front();
      initial_particle_distribution.dimension(i).front() -= spacing / 2;
      initial_particle_distribution.dimension(i).back() -= spacing / 2;
    }
    initial_particle_distribution.vertices().iterate_indices(
        [&](auto const... is) {
          particles.emplace_back(
              initial_particle_distribution.vertex_at(is...), t0,
              initial_particle_distribution.dimension(0).spacing() / 2
              // small_particle_size
          );
        });
    fill(std::forward<Flowmap>(flowmap), particles, t0 + tau, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const tau_step,
      std::vector<autonomous_particle<Real, NumDimensions>> const&
          initial_particles) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), initial_particles, t1, tau_step);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Flowmap>
  autonomous_particle_flowmap_discretization(
      Flowmap&& flowmap, arithmetic auto const t1,
      arithmetic auto const                           tau_step,
      autonomous_particle<Real, NumDimensions> const& initial_particle) {
    static_assert(
        std::decay_t<Flowmap>::num_dimensions() == NumDimensions,
        "Number of dimensions of flowmap does not match number of dimensions.");
    fill(std::forward<Flowmap>(flowmap), std::vector{initial_particle}, t1,
         tau_step);
  }
  //============================================================================
  auto samplers() const -> auto const& { return m_samplers; }
  //----------------------------------------------------------------------------
  auto hierarchy0() const -> auto const& {
    auto l = std::lock_guard{m_centers0_mutex};
    if (m_centers0 == nullptr) {
      m_centers0 = std::make_unique<pointset<Real, NumDimensions>>();
      for (auto const& sa : m_samplers) {
        m_centers0->insert_vertex(sa.ellipse0().center());
      }
    }
    return m_centers0;
  }
  //----------------------------------------------------------------------------
  auto hierarchy1() const -> auto const& {
    auto l = std::lock_guard{m_centers1_mutex};
    if (m_centers1 == nullptr) {
      m_centers1 = std::make_unique<pointset<Real, NumDimensions>>();
      for (auto const& sa : m_samplers) {
        m_centers1->insert_vertex(sa.ellipse1().center());
      }
    }
    return m_centers1;
  }
  //----------------------------------------------------------------------------
  template <typename Tag>
  auto hierarchy_mutex(Tag) const -> auto & {
    if constexpr (is_same<tag::forward_t, Tag>) {
      return m_centers0_mutex;
    } else if constexpr (is_same<tag::backward_t, Tag>) {
      return m_centers1_mutex;
    }
  }
  //----------------------------------------------------------------------------
  template <typename Tag>
  auto hierarchy(Tag) const -> auto const& {
    if constexpr (is_same<tag::forward_t, Tag>) {
      return hierarchy0();
    } else if constexpr (is_same<tag::backward_t, Tag>) {
      return hierarchy1();
    }
  }
  //============================================================================
  auto num_particles() const -> std::size_t {
    if (m_path) {
      auto file              = hdf5::file{*m_path};
      auto particles_on_disk = file.dataset<particle_t>("finished");
      return particles_on_disk.dataspace().current_resolution()[0];
    } else {
      return size(m_samplers);
    }
  }
 private:
  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto fill(Flowmap&& flowmap, range auto const& initial_particles,
            arithmetic auto const t1, arithmetic auto const tau_step) {
    if (m_path) {
      autonomous_particle<Real, NumDimensions>::advect_with_3_splits(
          std::forward<Flowmap>(flowmap), tau_step, t1, initial_particles,
          *m_path);
    } else {
      auto particles =
          autonomous_particle<Real, NumDimensions>::advect_with_3_splits(
              std::forward<Flowmap>(flowmap), tau_step, t1, initial_particles);
      m_samplers.clear();
      m_samplers.reserve(size(particles));
      std::transform(
          begin(particles), end(particles),
          std::back_inserter(m_samplers),
          [](auto const& p) { return p.sampler();});
    }
  }
  //----------------------------------------------------------------------------
  template <typename Tag, size_t... VertexSeq>
  [[nodiscard]] auto sample(pos_t const& x, Tag const tag,
                            std::index_sequence<VertexSeq...> /*seq*/) const {
    auto         shortest_distance = std::numeric_limits<real_t>::infinity();
    //auto         file              = hdf5::file{*m_path};
    //auto         particles_on_disk = file.dataset<particle_t>("finished");
    sampler_t    nearest_sampler;
    auto const& h = hierarchy(tag);
    std::pair<std::vector<int>, std::vector<Real>> nn;
    {
      auto l = std::lock_guard{hierarchy_mutex(tag)};
      nn     = h->nearest_neighbors_radius_raw(x, 0.1);
    }
    auto const& indices            = nn.first;
    auto const& physical_distances = nn.second;
    static auto m = std::mutex{};
#pragma omp parallel for
    for (size_t j = 0; j < size(indices); ++j) {
      auto const& sampler = m_samplers[indices[j]];
      auto const  local_dist =
          sampler.ellipse(tag).local_squared_euclidean_distance_to_center(x);
      {
        auto       l                 = std::lock_guard{m};
        auto const weighted_distance = physical_distances[j] * local_dist;
        if (weighted_distance < shortest_distance) {
          shortest_distance = weighted_distance;
          nearest_sampler   = sampler;
        }
      }
    }

    if (shortest_distance < std::numeric_limits<real_t>::infinity()) {
      return nearest_sampler.sample(x, tag);
    }
    return pos_t::fill(Real(0) / Real(0));
  }

 public:
  //----------------------------------------------------------------------------
  [[nodiscard]] auto sample_forward(pos_t const& x) const {
    return sample(x, tag::forward,
                  std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::forward_t /*tag*/) const {
    return sample_forward(x);
  }
  //----------------------------------------------------------------------------
  auto sample_backward(pos_t const& x) const {
    return sample(x, tag::backward,
                  std::make_index_sequence<NumDimensions + 1>{});
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_t const& x, tag::backward_t /*tag*/) const {
    return sample_backward(x);
  }
};
//==============================================================================
template <size_t NumDimensions>
using AutonomousParticleFlowmapDiscretization =
    autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
using autonomous_particle_flowmap_discretization_2 =
    AutonomousParticleFlowmapDiscretization<2>;
using autonomous_particle_flowmap_discretization_3 =
    AutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
template <typename Real, size_t NumDimensions>
using staggered_autonomous_particle_flowmap_discretization =
    staggered_flowmap_discretization<
        autonomous_particle_flowmap_discretization<Real, NumDimensions>>;
//------------------------------------------------------------------------------
template <size_t NumDimensions>
using StaggeredAutonomousParticleFlowmapDiscretization =
    staggered_autonomous_particle_flowmap_discretization<real_t, NumDimensions>;
using staggered_autonomous_particle_flowmap_discretization2 =
    StaggeredAutonomousParticleFlowmapDiscretization<2>;
using staggered_autonomous_particle_flowmap_discretization3 =
    StaggeredAutonomousParticleFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
