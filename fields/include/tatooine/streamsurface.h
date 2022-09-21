#ifndef TATOOINE_STREAMSURFACE_H
#define TATOOINE_STREAMSURFACE_H
//==============================================================================
#include <tatooine/bidiagonal_system_solver.h>
#include <tatooine/exceptions.h>
#include <tatooine/for_loop.h>
#include <tatooine/interpolation.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/ode/solver.h>
#include <tatooine/tensor.h>
#include <tatooine/unstructured_triangular_grid.h>

#include <algorithm>
#include <boost/functional.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <limits>
#include <list>
#include <map>
#include <memory>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Streamsurface>
struct hultquist_discretization;
//==============================================================================
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
struct streamsurface {
  using flowmap_type = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() -> std::size_t {
    return flowmap_type::num_dimensions();
  }
  using real_type      = typename flowmap_type::real_type;
  using this_type      = streamsurface<Flowmap, SeedcurveInterpolationKernel>;
  using seedcurve_type = line<real_type, num_dimensions()>;
  using seedcurve_interpolator_type =
      typename seedcurve_type::template vertex_property_sampler_type<
          seedcurve_type, SeedcurveInterpolationKernel>;
  using vec2     = vec<real_type, 2>;
  using pos_type = vec<real_type, num_dimensions()>;
  using vec_type = vec<real_type, num_dimensions()>;

 private:
  Flowmap                     m_flowmap;
  real_type                   m_t0_u0, m_t0_u1;
  seedcurve_type              m_seedcurve;
  seedcurve_interpolator_type m_seedcurve_interpolator;
  real_type                   m_min_u, m_max_u;

  //----------------------------------------------------------------------------
 public:
  streamsurface(convertible_to<Flowmap> auto&& flowmap, arithmetic auto t0u0,
                arithmetic auto t0u1, seedcurve_type const& seedcurve)
      : m_flowmap{std::forward<decltype(flowmap)>(flowmap)},
        m_t0_u0{static_cast<real_type>(t0u0)},
        m_t0_u1{static_cast<real_type>(t0u1)},
        m_seedcurve{seedcurve},
        m_seedcurve_interpolator{
            seedcurve.template sampler<SeedcurveInterpolationKernel>()},
        m_min_u{std::min(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())},
        m_max_u{std::max(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  streamsurface(convertible_to<Flowmap> auto&& flowmap, arithmetic auto t0,
                seedcurve_type const& seedcurve)
      : m_flowmap{std::forward<decltype(flowmap)>(flowmap)},
        m_t0_u0{static_cast<real_type>(t0)},
        m_t0_u1{static_cast<real_type>(t0)},
        m_seedcurve(seedcurve),
        m_seedcurve_interpolator{
            seedcurve.template sampler<SeedcurveInterpolationKernel>()},
        m_min_u{std::min(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())},
        m_max_u{std::max(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  streamsurface(convertible_to<Flowmap> auto&& flowmap,
                seedcurve_type const&          seedcurve)
      : m_flowmap{std::forward<decltype(flowmap)>(flowmap)},
        m_t0_u0{static_cast<real_type>(0)},
        m_t0_u1{static_cast<real_type>(0)},
        m_seedcurve(seedcurve),
        m_seedcurve_interpolator{
            seedcurve.template sampler<SeedcurveInterpolationKernel>()},
        m_min_u{std::min(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())},
        m_max_u{std::max(m_seedcurve.parameterization().front(),
                         m_seedcurve.parameterization().back())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // streamsurface(streamsurface const& other) = default;
  // streamsurface(streamsurface&& other)      = default;
  // streamsurface& operator=(streamsurface const& other) = default;
  // streamsurface& operator=(streamsurface&& other) = default;
  //============================================================================
  auto t0(real_type const u) const {
    return (u - m_seedcurve.parameterization().front()) /
               (m_seedcurve.parameterization().back() -
                m_seedcurve.parameterization().front()) *
               (m_t0_u1 - m_t0_u0) +
           m_t0_u0;
  }
  //----------------------------------------------------------------------------
  auto flowmap() -> auto& { return m_flowmap; }
  auto flowmap() const -> auto const& { return m_flowmap; }
  //----------------------------------------------------------------------------
  auto seedcurve() const -> auto const& { return m_seedcurve; }
  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  auto sample(real_type const u, real_type const v) const -> vec_type {
    if (u < m_min_u || u > m_max_u) {
      throw out_of_domain_error{};
    }
    if (v == t0(u)) {
      return m_seedcurve_interpolator(u);
    }
    try {
      return m_flowmap(m_seedcurve_interpolator(u), t0(u), v);
    } catch (std::exception&) {
      throw out_of_domain_error{};
    }
  }
  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  auto sample(vec2 const& uv) const -> vec_type { return sample(uv(0), uv(1)); }
  //----------------------------------------------------------------------------
  auto distance(vec2 const& uv0, vec2 const& uv1,
                std::size_t num_samples) const {
    auto      step = (uv1 - uv0) / (num_samples - 1);
    real_type d    = 0;
    for (std::size_t i = 0; i < num_samples - 1; ++i) {
      d += tatooine::euclidean_distance(sample(uv0 + step * i),
                                        sample(uv0 + step * (i + 1)));
    }
    return d;
  }
  //----------------------------------------------------------------------------
  auto operator()(real_type u, real_type v) const { return sample(u, v); }
  auto operator()(vec2 const& uv) const { return sample(uv); }
  //----------------------------------------------------------------------------
  template <template <typename>
            typename Discretization = hultquist_discretization,
            typename... Args>
  auto discretize(Args&&... args) {
    return Discretization<this_type>(this, std::forward<Args>(args)...);
  }
  //----------------------------------------------------------------------------
  constexpr auto min_u() const { return m_min_u; }
  constexpr auto max_u() const { return m_max_u; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Flowmap>
streamsurface(Flowmap const&, arithmetic auto u0t0, arithmetic auto u1t0,
              line<typename Flowmap::real_type,
                   Flowmap::num_dimensions()> const& seedcurve)
    -> streamsurface<Flowmap const&, interpolation::linear>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap>
streamsurface(Flowmap const&, arithmetic auto t0,
              line<typename Flowmap::real_type,
                   Flowmap::num_dimensions()> const& seedcurve)
    -> streamsurface<Flowmap const&, interpolation::linear>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap>
streamsurface(
    Flowmap const&,
    line<typename Flowmap::real_type, Flowmap::num_dimensions()> const&)
    -> streamsurface<Flowmap const&, interpolation::linear>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap>
streamsurface(Flowmap&&, arithmetic auto u0t0, arithmetic auto u1t0,
              line<typename Flowmap::real_type,
                   Flowmap::num_dimensions()> const& seedcurve)
    -> streamsurface<std::decay_t<Flowmap>, interpolation::linear>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap>
streamsurface(Flowmap&&, arithmetic auto t0,
              line<typename Flowmap::real_type,
                   Flowmap::num_dimensions()> const& seedcurve)
    -> streamsurface<std::decay_t<Flowmap>, interpolation::linear>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap>
streamsurface(
    Flowmap&&,
    line<typename Flowmap::real_type, Flowmap::num_dimensions()> const&)
    -> streamsurface<std::decay_t<Flowmap>, interpolation::linear>;
//==============================================================================
template <typename Streamsurface>
struct front_evolving_streamsurface_discretization
    : public unstructured_triangular_grid<typename Streamsurface::real_type,
                                          Streamsurface::num_dimensions()> {
  //============================================================================
  // typedefs
  //============================================================================
  static constexpr auto num_dimensions() -> std::size_t {
    return Streamsurface::num_dimensions();
  }
  using real_type = typename Streamsurface::real_type;
  using this_type = front_evolving_streamsurface_discretization<Streamsurface>;
  using parent_type = unstructured_triangular_grid<real_type, num_dimensions()>;
  using parent_type::at;
  using typename parent_type::pos_type;
  using parent_type::operator[];
  using typename parent_type::triangle_handle;
  using typename parent_type::vertex_handle;

  using vec2    = vec<real_type, 2>;
  using uv_type = vec2;
  using uv_property_type =
      typename parent_type::template typed_vertex_property_type<uv_type>;

  using vertex_vec_type           = std::vector<vertex_handle>;
  using vertex_list_type          = std::list<vertex_handle>;
  using vertex_list_iterator_type = typename vertex_list_type::const_iterator;
  using vertex_range_type =
      std::pair<vertex_list_iterator_type, vertex_list_iterator_type>;
  using streamsurface_type = Streamsurface;

  // a front is a list of lists, containing vertices and a range specifing
  // which vertices have to be triangulated from previous front
  using front_type = vertex_list_type;

  //============================================================================
  // members
  //============================================================================
 private:
  streamsurface_type const* m_streamsurface;
  uv_property_type*         m_uv_property;

 public:
  auto streamsurface() const -> auto const& { return *m_streamsurface; }

  //============================================================================
  // ctors
  //============================================================================
  front_evolving_streamsurface_discretization(
      streamsurface_type const* streamsurface)
      : m_streamsurface{streamsurface}, m_uv_property{&insert_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(this_type const& other)
      : parent_type{other},
        m_streamsurface{other.m_streamsurface},
        m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(this_type&& other) noexcept
      : parent_type{std::move(other)},
        m_streamsurface{other.m_streamsurface},
        m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  auto& operator=(this_type const& other) {
    parent_type::operator=(other);
    m_streamsurface = other.m_streamsurface;
    m_uv_property   = &find_uv_prop();
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(this_type&& other) noexcept {
    parent_type::operator=(std::move(other));
    m_streamsurface = other.m_streamsurface;
    m_uv_property   = &find_uv_prop();
    return *this;
  }
  //============================================================================
  // methods
  //============================================================================
 private:
  auto& insert_uv_prop() {
    return this->template vertex_property<uv_type>("uv");
  }
  auto& find_uv_prop() { return this->template vertex_property<uv_type>("uv"); }
  //----------------------------------------------------------------------------
 public:
  auto uv(vertex_handle v) -> auto& { return m_uv_property->at(v); }
  auto uv(vertex_handle v) const -> auto const& { return m_uv_property->at(v); }
  //----------------------------------------------------------------------------
  auto t0(real_type u) const { return streamsurface().t0(u); }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type const& p, uv_type const& p_uv) {
    auto v = parent_type::insert_vertex(p);
    uv(v)  = p_uv;
    return v;
  }
  //----------------------------------------------------------------------------
  auto triangulate_timeline(front_type const& front0,
                            front_type const& front1) {
    auto       new_face_indices = std::vector<triangle_handle>{};
    auto       left0            = begin(front0);
    auto const end0             = end(front0);
    auto       left1            = begin(front1);
    auto const end1             = end(front1);

    // while both lines are not fully traversed
    while (next(left0) != end0 || next(left1) != end1) {
      assert(left0 != end0);
      assert(left1 != end1);
      auto       lower_edge_len = std::numeric_limits<real_type>::max();
      auto       upper_edge_len = std::numeric_limits<real_type>::max();
      auto const right0         = next(left0);
      auto const right1         = next(left1);

      if (next(left0) != end0) {
        lower_edge_len = std::abs(uv(*left1)(0) - uv(*right0)(0));
      }

      if (next(left1) != end1) {
        upper_edge_len = std::abs(uv(*right1)(0) - uv(*left0)(0));
      }

      if (lower_edge_len < upper_edge_len) {
        new_face_indices.push_back(
            this->insert_triangle(*left0, *right0, *left1));
        if (next(left0) != end0) {
          ++left0;
        }

      } else {
        new_face_indices.push_back(
            this->insert_triangle(*left0, *right1, *left1));

        if (next(left1) != end1) {
          ++left1;
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto average_segment_length(vertex_list_type const& vertices) const {
    real_type dist_acc = 0;
    for (auto v = begin(vertices); v != prev(end(vertices)); ++v) {
      dist_acc += euclidean_distance(at(*v), at(*next(v)));
    }
    return dist_acc / (size(vertices) - 1);
  }
  //--------------------------------------------------------------------------
  auto seedcurve_to_front(std::size_t const seedline_resolution) {
    auto front = front_type{};
    for (auto u : linspace{streamsurface().min_u(), streamsurface().max_u(),
                           seedline_resolution}) {
      auto const t0u     = t0(u);
      auto const new_pos = streamsurface()(u, t0u);
      auto const v       = insert_vertex(new_pos, uv_type{u, t0u});
      front.push_back(v);
    }
    return front;
  }
  //--------------------------------------------------------------------------
  auto subdivide(front_type& front, real_type desired_spatial_dist) -> void {
    for (auto v = begin(front); v != prev(end(front)); ++v) {
      auto d = real_type{};
      try {
        d = streamsurface().distance(uv(*v), uv(*next(v)), 5);
      } catch (std::exception&) {
        d = euclidean_distance(at(*v), at(*next(v)));
      }

      auto stop = false;
      while (d > desired_spatial_dist * 1.5) {
        // split front if u distance to small
        auto new_uv = (uv(*v) + uv(*next(v))) * 0.5;
        try {
          auto new_pnt = streamsurface()(new_uv);
          auto new_v   = insert_vertex(new_pnt, uv_type{new_uv(0), new_uv(1)});
          front.insert(next(v), new_v);
          try {
            d = streamsurface().distance(uv(*v), uv(*next(v)), 5);
          } catch (std::exception&) {
            d = euclidean_distance(at(*v), at(*next(v)));
          }
        } catch (std::exception&) {
          stop = true;
          break;
        }
      }
      if (stop) {
        break;
      }
    }
  }
  //---------------------------------------------------------------------------
  void reduce(front_type& front, real_type desired_spatial_dist) {
    if (front.size() >= 3) {
      for (auto v = begin(front); v != prev(end(front), 2); ++v) {
        auto d = [&] {
          try {
            return streamsurface().distance(uv(*v), uv(*next(v, 2)), 7);
          } catch (std::exception&) {
            return euclidean_distance(at(*v), at(*next(v))) +
                   euclidean_distance(at(*next(v)), at(*next(v, 2)));
          }
        };
        while (next(v) != prev(end(front), 2) &&
               d() < desired_spatial_dist * 1.25) {
          front.erase(next(v));
        }
      }
    }
    if (front.size() > 2 &&
        norm(at(*prev(end(front), 1)) - at(*prev(end(front), 2))) <
            desired_spatial_dist * 0.5) {
      front.erase(prev(end(front), 2));
    }
  }
};
//==============================================================================
template <typename Streamsurface>
struct multi_front_evolving_streamsurface_discretization
    : public unstructured_triangular_grid<typename Streamsurface::real_type,
                                          Streamsurface::num_dimensions()> {
  //============================================================================
  // typedefs
  //============================================================================
  static constexpr auto num_dimensions() -> std::size_t {
    return Streamsurface::num_dimensions();
  }
  using real_type = typename Streamsurface::real_type;
  using this_type = multi_front_evolving_streamsurface_discretization<Streamsurface>;
  using parent_type = unstructured_triangular_grid<real_type, num_dimensions()>;
  using parent_type::at;
  using typename parent_type::pos_type;
  using parent_type::operator[];
  using typename parent_type::triangle_handle;
  using typename parent_type::vertex_handle;

  using vec2    = vec<real_type, 2>;
  using uv_type = vec2;
  using uv_property_type =
      typename parent_type::template typed_vertex_property_type<uv_type>;

  using vertex_vec_type           = std::vector<vertex_handle>;
  using vertex_list_type          = std::list<vertex_handle>;
  using vertex_list_iterator_type = typename vertex_list_type::const_iterator;
  using vertex_range_type =
      std::pair<vertex_list_iterator_type, vertex_list_iterator_type>;
  using subfront_type      = std::pair<vertex_list_type, vertex_range_type>;
  using streamsurface_type = Streamsurface;

  // a front is a list of lists, containing vertices and a range specifing
  // which vertices have to be triangulated from previous front
  using front_type = std::list<subfront_type>;

  //============================================================================
  // members
  //============================================================================
 private:
  streamsurface_type const* m_streamsurface;
  std::set<vertex_handle>   m_on_border;
  uv_property_type*         m_uv_property;

 public:
  auto streamsurface() const -> auto const& { return *m_streamsurface; }

  //============================================================================
  // ctors
  //============================================================================
  multi_front_evolving_streamsurface_discretization(
      streamsurface_type const* streamsurface)
      : m_streamsurface{streamsurface}, m_uv_property{&insert_uv_prop()} {}
  //----------------------------------------------------------------------------
  multi_front_evolving_streamsurface_discretization(this_type const& other)
      : parent_type{other},
        m_streamsurface{other.m_streamsurface},
        m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  multi_front_evolving_streamsurface_discretization(this_type&& other) noexcept
      : parent_type{std::move(other)},
        m_streamsurface{other.m_streamsurface},
        m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  auto& operator=(this_type const& other) {
    parent_type::operator=(other);
    m_streamsurface = other.m_streamsurface;
    m_uv_property   = &find_uv_prop();
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(this_type&& other) noexcept {
    parent_type::operator=(std::move(other));
    m_streamsurface = other.m_streamsurface;
    m_uv_property   = &find_uv_prop();
    return *this;
  }
  //============================================================================
  // methods
  //============================================================================
 private:
  auto& insert_uv_prop() {
    return this->template vertex_property<uv_type>("uv");
  }
  auto& find_uv_prop() { return this->template vertex_property<uv_type>("uv"); }
  //----------------------------------------------------------------------------
 public:
  auto& uv(vertex_handle v) { return m_uv_property->at(v); }
  auto uv(vertex_handle v) const -> auto const& { return m_uv_property->at(v); }
  //----------------------------------------------------------------------------
  auto t0(real_type u) const { return streamsurface().t0(u); }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type const& p, uv_type const& p_uv) {
    auto v = parent_type::insert_vertex(p);
    uv(v)  = p_uv;
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type&& p, uv_type const& p_uv) {
    auto v = parent_type::insert_vertex(std::move(p));
    uv(v)  = p_uv;
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type const& p, uv_type&& p_uv) {
    auto v = parent_type::insert_vertex(p);
    uv(v)  = std::move(p_uv);
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_type&& p, uv_type&& p_uv) {
    auto v = parent_type::insert_vertex(std::move(p));
    uv(v)  = std::move(p_uv);
    return v;
  }
  //----------------------------------------------------------------------------
  void triangulate_timeline(front_type const& front) {
    for (auto const& subfront : front) {
      auto        new_face_indices = std::vector<triangle_handle>{};
      auto const& vs               = subfront.first;
      auto [left0, end0]           = subfront.second;
      auto left1                   = begin(vs);
      auto end1                    = end(vs);

      if (left0 == end0) {
        continue;
      }

      // while both lines are not fully traversed
      while (next(left0) != end0 || next(left1) != end1) {
        assert(left0 != end0);
        assert(left1 != end1);
        real_type  lower_edge_len = std::numeric_limits<real_type>::max();
        real_type  upper_edge_len = std::numeric_limits<real_type>::max();
        auto const right0         = next(left0);
        auto const right1         = next(left1);

        if (next(left0) != end0) {
          lower_edge_len = std::abs(uv(*left1)(0) - uv(*right0)(0));
        }

        if (next(left1) != end1) {
          upper_edge_len = std::abs(uv(*right1)(0) - uv(*left0)(0));
        }

        if (lower_edge_len < upper_edge_len) {
          new_face_indices.push_back(
              this->insert_triangle(*left0, *right0, *left1));
          if (next(left0) != end0) {
            ++left0;
          }

        } else {
          new_face_indices.push_back(
              this->insert_triangle(*left0, *right1, *left1));

          if (next(left1) != end1) {
            ++left1;
          }
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  auto seedcurve_to_front(std::size_t seedline_resolution) {
    auto vs = std::vector<vertex_list_type>{};
    vs.emplace_back();
    for (auto u : linspace{streamsurface().min_u(), streamsurface().max_u(),
                           seedline_resolution}) {
      auto const t0u = t0(u);
      // if
      // (this->streamsurface().vectorfield().in_domain(this->streamsurface().seedcurve().sample(u),
      //                                       t0u)) {
      auto const new_pos = streamsurface().sample(u, t0u);
      auto const v       = insert_vertex(std::move(new_pos), uv_type{u, t0u});
      vs.back().push_back(v);
      //} else if (vs.back().size() > 1) {
      //  vs.emplace_back();
      //} else {
      //  vs.back().clear();
      //}
    }
    if (vs.back().size() <= 1) {
      vs.pop_back();
    }
    auto front = front_type{};
    for (auto&& vs : vs) {
      front.emplace_back(std::move(vs), std::pair{begin(vs), end(vs)});
    }
    return front;
  }
  //----------------------------------------------------------------------------
  real_type average_segment_length(subfront_type const& subfront) const {
    return average_segment_length(subfront.first);
  }
  //----------------------------------------------------------------------------
  real_type average_segment_length(vertex_list_type const& vs) const {
    real_type dist_acc = 0;
    for (auto v = begin(vs); v != prev(end(vs)); ++v) {
      dist_acc += norm(at(*v) - at(*next(v)));
    }

    return dist_acc / (vs.size() - 1);
  }
  //--------------------------------------------------------------------------
  void subdivide(front_type& front, real_type desired_spatial_dist) {
    auto find_best_predecessor = [this](auto const v, auto const& pred_range) {
      real_type min_u_dist = std::numeric_limits<real_type>::max();
      auto      best_it    = pred_range.second;
      for (auto it = pred_range.first; it != pred_range.second; ++it) {
        if (real_type u_dist = std::abs(uv(*v)(0) - uv(*it)(0));
            u_dist < min_u_dist) {
          min_u_dist = u_dist;
          best_it    = it;
          if (min_u_dist == 0) {
            break;
          }
        }
      }
      return best_it;
    };

    for (auto subfront = begin(front); subfront != end(front); ++subfront) {
      if (subfront->first.size() > 1) {
        auto& vs         = subfront->first;
        auto& pred_range = subfront->second;

        for (auto v = begin(vs); v != prev(end(vs)); ++v) {
          real_type d;
          try {
            d = streamsurface().distance(uv(*v), uv(*next(v)), 5);
          } catch (std::exception&) {
            d = euclidean_distance(at(*v), at(*next(v)));
          }

          bool stop = false;
          while (d > desired_spatial_dist * 1.5) {
            // split front if u distance to small
            real_type u_dist = std::abs(uv(*v)(0) - uv(*next(v))(0));
            if (u_dist < 1e-5) {
              auto const best_predecessor = find_best_predecessor(v, pred_range);
              front.emplace(
                  next(subfront), vertex_list_type{next(v), end(vs)},
                  vertex_range_type{next(best_predecessor), pred_range.second});
              pred_range.second = next(best_predecessor);
              vs.erase(next(v), end(vs));
              stop = true;
              break;
            }

            auto new_uv = (uv(*v) + uv(*next(v))) * 0.5;
            try {
              auto new_pnt = streamsurface()(new_uv);
              auto new_v =
                  insert_vertex(new_pnt, uv_type{new_uv(0), new_uv(1)});
              vs.insert(next(v), new_v);
              try {
                d = streamsurface().distance(uv(*v), uv(*next(v)), 5);
              } catch (std::exception&) {
                d = euclidean_distance(at(*v), at(*next(v)));
              }
            } catch (std::exception&) {
              if (next(v, 2) != end(vs)) {
                auto const best_predecessor = find_best_predecessor(v, pred_range);
                // front.emplace(next(subfront), vertex_list_type{next(v),
                // end(vs)},
                //              vertex_range_type{next(best_predecessor),
                //              pred_range.second});
                pred_range.second = next(best_predecessor);
                vs.erase(next(v), end(vs));
              }
              stop = true;
              break;
            }
          }
          if (stop) {
            break;
          }
        }
      }
    }
  }
  //---------------------------------------------------------------------------
  void reduce(front_type& front, real_type desired_spatial_dist) {
    for (auto& subfront : front) {
      auto& vs = subfront.first;
      if (vs.size() >= 3) {
        for (auto v = begin(vs); v != prev(end(vs), 2); ++v) {
          // if (m_on_border.find(*v) != end(m_on_border)) {
          //   continue;
          // }
          auto d = [&] {
            try {
              return streamsurface().distance(uv(*v), uv(*next(v, 2)), 7);
            } catch (std::exception&) {
              return euclidean_distance(at(*v), at(*next(v))) +
                     euclidean_distance(at(*next(v)), at(*next(v, 2)));
            }
          };
          while (next(v) != prev(end(vs), 2) &&
                 d() < desired_spatial_dist * 1.25) {
            vs.erase(next(v));
          }
        }
      }
      if (vs.size() > 2 && norm(at(*prev(end(vs), 1)) - at(*prev(end(vs), 2))) <
                               desired_spatial_dist * 0.5) {
        vs.erase(prev(end(vs), 2));
      }
    }
  }
};
//==============================================================================
template <typename Streamsurface>
struct naive_discretization
    : front_evolving_streamsurface_discretization<Streamsurface> {
  using real_type = typename Streamsurface::real_type;
  static constexpr auto num_dimensions() -> std::size_t {
    return Streamsurface::num_dimensions();
  }
  using parent_type =
      front_evolving_streamsurface_discretization<Streamsurface>;
  using front_type         = typename parent_type::front_type;
  using subfront_type      = typename parent_type::subfront_type;
  using streamsurface_type = typename parent_type::streamsurface_type;
  using vertex_vec_type    = typename parent_type::vertex_vec_type;
  using vertex_list_type   = typename parent_type::vertex_list_type;
  using vertex_handle      = typename parent_type::vertex_handle;
  using triangle_handle    = typename parent_type::triangle_handle;
  using vertex_list_iterator_type =
      typename parent_type::vertex_list_iterator_type;
  using vertex_range_type = typename parent_type::vertex_range_type;
  using parent_type::at;
  using parent_type::insert_vertex;
  using parent_type::t0;
  using parent_type::uv;
  using parent_type::streamsurface;
  //============================================================================
  naive_discretization(naive_discretization const& other)            = default;
  naive_discretization(naive_discretization&& other) noexcept        = default;
  naive_discretization& operator=(naive_discretization const& other) = default;
  naive_discretization& operator=(naive_discretization&& other) noexcept =
      default;
  ~naive_discretization() = default;
  //============================================================================
  naive_discretization(streamsurface_type* streamsurface,
                       std::size_t seedline_resolution, real_type stepsize,
                       real_type backward_tau, real_type forward_tau)
      : parent_type{streamsurface} {
    assert(forward_tau >= 0);
    assert(backward_tau <= 0);

    auto const seed_front = this->seedcurve_to_front(seedline_resolution);
    if (seed_front.empty()) {
      return;
    }

    if (backward_tau < 0) {
      auto      cur_stepsize  = stepsize;
      auto      cur_front     = seed_front;
      real_type advected_time = 0;
      while (advected_time > backward_tau) {
        if (advected_time - cur_stepsize < backward_tau) {
          cur_stepsize = std::abs(backward_tau - advected_time);
        }
        cur_front = evolve(cur_front, -cur_stepsize);
        advected_time -= cur_stepsize;
      }
    }

    if (forward_tau > 0) {
      auto      cur_stepsize  = stepsize;
      auto      cur_front     = seed_front;
      real_type advected_time = 0;
      while (advected_time < forward_tau) {
        if (advected_time + cur_stepsize > forward_tau) {
          cur_stepsize = forward_tau - advected_time;
        }
        cur_front = evolve(cur_front, cur_stepsize);
        advected_time += cur_stepsize;
      }
    }
  }

  //============================================================================
  auto evolve(front_type const& front, real_type const step) {
    auto advected_front          = front;
    auto& [vertices, range] = advected_front.front();
    range.first             = begin(front.front().first);
    range.second            = end(front.front().first);

    for (auto& v : vertices) {
      auto const& uv = parent_type::uv(v);
      vec const   new_uv{uv(0), uv(1) + step};
      auto        new_pos = streamsurface()(new_uv);

      v = insert_vertex(new_pos, {uv(0), uv(1) + step});
    }

    this->triangulate_timeline(front, advected_front);
    return advected_front;
  }
};
//==============================================================================
template <typename Streamsurface>
struct hultquist_discretization
    : front_evolving_streamsurface_discretization<Streamsurface> {
  using real_type = typename Streamsurface::real_type;
  using this_type = hultquist_discretization<Streamsurface>;
  using parent_type =
      front_evolving_streamsurface_discretization<Streamsurface>;
  using parent_type::at;
  using parent_type::insert_vertex;
  using parent_type::t0;
  using parent_type::uv;
  using parent_type::streamsurface;
  using typename parent_type::front_type;
  using typename parent_type::streamsurface_type;
  using typename parent_type::triangle_handle;
  using typename parent_type::uv_type;
  using typename parent_type::vertex_handle;
  using typename parent_type::vertex_list_iterator_type;
  using typename parent_type::vertex_list_type;
  using typename parent_type::vertex_range_type;
  using typename parent_type::vertex_vec_type;
  //----------------------------------------------------------------------------
  hultquist_discretization(streamsurface_type* streamsurface,
                           std::size_t seedline_resolution, real_type stepsize,
                           real_type backward_tau, real_type forward_tau)
      : parent_type(streamsurface) {
    assert(forward_tau >= 0);
    assert(backward_tau <= 0);

    auto const seed_front = this->seedcurve_to_front(seedline_resolution);
    if (seed_front.size() <= 1) {
      return;
    }
    auto const desired_spatial_dist = this->average_segment_length(seed_front);

    if (backward_tau < 0) {
      auto cur_stepsize  = stepsize;
      auto cur_front     = seed_front;
      auto advected_time = real_type{};
      while (advected_time > backward_tau) {
        if (advected_time - cur_stepsize < backward_tau) {
          cur_stepsize = std::abs(backward_tau - advected_time);
        }
        cur_front = evolve(cur_front, -cur_stepsize, desired_spatial_dist);
        advected_time -= cur_stepsize;
      }
    }

    if (forward_tau > 0) {
      auto cur_stepsize  = stepsize;
      auto cur_front     = seed_front;
      auto advected_time = real_type{};
      while (advected_time < forward_tau) {
        if (advected_time + cur_stepsize > forward_tau) {
          cur_stepsize = forward_tau - advected_time;
        }
        cur_front = evolve(cur_front, cur_stepsize, desired_spatial_dist);
        advected_time += cur_stepsize;
      }
    }
  }
  //----------------------------------------------------------------------------
  hultquist_discretization(this_type const& other)                    = default;
  hultquist_discretization(this_type&& other) noexcept                = default;
  auto operator=(this_type const& other) -> hultquist_discretization& = default;
  auto operator=(this_type&& other) noexcept
      -> hultquist_discretization&                                = default;
  ~hultquist_discretization()                                     = default;
  //============================================================================
  auto advect(front_type front, real_type step) {
    assert(step != 0);
    for (auto& v :front) {
      auto const& uv     = parent_type::uv(v);
      auto const  new_uv = vec{uv(0), uv(1) + step};
      auto new_pos = streamsurface()(new_uv);
      v = insert_vertex(new_pos, uv_type{new_uv(0), new_uv(1)});
    }
    return front;
  }

  //--------------------------------------------------------------------------
  auto evolve(front_type const& front, real_type step,
              real_type /*desired_spatial_dist*/) {
    auto advected_front = advect(front, step);

    //this->subdivide(advected_front, desired_spatial_dist);
    //this->reduce(advected_front, desired_spatial_dist);
    if (step > 0) {
      this->triangulate_timeline(front, advected_front);
    } else {
      this->triangulate_timeline(advected_front, front);
    }
    return advected_front;
  }
};

//==============================================================================
// template <typename Flowmap,
//          template <typename> typename SeedcurveInterpolationKernel>
// struct schulze_discretization : multi_front_evolving_streamsurface_discretization<
//                                    Flowmap, SeedcurveInterpolationKernel> {
//  using real_type = typename Flowmap::real_type;
//  static constexpr auto num_dimensions() -> std::size_t { return Flowmap::num_dimensions(); }
//  using parent_type =
//      multi_front_evolving_streamsurface_discretization<Flowmap,
//                                                  SeedcurveInterpolationKernel>;
//  using typename parent_type::triangle_handle;
//  using typename parent_type::front_type;
//  using typename parent_type::streamsurface_type;
//  using typename parent_type::subfront_type;
//  using typename parent_type::vertex_handle;
//  using typename parent_type::vertex_list_iterator_type;
//  using typename parent_type::vertex_list_type;
//  template <typename T>
//  using vertex_property_t = typename parent_type::template
//  vertex_property_t<T>; using parent_type::at; using
//  parent_type::insert_vertex; using parent_type::uv;
//  using parent_type::streamsurface;
//
//  vertex_property_t<real_type>& alpha_prop;
//  vertex_property_t<real_type>& second_derivate_alpha_prop;
//
//  //----------------------------------------------------------------------------
//  schulze_discretization(streamsurface_type* streamsurface, std::size_t
//  seedline_resolution,
//                         std::size_t num_iterations)
//      : parent_type(streamsurface),
//        alpha_prop(this->template add_vertex_property<real_type>("alpha")),
//        second_derivate_alpha_prop(
//            this->template add_vertex_property<real_type>(
//                "second_derivative_alpha")) {
//    auto const initial_front = this->seedcurve_to_front(seedline_resolution);
//    real_type  desired_spatial_dist =
//        this->average_segment_length(initial_front.front());
//
//    // evolve front
//    front_type cur_front = initial_front;
//    for (std::size_t i = 0; i < num_iterations; ++i) {
//      cur_front = evolve(cur_front, desired_spatial_dist);
//    }
//  }
//
//  //--------------------------------------------------------------------------
//  auto advect(subfront_type const& front) {
//    std::vector<subfront_type> new_subfronts{
//        {{}, {begin(front.first), end(front.first)}}};
//    auto const& [vs, pred_range] = front;
//    // advect each subfront
//    auto alpha = optimal_stepsizes(vs);
//    for (auto [v, i] = std::pair{begin(vs), std::size_t(0)}; v != end(vs);
//         ++v, ++i) {
//      alpha_prop[*v] = alpha[i];
//    }
//    auto splitted_front_ranges = detect_peaks(alpha, vs);
//    // no rip needed
//    if (size(splitted_front_ranges) == 1 &&
//        splitted_front_ranges[0].first == begin(vs) &&
//        splitted_front_ranges[0].second == end(vs)) {
//      auto& vertices1 = new_subfronts
//                            .emplace_back(std::list<vertex_handle>{},
//                                          std::pair{begin(vs), end(vs)})
//                            .first;
//
//      std::size_t i = 0;
//      for (auto const v : vs) {
//        auto const& uv = parent_type::uv(v);
//        vec2        new_uv{uv(0), uv(1) + alpha[i++]};
//        auto        new_pos = streamsurface()(new_uv);
//        vertices1.push_back(insert_vertex(new_pos, new_uv));
//        alpha_prop[v] = alpha[i - 1];
//      }
//
//    } else {
//      // rip needed
//      for (auto const& range : splitted_front_ranges) {
//        auto& [vertices1, pred_range] =
//            new_subfronts.emplace_back(std::list<vertex_handle>{}, range);
//
//        std::size_t                   i = 0;
//        std::list<vertex_handle> sub_front;
//        std::copy(pred_range.first, pred_range.second,
//                  std::back_inserter(sub_front));
//        auto sub_alpha = optimal_stepsizes(sub_front);
//        for (auto v = pred_range.first; v != pred_range.second; ++v) {
//          auto const& uv = parent_type::uv(*v);
//          vec2        new_uv{uv(0), uv(1) + sub_alpha[i++]};
//          auto        new_pos = streamsurface()(new_uv);
//          vertices1.push_back(insert_vertex(new_pos, new_uv));
//          alpha_prop[vertices1.back()] = sub_alpha[i - 1];
//        }
//      }
//    }
//    return new_subfronts;
//  }
//  //--------------------------------------------------------------------------
//  auto advect(front_type const& front) {
//    front_type advected_front;
//    for (auto const& subfront : front) {
//      if (subfront.first.size() > 1) {
//        boost::copy(advect(subfront),
//        std::back_inserter(advected_front));
//      }
//    }
//    return advected_front;
//  }
//  //----------------------------------------------------------------------------
//  auto evolve(front_type const& front, real_type desired_spatial_dist) {
//    auto advected_front = advect(front);
//    // triangulate
//    std::vector<std::vector<triangle_handle>> faces;
//    this->subdivide(advected_front, desired_spatial_dist);
//    this->reduce(advected_front, desired_spatial_dist);
//    this->triangulate_timeline(advected_front);
//    return advected_front;
//  }
//  //----------------------------------------------------------------------------
//  std::vector<real_type> optimal_stepsizes(vertex_list_type const& vs) {
//    auto const& v        = streamsurface().flowmap().vectorfield();
//    auto        jacobian = diff(v, 1e-7);
//
//    auto                   num_pnts = size(vs);
//    std::vector<real_type> p(num_pnts - 1), q(num_pnts - 1), null(num_pnts),
//        r(num_pnts);
//    std::vector<vec<real_type, num_dimensions()>> ps(num_pnts);
//
//    // TODO: get t0 at u, not at 0
//    std::size_t i = 0;
//    for (auto const vertex_handle : vs)
//      ps[i++] = v(at(vertex_handle), this->t0(0) + uv(vertex_handle)(1));
//
//    i                 = 0;
//    real_type avg_len = 0;
//    for (auto vertex_handle = begin(vs); vertex_handle != prev(end(vs));
//         ++vertex_handle, ++i) {
//      auto tm = this->t0(0) +
//                (uv(*vertex_handle)(1) + uv(*next(vertex_handle))(1)) * 0.5;
//      auto xm  = (at(*next(vertex_handle)) + at(*vertex_handle)) * 0.5;
//      auto dir = at(*next(vertex_handle)) - at(*vertex_handle);
//      auto vm  = v(xm, tm);
//      auto Jm  = jacobian(xm, tm);
//
//      p[i] = dot(dir * 0.5, Jm * ps[i]) - dot(ps[i], vm);
//      q[i] = dot(dir * 0.5, Jm * ps[i + 1]) + dot(ps[i + 1], vm);
//      r[i] = -dot(dir, vm);
//
//      avg_len += norm(dir);
//    }
//    avg_len /= num_pnts - 1;
//    solve_qr(num_pnts - 1, &p[0], &q[0], &r[0], &null[0]);
//
//    // real_type nrm{0};
//    // for (auto x : null) nrm += x * x;
//    // nrm = sqrt(nrm);
//    // for (std::size_t i = 0; i < null.size(); ++i) null[i] /= nrm;
//
//    // count positive entries in nullspace
//    std::size_t num_pos_null = 0;
//    for (auto c : null)
//      if (c > 0) ++num_pos_null;
//    int k_plus_factor = (num_pos_null < null.size() / 2) ? -1 : 1;
//
//    for (std::size_t i = 0; i < r.size(); ++i)
//      r[i] += k_plus_factor * null[i];
//    // r[i] += (num_pnts / 10.0) * k_plus_factor * null[i];
//
//    // apply step width
//    real_type h = std::numeric_limits<real_type>::max();
//    for (std::size_t i = 0; i < num_pnts; ++i)
//      h = std::min(h, avg_len / (std::abs(r[i]) * norm(ps[i])));
//    for (std::size_t i = 0; i < r.size(); ++i)
//      r[i] *= h;
//    // for (std::size_t i = 0; i < r.size(); ++i) r[i] = std::max(r[i],1e-3);
//    return r;
//  }
//
//  //----------------------------------------------------------------------------
//  auto detect_peaks(std::vector<real_type> const& alpha,
//                    vertex_list_type const& vs, real_type threshold = 100) {
//    // calculate second derivative
//    std::vector<real_type> snd_der(alpha.size(), 0);
//    auto                   v = next(begin(vs));
//    for (std::size_t i = 1; i < alpha.size() - 1; ++i, ++v) {
//      mat<real_type, 3, 3> A{
//          {real_type(1), uv(*prev(v))(0), uv(*prev(v))(0) * uv(*prev(v))(0)},
//          {real_type(1), uv(*v)(0), uv(*v)(0) * uv(*v)(0)},
//          {real_type(1), uv(*next(v))(0), uv(*next(v))(0) * uv(*next(v))(0)}};
//      vec<real_type, 3> b{alpha[i - 1], alpha[i], alpha[i + 1]};
//      snd_der[i]                     = 2 * solve(A, b)(2);
//      second_derivate_alpha_prop[*v] = std::abs(snd_der[i]);
//    }
//
//    std::vector<std::pair<vertex_list_iterator_type,
//    vertex_list_iterator_type>>
//        splitted_front_ranges;
//    splitted_front_ranges.emplace_back(begin(vs), end(vs));
//
//    auto v_it = next(begin(vs));
//    for (std::size_t i = 1; i < snd_der.size() - 1; ++i, ++v_it)
//      if (std::abs(snd_der[i]) > threshold) {
//        if (splitted_front_ranges.back().first == v_it)
//          // shift left border of split to the right
//          ++splitted_front_ranges.back().first;
//        else {
//          // insert new subfront
//          splitted_front_ranges.back().second = next(v_it);
//          if (splitted_front_ranges.back().first ==
//              splitted_front_ranges.back().second)
//            splitted_front_ranges.pop_back();
//
//          splitted_front_ranges.emplace_back(next(v_it), end(vs));
//        }
//      }
//    if (splitted_front_ranges.back().first ==
//        splitted_front_ranges.back().second) {
//      splitted_front_ranges.pop_back();
//    }
//    return splitted_front_ranges;
//  }
//};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
