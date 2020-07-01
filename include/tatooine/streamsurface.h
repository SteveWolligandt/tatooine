#ifndef TATOOINE_STREAMSURFACE_H
#define TATOOINE_STREAMSURFACE_H
//==============================================================================
#include <tatooine/bidiagonal_system_solver.h>
#include <tatooine/exceptions.h>
#include <tatooine/for_loop.h>
#include <tatooine/interpolation.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/ode/solver.h>
#include <tatooine/simple_tri_mesh.h>
#include <tatooine/tensor.h>

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
template <typename, template <typename> typename>
struct hultquist_discretization;
//==============================================================================
template <flowmap_c Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
struct streamsurface {
  using flowmap_t = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using real_t      = typename flowmap_t::real_t;
  using this_t      = streamsurface<Flowmap, SeedcurveInterpolationKernel>;
  using seedcurve_t = parameterized_line<real_t, num_dimensions(),
                                         SeedcurveInterpolationKernel>;
  using vec2        = vec<real_t, 2>;
  using pos_t       = vec<real_t, num_dimensions()>;
  using vec_t       = vec<real_t, num_dimensions()>;

 private:
  Flowmap     m_flowmap;
  real_t      m_t0_u0, m_t0_u1;
  seedcurve_t m_seedcurve;
  real_t      m_min_u, m_max_u;

  //----------------------------------------------------------------------------
 public:
  template <flowmap_c _Flowmap>
  streamsurface(_Flowmap&& flowmap, real_number auto t0u0,
                real_number auto t0u1, const seedcurve_t& seedcurve)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_t0_u0{static_cast<real_t>(t0u0)},
        m_t0_u1{static_cast<real_t>(t0u1)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <flowmap_c _Flowmap>
  streamsurface(_Flowmap&& flowmap, real_number auto t0,
                const seedcurve_t& seedcurve)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_t0_u0{static_cast<real_t>(t0)},
        m_t0_u1{static_cast<real_t>(t0)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  template <flowmap_c _Flowmap>
  streamsurface(_Flowmap&& flowmap, const seedcurve_t& seedcurve)
      : m_flowmap{std::forward<_Flowmap>(flowmap)},
        m_t0_u0{static_cast<real_t>(0)},
        m_t0_u1{static_cast<real_t>(0)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  template <typename V>
  streamsurface(const vectorfield<V, real_t, num_dimensions()>& v,
                real_number auto t0u0, real_number auto t0u1,
                const seedcurve_t& seedcurve)
      : m_flowmap{tatooine::flowmap(v)},
        m_t0_u0{static_cast<real_t>(t0u0)},
        m_t0_u1{static_cast<real_t>(t0u1)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename V>
  streamsurface(const vectorfield<V, real_t, num_dimensions()>& v,
                real_number auto t0, const seedcurve_t& seedcurve)
      : m_flowmap{tatooine::flowmap(v)},
        m_t0_u0{static_cast<real_t>(t0)},
        m_t0_u1{static_cast<real_t>(t0)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename V>
  streamsurface(const vectorfield<V, real_t, num_dimensions()>& v,
                const seedcurve_t& seedcurve)
      : m_flowmap{tatooine::flowmap(v)},
        m_t0_u0{real_t(0)},
        m_t0_u1{real_t(0)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  streamsurface(const streamsurface& other) = default;
  streamsurface(streamsurface&& other)      = default;
  streamsurface& operator=(const streamsurface& other) = default;
  streamsurface& operator=(streamsurface&& other) = default;
  //============================================================================
  auto t0(real_t u) const {
    return (u - m_seedcurve.front_parameterization()) /
               (m_seedcurve.back_parameterization() -
                m_seedcurve.front_parameterization()) *
               (m_t0_u1 - m_t0_u0) +
           m_t0_u0;
  }
  //----------------------------------------------------------------------------
  auto&       flowmap() { return m_flowmap; }
  const auto& flowmap() const { return m_flowmap; }
  //----------------------------------------------------------------------------
  const auto& seedcurve() const { return m_seedcurve; }
  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  vec_t sample(real_t u, real_t v) const {
    if (u < m_min_u || u > m_max_u) { throw out_of_domain_error{}; }
    if (v == t0(u)) { return m_seedcurve.sample(u); }
    try {
      return m_flowmap(m_seedcurve.sample(u), t0(u), v);
    } catch (std::exception&) { throw out_of_domain_error{}; }
  }
  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  vec_t sample(const vec2& uv) const { return sample(uv(0), uv(1)); }
  //----------------------------------------------------------------------------
  auto distance(const vec2& uv0, const vec2& uv1, size_t num_samples) const {
    auto   step = (uv1 - uv0) / (num_samples - 1);
    real_t d    = 0;
    for (size_t i = 0; i < num_samples - 1; ++i) {
      d += tatooine::distance(sample(uv0 + step * i),
                              sample(uv0 + step * (i + 1)));
    }
    return d;
  }
  //----------------------------------------------------------------------------
  auto operator()(real_t u, real_t v) const { return sample(u, v); }
  //----------------------------------------------------------------------------
  template <template <typename, template <typename> typename>
            typename Discretization = hultquist_discretization,
            typename... Args>
  auto discretize(Args&&... args) {
    return Discretization<Flowmap, SeedcurveInterpolationKernel>(
        this, std::forward<Args>(args)...);
  }
  //----------------------------------------------------------------------------
  constexpr auto min_u() const { return m_min_u; }
  constexpr auto max_u() const { return m_max_u; }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename V, floating_point Real, size_t N,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(
    const vectorfield<V, Real, N>& v, real_number auto u0t0,
    real_number auto                                                 u1t0,
    const parameterized_line<Real, N, SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<std::decay_t<decltype(flowmap(v))>,
                     SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, floating_point Real, size_t N,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(const vectorfield<V, Real, N>& v, real_number auto t0,
              const parameterized_line<typename V::real_t, N,
                                       SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<std::decay_t<decltype(flowmap(v))>,
                     SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, floating_point Real, size_t N,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(
    const vectorfield<V, Real, N>&                                   v,
    const parameterized_line<Real, N, SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<std::decay_t<decltype(flowmap(v))>,
                     SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap const&, real_number auto u0t0, real_number auto u1t0,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<Flowmap const&, SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap const&, real_number auto t0,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<Flowmap const&, SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap const&,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>&)
    -> streamsurface<Flowmap const&, SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap&&, real_number auto u0t0, real_number auto u1t0,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<std::decay_t<Flowmap>, SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap&&, real_number auto t0,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>& seedcurve)
    -> streamsurface<std::decay_t<Flowmap>, SeedcurveInterpolationKernel>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
streamsurface(Flowmap&&,
              const parameterized_line<typename Flowmap::real_t,
                                       Flowmap::num_dimensions(),
                                       SeedcurveInterpolationKernel>&)
    -> streamsurface<std::decay_t<Flowmap>, SeedcurveInterpolationKernel>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// template <typename V, typename Real, size_t N, typename T0Real,
//          typename GridReal, typename... Args>
// streamsurface(const field<V, Real, N, N>& v, T0Real t0,
//              parameterized_line<Real, N>&& seedcurve, Args&&... args)
//    ->streamsurface<TATOOINE_DEFAULT_INTEGRATOR, interpolation::hermite,
//                    interpolation::linear, V, Real, N>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <typename V, typename Real, size_t N, typename T0Real,
//          typename GridReal, typename... Args>
// streamsurface(const field<V, Real, N, N>&, T0Real,
//              const grid_edge<GridReal, N>&, Args&&... args)
//    ->streamsurface<TATOOINE_DEFAULT_INTEGRATOR, interpolation::linear,
//                    interpolation::linear, V, Real, N>;

//==============================================================================
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
struct front_evolving_streamsurface_discretization
    : public simple_tri_mesh<typename Flowmap::real_t,
                             Flowmap::num_dimensions()> {
  //============================================================================
  // typedefs
  //============================================================================
  static constexpr auto num_dimensions() { return Flowmap::num_dimensions(); }
  using real_t = typename Flowmap::real_t;
  using this_t =
      front_evolving_streamsurface_discretization<Flowmap,
                                                  SeedcurveInterpolationKernel>;
  using parent_t = simple_tri_mesh<real_t, num_dimensions()>;
  using parent_t::at;
  using typename parent_t::pos_t;
  using parent_t::operator[];
  using typename parent_t::face;
  using typename parent_t::vertex;

  using vec2          = vec<real_t, 2>;
  using uv_t          = vec2;
  using uv_property_t = typename parent_t::template vertex_property_t<uv_t>;

  using vertex_vec_t     = std::vector<vertex>;
  using vertex_list_t    = std::list<vertex>;
  using vertex_list_it_t = typename vertex_list_t::const_iterator;
  using vertex_range_t   = std::pair<vertex_list_it_t, vertex_list_it_t>;
  using subfront_t       = std::pair<vertex_list_t, vertex_range_t>;
  using ssf_t            = streamsurface<Flowmap, SeedcurveInterpolationKernel>;

  // a front is a list of lists, containing vertices and a range specifing
  // which vertices have to be triangulated from previous front
  using front_t = std::list<subfront_t>;

  //============================================================================
  // members
  //============================================================================
  ssf_t*           ssf;
  std::set<vertex> m_on_border;
  uv_property_t*   m_uv_property;

  //============================================================================
  // ctors
  //============================================================================
  front_evolving_streamsurface_discretization(ssf_t* _ssf)
      : ssf{_ssf}, m_uv_property{&add_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(const this_t& other)
      : parent_t{other}, ssf{other.ssf}, m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(this_t&& other) noexcept
      : parent_t{std::move(other)},
        ssf{other.ssf},
        m_uv_property{&find_uv_prop()} {}
  //----------------------------------------------------------------------------
  auto& operator=(const this_t& other) {
    parent_t::operator=(other);
    ssf               = other.ssf;
    m_uv_property     = &find_uv_prop();
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(this_t&& other) noexcept {
    parent_t::operator=(std::move(other));
    ssf               = other.ssf;
    m_uv_property     = &find_uv_prop();
    return *this;
  }
  //============================================================================
  // methods
  //============================================================================
 private:
  auto& add_uv_prop() { return this->template add_vertex_property<uv_t>("uv"); }
  auto& find_uv_prop() { return this->template vertex_property<uv_t>("uv"); }
  //----------------------------------------------------------------------------
 public:
  auto&       uv(vertex v) { return m_uv_property->at(v); }
  const auto& uv(vertex v) const { return m_uv_property->at(v); }
  //----------------------------------------------------------------------------
  auto t0(real_t u) const { return ssf->t0(u); }
  //----------------------------------------------------------------------------
  auto insert_vertex(const pos_t& p, const uv_t& p_uv) {
    auto v = parent_t::insert_vertex(p);
    uv(v)  = p_uv;
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& p, const uv_t& p_uv) {
    auto v = parent_t::insert_vertex(std::move(p));
    uv(v)  = p_uv;
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(const pos_t& p, uv_t&& p_uv) {
    auto v = parent_t::insert_vertex(p);
    uv(v)  = std::move(p_uv);
    return v;
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(pos_t&& p, uv_t&& p_uv) {
    auto v = parent_t::insert_vertex(std::move(p));
    uv(v)  = std::move(p_uv);
    return v;
  }
  //----------------------------------------------------------------------------
  void triangulate_timeline(const front_t& front) {
    for (const auto& subfront : front) {
      std::vector<face> new_face_indices;
      const auto&       vs = subfront.first;
      auto [left0, end0]   = subfront.second;
      auto left1           = begin(vs);
      auto end1            = end(vs);

      if (left0 == end0) { continue; }

      // while both lines are not fully traversed
      while (next(left0) != end0 || next(left1) != end1) {
        assert(left0 != end0);
        assert(left1 != end1);
        real_t     lower_edge_len = std::numeric_limits<real_t>::max();
        real_t     upper_edge_len = std::numeric_limits<real_t>::max();
        const auto right0         = next(left0);
        const auto right1         = next(left1);

        if (next(left0) != end0) {
          lower_edge_len = std::abs(uv(*left1)(0) - uv(*right0)(0));
        }

        if (next(left1) != end1) {
          upper_edge_len = std::abs(uv(*right1)(0) - uv(*left0)(0));
        }

        if (lower_edge_len < upper_edge_len) {
          new_face_indices.push_back(
              this->insert_face(*left0, *right0, *left1));
          if (next(left0) != end0) { ++left0; }

        } else {
          new_face_indices.push_back(
              this->insert_face(*left0, *right1, *left1));

          if (next(left1) != end1) { ++left1; }
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  auto seedcurve_to_front(size_t seedline_resolution) {
    std::vector<vertex_list_t> vs;
    vs.emplace_back();
    for (auto u : linspace{ssf->min_u(), ssf->max_u(), seedline_resolution}) {
      const auto t0u = t0(u);
      //if (this->ssf->vectorfield().in_domain(this->ssf->seedcurve().sample(u),
      //                                       t0u)) {
      const auto new_pos = ssf->sample(u, t0u);
      const auto v       = insert_vertex(std::move(new_pos), uv_t{u, t0u});
      vs.back().push_back(v);
      //} else if (vs.back().size() > 1) {
      //  vs.emplace_back();
      //} else {
      //  vs.back().clear();
      //}
    }
    if (vs.back().size() <= 1) { vs.pop_back(); }
    front_t front;
    for (auto&& vs : vs) {
      front.emplace_back(std::move(vs), std::pair{begin(vs), end(vs)});
    }
    return front;
  }
  //----------------------------------------------------------------------------
  real_t average_segment_length(const subfront_t& subfront) const {
    return average_segment_length(subfront.first);
  }
  //----------------------------------------------------------------------------
  real_t average_segment_length(const vertex_list_t& vs) const {
    real_t dist_acc = 0;
    for (auto v = begin(vs); v != prev(end(vs)); ++v) {
      dist_acc += norm(at(*v) - at(*next(v)));
    }

    return dist_acc / (vs.size() - 1);
  }
  //--------------------------------------------------------------------------
  void subdivide(front_t& front, real_t desired_spatial_dist) {
    auto find_best_predecessor = [this](const auto v, const auto& pred_range) {
      real_t min_u_dist = std::numeric_limits<real_t>::max();
      auto   best_it    = pred_range.second;
      for (auto it = pred_range.first; it != pred_range.second; ++it) {
        if (real_t u_dist = std::abs(uv(*v)(0) - uv(*it)(0));
            u_dist < min_u_dist) {
          min_u_dist = u_dist;
          best_it    = it;
          if (min_u_dist == 0) { break; }
        }
      }
      return best_it;
    };

    for (auto subfront = begin(front); subfront != end(front); ++subfront) {
      if (subfront->first.size() > 1) {
        auto& vs         = subfront->first;
        auto& pred_range = subfront->second;

        for (auto v = begin(vs); v != prev(end(vs)); ++v) {
          real_t d;
          try {
            d = this->ssf->distance(uv(*v), uv(*next(v)), 5);
          } catch (std::exception&) { d = distance(at(*v), at(*next(v))); }

          bool stop = false;
          while (d > desired_spatial_dist * 1.5) {
            // split front if u distance to small
            real_t u_dist = std::abs(uv(*v)(0) - uv(*next(v))(0));
            if (u_dist < 1e-5) {
              const auto best_pred = find_best_predecessor(v, pred_range);
              front.emplace(next(subfront), vertex_list_t{next(v), end(vs)},
                            vertex_range_t{next(best_pred), pred_range.second});
              pred_range.second = next(best_pred);
              vs.erase(next(v), end(vs));
              stop = true;
              break;
            }

            auto new_uv = (uv(*v) + uv(*next(v))) * 0.5;
            try {
              auto new_pnt = this->ssf->sample(new_uv);
              auto new_v   = insert_vertex(new_pnt, uv_t{new_uv(0), new_uv(1)});
              vs.insert(next(v), new_v);
              try {
                d = this->ssf->distance(uv(*v), uv(*next(v)), 5);
              } catch (std::exception&) { d = distance(at(*v), at(*next(v))); }
            } catch (std::exception&) {
              if (next(v, 2) != end(vs)) {
                const auto best_pred = find_best_predecessor(v, pred_range);
                // front.emplace(next(subfront), vertex_list_t{next(v),
                // end(vs)},
                //              vertex_range_t{next(best_pred),
                //              pred_range.second});
                pred_range.second = next(best_pred);
                vs.erase(next(v), end(vs));
              }
              stop = true;
              break;
            }
          }
          if (stop) { break; }
        }
      }
    }
  }
  //---------------------------------------------------------------------------
  void reduce(front_t& front, real_t desired_spatial_dist) {
    for (auto& subfront : front) {
      auto& vs = subfront.first;
      if (vs.size() >= 3) {
        for (auto v = begin(vs); v != prev(end(vs), 2); ++v) {
          if (m_on_border.find(*v) != end(m_on_border)) { continue; }
          auto d = [&] {
            try {
              return this->ssf->distance(uv(*v), uv(*next(v, 2)), 7);
            } catch (std::exception&) {
              return distance(at(*v), at(*next(v))) +
                     distance(at(*next(v)), at(*next(v, 2)));
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

//template <typename Flowmap,
//          template <typename> typename SeedcurveInterpolationKernel>
//struct simple_discretization : front_evolving_streamsurface_discretization<
//                                   Flowmap, SeedcurveInterpolationKernel> {
//  using real_t = typename Flowmap::real_t;
//  static constexpr auto num_dimensions() { return Flowmap::num_dimensions(); }
//  using parent_t =
//      front_evolving_streamsurface_discretization<Flowmap,
//                                                  SeedcurveInterpolationKernel>;
//  using front_t          = typename parent_t::front_t;
//  using subfront_t       = typename parent_t::subfront_t;
//  using ssf_t            = typename parent_t::ssf_t;
//  using vertex_vec_t     = typename parent_t::vertex_vec_t;
//  using vertex_list_t    = typename parent_t::vertex_list_t;
//  using vertex           = typename parent_t::vertex;
//  using face             = typename parent_t::face;
//  using vertex_list_it_t = typename parent_t::vertex_list_it_t;
//  using vertex_range_t   = typename parent_t::vertex_range_t;
//  using parent_t::at;
//  using parent_t::insert_vertex;
//  using parent_t::t0;
//  using parent_t::uv;
//
//  //============================================================================
//  simple_discretization(const simple_discretization& other)     = default;
//  simple_discretization(simple_discretization&& other) noexcept = default;
//  simple_discretization& operator=(const simple_discretization& other) =
//      default;
//  simple_discretization& operator=(simple_discretization&& other) noexcept =
//      default;
//  ~simple_discretization() = default;
//  //============================================================================
//  simple_discretization(ssf_t* ssf, size_t seedline_resolution, real_t stepsize,
//                        real_t backward_tau, real_t forward_tau)
//      : parent_t{ssf} {
//    assert(forward_tau >= 0);
//    assert(backward_tau <= 0);
//
//    const auto seed_front = this->seedcurve_to_front(seedline_resolution,
//                                                     backward_tau, forward_tau);
//    if (seed_front.empty()) { return; }
//
//    if (backward_tau < 0) {
//      auto   cur_stepsize    = stepsize;
//      auto   cur_front       = seed_front;
//      real_t integrated_time = 0;
//      while (integrated_time > backward_tau) {
//        if (integrated_time - cur_stepsize < backward_tau) {
//          cur_stepsize = std::abs(backward_tau - integrated_time);
//        }
//        cur_front = evolve(cur_front, -cur_stepsize, backward_tau, 0);
//        integrated_time -= cur_stepsize;
//      }
//    }
//
//    if (forward_tau > 0) {
//      auto   cur_stepsize    = stepsize;
//      auto   cur_front       = seed_front;
//      real_t integrated_time = 0;
//      while (integrated_time < forward_tau) {
//        if (integrated_time + cur_stepsize > forward_tau) {
//          cur_stepsize = forward_tau - integrated_time;
//        }
//        cur_front = evolve(cur_front, cur_stepsize, 0, forward_tau);
//        integrated_time += cur_stepsize;
//      }
//    }
//  }
//
//  //============================================================================
//  auto evolve(const front_t& front, real_t step) {
//    auto integrated_front = integrate(front, step);
//
//    this->triangulate_timeline(integrated_front);
//    return integrated_front;
//  }
//
//  //============================================================================
//  auto integrate(const front_t& old_front, real_t step, real_t backward_tau,
//                 real_t forward_tau) {
//    auto new_front          = old_front;
//    auto& [vertices, range] = new_front.front();
//    range.first             = begin(old_front.front().first);
//    range.second            = end(old_front.front().first);
//
//    for (auto& v : vertices) {
//      const auto& uv = parent_t::uv(v);
//      const vec   new_uv{uv(0), uv(1) + step};
//      auto new_pos = this->ssf->sample(new_uv, backward_tau, forward_tau);
//
//      v = insert_vertex(new_pos, {uv(0), uv(1) + step});
//    }
//    return new_front;
//  }
//};
//==============================================================================
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
struct hultquist_discretization : front_evolving_streamsurface_discretization<
                                      Flowmap, SeedcurveInterpolationKernel> {
  using real_t = typename Flowmap::real_t;
  using this_t =
      hultquist_discretization<Flowmap, SeedcurveInterpolationKernel>;
  using parent_t =
      front_evolving_streamsurface_discretization<Flowmap,
                                                  SeedcurveInterpolationKernel>;
  using parent_t::at;
  using parent_t::insert_vertex;
  using parent_t::t0;
  using parent_t::uv;
  using typename parent_t::face;
  using typename parent_t::front_t;
  using typename parent_t::ssf_t;
  using typename parent_t::subfront_t;
  using typename parent_t::uv_t;
  using typename parent_t::vertex;
  using typename parent_t::vertex_list_it_t;
  using typename parent_t::vertex_list_t;
  using typename parent_t::vertex_range_t;
  using typename parent_t::vertex_vec_t;
  //----------------------------------------------------------------------------
  hultquist_discretization(ssf_t* ssf, size_t seedline_resolution,
                           real_t stepsize, real_t backward_tau,
                           real_t forward_tau)
      : parent_t(ssf) {
    assert(forward_tau >= 0);
    assert(backward_tau <= 0);

    const auto seed_front = this->seedcurve_to_front(seedline_resolution/*,*/
                                                     /*backward_tau, forward_tau*/);
    if (seed_front.empty()) { return; }
    real_t desired_spatial_dist =
        this->average_segment_length(seed_front.front());

    if (backward_tau < 0) {
      auto   cur_stepsize    = stepsize;
      auto   cur_front       = seed_front;
      real_t integrated_time = 0;
      while (integrated_time > backward_tau) {
        if (integrated_time - cur_stepsize < backward_tau) {
          cur_stepsize = std::abs(backward_tau - integrated_time);
        }
        cur_front = evolve(cur_front, -cur_stepsize, desired_spatial_dist);
        integrated_time -= cur_stepsize;
      }
    }

    if (forward_tau > 0) {
      auto   cur_stepsize    = stepsize;
      auto   cur_front       = seed_front;
      real_t integrated_time = 0;
      while (integrated_time < forward_tau) {
        if (integrated_time + cur_stepsize > forward_tau) {
          cur_stepsize = forward_tau - integrated_time;
        }
        cur_front = evolve(cur_front, cur_stepsize, desired_spatial_dist);
        integrated_time += cur_stepsize;
      }
    }
  }
  //----------------------------------------------------------------------------
  hultquist_discretization(const this_t& other)     = default;
  hultquist_discretization(this_t&& other) noexcept = default;
  hultquist_discretization& operator=(const this_t& other) = default;
  hultquist_discretization& operator=(this_t&& other) noexcept = default;
  ~hultquist_discretization()                                  = default;
  //============================================================================
  auto integrate(const subfront_t& subfront, real_t step) {
    assert(step != 0);
    struct integrated_t {
      vertex           v;
      bool             moved, on_border, resampled;
      vertex_list_it_t start_vertex;
    };
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // integrate vertices with domain border detection
    std::list<integrated_t> integrated_vertices;
    for (auto v_it = begin(subfront.first); v_it != end(subfront.first);
         ++v_it) {
      auto        v  = *v_it;
      const auto& uv = parent_t::uv(v);
      const vec   new_uv{uv(0), uv(1) + step};
      try {
        if (this->m_on_border.find(v) == end(this->m_on_border)) {
          auto new_pos = this->ssf->sample(new_uv);
          integrated_vertices.push_back(
              {insert_vertex(new_pos, uv_t{new_uv(0), new_uv(1)}), true, false,
               false, v_it});
        } else {
          integrated_vertices.push_back({v, false, true, false, v_it});
        }

      } catch (std::exception&) {
        //const auto& streamline =
        //    this->ssf->streamline_at(uv(0));
        //if (!streamline.empty()) {
        //  const auto& border_point =
        //      step > 0 ? streamline.back() : streamline.front();
        //  auto new_v = insert_vertex(border_point.first,
        //                             uv_t{uv(0), border_point.second});
        //  integrated_vertices.push_back({new_v, true, true, false, v_it});
        //  this->m_on_border.insert(new_v);
        //}
      }
    }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // insert front vertices on actual front intersecting domain border
    for (auto it = begin(integrated_vertices);
         it != prev(end(integrated_vertices)); ++it) {
      if (it->on_border != next(it)->on_border) {
        // find point between current and next that hits the border and is on
        // integrated subfront
        const auto& uv = next(it)->on_border ? parent_t::uv(it->v)
                                             : parent_t::uv(next(it)->v);
        const auto fix_v     = uv(1);
        auto       walking_u = uv(0);
        const auto dist =
            std::abs(parent_t::uv(it->v)(0) - parent_t::uv(next(it)->v)(0));
        auto step = dist / 4;
        if (it->on_border) { step = -step; }
        bool found = false;
        while (std::abs(step) > 1e-10) {
          try {
            this->ssf->sample(walking_u + step, fix_v);
            walking_u += step;
            if (!found) { found = true; }
          } catch (std::exception&) { step /= 2; }
        }
        if (found) {
          auto new_v = insert_vertex(
              this->ssf->sample(walking_u, fix_v),
              uv_t{walking_u, fix_v});
          integrated_vertices.insert(
              next(it), {new_v, true, true, true,
                         next(it)->on_border ? next(it)->start_vertex
                                             : it->start_vertex});
          ++it;
        }
      }
    }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // create new subfronts with domain border detection
    std::vector<subfront_t> new_subfronts{
        {{}, {begin(subfront.first), end(subfront.first)}}};

    for (auto it = begin(integrated_vertices);
         it != prev(end(integrated_vertices)); ++it) {
      if (it->moved) {
        new_subfronts.back().first.push_back(it->v);
      }

      else if (new_subfronts.back().first.empty()) {
        if (it != begin(integrated_vertices)) {
          ++new_subfronts.back().second.first;
        }

      } else {
        new_subfronts.back().second.second = next(it)->start_vertex;
        new_subfronts.emplace_back(
            vertex_list_t{},
            vertex_range_t{it->start_vertex, end(subfront.first)});
      }
    }
    if (integrated_vertices.back().moved) {
      new_subfronts.back().first.push_back(integrated_vertices.back().v);
    }

    if (new_subfronts.back().first.empty()) { new_subfronts.pop_back(); }
    return new_subfronts;
  }

  //--------------------------------------------------------------------------
  auto integrate(const front_t& front, real_t step) {
    front_t new_front;
    for (const auto& subfront : front) {
      if (subfront.first.size() > 1) {
        boost::copy(integrate(subfront, step),
                    std::back_inserter(new_front));
      }
    }
    return new_front;
  }
  //----------------------------------------------------------------------------
  auto evolve(const front_t& front, real_t step, real_t desired_spatial_dist) {
    auto integrated_front = integrate(front, step);

    this->subdivide(integrated_front, desired_spatial_dist);
    this->reduce(integrated_front, desired_spatial_dist);
    this->triangulate_timeline(integrated_front);
    return integrated_front;
  }
};

//==============================================================================
template <typename Flowmap,
          template <typename> typename SeedcurveInterpolationKernel>
struct schulze_discretization : front_evolving_streamsurface_discretization<
                                    Flowmap, SeedcurveInterpolationKernel> {
  using real_t = typename Flowmap::real_t;
  static constexpr auto num_dimensions() { return Flowmap::num_dimensions(); }
  using parent_t =
      front_evolving_streamsurface_discretization<Flowmap,
                                                  SeedcurveInterpolationKernel>;
  using typename parent_t::face;
  using typename parent_t::front_t;
  using typename parent_t::ssf_t;
  using typename parent_t::subfront_t;
  using typename parent_t::vertex;
  using typename parent_t::vertex_list_t;
  using typename parent_t::vertex_list_it_t;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  using parent_t::at;
  using parent_t::insert_vertex;
  using parent_t::uv;

  vertex_property_t<real_t>& alpha_prop;
  vertex_property_t<real_t>& second_derivate_alpha_prop;

  //----------------------------------------------------------------------------
  schulze_discretization(ssf_t* ssf, size_t seedline_resolution,
                         size_t num_iterations)
      : parent_t(ssf),
        alpha_prop(this->template add_vertex_property<real_t>("alpha")),
        second_derivate_alpha_prop(this->template add_vertex_property<real_t>(
            "second_derivative_alpha")) {
    auto const initial_front = this->seedcurve_to_front(seedline_resolution);
    real_t desired_spatial_dist =
        this->average_segment_length(initial_front.front());

    // evolve front
    front_t cur_front = initial_front;
    for (size_t i = 0; i < num_iterations; ++i) {
      cur_front = evolve(cur_front, desired_spatial_dist);
    }
  }

  //--------------------------------------------------------------------------
  auto integrate(const subfront_t& front, real_t step) {
    std::vector<subfront_t> new_subfronts{
        {{}, {begin(front.first), end(front.first)}}};
    const auto& [vs, pred_range] = front;
    // integrate each subfront
    auto alpha = optimal_stepsizes(vs);
    for (auto [v, i] = std::pair{begin(vs), size_t(0)}; v != end(vs);
         ++v, ++i) {
      alpha_prop[*v] = alpha[i];
    }
    auto splitted_front_ranges = detect_peaks(alpha, vs);
    // no rip needed
    if (size(splitted_front_ranges) == 1 &&
        splitted_front_ranges[0].first == begin(vs) &&
        splitted_front_ranges[0].second == end(vs)) {
      auto& vertices1 =
          new_subfronts
              .emplace_back(std::list<vertex>{}, std::pair{begin(vs), end(vs)})
              .first;

      size_t i = 0;
      for (const auto v : vs) {
        const auto& uv = parent_t::uv(v);
        vec2        new_uv{uv(0), uv(1) + alpha[i++]};
        auto        new_pos = this->ssf->sample(new_uv);
        vertices1.push_back(insert_vertex(new_pos, new_uv));
        alpha_prop[v] = alpha[i - 1];
      }

    } else {
      // rip needed
      for (const auto& range : splitted_front_ranges) {
        auto& [vertices1, pred_range] =
            new_subfronts.emplace_back(std::list<vertex>{}, range);

        size_t            i = 0;
        std::list<vertex> sub_front;
        std::copy(pred_range.first, pred_range.second,
                  std::back_inserter(sub_front));
        auto sub_alpha = optimal_stepsizes(sub_front);
        for (auto v = pred_range.first; v != pred_range.second; ++v) {
          const auto& uv = parent_t::uv(*v);
          vec2        new_uv{uv(0), uv(1) + sub_alpha[i++]};
          auto        new_pos = this->ssf->sample(new_uv);
          vertices1.push_back(insert_vertex(new_pos, new_uv));
          alpha_prop[vertices1.back()] = sub_alpha[i - 1];
        }
      }
    }
    return new_subfronts;
  }
  //--------------------------------------------------------------------------
  auto integrate(const front_t& front, real_t step) {
    front_t integrated_front;
    for (const auto& subfront : front) {
      if (subfront.first.size() > 1) {
        boost::copy(integrate(subfront, step),
                    std::back_inserter(integrated_front));
      }
    }
    return integrated_front;
  }
  //----------------------------------------------------------------------------
  auto evolve(const front_t& front, real_t desired_spatial_dist) {
    auto integrated_front = integrate(front, desired_spatial_dist);
    // triangulate
    std::vector<std::vector<face>> faces;
    this->subdivide(integrated_front, desired_spatial_dist);
    this->reduce(integrated_front, desired_spatial_dist);
    this->triangulate_timeline(integrated_front);
    return integrated_front;
  }
  //----------------------------------------------------------------------------
  std::vector<real_t> optimal_stepsizes(const vertex_list_t& vs) {
    const auto& v        = this->ssf->flowmap().vectorfield();
    auto        jacobian = diff(v, 1e-7);

    auto              num_pnts = size(vs);
    std::vector<real_t> p(num_pnts - 1), q(num_pnts - 1), null(num_pnts),
        r(num_pnts);
    std::vector<vec<real_t, num_dimensions()>> ps(num_pnts);

    // TODO: get t0 at u, not at 0
    size_t i = 0;
    for (const auto vertex : vs)
      ps[i++] = v(at(vertex), this->t0(0) + uv(vertex)(1));

    i            = 0;
    real_t avg_len = 0;
    for (auto vertex = begin(vs); vertex != prev(end(vs)); ++vertex, ++i) {
      auto tm  = this->t0(0) + (uv(*vertex)(1) + uv(*next(vertex))(1)) * 0.5;
      auto xm  = (at(*next(vertex)) + at(*vertex)) * 0.5;
      auto dir = at(*next(vertex)) - at(*vertex);
      auto vm  = v(xm, tm);
      auto Jm  = jacobian(xm, tm);

      p[i] = dot(dir * 0.5, Jm * ps[i]) - dot(ps[i], vm);
      q[i] = dot(dir * 0.5, Jm * ps[i + 1]) + dot(ps[i + 1], vm);
      r[i] = -dot(dir, vm);

      avg_len += norm(dir);
    }
    avg_len /= num_pnts - 1;
    solve_qr(num_pnts - 1, &p[0], &q[0], &r[0], &null[0]);

    // real_t nrm{0};
    // for (auto x : null) nrm += x * x;
    // nrm = sqrt(nrm);
    // for (size_t i = 0; i < null.size(); ++i) null[i] /= nrm;

    // count positive entries in nullspace
    size_t num_pos_null = 0;
    for (auto c : null)
      if (c > 0) ++num_pos_null;
    int k_plus_factor = (num_pos_null < null.size() / 2) ? -1 : 1;

    for (size_t i = 0; i < r.size(); ++i) r[i] += k_plus_factor * null[i];
    // r[i] += (num_pnts / 10.0) * k_plus_factor * null[i];

    // apply step width
    real_t h = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < num_pnts; ++i)
      h = std::min(h, avg_len / (std::abs(r[i]) * norm(ps[i])));
    for (size_t i = 0; i < r.size(); ++i) r[i] *= h;
    // for (size_t i = 0; i < r.size(); ++i) r[i] = std::max(r[i],1e-3);
    return r;
  }

  //----------------------------------------------------------------------------
  auto detect_peaks(const std::vector<real_t>& alpha, const vertex_list_t& vs,
                    real_t threshold = 100) {
    // calculate second derivative
    std::vector<real_t> snd_der(alpha.size(), 0);
    auto              v = next(begin(vs));
    for (size_t i = 1; i < alpha.size() - 1; ++i, ++v) {
      mat<real_t, 3, 3> A{
          {real_t(1), uv(*prev(v))(0), uv(*prev(v))(0) * uv(*prev(v))(0)},
          {real_t(1), uv(*v)(0), uv(*v)(0) * uv(*v)(0)},
          {real_t(1), uv(*next(v))(0), uv(*next(v))(0) * uv(*next(v))(0)}};
      vec<real_t, 3> b{alpha[i - 1], alpha[i], alpha[i + 1]};
      snd_der[i]                     = 2 * solve(A, b)(2);
      second_derivate_alpha_prop[*v] = std::abs(snd_der[i]);
    }

    std::vector<std::pair<vertex_list_it_t, vertex_list_it_t>> splitted_front_ranges;
    splitted_front_ranges.emplace_back(begin(vs), end(vs));

    auto v_it = next(begin(vs));
    for (size_t i = 1; i < snd_der.size() - 1; ++i, ++v_it)
      if (std::abs(snd_der[i]) > threshold) {
        if (splitted_front_ranges.back().first == v_it)
          // shift left border of split to the right
          ++splitted_front_ranges.back().first;
        else {
          // insert new subfront
          splitted_front_ranges.back().second = next(v_it);
          if (splitted_front_ranges.back().first ==
              splitted_front_ranges.back().second)
            splitted_front_ranges.pop_back();

          splitted_front_ranges.emplace_back(next(v_it), end(vs));
        }
      }
    if (splitted_front_ranges.back().first ==
        splitted_front_ranges.back().second) {
      splitted_front_ranges.pop_back();
    }
    return splitted_front_ranges;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
