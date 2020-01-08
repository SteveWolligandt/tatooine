#ifndef TATOOINE_STREAMSURFACE_H
#define TATOOINE_STREAMSURFACE_H

#include <algorithm>
#include <boost/functional.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include "bidiagonal_system_solver.h"
#include "grid_edge.h"
#include "integration/integrator.h"
#include "interpolation.h"
#include "line.h"
#include "linspace.h"
#include "for_loop.h"
#include "simple_tri_mesh.h"
#include "tensor.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <template <typename, size_t> typename Integrator,
          template <typename> typename SeedcurveInterpolator,
          template <typename> typename StreamlineInterpolator, typename V,
          typename Real, size_t N>
struct hultquist_discretization;

template <template <typename, size_t> typename Integrator,
          template <typename> typename SeedcurveInterpolator,
          template <typename> typename StreamlineInterpolator, typename V,
          typename Real, size_t N>
struct streamsurface {
  static constexpr auto num_dimensions() { return N; }
  using real_t       = Real;
  using this_t       = streamsurface<Integrator, SeedcurveInterpolator,
                               StreamlineInterpolator, V, Real, N>;
  using line_t       = parameterized_line<Real, N>;
  using vec2         = vec<Real, 2>;
  using pos_t        = vec<Real, N>;
  using vec_t        = vec<Real, N>;
  using integrator_t = Integrator<Real, N>;

  struct out_of_domain : std::exception {};

 private:
  V            m_v;
  Real         m_t0;
  line_t       m_seedcurve;
  Real         m_min_u, m_max_u;
  integrator_t m_integrator;

  //----------------------------------------------------------------------------
 public:
  template <typename T0Real, typename... Args>
  streamsurface(const field<V, Real, N, N>& v, T0Real t0,
                const line_t& seedcurve, const Integrator<Real, N>& integrator,
                SeedcurveInterpolator<Real>, StreamlineInterpolator<Real>)
      : m_v{v.as_derived()},
        m_t0{static_cast<Real>(t0)},
        m_seedcurve(seedcurve),
        m_min_u{std::min(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_max_u{std::max(m_seedcurve.front_parameterization(),
                         m_seedcurve.back_parameterization())},
        m_integrator{integrator} {}

  //============================================================================
  auto t0() const { return m_t0; }

  //----------------------------------------------------------------------------
  auto&       integrator() { return m_integrator; }
  const auto& integrator() const { return m_integrator; }

  //----------------------------------------------------------------------------
  const auto& streamline_at(Real u, Real cache_bw_tau,
                            Real cache_fw_tau) const {
    return m_integrator.integrate(
        m_v, m_seedcurve.template sample<SeedcurveInterpolator>(u), m_t0,
        cache_bw_tau, cache_fw_tau);
  }

  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  vec_t sample(Real u, Real v, Real cache_bw_tau, Real cache_fw_tau) const {
    if (u < m_min_u || u > m_max_u) { throw out_of_domain{}; }
    if (v == 0) {
      return m_seedcurve.template sample<SeedcurveInterpolator>(u);
    }
    try {
      return streamline_at(u, cache_bw_tau, cache_fw_tau)
          .template sample<StreamlineInterpolator>(m_t0 + v);
    } catch (std::exception&) { throw out_of_domain{}; }
  }

  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  vec_t sample(Real u, Real v) const {
    if (v < 0) { return sample(u, v, v, 0); }
    if (v > 0) { return sample(u, v, 0, v); }
    return m_seedcurve.template sample<SeedcurveInterpolator>(u);
  }

  //----------------------------------------------------------------------------
  /// calculates position of streamsurface
  vec_t sample(const vec2& uv) const { return sample(uv(0), uv(1)); }
  vec_t sample(const vec2& uv, Real cache_bw_tau, Real cache_fw_tau) const {
    return sample(uv(0), uv(1), cache_bw_tau, cache_fw_tau);
  }

  //----------------------------------------------------------------------------
  auto distance(const vec2& uv0, const vec2& uv1, size_t num_samples,
                Real cache_bw_tau, Real cache_fw_tau) const {
    auto step = (uv1 - uv0) / (num_samples - 1);
    Real d    = 0;
    for (size_t i = 0; i < num_samples - 1; ++i) {
      d += ::tatooine::distance(
          sample(uv0 + step * i, cache_bw_tau, cache_fw_tau),
          sample(uv0 + step * (i + 1), cache_bw_tau, cache_fw_tau));
    }
    return d;
  }

  //----------------------------------------------------------------------------
  auto operator()(Real u, Real v) const { return sample(u, v); }
  auto operator()(Real u, Real v, Real cache_bw_tau, Real cache_fw_tau) const {
    return sample(u, v, cache_bw_tau, cache_fw_tau);
  }

  //----------------------------------------------------------------------------
  const auto& seedcurve() const { return m_seedcurve; }
  const auto& vectorfield() const { return m_v; }
  auto&       vectorfield() { return m_v; }

  //----------------------------------------------------------------------------
  template <template <template <typename, size_t> typename,
                      template <typename> typename,
                      template <typename> typename, typename, typename, size_t>
            typename Discretization = hultquist_discretization,
            typename... Args>
  auto discretize(Args&&... args) {
    return Discretization<Integrator, SeedcurveInterpolator,
                          StreamlineInterpolator, V, Real, N>(
        this, std::forward<Args>(args)...);
  }

  //----------------------------------------------------------------------------
  constexpr auto min_u() const { return m_min_u; }
  constexpr auto max_u() const { return m_max_u; }
};

////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// template <typename V, typename Real, size_t N, typename T0Real,
//          typename GridReal, typename... Args>
// streamsurface(const field<V, Real, N, N>& v, T0Real t0,
//              const parameterized_line<Real, N>& seedcurve, Args&&... args)
//    ->streamsurface<TATOOINE_DEFAULT_INTEGRATOR, interpolation::hermite,
//                    interpolation::linear, V, Real, N>;
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
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
template <template <typename, size_t> typename Integrator,
          template <typename> typename SeedcurveInterpolator,
          template <typename> typename StreamlineInterpolator, typename V,
          typename Real, size_t N>
struct front_evolving_streamsurface_discretization
    : public simple_tri_mesh<Real, N> {
  static constexpr auto num_dimensions() { return N; }
  using real_t = Real;
  using this_t = front_evolving_streamsurface_discretization<
      Integrator, SeedcurveInterpolator, StreamlineInterpolator, V, Real, N>;
  using parent_t = simple_tri_mesh<Real, N>;
  using parent_t::at;
  using parent_t::insert_vertex;
  using parent_t::operator[];
  using typename parent_t::face;
  using typename parent_t::vertex;

  using vec2             = vec<Real, 2>;

  using vertex_vec_t     = std::vector<vertex>;
  using vertex_list_t    = std::list<vertex>;
  using vertex_list_it_t = typename vertex_list_t::const_iterator;
  using vertex_range_t   = std::pair<vertex_list_it_t, vertex_list_it_t>;
  using subfront_t       = std::pair<vertex_list_t, vertex_range_t>;
  using ssf_t            = streamsurface<Integrator, SeedcurveInterpolator,
                              StreamlineInterpolator, V, Real, N>;
  using uv_t             = vec2;
  using uv_property_t    = typename parent_t::template vertex_property_t<uv_t>;

  // a front is a list of lists, containing vertices and a range specifing
  // which vertices have to be triangulated from previous front
  using front_t = std::list<subfront_t>;

  //============================================================================
  ssf_t*           ssf;
  std::set<vertex> m_on_border;
  uv_property_t*       m_uv_property;

  //============================================================================
  front_evolving_streamsurface_discretization(ssf_t* _ssf)
      : ssf{_ssf}, m_uv_property{&add_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(const this_t& other)
      : parent_t{other}, ssf{other.ssf}, m_uv_property{&add_uv_prop()} {}
  //----------------------------------------------------------------------------
  front_evolving_streamsurface_discretization(this_t&& other) noexcept
      : parent_t(std::move(other)), ssf(other.ssf), m_uv_property{&add_uv_prop()} {}
  //----------------------------------------------------------------------------
  auto& operator=(const this_t& other) {
    parent_t::operator=(other);
    ssf               = other.ssf;
    m_uv_property         = &uv_prop();
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(this_t&& other) noexcept {
    parent_t::operator=(std::move(other));
    ssf               = other.ssf;
    m_uv_property         = &uv_prop();
    return *this;
  }
  //============================================================================
 private:
  auto& add_uv_prop() { return this->template add_vertex_property<uv_t>("uv"); }
  //----------------------------------------------------------------------------
 public:
  auto& uv_prop() { return this->template vertex_property<uv_t>("uv"); }
  //----------------------------------------------------------------------------
  auto&       uv(vertex v) { return m_uv_property->at(v); }
  const auto& uv(vertex v) const { return m_uv_property->at(v); }
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
        Real       lower_edge_len = std::numeric_limits<Real>::max();
        Real       upper_edge_len = std::numeric_limits<Real>::max();
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
  auto seedcurve_to_front(size_t seedline_resolution, Real cache_bw_tau,
                          Real cache_fw_tau) {
    std::vector<vertex_list_t> vs;
    vs.emplace_back();
    for (auto u : linspace{ssf->min_u(), ssf->max_u(), seedline_resolution}) {
      if (this->ssf->vectorfield().in_domain(
              this->ssf->seedcurve().template sample<SeedcurveInterpolator>(u),
              ssf->t0())) {
        vs.back().push_back(insert_vertex(
            ssf->sample(u, 0, cache_bw_tau, cache_fw_tau)));
        uv(vs.back().back()) = {u, 0};
      } else if (vs.back().size() > 1) {
        vs.emplace_back();
      } else {
        vs.back().clear();
      }
    }
    if (vs.back().size() <= 1) { vs.pop_back(); }
    front_t front;
    for (auto&& vs : vs) {
      front.emplace_back(std::move(vs), std::pair{begin(vs), end(vs)});
    }
    return front;
  }

  //----------------------------------------------------------------------------
  Real average_segment_length(const subfront_t& subfront) const {
    return average_segment_length(subfront.first);
  }

  //----------------------------------------------------------------------------
  Real average_segment_length(const vertex_list_t& vs) const {
    Real dist_acc = 0;
    for (auto v = begin(vs); v != prev(end(vs)); ++v) {
      dist_acc += norm(at(*v) - at(*next(v)));
    }

    return dist_acc / (vs.size() - 1);
  }

  //--------------------------------------------------------------------------
  void subdivide(front_t& front, Real desired_spatial_dist, Real cache_bw_tau,
                 Real cache_fw_tau) {
    auto find_best_predecessor = [this](const auto v, const auto& pred_range) {
      Real min_u_dist = std::numeric_limits<Real>::max();
      auto best_it    = pred_range.second;
      for (auto it = pred_range.first; it != pred_range.second; ++it) {
        if (Real u_dist = std::abs(uv(*v)(0) - uv(*it)(0));
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
          Real d;
          try {
            d = this->ssf->distance(uv(*v), uv(*next(v)), 5, cache_bw_tau,
                                    cache_fw_tau);
          } catch (std::exception&) {
            d = distance(at(*v), at(*next(v)));
          }

          bool stop = false;
          while (d > desired_spatial_dist * 1.5) {
            // split front if u distance to small
            Real u_dist = std::abs(uv(*v)(0) - uv(*next(v))(0));
            if (u_dist < 1e-6) {
              auto pred_split_it = find_best_predecessor(v, pred_range);
              // auto new_sub =
              //     front.insert(next(subfront),
              //                  subfront_t{{next(v), end(vs)},
              //                             {pred_split_it,
              //                             pred_range.second}});
              pred_range.second = next(pred_split_it);
              vs.erase(next(v), end(vs));
              stop = true;
              break;
            }

            auto new_uv = (uv(*v) + uv(*next(v))) * 0.5;
            try {
              auto new_pnt =
                  this->ssf->sample(new_uv, cache_bw_tau, cache_fw_tau);
              auto new_v      = insert_vertex(new_pnt);
              this->uv(new_v) = new_uv;
              vs.insert(next(v), new_v);
              try {
                d = this->ssf->distance(uv(*v), uv(*next(v)), 5, cache_bw_tau,
                                        cache_fw_tau);
              } catch (std::exception&) {
                d = distance(at(*v), at(*next(v)));
              }
            } catch (std::exception&) {
              if (next(v, 2) != end(vs)) {
                auto pred_split_it = find_best_predecessor(v, pred_range);
                // auto new_sub       = front.insert(
                //     next(subfront),
                //     subfront_t{{next(v), end(vs)},
                //                {pred_split_it, pred_range.second}});
                pred_range.second = next(pred_split_it);
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
  void reduce(front_t& front, Real desired_spatial_dist, Real cache_bw_tau,
              Real cache_fw_tau) {
    for (auto& subfront : front) {
      auto& vs = subfront.first;
      if (vs.size() >= 3) {
        for (auto v = begin(vs); v != prev(end(vs), 2); ++v) {
          if (m_on_border.find(*v) != end(m_on_border)) { continue; }
          auto d = [&] {
            try {
              return this->ssf->distance(uv(*v), uv(*next(v, 2)), 7,
                                         cache_bw_tau, cache_fw_tau);
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

template <template <typename, size_t> typename Integrator,
          template <typename> typename SeedcurveInterpolator,
          template <typename> typename StreamlineInterpolator, typename V,
          typename Real, size_t N>
struct simple_discretization : front_evolving_streamsurface_discretization<
                                   Integrator, SeedcurveInterpolator,
                                   StreamlineInterpolator, V, Real, N> {
  using parent_t = front_evolving_streamsurface_discretization<
      Integrator, SeedcurveInterpolator, StreamlineInterpolator, V, Real, N>;
  using front_t          = typename parent_t::front_t;
  using subfront_t       = typename parent_t::subfront_t;
  using ssf_t            = typename parent_t::ssf_t;
  using vertex_vec_t     = typename parent_t::vertex_vec_t;
  using vertex_list_t    = typename parent_t::vertex_list_t;
  using vertex           = typename parent_t::vertex;
  using face             = typename parent_t::face;
  using vertex_list_it_t = typename parent_t::vertex_list_it_t;
  using vertex_range_t   = typename parent_t::vertex_range_t;
  using parent_t::at;
  using parent_t::insert_vertex;
  using parent_t::uv;

  //============================================================================
  simple_discretization(const simple_discretization& other)     = default;
  simple_discretization(simple_discretization&& other) noexcept = default;
  simple_discretization& operator=(const simple_discretization& other) =
      default;
  simple_discretization& operator=(simple_discretization&& other) noexcept =
      default;
  ~simple_discretization() = default;
  //============================================================================
  simple_discretization(ssf_t* ssf, size_t seedline_resolution, Real stepsize,
                        Real backward_tau, Real forward_tau)
      : parent_t{ssf} {
    assert(forward_tau >= 0);
    assert(backward_tau <= 0);

    const auto seed_front = this->seedcurve_to_front(seedline_resolution,
                                                     backward_tau, forward_tau);
    if (seed_front.empty()) { return; }

    if (backward_tau < 0) {
      auto cur_stepsize    = stepsize;
      auto cur_front       = seed_front;
      Real integrated_time = 0;
      while (integrated_time > backward_tau) {
        if (integrated_time - cur_stepsize < backward_tau) {
          cur_stepsize = std::abs(backward_tau - integrated_time);
        }
        cur_front = evolve(cur_front, -cur_stepsize, backward_tau, 0);
        integrated_time -= cur_stepsize;
      }
    }

    if (forward_tau > 0) {
      auto cur_stepsize    = stepsize;
      auto cur_front       = seed_front;
      Real integrated_time = 0;
      while (integrated_time < forward_tau) {
        if (integrated_time + cur_stepsize > forward_tau) {
          cur_stepsize = forward_tau - integrated_time;
        }
        cur_front = evolve(cur_front, cur_stepsize, 0, forward_tau);
        integrated_time += cur_stepsize;
      }
    }
  }

  //============================================================================
  auto evolve(const front_t& front, Real step, Real cache_bw_tau,
              Real cache_fw_tau) {
    auto integrated_front = integrate(front, step, cache_bw_tau, cache_fw_tau);

    this->triangulate_timeline(integrated_front);
    return integrated_front;
  }

  //============================================================================
  auto integrate(const front_t& old_front, Real step, Real backward_tau,
                 Real forward_tau) {
    auto new_front          = old_front;
    auto& [vertices, range] = new_front.front();
    range.first             = begin(old_front.front().first);
    range.second            = end(old_front.front().first);

    for (auto& v : vertices) {
      const auto& uv = parent_t::uv(v);
      const vec   new_uv{uv(0), uv(1) + step};
      auto new_pos = this->ssf->sample(new_uv, backward_tau, forward_tau);

      v = insert_vertex(new_pos, new_uv);
    }
    return new_front;
  }
};
//==============================================================================
template <template <typename, size_t> typename Integrator,
          template <typename> typename SeedcurveInterpolator,
          template <typename> typename StreamlineInterpolator, typename V,
          typename Real, size_t N>
struct hultquist_discretization : front_evolving_streamsurface_discretization<
                                      Integrator, SeedcurveInterpolator,
                                      StreamlineInterpolator, V, Real, N> {
  using parent_t = front_evolving_streamsurface_discretization<
      Integrator, SeedcurveInterpolator, StreamlineInterpolator, V, Real, N>;
  using front_t          = typename parent_t::front_t;
  using subfront_t       = typename parent_t::subfront_t;
  using ssf_t            = typename parent_t::ssf_t;
  using vertex_vec_t     = typename parent_t::vertex_vec_t;
  using vertex_list_t    = typename parent_t::vertex_list_t;
  using vertex           = typename parent_t::vertex;
  using face             = typename parent_t::face;
  using vertex_list_it_t = typename parent_t::vertex_list_it_t;
  using vertex_range_t   = typename parent_t::vertex_range_t;
  using parent_t::at;
  using parent_t::insert_vertex;
  using parent_t::uv;

  //----------------------------------------------------------------------------
  hultquist_discretization(ssf_t* ssf, size_t seedline_resolution,
                           Real stepsize, Real backward_tau, Real forward_tau)
      : parent_t(ssf) {
    assert(forward_tau >= 0);
    assert(backward_tau <= 0);

    const auto seed_front = this->seedcurve_to_front(seedline_resolution,
                                                     backward_tau, forward_tau);
    if (seed_front.empty()) { return; }
    Real desired_spatial_dist =
        this->average_segment_length(seed_front.front());

    if (backward_tau < 0) {
      auto cur_stepsize    = stepsize;
      auto cur_front       = seed_front;
      Real integrated_time = 0;
      while (integrated_time > backward_tau) {
        if (integrated_time - cur_stepsize < backward_tau) {
          cur_stepsize = std::abs(backward_tau - integrated_time);
        }
        cur_front = evolve(cur_front, -cur_stepsize, desired_spatial_dist,
                           backward_tau, 0);
        integrated_time -= cur_stepsize;
      }
    }

    if (forward_tau > 0) {
      auto cur_stepsize    = stepsize;
      auto cur_front       = seed_front;
      Real integrated_time = 0;
      while (integrated_time < forward_tau) {
        if (integrated_time + cur_stepsize > forward_tau) {
          cur_stepsize = forward_tau - integrated_time;
        }
        cur_front = evolve(cur_front, cur_stepsize, desired_spatial_dist, 0,
                           forward_tau);
        integrated_time += cur_stepsize;
      }
    }
  }

  //----------------------------------------------------------------------------
  hultquist_discretization(const hultquist_discretization& other)     = default;
  hultquist_discretization(hultquist_discretization&& other) noexcept = default;
  hultquist_discretization& operator=(const hultquist_discretization& other) =
      default;
  hultquist_discretization& operator             =(
      hultquist_discretization&& other) noexcept = default;
  ~hultquist_discretization()                    = default;

  //============================================================================
  auto integrate(const subfront_t& subfront, Real step, Real backward_tau,
                 Real forward_tau) {
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
      const auto& v  = *v_it;
      const auto& uv = parent_t::uv(v);
      const vec   new_uv{uv(0), uv(1) + step};
      try {
        if (this->m_on_border.find(v) == end(this->m_on_border)) {
          auto new_pos = this->ssf->sample(new_uv, backward_tau, forward_tau);
          integrated_vertices.push_back(
              {insert_vertex(new_pos), true, false, false, v_it});
          this->uv(integrated_vertices.back().v) = new_uv;
        } else {
          integrated_vertices.push_back({v, false, true, false, v_it});
        }

      } catch (std::exception&) {
        const auto& streamline =
            this->ssf->streamline_at(uv(0), backward_tau, forward_tau);
        if (!streamline.empty()) {
          const auto& border_point =
              step > 0 ? streamline.back() : streamline.front();
          auto new_v = insert_vertex(border_point.first);
          this->uv(new_v)  = {uv(0), border_point.second - this->ssf->t0()};
          integrated_vertices.push_back({new_v, true, true, false, v_it});
          this->m_on_border.insert(new_v);
        }
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
            this->ssf->sample(walking_u + step, fix_v, backward_tau,
                              forward_tau);
            walking_u += step;
            if (!found) { found = true; }
          } catch (std::exception&) { step /= 2; }
        }
        if (found) {
          auto new_v = insert_vertex(
              this->ssf->sample(walking_u, fix_v, backward_tau, forward_tau));
          this->uv(new_v) = {walking_u, fix_v};
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
  auto integrate(const front_t& front, Real step, Real backward_tau,
                 Real forward_tau) {
    front_t new_front;

    for (const auto& subfront : front) {
      if (subfront.first.size() > 1) {
        boost::copy(integrate(subfront, step, backward_tau, forward_tau),
                    std::back_inserter(new_front));
      }
    }

    return new_front;
  }

  //--------------------------------------------------------------------------
  auto evolve(const front_t& front, Real step, Real desired_spatial_dist,
              Real cache_bw_tau, Real cache_fw_tau) {
    auto integrated_front = integrate(front, step, cache_bw_tau, cache_fw_tau);

    this->subdivide(integrated_front, desired_spatial_dist, cache_bw_tau,
                    cache_fw_tau);
    this->reduce(integrated_front, desired_spatial_dist, cache_bw_tau,
                 cache_fw_tau);
    this->triangulate_timeline(integrated_front);
    return integrated_front;
  }
};

//==============================================================================
// template <template <typename, size_t> typename Integrator,
//          template <typename> typename SeedcurveInterpolator,
//          template <typename> typename StreamlineInterpolator,
//          typename V, typename Real, size_t N>
// struct schulze_discretization
//     : front_evolving_streamsurface_discretization<V, Integrator,
//                                                SeedcurveInterpolator,
//                                                StreamlineInterpolator> {
// static constexpr auto num_dimensions() {
//  return N;
//}
//   using parent_t          = front_evolving_streamsurface_discretization<
//       V, Integrator, SeedcurveInterpolator,
//       StreamlineInterpolator>;
//   using front_t       = typename parent_t::front_t;
//   using subfront_t    = typename parent_t::subfront_t;
//   using ssf_t         = typename parent_t::ssf_t;
//   using vertex_list_t = typename parent_t::vertex_list_t;
//   using vertex        = typename parent_t::vertex;
//   using face          = typename parent_t::face;
//   template <typename T>
//   using VertexProperty = typename parent_t::template VertexProperty<T>;
//   using parent_t::uv;
//   using parent_t::at;
//   using parent_t::insert_vertex;
//
//   VertexProperty<Real>& alpha_prop;
//   VertexProperty<Real>& second_derivate_alpha_prop;
//
//   //----------------------------------------------------------------------------
//   schulze_discretization(const ssf_t* ssf, size_t seedline_resolution,
//                         size_t num_iterations)
//       : parent_t(ssf),
//         alpha_prop(this->template add_vertex_property<Real>("alpha")),
//         second_derivate_alpha_prop(this->template
//         add_vertex_property<Real>(
//             "second_derivative_alpha")) {
//     this->fronts.push_back(this->seedcurve_to_front(seedline_resolution));
//     Real desired_spatial_dist =
//         this->average_segment_length(this->fronts.front().front());
//
//     // evolve front
//     for (size_t i = 0; i < num_iterations; ++i)
//       forward_evolve(desired_spatial_dist);
//   }
//
//   //----------------------------------------------------------------------------
//   auto forward_evolve(Real desired_spatial_dist) {
//     const auto& front     = this->fronts.back();
//     auto&       new_front = this->fronts.emplace_back();
//
//     // integrate each subfront
//     for (const auto& [vs, pred_range] : front) {
//       auto alpha = optimal_stepsizes(vs);
//       for (auto [v, i] = std::pair{begin(vs), size_t(0)};
//            v != end(vs); ++v, ++i)
//         alpha_prop[*v] = alpha[i];
//       auto splitted_front_indices = detect_peaks(alpha, vs);
//       // no rip needed
//       if (splitted_front_indices.size() == 1 &&
//           splitted_front_indices[0].first == 0 &&
//           splitted_front_indices[0].second == alpha.size() - 1) {
//         auto& vertices1 =
//             new_front
//                 .emplace_back(std::list<vertex>{},
//                               std::pair{begin(vs), end(vs)})
//                 .first;
//
//         size_t i = 0;
//         for (const auto v : vs) {
//           const auto& uv = parent_t::uv(v);
//           vec2        new_uv{uv(0), uv(1) + alpha[i++]};
//           auto        new_pos = this->ssf->sample(new_uv);
//           vertices1.push_back(insert_vertex(new_pos, new_uv));
//           alpha_prop[v] = alpha[i - 1];
//         }
//
//         // rip needed
//       } else {
//         for (const auto& [first, last] : splitted_front_indices) {
//           auto& [vertices1, pred_range] = new_front.emplace_back(
//               std::list<vertex>{}, std::pair{next(begin(vs), first),
//                                              next(begin(vs), last +
//                                              1)});
//
//           size_t            i = 0;
//           std::list<vertex> sub_front;
//           std::copy(pred_range.first, pred_range.second,
//           std::back_inserter(sub_front)); auto sub_alpha =
//           optimal_stepsizes(sub_front); for (auto v = pred_range.first; v !=
//           pred_range.second; ++v) {
//             const auto& uv = parent_t::uv(*v);
//             vec2        new_uv{uv(0), uv(1) + sub_alpha[i++]};
//             auto        new_pos = this->ssf->sample(new_uv);
//             vertices1.push_back(insert_vertex(new_pos, new_uv));
//             alpha_prop[vertices1.back()] = sub_alpha[i - 1];
//           }
//         }
//       }
//     }
//
//     // triangulate
//     std::vector<std::vector<face>> faces;
//     for (auto& subfront : this->fronts.back()) {
//       this->subdivide(subfront, desired_spatial_dist);
//       this->reduce(subfront, desired_spatial_dist);
//       faces.push_back(this->triangulate_timeline(subfront));
//     }
//     return faces;
//   }
//
//   //----------------------------------------------------------------------------
//   std::vector<Real> optimal_stepsizes(const vertex_list_t& vs) {
//     const auto& v       = this->ssf->vectorfield();
//     auto        jacobian = make_jacobian(v);
//
//     auto                num_pnts = vs.size();
//     std::vector<Real> p(num_pnts - 1), q(num_pnts - 1), null(num_pnts),
//         r(num_pnts);
//     std::vector<vec<Real, N>> v(num_pnts);
//
//     size_t i = 0;
//     for (const auto vertex : vs)
//       v[i++] = v(at(vertex), this->ssf->t0() + uv(vertex)(1));
//
//     i              = 0;
//     Real avg_len = 0;
//     for (auto vertex = begin(vs); vertex != prev(end(vs));
//          ++vertex, ++i) {
//       auto tm = this->ssf->t0() + (uv(*vertex)(1) + uv(*next(vertex))(1)) *
//       0.5; auto xm = (at(*next(vertex)) + at(*vertex)) * 0.5; auto dir =
//       at(*next(vertex)) - at(*vertex); auto vm  = v(xm, tm); auto Jm  =
//       jacobian(xm, tm);
//
//       p[i] = dot(dir * 0.5, Jm * v[i]) - dot(v[i], vm);
//       q[i] = dot(dir * 0.5, Jm * v[i + 1]) + dot(v[i + 1], vm);
//       r[i] = -dot(dir, vm);
//
//       avg_len += norm(dir);
//     }
//     avg_len /= num_pnts - 1;
//     solve_qr(num_pnts - 1, &p[0], &q[0], &r[0], &null[0]);
//
//     // Real nrm{0};
//     // for (auto x : null) nrm += x * x;
//     // nrm = sqrt(nrm);
//     // for (size_t i = 0; i < null.size(); ++i) null[i] /= nrm;
//
//     // count positive entries in nullspace
//     size_t num_pos_null = 0;
//     for (auto c : null)
//       if (c > 0) ++num_pos_null;
//     int k_plus_factor = (num_pos_null < null.size() / 2) ? -1 : 1;
//
//     for (size_t i = 0; i < r.size(); ++i) r[i] += k_plus_factor * null[i];
//     // r[i] += (num_pnts / 10.0) * k_plus_factor * null[i];
//
//     // apply step width
//     Real h = std::numeric_limits<Real>::max();
//     for (size_t i = 0; i < num_pnts; ++i)
//       h = std::min(h, avg_len / (std::abs(r[i]) * norm(v[i])));
//     for (size_t i = 0; i < r.size(); ++i) r[i] *= h;
//     // for (size_t i = 0; i < r.size(); ++i) r[i] = std::max(r[i],1e-3);
//     return r;
//   }
//
//   //----------------------------------------------------------------------------
//   auto detect_peaks(const std::vector<Real>& alpha,
//                     const vertex_list_t& vs, Real threshold = 100) {
//     // calculate second derivative
//     std::vector<Real> snd_der(alpha.size(), 0);
//     auto                v = next(begin(vs));
//     for (size_t i = 1; i < alpha.size() - 1; ++i, ++v) {
//       Mat<Real, 3, 3> A{
//           1,
//           1,
//           1,
//
//           uv(*prev(v))(0),
//           uv(*v)(0),
//           uv(*next(v))(0),
//
//           uv(*prev(v))(0) * uv(*prev(v))(0),
//           uv(*v)(0) * uv(*v)(0),
//           uv(*next(v))(0) * uv(*next(v))(0),
//       };
//       vec<Real, 3> b{alpha[i - 1], alpha[i], alpha[i + 1]};
//       snd_der[i]                     = 2 * solve(A, b)(2);
//       second_derivate_alpha_prop[*v] = std::abs(snd_der[i]);
//     }
//
//     std::vector<std::pair<size_t, size_t>> splitted_front_indices;
//     splitted_front_indices.emplace_back(0, alpha.size() - 1);
//
//     for (size_t i = 1; i < snd_der.size() - 1; ++i)
//       if (std::abs(snd_der[i]) > threshold) {
//         if (splitted_front_indices.back().first == i)
//           // shift left border of split to the right
//           ++splitted_front_indices.back().first;
//         else {
//           // insert new subfront
//           splitted_front_indices.back().second = i - 1;
//           if (splitted_front_indices.back().first ==
//               splitted_front_indices.back().second)
//             splitted_front_indices.pop_back();
//
//           splitted_front_indices.emplace_back(i + 1, alpha.size() - 1);
//         }
//       }
//     if (splitted_front_indices.back().first ==
//         splitted_front_indices.back().second)
//       splitted_front_indices.pop_back();
//     return splitted_front_indices;
//   }
// };

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
