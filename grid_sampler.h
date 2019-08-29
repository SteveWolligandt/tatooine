#ifndef TATOOINE_GRID_SAMPLER_H
#define TATOOINE_GRID_SAMPLER_H

//==============================================================================
#include <array>
#include <boost/range/algorithm.hpp>
#include <cmath>
#include <png++/png.hpp>
#include <random>
#include <vector>
#include "amira_file.h"
#include "boundingbox.h"
#include "chunked_data.h"
#include "crtp.h"
#include "grid.h"
#include "interpolators.h"
#include "parallel_for.h"
#include "sampled_field.h"
#include "scalarfield.h"
#include "traits.h"
#include "vtk_legacy.h"

#include "concept_defines.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename T>
struct num_components;
template <typename T>
static constexpr size_t num_components_v = num_components<T>::value;

template <typename real_t, size_t n>
struct num_components<vec<real_t, n>> : std::integral_constant<size_t, n> {};

template <>
struct num_components<double> : std::integral_constant<size_t, 1> {};

template <>
struct num_components<float> : std::integral_constant<size_t, 1> {};

template <typename T>
struct internal_data_type;

template <typename real_t, size_t n>
struct internal_data_type<vec<real_t, n>> {
  using type = real_t;
};

template <>
struct internal_data_type<double> {
  using type = double;
};

template <>
struct internal_data_type<float> {
  using type = float;
};

template <typename T>
using internal_data_type_t = typename internal_data_type<T>::type;

template <size_t n, typename real_t, typename data_t,
          template <typename> typename head_interpolator_t,
          template <typename> typename... tail_interpolator_ts>
struct grid_sampler;

//==============================================================================
template <size_t n, typename real_t, typename data_t, typename top_grid_t,
          template <typename> typename head_interpolator_t,
          template <typename> typename... tail_interpolator_ts>
struct grid_sampler_view;

//==============================================================================
template <size_t n, typename real_t, typename data_t, typename grid_t,
          template <typename> typename... interpolator_ts>
struct grid_sampler_iterator;

//==============================================================================
template <size_t n, typename real_t, typename data_t, typename grid_t,
          template <typename> typename... interpolator_ts>
struct base_grid_sampler_at {
  using type =
      grid_sampler_view<n - 1, real_t, data_t, grid_t, interpolator_ts...>;
  using const_type = grid_sampler_view<n - 1, real_t, data_t, const grid_t,
                                       interpolator_ts...>;
};

//==============================================================================
template <typename real_t, typename data_t, typename grid_t,
          template <typename> typename... interpolator_ts>
struct base_grid_sampler_at<1, real_t, data_t, grid_t, interpolator_ts...> {
  using type       = std::decay_t<data_t>&;
  using const_type = const std::decay_t<data_t>;
};

//==============================================================================
template <size_t n, typename real_t, typename data_t, typename T,
          template <typename> typename... interpolator_ts>
using base_grid_sampler_at_t =
    typename base_grid_sampler_at<n, real_t, data_t, T,
                                  interpolator_ts...>::type;

//==============================================================================
template <size_t n, typename real_t, typename data_t, typename T,
          template <typename> typename... interpolator_ts>
using base_grid_sampler_at_ct =
    typename base_grid_sampler_at<n, real_t, data_t, T,
                                  interpolator_ts...>::const_type;

//==============================================================================
//! CRTP inheritance class for grid_sampler and grid_sampler_view
template <typename derived_t, size_t N, typename Real, typename Data,
          template <typename> typename head_interpolator_t,
          template <typename> typename... tail_interpolator_ts>
struct base_grid_sampler : crtp<derived_t>, grid<N, Real> {
  static_assert(N > 0, "grid_sampler must have at least one dimension");
  static_assert(
      N == sizeof...(tail_interpolator_ts) + 1,
      "number of interpolator kernels does not match number of dimensions");

  //----------------------------------------------------------------------------
  static constexpr auto n                = N;
  using real_t                           = Real;
  using data_t                           = Data;
  using internal_data_t                  = internal_data_type_t<Data>;
  static constexpr size_t num_components = num_components_v<Data>;
  using this_t =
      base_grid_sampler<derived_t, n, real_t, data_t, head_interpolator_t,
                        tail_interpolator_ts...>;
  using grid_t           = grid<n, real_t>;
  using iterator         = grid_sampler_iterator<n, real_t, data_t, this_t>;
  using indexing_t       = base_grid_sampler_at_t<n, real_t, data_t, this_t,
                                            tail_interpolator_ts...>;
  using const_indexing_t = base_grid_sampler_at_ct<n, real_t, data_t, this_t,
                                                   tail_interpolator_ts...>;
  using crtp<derived_t>::as_derived;
  using grid_t::dimension;
  using grid_t::dimensions;
  using grid_t::resolution;

  //----------------------------------------------------------------------------
  struct out_of_domain : std::runtime_error {
    out_of_domain(const std::string& err) : std::runtime_error{err} {}
  };

  base_grid_sampler() = default;
  base_grid_sampler(const grid_t& g) : grid_t{g} {}
  base_grid_sampler(grid_t&& g) : grid_t{std::move(g)} {}
  template <typename... real_ts>
  base_grid_sampler(const linspace<real_ts>&... linspaces)
      : grid_t{linspaces...} {}
  base_grid_sampler(const base_grid_sampler& other) : grid_t{other} {}
  base_grid_sampler(base_grid_sampler&& other) noexcept
      : grid_t{std::move(other)} {}
  auto& operator=(const base_grid_sampler& other) {
    grid_t::operator=(other);
    return *this;
  }
  auto& operator=(base_grid_sampler&& other) noexcept {
    grid_t::operator=(std::move(other));
    return *this;
  }
  template <typename other_real_t>
  auto& operator=(const grid<n, other_real_t>& other) {
    grid_t::operator=(other);
    return *this;
  }
  template <typename other_real_t>
  auto& operator=(grid<n, other_real_t>&& other) {
    grid_t::operator=(std::move(other));
    return *this;
  }

  //! data at specified indices is...
  //! CRTP-virtual method
  template <typename... Is,
            typename = std::enable_if_t<are_integral_v<std::decay_t<Is>...>>>
  data_t& data(Is&&... is) {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return as_derived().data(is...);
  }

  //! data at specified indices is...
  //! CRTP-virtual method
  template <typename... Is,
            typename = std::enable_if_t<are_integral_v<std::decay_t<Is>...>>>
  data_t data(Is&&... is) const {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return as_derived().data(is...);
  }

  //----------------------------------------------------------------------------
  //! indexing of data.
  //! if n == 1 returns actual data otherwise returns a grid_sampler_view with i
  //! as fixed index
  const_indexing_t at(size_t i) const {
    if constexpr (n > 1) {
      return const_indexing_t{this, i};
    } else {
      return data(i);
    }
  }
  const_indexing_t operator[](size_t i) const { return at(i); }

  indexing_t at(size_t i) {
    if constexpr (n > 1) {
      return indexing_t{this, i};
    } else {
      return data(i);
    }
  }
  indexing_t operator[](size_t i) { return at(i); }

  //----------------------------------------------------------------------------
  template <typename... Xs,
            typename = std::enable_if_t<are_arithmetic_v<std::decay_t<Xs>...>>>
  auto operator()(Xs&&... xs) const {
    static_assert(sizeof...(Xs) == n,
                  "number of coordinates does not match number of dimensions");
    return sample(std::forward<Xs>(xs)...);
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  auto operator()(const std::array<real_t, n>& xs) const { return sample(xs); }

  //----------------------------------------------------------------------------
  //! sampling by interpolating using head_interpolator_t and
  //! grid_iterators
  template <typename... Xs>
  auto sample(real_t x, Xs&&... xs) const {
    static_assert(sizeof...(Xs) + 1 == n,
                  "number of coordinates does not match number of dimensions");
    x        = domain_to_global(x, 0);
    size_t i = std::floor(x);
    real_t t = x - std::floor(x);
    if (begin() + i + 1 == end()) {
      return head_interpolator_t<real_t>::template interpolate_iter(
          begin() + i, begin() + i, begin(), end(), t, std::forward<Xs>(xs)...);
    }
    return head_interpolator_t<real_t>::template interpolate_iter(
        begin() + i, begin() + i + 1, begin(), end(), t,
        std::forward<Xs>(xs)...);
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <std::size_t... Is>
  auto sample(const std::array<real_t, n>& xs,
              std::index_sequence<Is...> /*is*/) const {
    return sample(xs[Is]...);
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  auto sample(const std::array<real_t, n>& xs) const {
    return sample(xs, std::make_index_sequence<n>{});
  }

  //----------------------------------------------------------------------------
  auto domain_to_global(real_t x, size_t i) const {
    auto converted =
        (x - dimension(i).min) / (dimension(i).max - dimension(i).min);
    if (converted < 0 || converted > 1) {
      throw out_of_domain{std::to_string(x) + " in dimension " +
                          std::to_string(i)};
    }
    return converted * (dimension(i).resolution - 1);
  }

  //============================================================================
  //! converts index list to global index in m_data
  template <typename... Is, typename = std::enable_if_t<are_integral_v<Is...>>>
  auto global_index(Is... is) const {
    static_assert(sizeof...(Is) == n);
    static_assert(are_integral_v<Is...>);
    return global_index(std::array<size_t, n>{size_t(is)...});
  }

  //----------------------------------------------------------------------------
  //! converts index list to global index in m_data
  auto global_index(const std::array<size_t, n>& is) const {
    size_t multiplier = 1;
    size_t gi         = 0;
    size_t counter    = 0;
    for (auto i : is) {
      gi += multiplier * i;
      multiplier *= dimension(counter++).resolution;
    }
    return gi;
  }

  //----------------------------------------------------------------------------

  auto begin() const { return iterator{this, 0}; }
  auto end() const { return iterator{this, resolution().front()}; }
};

//==============================================================================
make_type_assert(reg_grid_write_png_type_assert, "type not allowed", float,
                 double, vec<float, 4>, vec<double, 4>)

    //==============================================================================
    //! holds actual data
    template <size_t N, typename Real, typename Data,
              template <typename> typename head_interpolator_t,
              template <typename> typename... tail_interpolator_ts>
    struct grid_sampler
    : base_grid_sampler<grid_sampler<N, Real, Data, head_interpolator_t,
                                     tail_interpolator_ts...>,
                        N, Real, Data, head_interpolator_t,
                        tail_interpolator_ts...> {
  using real_t            = Real;
  using data_t            = Data;
  static constexpr auto n = N;
  using vec_t             = vec<real_t, n>;
  using this_t            = grid_sampler<n, real_t, data_t, head_interpolator_t,
                              tail_interpolator_ts...>;
  using parent_t =
      base_grid_sampler<this_t, n, real_t, data_t, head_interpolator_t,
                        tail_interpolator_ts...>;
  using internal_data_t                  = typename parent_t::internal_data_t;
  static constexpr size_t num_components = parent_t::num_components;
  using iterator =
      grid_sampler_iterator<n, real_t, data_t, this_t, tail_interpolator_ts...>;
  using parent_t::dimension;
  using parent_t::dimensions;
  using parent_t::resolution;

  //============================================================================
 private:
  chunked_data<Data, n> m_data;

  //============================================================================
 public:
  template <std::size_t... Is>
  grid_sampler(std::index_sequence<Is...> /*is*/)
      : parent_t{((void)Is, linspace<real_t>{0, 1, 1})...},
        m_data{((void)Is, 1)...} {}

  //----------------------------------------------------------------------------
  grid_sampler() : grid_sampler{std::make_index_sequence<n>{}} {}

  //----------------------------------------------------------------------------
  grid_sampler(const grid_sampler& other)
      : parent_t{other}, m_data{other.m_data} {}

  //----------------------------------------------------------------------------
  grid_sampler(grid_sampler&& other) noexcept
      : parent_t{other}, m_data{std::move(other.m_data)} {}

  //----------------------------------------------------------------------------
  auto& operator=(const grid_sampler& other) {
    parent_t::operator=(other);
    m_data            = other.m_data;
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(grid_sampler&& other) noexcept {
    parent_t::operator=(std::move(other));
    return *this;
  }

  //============================================================================
  template <
      typename... Resolution,
      typename = std::enable_if_t<are_integral_v<std::decay_t<Resolution>...>>>
  grid_sampler(Resolution... resolution)
      : parent_t{linspace<real_t>{0, 1, std::size_t(resolution)}...},
        m_data(resolution...) {}

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  grid_sampler(const linspace<real_ts>&... linspaces)
      : parent_t{linspaces...}, m_data{linspaces.resolution...} {}

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  grid_sampler(const std::array<std::size_t, n>& resolution,
               std::index_sequence<Is...> /*is*/)
      : parent_t{linspace<real_t>{0, 1, std::size_t(resolution[Is])}...},
        m_data(resolution[Is]...) {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const std::array<std::size_t, n>& resolution)
      : grid_sampler(resolution, std::make_index_sequence<n>{}) {}

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  grid_sampler(const grid<n, real_t>& domain, std::index_sequence<Is...> /*is*/)
      : parent_t{domain}, m_data{domain.dimension(Is).resolution...} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const grid<n, real_t>& domain)
      : grid_sampler(domain, std::make_index_sequence<n>{}) {}

  //============================================================================
  template <
      typename... Resolution,
      typename = std::enable_if_t<are_integral_v<std::decay_t<Resolution>...>>>
  grid_sampler(const std::vector<real_t>& data, Resolution... resolution)
      : parent_t{linspace<real_t>{0, 1, std::size_t(resolution)}...},
        m_data{data, resolution...} {}

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  grid_sampler(const std::vector<real_t>& data,
               const linspace<real_ts>&... linspaces)
      : parent_t{linspaces...}, m_data{data, linspaces.resolution...} {}

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  grid_sampler(const std::vector<real_t>&        data,
               const std::array<std::size_t, n>& resolution,
               std::index_sequence<Is...> /*is*/)
      : parent_t{linspace<real_t>{0, 1, std::size_t(resolution[Is])}...},
        m_data{data, resolution} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const std::vector<real_t>&        data,
               const std::array<std::size_t, n>& resolution)
      : grid_sampler(data, resolution, std::make_index_sequence<n>{}) {}

  //----------------------------------------------------------------------------
  // template <std::size_t... Is>
  // grid_sampler(const std::vector<real_t>& data, const grid<n, real_t>&
  // domain,
  //              std::index_sequence<Is...>)
  //     : parent_t{domain}, m_data{data} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // grid_sampler(const std::vector<real_t>& data, const grid<n, real_t>&
  // domain)
  //     : grid_sampler(data, domain, std::make_index_sequence<n>{}) {}

  //============================================================================
  grid_sampler(const std::string& filename) : grid_sampler{} { read(filename); }

  //----------------------------------------------------------------------------
  // grid_sampler(const std::vector<std::string>& filenames)
  //     : m_size(0), m_domain{vec_t{fill{0}}, vec_t{fill{1}}} {
  //   read(filenames);
  // }

  //----------------------------------------------------------------------------
  template <typename... Is, typename = std::enable_if_t<are_integral_v<Is...>>>
  decltype(auto) data(Is... is) const {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return m_data(is...);
  }
  template <typename... Is, typename = std::enable_if_t<are_integral_v<Is...>>>
  decltype(auto) data(Is... is) {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return m_data(is...);
  }

  //----------------------------------------------------------------------------
  const auto& data() const { return m_data; }
  auto&       data() { return m_data; }

  //============================================================================
  template <
      typename... Resolution,
      typename = std::enable_if_t<are_integral_v<std::decay_t<Resolution>...>>>
  void resize(Resolution... resolution) {
    parent_t::operator=
        (grid{linspace<real_t>{0, 1, std::size_t(resolution)}...});
    m_data.resize((resolution * ...));
  }

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  void resize(const linspace<real_ts>&... linspaces) {
    parent_t::operator=(grid{linspaces...});
    m_data.resize(linspaces.resolution...);
  }

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  void resize(const std::array<std::size_t, n>& resolution,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator-
        (grid{linspace<real_t>{0, 1, std::size_t(resolution[Is])}...});
    m_data.resize((resolution[Is] * ...));
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  void resize(const std::array<std::size_t, n>& resolution) {
    resize(resolution, std::make_index_sequence<n>{});
  }

  //----------------------------------------------------------------------------
  template <typename other_real_t, std::size_t... Is>
  void resize(const grid<n, other_real_t>& domain,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=(domain);
    m_data.resize((domain.resolution()[Is] * ...));
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename other_real_t, std::size_t... Is>
  void resize(const grid<n, other_real_t>& domain) {
    resize(domain, std::make_index_sequence<n>{});
  }

  //============================================================================
  template <
      typename... Resolution,
      typename = std::enable_if_t<are_integral_v<std::decay_t<Resolution>...>>>
  void resize(const std::vector<data_t>& data, Resolution... resolution) {
    parent_t::operator=
        (grid{linspace<real_t>{0, 1, std::size_t(resolution)}...});
    m_data = data;
  }

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  void resize(const std::vector<data_t>& data,
              const linspace<real_ts>&... linspaces) {
    parent_t::operator=(grid{linspaces...});
    m_data            = data;
  }

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  void resize(const std::vector<data_t>&        data,
              const std::array<std::size_t, n>& resolution,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=
        (grid{linspace<real_t>{0, 1, std::size_t(resolution[Is])}...});
    m_data = data;
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  void resize(const std::vector<data_t>&        data,
              const std::array<std::size_t, n>& resolution) {
    resize(data, resolution, std::make_index_sequence<n>{});
  }

  //----------------------------------------------------------------------------
  template <typename other_real_t, std::size_t... Is>
  void resize(const std::vector<data_t>&   data,
              const grid<n, other_real_t>& domain,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=(domain);
    m_data.resize(domain.resolution(Is)...);
    m_data = data;
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename other_real_t>
  void resize(const std::vector<data_t>&   data,
              const grid<n, other_real_t>& domain) {
    resize(data, domain, std::make_index_sequence<n>{});
  }

  //============================================================================
  //   template <typename _data_t = data_t,
  //             typename = std::enable_if_t<std::is_same_v<_data_t, real_t>>>
  //   auto min_value() {
  //     return *std::min_element(begin(m_data), end(m_data));
  //   }
  //
  //   //----------------------------------------------------------------------------
  //   template <typename _data_t = data_t,
  //             typename = std::enable_if_t<std::is_same_v<_data_t, real_t>>>
  //   auto max_value() {
  //     return *std::max_element(begin(m_data), end(m_data));
  //   }
  //
  //   //----------------------------------------------------------------------------
  //   template <typename _data_t = data_t,
  //             typename = std::enable_if_t<std::is_same_v<_data_t, real_t>>>
  //   auto minmax_value() {
  //     auto [min_it, max_it] = std::minmax_element(begin(m_data),
  //     end(m_data)); return std::pair{*min_it, *max_it};
  //   }
  //
  //   //----------------------------------------------------------------------------
  //   template <typename other_real_t, size_t _n = n, typename _data_t =
  //   data_t,
  //             typename = std::enable_if_t<std::is_arithmetic_v<_data_t>>,
  //             typename = std::enable_if_t<_n == 2>>
  //   auto filter(const Mat<other_real_t, 3, 3>& kernel) {
  //     auto copy = *this;
  // #pragma omp parallel for collapse(2)
  //     for (size_t x = 1; x < dimension(0).resolution - 1; ++x)
  //       for (size_t y = 1; y < dimension(1).resolution - 1; ++y) {
  //         // copy neighbour data
  //         Mat<real_t, 3, 3> neighbour_data{
  //             data(x - 1, y + 1), data(x - 1, y), data(x - 1, y - 1),  //
  //             data(x, y + 1),     data(x, y),     data(x, y - 1),      //
  //             data(x + 1, y + 1), data(x + 1, y), data(x + 1, y - 1)};
  //         copy.data(x, y) = 0;
  //         for (size_t i = 0; i < 3; ++i)
  //           for (size_t j = 0; j < 3; ++j)
  //             copy.data(x, y) += neighbour_data(i, j) * kernel(i, j);
  //       }
  //     return copy;
  //   }
  //
  //   //----------------------------------------------------------------------------
  //   void remap(real_t small, real_t large) {
  //     if constexpr (std::is_same_v<real_t, data_t>)
  //       boost::transform(m_data, begin(m_data), [small, large](auto d) {
  //         return (d - small) / (large - small);
  //       });
  //   }
  //
  //   //----------------------------------------------------------------------------
  //   void remap() {
  //     if constexpr (std::is_same_v<real_t, data_t>) {
  //       auto [min, max] = minmax_value();
  //       std::cerr << min << ", "<< max << '\n';
  //       remap(min, max);
  //     }
  //   }
  //
  //----------------------------------------------------------------------------
  template <typename random_engine_t = std::mt19937, typename _Data = data_t,
            typename = std::enable_if_t<std::is_arithmetic_v<_Data>>>
  void randu(real_t lower_boundary, real_t upper_boundary,
             random_engine_t&& random_engine = random_engine_t{
                 std::random_device{}()}) {
    m_data.randu(lower_boundary, upper_boundary,
                 std::forward<random_engine_t>(random_engine));
  }
  //----------------------------------------------------------------------------
  template <typename random_engine_t = std::mt19937, typename _Data = data_t,
            typename = std::enable_if_t<std::is_arithmetic_v<_Data>>>
  void randu(random_engine_t&& random_engine = random_engine_t{
                 std::random_device{}()}) {
    randu(0, 1, std::forward<random_engine_t>(random_engine));
  }

  //----------------------------------------------------------------------------
  void read(const std::string& filename) {
    auto ext = filename.substr(filename.find_last_of('.') + 1);
    if constexpr (n == 3) {
      if (ext == "am") {
        read_amira(filename);
        return;
      }
    }

    if (ext == "vtk") {
      read_vtk(filename);
      return;
    }

    throw std::runtime_error("unknown file extension");
  }

  //----------------------------------------------------------------------------
  template <size_t n_ = n>
  void read_amira(const std::string& filename) {
    static_assert(n_ == 3,
                  "sampler must have 3 dimensions for reading amira files");
    auto [data, dims, domain, amira_num_components] = amira::read(filename);
    if constexpr (std::is_same_v<float, real_t>) {
      resize(std::move(data), grid{domain, dims});
    } else {
      std::vector<data_t> casted_data(data.size() / amira_num_components);
      auto ptr = reinterpret_cast<internal_data_t*>(casted_data.data());
      for (size_t i = 0; i < casted_data.size(); ++i) {
        for (size_t j = 0; j < num_components; ++j) {
          ptr[i * num_components + j] = data[i * amira_num_components + j];
        }
      }
      resize(std::move(casted_data), grid<n, real_t>{domain, dims});
    }
  }

  //----------------------------------------------------------------------------
  void read_vtk(const std::string& filename) {
    read_vtk(filename, std::make_index_sequence<n>{});
  }
  template <std::size_t... Is>
  void read_vtk(const std::string& filename,
                std::index_sequence<Is...> /*is*/) {
    struct listener_t : vtk::LegacyFileListener {
      std::array<size_t, 3>              dims;
      std::array<real_t, 3>              origin, spacing;
      std::vector<std::array<real_t, 3>> data;
      std::array<real_t, 3>              min_coord, max_coord;
      vtk::DatasetType                   type = vtk::UNKNOWN_TYPE;

      void on_dataset_type(vtk::DatasetType _type) override { type = _type; }
      void on_dimensions(size_t x, size_t y, size_t z) override {
        if constexpr (n == 2) {
          dims = {x, y};

        } else if constexpr (n == 3) {
          dims = {x, y, z};
        }
      }
      void on_x_coordinates(const std::vector<real_t>& xs) override {
        min_coord[0] = xs.front();
        max_coord[0] = xs.back();
      }
      void on_y_coordinates(const std::vector<real_t>& ys) override {
        min_coord[1] = ys.front();
        max_coord[1] = ys.back();
      }
      void on_z_coordinates(const std::vector<real_t>& zs) override {
        min_coord[2] = zs.front();
        max_coord[2] = zs.back();
      }
      void on_spacing(real_t x, real_t y, real_t z) override {
        spacing = {x, y, z};
      }
      void on_origin(real_t x, real_t y, real_t z) override {
        origin = {x, y, z};
      }
      void on_vectors(const std::string& /* name */,
                      const std::vector<std::array<real_t, 3>>& vectors,
                      vtk::ReaderData /*data*/) override {
        data = vectors;
      }
    } listener;

    vtk::LegacyFile file(filename);
    file.add_listener(listener);
    file.read();

    if (listener.type == vtk::STRUCTURED_POINTS) {
      resize(linspace{
          listener.origin[Is],
          listener.origin[Is] + listener.spacing[Is] * (listener.dims[Is] - 1),
          listener.dims[Is]}...);

    } else if (listener.type == vtk::RECTILINEAR_GRID) {
      resize(linspace{listener.min_coord[Is], listener.max_coord[Is],
                      listener.dims[Is]}...);

    } else {
      throw std::runtime_error{"type not known"};
    }

    std::vector<data_t> casted_data(listener.data.size());
    // auto ptr = reinterpret_cast<internal_data_t*>(casted_data.data());
    // for (size_t i = 0; i < listener.data.size(); ++i) { ptr[i] =
    // listener.data[i]; }
    boost::transform(listener.data, begin(casted_data), [](const auto& v) {
      return vec{v[0], v[1]};
    });
    m_data = std::move(casted_data);
  }

  //----------------------------------------------------------------------------
  template <std::size_t _n = n>
  void write_png(const std::string& filepath) {
    static_assert(_n == 2,
                  "cannot write sampler of dimenion other than 2 to png");

    if constexpr (std::is_same_v<data_t, real_t>) {
      png::image<png::rgb_pixel> image(dimension(0).resolution,
                                       dimension(1).resolution);
      for (unsigned int y = 0; y < image.get_height(); ++y) {
        for (png::uint_32 x = 0; x < image.get_width(); ++x) {
          unsigned int idx = x + dimension(0).resolution * y;

          image[image.get_height() - 1 - y][x].red =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx])) * 255;
          image[image.get_height() - 1 - y][x].green =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx])) * 255;
          image[image.get_height() - 1 - y][x].blue =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx])) * 255;
        }
      }
      image.write(filepath);

    } else if constexpr (std::is_same_v<data_t, vec<real_t, 4>>) {
      png::image<png::rgba_pixel> image(dimension(0).resolution,
                                        dimension(1).resolution);
      for (unsigned int y = 0; y < image.get_height(); ++y) {
        for (png::uint_32 x = 0; x < image.get_width(); ++x) {
          unsigned int idx = x + dimension(0).resolution * y;

          image[image.get_height() - 1 - y][x].red =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx * 4 + 0])) *
              255;
          image[image.get_height() - 1 - y][x].green =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx * 4 + 1])) *
              255;
          image[image.get_height() - 1 - y][x].blue =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx * 4 + 2])) *
              255;
          image[image.get_height() - 1 - y][x].alpha =
              std::max<real_t>(0, std::min<real_t>(1, m_data[idx * 4 + 3])) *
              255;
        }
      }
      image.write(filepath);
    } else {
      reg_grid_write_png_type_assert<data_t>();
    }
  }
};  // namespace tatooine

//==============================================================================
//! holds an object of type top_grid_t which can either be
//! grid_sampler or grid_sampler_view and a fixed index of the top
//! grid_sampler
template <size_t n, typename real_t, typename data_t, typename top_grid_t,
          template <typename> typename head_interpolator_t,
          template <typename> typename... tail_interpolator_ts>
struct grid_sampler_view
    : base_grid_sampler<
          grid_sampler_view<n, real_t, data_t, top_grid_t, head_interpolator_t,
                            tail_interpolator_ts...>,
          n, real_t, data_t, head_interpolator_t, tail_interpolator_ts...> {
  using parent_t = base_grid_sampler<
      grid_sampler_view<n, real_t, data_t, top_grid_t, head_interpolator_t,
                        tail_interpolator_ts...>,
      n, real_t, data_t, head_interpolator_t, tail_interpolator_ts...>;

  top_grid_t* top_grid;
  size_t      fixed_index;

  //----------------------------------------------------------------------------

  template <std::size_t... Is>
  grid_sampler_view(top_grid_t* _top_grid, size_t _fixed_index,
                    std::index_sequence<Is...> /*is*/)
      : parent_t{_top_grid->dimension(Is + 1)...},
        top_grid{_top_grid},
        fixed_index{_fixed_index} {}

  grid_sampler_view(top_grid_t* _top_grid, size_t _fixed_index)
      : grid_sampler_view{_top_grid, _fixed_index,
                          std::make_index_sequence<n>{}} {}

  //! returns data of top grid at fixed_index and index list is...
  template <typename T = top_grid_t,
            typename   = std::enable_if_t<!std::is_const_v<T>>, typename... Is,
            typename   = std::enable_if_t<are_integral_v<std::decay_t<Is>...>>>
  data_t& data(Is&&... is) {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return top_grid->data(fixed_index, is...);
  }

  //! returns data of top grid at fixed_index and index list is...
  template <typename... Is,
            typename = std::enable_if_t<are_integral_v<std::decay_t<Is>...>>>
  decltype(auto) data(Is&&... is) const {
    static_assert(sizeof...(Is) == n,
                  "number of indices is not equal to number of dimensions");
    return top_grid->data(fixed_index, is...);
  }
};

//==============================================================================
//! holds an object of type grid_t which either can be
//! grid_sampler or grid_sampler_view and an index of that grid
template <size_t n, typename real_t, typename data_t, typename grid_t,
          template <typename> typename... tail_interpolator_ts>
struct grid_sampler_iterator {
  using this_t =
      grid_sampler_iterator<n, real_t, data_t, grid_t, tail_interpolator_ts...>;

  //----------------------------------------------------------------------------

  const grid_t* m_grid;
  size_t        m_index;

  //----------------------------------------------------------------------------

  auto operator*() const { return m_grid->at(m_index); }

  auto& operator++() {
    ++m_index;
    return *this;
  }

  auto& operator--() {
    --m_index;
    return *this;
  }

  bool operator==(const this_t& other) const {
    return m_grid == other.m_grid && m_index == other.m_index;
  }

  bool operator!=(const this_t& other) const {
    return m_grid != other.m_grid || m_index != other.m_index;
  }
  bool operator<(const this_t& other) const { return m_index < other.m_index; }
  bool operator>(const this_t& other) const { return m_index > other.m_index; }
  bool operator<=(const this_t& other) const {
    return m_index <= other.m_index;
  }
  bool operator>=(const this_t& other) const {
    return m_index >= other.m_index;
  }

  auto operator+(size_t rhs) { return this_t{m_grid, m_index + rhs}; }
  auto operator-(size_t rhs) { return this_t{m_grid, m_index - rhs}; }

  auto& operator+=(size_t rhs) {
    m_index += rhs;
    return *this;
  }
  auto& operator-=(size_t rhs) {
    m_index -= rhs;
    return *this;
  }
};

//==============================================================================
//! next specification for grid_sampler_iterator
template <size_t n, typename real_t, typename data_t, typename grid_t,
          template <typename> typename... tail_interpolator_ts>
auto next(const grid_sampler_iterator<n, real_t, data_t, grid_t,
                                      tail_interpolator_ts...>& it,
          size_t                                                x = 1) {
  return grid_sampler_iterator<n, real_t, data_t, grid_t,
                               tail_interpolator_ts...>{it.m_grid,
                                                        it.m_index + x};
}

//------------------------------------------------------------------------------
//! prev specification for grid_sampler_iterator
template <size_t n, typename real_t, typename data_t, typename grid_t,
          template <typename> typename... tail_interpolator_ts>
auto prev(const grid_sampler_iterator<n, real_t, data_t, grid_t,
                                      tail_interpolator_ts...>& it,
          size_t                                                x = 1) {
  return grid_sampler_iterator<n, real_t, data_t, grid_t,
                               tail_interpolator_ts...>{it.m_grid,
                                                        it.m_index - x};
}

//==============================================================================
//! resamples a grid_sampler
template <size_t n, typename real_t, typename data_t,
          template <typename> typename... interpolator_ts,
          typename... Resolution,
          typename = std::enable_if_t<sizeof...(Resolution) == n>>
auto resample(
    const grid_sampler<n, real_t, data_t, interpolator_ts...>& sampler,
    Resolution&&... resolution) {
  static_assert(n > 0);
  static_assert(n <= 4);
  std::array<size_t, n> resampled_size{size_t(resolution)...};
  std::vector<real_t>   resampled_data;
  resampled_data.reserve((resolution * ...));

  // 1D specification
  if constexpr (n == 1) {
    for (size_t x = 0; x < resampled_size[0]; ++x) {
      resampled_data.push_back(
          sampler(real_t(x) / (resampled_size[0] - 1) * (sampler.size(0) - 1)));
    }

    // 2D specification
  } else if constexpr (n == 2) {
    for (size_t y = 0; y < resampled_size[1]; ++y) {
      for (size_t x = 0; x < resampled_size[0]; ++x) {
        resampled_data.push_back(sampler(
            real_t(x) / (resampled_size[0] - 1) * (sampler.size(0) - 1),
            real_t(y) / (resampled_size[1] - 1) * (sampler.size(1) - 1)));
      }
    }

    // 3D specification
  } else if constexpr (n == 3) {
    for (size_t z = 0; z < resampled_size[2]; ++z) {
      for (size_t y = 0; y < resampled_size[1]; ++y) {
        for (size_t x = 0; x < resampled_size[0]; ++x) {
          resampled_data.push_back(sampler(
              real_t(x) / (resampled_size[0] - 1) * (sampler.size(0) - 1),
              real_t(y) / (resampled_size[1] - 1) * (sampler.size(1) - 1),
              real_t(z) / (resampled_size[2] - 1) * (sampler.size(2) - 1)));
        }
      }
    }

    // 4D specification
  } else if constexpr (n == 4) {
    for (size_t w = 0; w < resampled_size[3]; ++w) {
      for (size_t z = 0; z < resampled_size[2]; ++z) {
        for (size_t y = 0; y < resampled_size[1]; ++y) {
          for (size_t x = 0; x < resampled_size[0]; ++x) {
            resampled_data.push_back(sampler(
                real_t(x) / (resampled_size[0] - 1) * (sampler.size(0) - 1),
                real_t(y) / (resampled_size[1] - 1) * (sampler.size(1) - 1),
                real_t(z) / (resampled_size[2] - 1) * (sampler.size(2) - 1),
                real_t(w) / (resampled_size[3] - 1) * (sampler.size(3) - 1)));
          }
        }
      }
    }
  }

  return grid_sampler<n, real_t, data_t, interpolator_ts...>(
      std::move(resampled_data), resampled_size);
}

//==============================================================================
template <size_t n, typename real_t,
          template <typename> typename... interpolator_ts>
using SteadyGridSamplerSF =
    sampled_scalarfield<n, real_t,
                        grid_sampler<n, real_t, real_t, interpolator_ts...>>;

//==============================================================================
template <size_t n, typename real_t, size_t vec_dim,
          template <typename> typename... interpolator_ts>
using SteadyGridSamplerVF = sampled_vectorfield<
    n, real_t,
    grid_sampler<n, real_t, vec<real_t, vec_dim>, interpolator_ts...>, vec_dim>;

//==============================================================================
template <size_t n, typename real_t, size_t vec_dim,
          template <typename> typename... interpolator_ts>
using UnsteadyGridSamplerVF = sampled_vectorfield<
    n, real_t,
    grid_sampler<n + 1, real_t, vec<real_t, vec_dim>, interpolator_ts...>,
    vec_dim>;

//==============================================================================
//! resamples a scalarfield
template <template <typename> typename... interpolator_ts, typename sf_t,
          typename real_t, size_t n, typename grid_real_t, typename time_real_t>
auto resample(const scalarfield<n, real_t, sf_t>& sf,
              const grid<n, grid_real_t>& g, time_real_t t) {
  static_assert(n > 0, "number of dimensions must be greater than 0");
  static_assert(sizeof...(interpolator_ts) == n,
                "number of interpolators does not match number of dimensions");
  std::vector<real_t> resampled_data(g.num_points());

  parallel_for(g.vertices(), [&](auto v) {
    // for(auto v :g.vertices()) {
    const auto i      = v.global_index();
    resampled_data[i] = [&]() {
      try {
        return sf(v.to_vec(), t);
      } catch (std::exception& /*e*/) { return 0.0 / 0.0; }
    }();
  });

  return SteadyGridSamplerSF<n, real_t, interpolator_ts...>(
      std::move(resampled_data), g);
}

//==============================================================================
//! resamples a vectorfield
template <template <typename> typename... interpolator_ts, typename vf_t,
          typename real_t, size_t n, typename data_t = typename vf_t::vec_t,
          typename grid_real_t, typename time_real_t>
auto resample(const vectorfield<n, real_t, vf_t>& vf,
              const grid<n, grid_real_t>& g, time_real_t t) {
  static_assert(n > 0, "number of dimensions must be greater than 0");
  static_assert(sizeof...(interpolator_ts) == n,
                "number of interpolators does not match number of dimensions");
  using vec_t = typename vectorfield<n, real_t, vf_t>::vec_t;
  SteadyGridSamplerVF<n, real_t, vf_t::vec_dim, interpolator_ts...> field(g);
  auto& data = field.sampler().data();

  for (auto v : g.vertices()) {
    auto indices = v.to_indices();
    try {
      data(indices) = vf(v.to_vec(), t);
    } catch (std::exception& /*e*/) { data(indices) = vec_t{0.0 / 0.0}; }
  }

  return field;
}

//==============================================================================
//! resamples a vectorfield
template <template <typename> typename... interpolator_ts, typename vf_t,
          typename real_t, size_t n, typename grid_real_t, typename time_real_t>
auto resample(const vectorfield<n, real_t, vf_t>& vf,
              const grid<n, grid_real_t>& g, const linspace<time_real_t>& ts) {
  static_assert(n > 0, "number of dimensions must be greater than 0");
  static_assert(sizeof...(interpolator_ts) == n + 1,
                "number of interpolators does not match number of dimensions");
  assert(ts.resolution > 0);
  using vec_t = typename vectorfield<n, real_t, vf_t>::vec_t;
  UnsteadyGridSamplerVF<n, real_t, vf_t::vec_dim, interpolator_ts...> field(g +
                                                                            ts);
  auto& data = field.sampler().data();

  auto indices = make_array<size_t, n + 1>(0);
  for (auto v : g.vertices()) {
    boost::copy(v.to_indices(), begin(indices));
    indices.back() = 0;
    for (auto t : ts) {
      try {
        data(indices) = vf(v.to_vec(), t);
      } catch (std::exception& /*e*/) { data(indices) = vec_t{0.0 / 0.0}; }
      ++indices.back();
    }
  }

  return field;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#include "concept_undefs.h"
#endif
