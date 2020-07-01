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
#include "chunked_data.h"
#include "crtp.h"
#include "field.h"
#include "for_loop.h"
#include "grid.h"
#include "interpolation.h"
#include "sampled_field.h"
#include "type_traits.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N, typename Data,
          template <typename> typename HeadInterpolator,
          template <typename> typename... TailInterpolators>
struct grid_sampler;

//==============================================================================
template <typename Real, size_t N, typename Data, typename top_grid_t,
          template <typename> typename HeadInterpolator,
          template <typename> typename... TailInterpolators>
struct grid_sampler_view;

//==============================================================================
template <typename Real, size_t N, typename Data, typename Grid,
          template <typename> typename... Interpolators>
struct grid_sampler_iterator;

//==============================================================================
template <typename Real, size_t N, typename Data, typename Grid,
          template <typename> typename... Interpolators>
struct base_grid_sampler_at {
  using type = grid_sampler_view<Real, N - 1, Data, Grid, Interpolators...>;
  using const_type =
      grid_sampler_view<Real, N - 1, Data, const Grid, Interpolators...>;
};

//==============================================================================
template <typename Real, typename Data, typename Grid,
          template <typename> typename... Interpolators>
struct base_grid_sampler_at<Real, 1, Data, Grid, Interpolators...> {
  using type       = std::decay_t<Data>&;
  using const_type = std::decay_t<Data>;
};

//==============================================================================
template <typename Real, size_t N, typename Data, typename T,
          template <typename> typename... Interpolators>
using base_grid_sampler_at_t =
    typename base_grid_sampler_at<Real, N, Data, T, Interpolators...>::type;

//==============================================================================
template <typename Real, size_t N, typename Data, typename T,
          template <typename> typename... Interpolators>
using base_grid_sampler_at_ct =
    typename base_grid_sampler_at<Real, N, Data, T,
                                  Interpolators...>::const_type;

//==============================================================================
/// CRTP inheritance class for grid_sampler and grid_sampler_view
template <typename Derived, typename Real, size_t N, typename Data,
          template <typename> typename HeadInterpolator,
          template <typename> typename... TailInterpolators>
struct base_grid_sampler : crtp<Derived>, grid<Real, N> {
  //----------------------------------------------------------------------------
  // static assertions
  //----------------------------------------------------------------------------
  static_assert(N > 0, "grid_sampler must have at least one dimension");
  static_assert(
      N == sizeof...(TailInterpolators) + 1,
      "number of interpolator kernels does not match number of dimensions");

  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return N; }
  using real_t                           = Real;
  using data_t                           = Data;
  using internal_data_t                  = internal_data_type_t<Data>;
  static constexpr size_t num_components = num_components_v<Data>;
  using this_t   = base_grid_sampler<Derived, Real, N, Data, HeadInterpolator,
                                   TailInterpolators...>;
  using parent_t = grid<Real, N>;
  using iterator = grid_sampler_iterator<Real, N, Data, this_t>;
  using indexing_t =
      base_grid_sampler_at_t<Real, N, Data, this_t, TailInterpolators...>;
  using const_indexing_t =
      base_grid_sampler_at_ct<Real, N, Data, this_t, TailInterpolators...>;
  using crtp<Derived>::as_derived;
  using parent_t::dimension;
  using parent_t::dimensions;
  using parent_t::size;
  struct out_of_domain : std::runtime_error {
    out_of_domain(const std::string& err) : std::runtime_error{err} {}
  };

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  base_grid_sampler() = default;
  base_grid_sampler(const parent_t& g) : parent_t{g} {}
  base_grid_sampler(parent_t&& g) : parent_t{std::move(g)} {}
  template <typename... real_ts>
  base_grid_sampler(const linspace<real_ts>&... linspaces)
      : parent_t{linspaces...} {}
  base_grid_sampler(const base_grid_sampler& other) : parent_t{other} {}
  base_grid_sampler(base_grid_sampler&& other) noexcept
      : parent_t{std::move(other)} {}
  auto& operator=(const base_grid_sampler& other) {
    parent_t::operator=(other);
    return *this;
  }
  auto& operator=(base_grid_sampler&& other) noexcept {
    parent_t::operator=(std::move(other));
    return *this;
  }
  template <typename OtherReal>
  auto& operator=(const grid<OtherReal, N>& other) {
    parent_t::operator=(other);
    return *this;
  }
  template <typename OtherReal>
  auto& operator=(grid<OtherReal, N>&& other) {
    parent_t::operator=(std::move(other));
    return *this;
  }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
  template <typename... Is, enable_if_integral<Is...> = true>
  decltype(auto) data(Is... is) {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return as_derived().data(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// data at specified indices is...
  /// CRTP-virtual method
  template <typename... Is, enable_if_integral<Is...> = true>
  Data data(Is... is) const {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return as_derived().data(is...);
  }
  //----------------------------------------------------------------------------
  /// indexing of data.
  /// if N == 1 returns actual data otherwise returns a grid_sampler_view with i
  /// as fixed index
  decltype(auto) at(size_t i) {
    if constexpr (N > 1) {
      return indexing_t{this, i};
    } else {
      return data(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if N == 1 returns actual data otherwise returns a grid_sampler_view with i
  /// as fixed index
  auto at(size_t i) const {
    if constexpr (N > 1) {
      return const_indexing_t{this, i};
    } else {
      return data(i);
    }
  }
  //----------------------------------------------------------------------------
  /// indexing of data.
  /// if N == 1 returns actual data otherwise returns a grid_sampler_view with i
  /// as fixed index
  decltype(auto) operator[](size_t i) { return at(i); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if N == 1 returns actual data otherwise returns a grid_sampler_view with i
  /// as fixed index
  auto operator[](size_t i) const { return at(i); }
  //----------------------------------------------------------------------------
  template <typename... Xs, enable_if_arithmetic<std::decay_t<Xs>...> = true>
  auto operator()(Xs... xs) const {
    static_assert(sizeof...(Xs) == N,
                  "number of coordinates does not match number of dimensions");
    return sample(xs...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(const std::array<Real, N>& xs) const { return sample(xs); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  auto operator()(const base_tensor<Tensor, TensorReal, N>& xs) const {
    return sample(xs);
  }
  //----------------------------------------------------------------------------
  /// sampling by interpolating using HeadInterpolator and
  /// grid_iterators
  template <typename... Xs, enable_if_arithmetic<std::decay_t<Xs>...> = true>
  auto sample(Real x, Xs... xs) const {
    using interpolator = HeadInterpolator<Data>;
    static_assert(sizeof...(Xs) + 1 == N,
                  "number of coordinates does not match number of dimensions");
    x        = domain_to_global(x, 0);
    size_t i = std::floor(x);
    Real   t = x - std::floor(x);
    if (begin() + i + 1 == end()) {
      if constexpr (interpolator::needs_first_derivative) {
        return interpolator::from_iterators(begin() + i, begin() + i, begin(),
                                            end(), t, xs...);
      } else {
        return interpolator::from_iterators(begin() + i, begin() + i, t,
                                            xs...);
      }
    }
    if constexpr (interpolator::needs_first_derivative) {
      return interpolator::from_iterators(begin() + i, begin() + i + 1, begin(),
                                          end(), t, xs...);
    } else {
      return interpolator::from_iterators(begin() + i, begin() + i + 1, t,
                                          xs...);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto sample(const std::array<Real, N>& pos) const {
    return invoke_unpacked(
        [&pos, this](const auto... xs) { return sample(xs...); }, unpack(pos));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  auto sample(const base_tensor<Tensor, TensorReal, N>& pos) const {
    return invoke_unpacked(
        [&pos, this](const auto... xs) { return sample(xs...); }, unpack(pos));
  }

  //----------------------------------------------------------------------------
  auto domain_to_global(Real x, size_t i) const {
    auto converted = (x - dimension(i).front()) /
                     (dimension(i).back() - dimension(i).front());
    if (converted < 0 || converted > 1) {
      throw out_of_domain{std::to_string(x) + " in dimension " +
                          std::to_string(i)};
    }
    return converted * (dimension(i).size() - 1);
  }

  //============================================================================
  /// converts index list to global index in m_data
  template <typename... Is,
            typename = std::enable_if_t<(std::is_integral_v<Is> && ...)>>
  auto global_index(Is... is) const {
    static_assert(sizeof...(Is) == N);
    static_assert((std::is_integral_v<Is> && ...));
    return global_index(std::array<size_t, N>{size_t(is)...});
  }

  //----------------------------------------------------------------------------
  /// converts index list to global index in m_data
  auto global_index(const std::array<size_t, N>& is) const {
    size_t multiplier = 1;
    size_t gi         = 0;
    size_t counter    = 0;
    for (auto i : is) {
      gi += multiplier * i;
      multiplier *= dimension(counter++).size();
    }
    return gi;
  }

  //----------------------------------------------------------------------------

  auto begin() const { return iterator{this, 0}; }
  auto end() const { return iterator{this, size(0)}; }
};

//==============================================================================
/// holds actual data
template <typename Real, size_t N, typename Data,
          template <typename> typename HeadInterpolator,
          template <typename> typename... TailInterpolators>
struct grid_sampler
    : base_grid_sampler<
          grid_sampler<Real, N, Data, HeadInterpolator, TailInterpolators...>,
          Real, N, Data, HeadInterpolator, TailInterpolators...> {
  using real_t = Real;
  using data_t = Data;
  static constexpr auto num_dimensions() { return N; }
  using vec_t = vec<Real, N>;
  using this_t =
      grid_sampler<Real, N, Data, HeadInterpolator, TailInterpolators...>;
  using parent_t = base_grid_sampler<this_t, Real, N, Data, HeadInterpolator,
                                     TailInterpolators...>;
  using internal_data_t                  = typename parent_t::internal_data_t;
  static constexpr size_t num_components = parent_t::num_components;
  using iterator =
      grid_sampler_iterator<Real, N, Data, this_t, TailInterpolators...>;
  using parent_t::dimension;
  using parent_t::dimensions;
  using parent_t::size;

  //============================================================================
 private:
  chunked_data<Data, N> m_data;

  //============================================================================
 public:
  template <size_t... Is>
  grid_sampler(std::index_sequence<Is...> /*is*/)
      : parent_t{((void)Is, linspace<Real>{0, 1, 1})...},
        m_data{((void)Is, 1)...} {}

  //----------------------------------------------------------------------------
  grid_sampler() : grid_sampler{std::make_index_sequence<N>{}} {}

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
  template <typename... Resolution,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Resolution>> && ...)>>
  grid_sampler(Resolution... resolution)
      : parent_t{linspace<Real>{0, 1, size_t(resolution)}...},
        m_data(resolution...) {}

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  grid_sampler(const linspace<real_ts>&... linspaces)
      : parent_t{linspaces...}, m_data{linspaces.size()...} {}

  //----------------------------------------------------------------------------
  template <size_t... Is>
  grid_sampler(const std::array<size_t, N>& resolution,
               std::index_sequence<Is...> /*is*/)
      : parent_t{linspace<Real>{0, 1, size_t(resolution[Is])}...},
        m_data(resolution[Is]...) {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const std::array<size_t, N>& resolution)
      : grid_sampler(resolution, std::make_index_sequence<N>{}) {}

  //----------------------------------------------------------------------------
  template <size_t... Is>
  grid_sampler(const grid<Real, N>& domain, std::index_sequence<Is...> /*is*/)
      : parent_t{domain}, m_data{domain.dimension(Is).size()...} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const grid<Real, N>& domain)
      : grid_sampler(domain, std::make_index_sequence<N>{}) {}

  //============================================================================
  template <typename... Resolution,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Resolution>> && ...)>>
  grid_sampler(const std::vector<Real>& data, Resolution... resolution)
      : parent_t{linspace<Real>{0, 1, size_t(resolution)}...},
        m_data{data, resolution...} {}

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  grid_sampler(const std::vector<Real>& data,
               const linspace<real_ts>&... linspaces)
      : parent_t{linspaces...}, m_data{data, linspaces.size()...} {}

  //----------------------------------------------------------------------------
  template <size_t... Is>
  grid_sampler(const std::vector<Real>&     data,
               const std::array<size_t, N>& resolution,
               std::index_sequence<Is...> /*is*/)
      : parent_t{linspace<Real>{0, 1, size_t(resolution[Is])}...},
        m_data{data, resolution} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  grid_sampler(const std::vector<Real>&     data,
               const std::array<size_t, N>& resolution)
      : grid_sampler(data, resolution, std::make_index_sequence<N>{}) {}

  //----------------------------------------------------------------------------
  // template <size_t... Is>
  // grid_sampler(const std::vector<Real>& data, const grid<Real, N>&
  // domain,
  //              std::index_sequence<Is...>)
  //     : parent_t{domain}, m_data{data} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // grid_sampler(const std::vector<Real>& data, const grid<Real, N>&
  // domain)
  //     : grid_sampler(data, domain, std::make_index_sequence<N>{}) {}

  //============================================================================
  grid_sampler(const std::string& filename) : grid_sampler{} { read(filename); }

  //----------------------------------------------------------------------------
  // grid_sampler(const std::vector<std::string>& filenames)
  //     : m_size(0), m_domain{vec_t{tag::fill{0}}, vec_t{tag::fill{1}}} {
  //   read(filenames);
  // }

  //----------------------------------------------------------------------------
  template <typename... Is,
            typename = std::enable_if_t<(std::is_integral_v<Is> && ...)>>
  Data& data(Is... is) {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return m_data(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is,
            typename = std::enable_if_t<(std::is_integral_v<Is> && ...)>>
  Data data(Is... is) const {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return m_data(is...);
  }
  //----------------------------------------------------------------------------
  const auto& data() const { return m_data; }
  auto&       data() { return m_data; }
  //============================================================================
  template <typename... Resolution,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Resolution>> && ...)>>
  void resize(Resolution... resolution) {
    parent_t::operator=(grid{linspace<Real>{0, 1, size_t(resolution)}...});
    m_data.resize((resolution * ...));
  }

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  void resize(const linspace<real_ts>&... linspaces) {
    parent_t::operator=(grid{linspaces...});
    m_data.resize(linspaces.size()...);
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  void resize(const std::array<size_t, N>& resolution,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator-(grid{linspace<Real>{0, 1, size_t(resolution[Is])}...});
    m_data.resize((resolution[Is] * ...));
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  void resize(const std::array<size_t, N>& resolution) {
    resize(resolution, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t... Is>
  void resize(const grid<OtherReal, N>& domain,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=(domain);
    m_data.resize((domain.size()[Is] * ...));
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename OtherReal, size_t... Is>
  void resize(const grid<OtherReal, N>& domain) {
    resize(domain, std::make_index_sequence<N>{});
  }

  //============================================================================
  template <typename... Resolution,
            typename = std::enable_if_t<
                (std::is_integral_v<std::decay_t<Resolution>> && ...)>>
  void resize(const std::vector<Data>& data, Resolution... resolution) {
    parent_t::operator=(grid{linspace<Real>{0, 1, size_t(resolution)}...});
    m_data            = data;
  }

  //----------------------------------------------------------------------------
  template <typename... real_ts>
  void resize(const std::vector<Data>& data,
              const linspace<real_ts>&... linspaces) {
    parent_t::operator=(grid{linspaces...});
    m_data            = data;
  }

  //----------------------------------------------------------------------------
  template <size_t... Is>
  void resize(const std::vector<Data>&     data,
              const std::array<size_t, N>& resolution,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=(grid{linspace<Real>{0, 1, size_t(resolution[Is])}...});
    m_data            = data;
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  void resize(const std::vector<Data>&     data,
              const std::array<size_t, N>& resolution) {
    resize(data, resolution, std::make_index_sequence<N>{});
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t... Is>
  void resize(const std::vector<Data>& data, const grid<OtherReal, N>& domain,
              std::index_sequence<Is...> /*is*/) {
    parent_t::operator=(domain);
    m_data.resize(domain.dimension(Is).size()...);
    m_data = data;
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <typename OtherReal>
  void resize(const std::vector<Data>& data, const grid<OtherReal, N>& domain) {
    resize(data, domain, std::make_index_sequence<N>{});
  }

  //============================================================================
  template <typename _Data = Data, enable_if_arithmetic<_Data>...>
  auto min_value() {
    return m_data.min_value();
  }
  //----------------------------------------------------------------------------
  template <typename _Data = Data, enable_if_arithmetic<_Data>...>
  auto max_value() {
    return m_data.max_value();
  }
  //----------------------------------------------------------------------------
  template <typename _Data = Data, enable_if_arithmetic<_Data>...>
  auto minmax_value() {
    return m_data.minmax_value();
  }
  //----------------------------------------------------------------------------
  template <typename _Data = Data, enable_if_arithmetic<_Data>...>
  auto normalize() {
    return m_data.normalize();
  }

  //   //----------------------------------------------------------------------------
  //   template <typename OtherReal, size_t _n = N, typename _data_t =
  //   Data,
  //             typename = std::enable_if_t<std::is_arithmetic_v<_data_t>>,
  //             typename = std::enable_if_t<_n == 2>>
  //   auto filter(const Mat<OtherReal, 3, 3>& kernel) {
  //     auto copy = *this;
  // #pragma omp parallel for collapse(2)
  //     for (size_t x = 1; x < dimension(0).size() - 1; ++x)
  //       for (size_t y = 1; y < dimension(1).size() - 1; ++y) {
  //         // copy neighbour data
  //         Mat<Real, 3, 3> neighbour_data{
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

  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937, typename _Data = Data,
            enable_if_arithmetic<_Data> = true>
  void randu(Real lower_boundary, Real upper_boundary,
             RandomEngine&& random_engine = RandomEngine{
                 std::random_device{}()}) {
    m_data.randu(lower_boundary, upper_boundary,
                 std::forward<RandomEngine>(random_engine));
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937, typename _Data = Data,
            enable_if_arithmetic<_Data> = true>
  void randu(RandomEngine&& random_engine = RandomEngine{
                 std::random_device{}()}) {
    randu(0, 1, std::forward<RandomEngine>(random_engine));
  }

  //----------------------------------------------------------------------------
  void read(const std::string& filename) {
    auto ext = filename.substr(filename.find_last_of('.') + 1);
    if constexpr (N == 3) {
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
  template <size_t n_ = N>
  void read_amira(const std::string& filename) {
    static_assert(n_ == 3,
                  "sampler must have 3 dimensions for reading amira files");
    auto [data, dims, domain, amira_num_components] = amira::read(filename);
    if constexpr (std::is_same_v<float, Real>) {
      resize(std::move(data), grid{domain, dims});
    } else {
      std::vector<Data> casted_data(data.size() / amira_num_components);
      auto ptr = reinterpret_cast<internal_data_t*>(casted_data.data());
      for (size_t i = 0; i < casted_data.size(); ++i) {
        for (size_t j = 0; j < num_components; ++j) {
          ptr[i * num_components + j] = data[i * amira_num_components + j];
        }
      }
      resize(std::move(casted_data), grid<Real, N>{domain, dims});
    }
  }

  //----------------------------------------------------------------------------
  void read_vtk_scalars(const std::string& filename, const std::string& data_name) {
    read_vtk_scalars(filename,data_name, std::make_index_sequence<N>{});
  }
  template <size_t... Is>
  void read_vtk_scalars(const std::string& filename, const std::string& data_name,
                std::index_sequence<Is...> /*is*/) {
    struct listener_t : vtk::legacy_file_listener {
      std::map<std::string, std::vector<Real>> scalars;
      std::array<size_t, 3>            dims;
      std::map<std::string, size_t>            scalars_num_comps;
      std::array<Real, 3>              origin, spacing;
      vtk::DatasetType                 type = vtk::UNKNOWN_TYPE;

      void on_dimensions(size_t x, size_t y, size_t z) override {
        if constexpr (N == 2) {
          dims = {x, y};

        } else if constexpr (N == 3) {
          dims = {x, y, z};
        }
      }
      void on_dataset_type(vtk::DatasetType _type) override { type = _type; }
      void on_spacing(Real x, Real y, Real z) override { spacing = {x, y, z}; }
      void on_origin(Real x, Real y, Real z) override { origin = {x, y, z}; }
      void on_scalars(const std::string& data_name,
                      const std::string& /*lookup_table_name*/,
                      size_t num_comps, const std::vector<Real>& data,
                      vtk::ReaderData) override {
        scalars[data_name] = data, scalars_num_comps[data_name] = num_comps;
      }
    } listener;

    vtk::legacy_file file(filename);
    file.add_listener(listener);
    file.read();

    if (listener.type == vtk::STRUCTURED_POINTS) {
      resize(linspace{
          listener.origin[Is],
          listener.origin[Is] + listener.spacing[Is] * (listener.dims[Is] - 1),
          listener.dims[Is]}...);

    } else {
      throw std::runtime_error{"structured points needed"};
    }
    if (listener.scalars.find(data_name) == end(listener.scalars)) {
      throw std::runtime_error{data_name + " not found"};
    }
    if (listener.scalars_num_comps[data_name] != num_components) {
      throw std::runtime_error{"number of components do not match"};
    }
    std::vector<Data> casted_data;
    casted_data.reserve(listener.scalars[data_name].size() /
                        listener.scalars_num_comps[data_name]);
    size_t      cnt       = 0;
    const auto& data      = listener.scalars[data_name];
    auto        num_comps = listener.scalars_num_comps[data_name];
    for (size_t i = 0; i < data.size(); i += num_comps) {
      casted_data.emplace_back();
      for (size_t j = 0; j < num_comps; ++j) {
        casted_data.back()(j) = data[cnt++];
      }
    }
    m_data = std::move(casted_data);
  }
  void read_vtk(const std::string& filename) {
    read_vtk(filename, std::make_index_sequence<N>{});
  }
  template <size_t... Is>
  void read_vtk(const std::string& filename,
                std::index_sequence<Is...> /*is*/) {
    struct listener_t : vtk::legacy_file_listener {
      std::array<size_t, 3>            dims;
      std::array<Real, 3>              origin, spacing;
      std::vector<std::array<Real, 3>> data;
      std::array<Real, 3>              min_coord, max_coord;
      vtk::DatasetType                 type = vtk::UNKNOWN_TYPE;
      std::map<std::string, std::vector<Real>> scalars;
      std::map<std::string, size_t>            scalars_num_comps;

      void on_dataset_type(vtk::DatasetType _type) override { type = _type; }
      void on_dimensions(size_t x, size_t y, size_t z) override {
        if constexpr (N == 2) {
          dims = {x, y};

        } else if constexpr (N == 3) {
          dims = {x, y, z};
        }
      }
      void on_x_coordinates(const std::vector<Real>& xs) override {
        min_coord[0] = xs.front();
        max_coord[0] = xs.back();
      }
      void on_y_coordinates(const std::vector<Real>& ys) override {
        min_coord[1] = ys.front();
        max_coord[1] = ys.back();
      }
      void on_z_coordinates(const std::vector<Real>& zs) override {
        min_coord[2] = zs.front();
        max_coord[2] = zs.back();
      }
      void on_spacing(Real x, Real y, Real z) override { spacing = {x, y, z}; }
      void on_origin(Real x, Real y, Real z) override { origin = {x, y, z}; }
      void on_vectors(const std::string& /* name */,
                      const std::vector<std::array<Real, 3>>& vectors,
                      vtk::ReaderData /*data*/) override {
        data = vectors;
      }

      void on_scalars(const std::string& data_name,
                      const std::string& /*lookup_table_name*/,
                      size_t num_comps, const std::vector<Real>& data,
                      vtk::ReaderData) override {
        scalars[data_name] = data, scalars_num_comps[data_name] = num_comps;
      }
    } listener;

    vtk::legacy_file file(filename);
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

    std::vector<Data> casted_data(listener.data.size());
    // auto ptr = reinterpret_cast<internal_data_t*>(casted_data.data());
    // for (size_t i = 0; i < listener.data.size(); ++i) { ptr[i] =
    // listener.data[i]; }
    boost::transform(listener.data, begin(casted_data), [](const auto& v) {
      return vec{v[0], v[1]};
    });
    m_data = std::move(casted_data);
  }

  //============================================================================
 private:
  void write_vtk_1(const std::string& filepath) {
    vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
    if (writer.is_open()) {
      writer.set_title("tatooine grid sampler");
      writer.write_header();

      writer.write_dimensions(dimension(0).size(), 1, 1);
      writer.write_origin(dimension(0).front(), 0, 0);
      writer.write_spacing(dimension(0).spacing(), 0, 0);
      writer.write_point_data(dimension(0).size());

      // write data
      std::vector<std::vector<double>> field_data;
      field_data.reserve(dimension(0).size());
      for (auto v : this->vertices()) {
        const auto& d = m_data(v.indices());
        field_data.emplace_back();
        if constexpr (!std::is_arithmetic_v<Data>) {
          for (const auto& c : d) { field_data.back().push_back(c); }
        } else {
          field_data.back().push_back(d);
        }
      }
      writer.write_scalars("field_data", field_data);
      writer.close();
    }
  }
  //============================================================================
  void write_vtk_2(const std::string& filepath) {
    vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
    if (writer.is_open()) {
      writer.set_title("tatooine grid sampler");
      writer.write_header();

      writer.write_dimensions(dimension(0).size(), dimension(1).size(), 1);
      writer.write_origin(dimension(0).front(), dimension(1).front(), 0);
      writer.write_spacing(dimension(0).spacing(), dimension(1).spacing(), 0);
      writer.write_point_data(dimension(0).size() * dimension(1).size());

      // write data
      std::vector<Data> field_data;
      field_data.reserve(dimension(0).size() * dimension(1).size());
      for (auto v : this->vertices()) {
        field_data.push_back(m_data(v.indices()));
      }
      writer.write_scalars("field_data", field_data);
      writer.close();
    }
  }
  //============================================================================
  void write_vtk_3(const std::string& filepath) {
    vtk::legacy_file_writer writer(filepath, vtk::STRUCTURED_POINTS);
    if (writer.is_open()) {
      writer.set_title("tatooine grid sampler");
      writer.write_header();

      writer.write_dimensions(dimension(0).size(), dimension(1).size(),
                              dimension(2).size());
      writer.write_origin(dimension(0).front(), dimension(1).front(),
                          dimension(2).front());
      writer.write_spacing(dimension(0).spacing(), dimension(1).spacing(),
                           dimension(2).spacing());
      writer.write_point_data(dimension(0).size() * dimension(1).size() *
                              dimension(2).size());

      // write data
      std::vector<Data> field_data;
      field_data.reserve(dimension(0).size() * dimension(1).size() *
                         dimension(2).size());
      for (auto v : this->vertices()) {
        field_data.push_back(m_data(v.indices()));
      }
      writer.write_scalars("field_data", field_data);
      writer.close();
    }
  }
  //----------------------------------------------------------------------------
 public:
  void write_vtk(const std::string& filepath) {
    if constexpr (N == 1) { write_vtk_1(filepath); }
    if constexpr (N == 2) { write_vtk_2(filepath); }
    if constexpr (N == 3) { write_vtk_3(filepath); }
  }

  //----------------------------------------------------------------------------
  template <size_t _n = N>
  void write_png(const std::string& filepath) {
    static_assert(_n == 2,
                  "cannot write sampler of dimenion other than 2 to png");

    if constexpr (std::is_same_v<Data, Real>) {
      png::image<png::rgb_pixel> image(dimension(0).size(),
                                       dimension(1).size());
      for (unsigned int y = 0; y < image.get_height(); ++y) {
        for (png::uint_32 x = 0; x < image.get_width(); ++x) {
          unsigned int idx = x + dimension(0).size() * y;

          image[image.get_height() - 1 - y][x].red =
              std::max<Real>(0, std::min<Real>(1, m_data[idx])) * 255;
          image[image.get_height() - 1 - y][x].green =
              std::max<Real>(0, std::min<Real>(1, m_data[idx])) * 255;
          image[image.get_height() - 1 - y][x].blue =
              std::max<Real>(0, std::min<Real>(1, m_data[idx])) * 255;
        }
      }
      image.write(filepath);

    } else if constexpr (std::is_same_v<Data, vec<Real, 4>>) {
      png::image<png::rgba_pixel> image(dimension(0).size(),
                                        dimension(1).size());
      for (unsigned int y = 0; y < image.get_height(); ++y) {
        for (png::uint_32 x = 0; x < image.get_width(); ++x) {
          unsigned int idx = x + dimension(0).size() * y;

          image[image.get_height() - 1 - y][x].red =
              std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 0])) * 255;
          image[image.get_height() - 1 - y][x].green =
              std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 1])) * 255;
          image[image.get_height() - 1 - y][x].blue =
              std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 2])) * 255;
          image[image.get_height() - 1 - y][x].alpha =
              std::max<Real>(0, std::min<Real>(1, m_data[idx * 4 + 3])) * 255;
        }
      }
      image.write(filepath);
    }
  }
};

//==============================================================================
/// holds an object of type top_grid_t which can either be
/// grid_sampler or grid_sampler_view and a fixed index of the top
/// grid_sampler
template <typename Real, size_t N, typename Data, typename top_grid_t,
          template <typename> typename HeadInterpolator,
          template <typename> typename... TailInterpolators>
struct grid_sampler_view
    : base_grid_sampler<
          grid_sampler_view<Real, N, Data, top_grid_t, HeadInterpolator,
                            TailInterpolators...>,
          Real, N, Data, HeadInterpolator, TailInterpolators...> {
  using parent_t = base_grid_sampler<
      grid_sampler_view<Real, N, Data, top_grid_t, HeadInterpolator,
                        TailInterpolators...>,
      Real, N, Data, HeadInterpolator, TailInterpolators...>;

  top_grid_t* top_grid;
  size_t      fixed_index;

  //----------------------------------------------------------------------------

  template <size_t... Is>
  grid_sampler_view(top_grid_t* _top_grid, size_t _fixed_index,
                    std::index_sequence<Is...> /*is*/)
      : parent_t{_top_grid->dimension(Is + 1)...},
        top_grid{_top_grid},
        fixed_index{_fixed_index} {}

  grid_sampler_view(top_grid_t* _top_grid, size_t _fixed_index)
      : grid_sampler_view{_top_grid, _fixed_index,
                          std::make_index_sequence<N>{}} {}
  //----------------------------------------------------------------------------
  /// returns data of top grid at fixed_index and index list is...
  template <typename T = top_grid_t, typename... Is,
            std::enable_if_t<!std::is_const_v<T>, bool> = true,
            enable_if_integral<Is...>                   = true>
  Data& data(Is... is) {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return top_grid->data(fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// returns data of top grid at fixed_index and index list is...
  template <typename... Is, enable_if_integral<Is...> = true>
  Data data(Is... is) const {
    static_assert(sizeof...(Is) == N,
                  "number of indices is not equal to number of dimensions");
    return top_grid->data(fixed_index, is...);
  }
};

//==============================================================================
/// holds an object of type Grid which either can be
/// grid_sampler or grid_sampler_view and an index of that grid
template <typename Real, size_t N, typename Data, typename Grid,
          template <typename> typename... TailInterpolators>
struct grid_sampler_iterator {
  using this_t =
      grid_sampler_iterator<Real, N, Data, Grid, TailInterpolators...>;
  //----------------------------------------------------------------------------
  const Grid* m_grid;
  size_t      m_index;
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
/// next specification for grid_sampler_iterator
template <typename Real, size_t N, typename Data, typename Grid,
          template <typename> typename... TailInterpolators>
auto next(
    const grid_sampler_iterator<Real, N, Data, Grid, TailInterpolators...>& it,
    size_t x = 1) {
  return grid_sampler_iterator<Real, N, Data, Grid, TailInterpolators...>{
      it.m_grid, it.m_index + x};
}

//------------------------------------------------------------------------------
/// prev specification for grid_sampler_iterator
template <typename Real, size_t N, typename Data, typename Grid,
          template <typename> typename... TailInterpolators>
auto prev(
    const grid_sampler_iterator<Real, N, Data, Grid, TailInterpolators...>& it,
    size_t x = 1) {
  return grid_sampler_iterator<Real, N, Data, Grid, TailInterpolators...>{
      it.m_grid, it.m_index - x};
}

//==============================================================================
/// resamples a time step of a field
template <template <typename> typename... Interpolators, typename Field,
          typename FieldReal, size_t N, size_t... TensorDims, typename GridReal,
          typename TimeReal>
auto resample(const field<Field, FieldReal, N, TensorDims...>& f,
              const grid<GridReal, N>& g, TimeReal t) {
  static_assert(sizeof...(Interpolators) > 0, "please specify interpolators");
  static_assert(N > 0, "number of dimensions must be greater than 0");
  static_assert(sizeof...(Interpolators) == N,
                "number of interpolators does not match number of dimensions");
  using real_t   = promote_t<FieldReal, GridReal>;
  using tensor_t = typename field<Field, real_t, N, TensorDims...>::tensor_t;

  sampled_field<
      grid_sampler<real_t, N, typename Field::tensor_t, Interpolators...>,
      real_t, N, TensorDims...>
      resampled{g};

  auto& data = resampled.sampler().data();

  for (auto v : g.vertices()) {
    auto is = v.indices();
    try {
      data(is) = f(v.position(), t);
    } catch (std::exception& /*e*/) {
      if constexpr (std::is_arithmetic_v<tensor_t>) {
        data(is) = 0.0 / 0.0;
      } else {
        data(is) = tensor_t{tag::fill{0.0 / 0.0}};
      }
    }
  }
  return resampled;
}

//==============================================================================
/// resamples multiple time steps of a field
template <template <typename> typename... Interpolators, typename Field,
          typename FieldReal, size_t N, typename GridReal, typename TimeReal,
          size_t... TensorDims>
auto resample(const field<Field, FieldReal, N, TensorDims...>& f,
              const grid<GridReal, N>& g, const linspace<TimeReal>& ts) {
  static_assert(N > 0, "number of dimensions must be greater than 0");
  static_assert(sizeof...(Interpolators) == N + 1,
                "number of interpolators does not match number of dimensions");
  assert(ts.size() > 0);
  using real_t   = promote_t<FieldReal, GridReal>;
  using tensor_t = typename field<Field, real_t, N, TensorDims...>::tensor_t;

  sampled_field<grid_sampler<real_t, N + 1, tensor<real_t, TensorDims...>,
                             Interpolators...>,
                real_t, N, TensorDims...>
        resampled{g + ts};
  auto& data = resampled.sampler().data();

  vec<size_t, N + 1> is{tag::zeros};
  for (auto v : g.vertices()) {
    for (size_t i = 0; i < N; ++i) { is(i) = v[i].i(); }
    for (auto t : ts) {
      try {
        data(is) = f(v.position(), t);
      } catch (std::exception& /*e*/) {
        if constexpr (std::is_arithmetic_v<tensor_t>) {
          data(is) = 0.0 / 0.0;
        } else {
          data(is) = tensor_t{tag::fill{0.0 / 0.0}};
        }
      }
      ++is(N);
    }
    is(N) = 0;
  }

  return resampled;
}
//==============================================================================
template <typename Real, size_t N,
          template <typename> typename... Interpolators>
void write_png(grid_sampler<Real, 2, Real, Interpolators...> const& sampler,
               std::string const&                                   path) {
  sampler.write_png(path);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
