#ifndef TATOOINE_GRID_H
#define TATOOINE_GRID_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/grid_vertex_container.h>
#include <tatooine/grid_vertex_iterator.h>
#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>
#include <tatooine/multidim_property.h>
#include <tatooine/random.h>
#include <tatooine/template_helper.h>
#include <tatooine/vec.h>

#include <map>
#include <memory>
#include <tuple>
//==============================================================================
namespace tatooine {
//==============================================================================
/// When using GCC you have to specify Dimensions types by hand. This is a known
/// GCC bug (80438)
template <indexable_space... Dimensions>
class grid {
  static_assert(sizeof...(Dimensions) > 0,
                "Grid needs at least one dimension.");

 public:
  static constexpr bool is_regular =
      (is_linspace_v<std::decay_t<Dimensions>> && ...);
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  using this_t = grid<Dimensions...>;
  using real_t = promote_t<typename Dimensions::value_type...>;
  using vec_t  = vec<real_t, num_dimensions()>;
  using pos_t  = vec_t;
  using seq_t  = std::make_index_sequence<num_dimensions()>;

  using dimensions_t = std::tuple<std::decay_t<Dimensions>...>;

  using vertex_iterator  = grid_vertex_iterator<Dimensions...>;
  using vertex_container = grid_vertex_container<Dimensions...>;

  // general property types
  using property_t = multidim_property<this_t>;
  template <typename ValueType>
  using typed_property_t = typed_multidim_property<this_t, ValueType>;
  template <typename Container>
  using typed_property_impl_t =
      typed_multidim_property_impl<this_t, typename Container::value_type,
                                   Container>;
  using property_ptr_t       = std::unique_ptr<property_t>;
  using property_container_t = std::map<std::string, property_ptr_t>;
  //============================================================================
 private:
  dimensions_t         m_dimensions;
  property_container_t m_vertex_properties;
  mutable bool         m_diff_stencil_coefficients_created_once = false;
  mutable std::array<std::vector<std::vector<double>>, num_dimensions()>
      m_diff_stencil_coefficients_n1_0_p1, m_diff_stencil_coefficients_n2_n1_0,
      m_diff_stencil_coefficients_0_p1_p2, m_diff_stencil_coefficients_0_p1,
      m_diff_stencil_coefficients_n1_0;
  //============================================================================
 public:
  constexpr grid() = default;
  constexpr grid(grid const& other)
      : m_dimensions{other.m_dimensions},
        m_diff_stencil_coefficients_n1_0_p1{
            other.m_diff_stencil_coefficients_n1_0_p1},
        m_diff_stencil_coefficients_n2_n1_0{
            other.m_diff_stencil_coefficients_n2_n1_0},
        m_diff_stencil_coefficients_0_p1_p2{
            other.m_diff_stencil_coefficients_0_p1_p2},
        m_diff_stencil_coefficients_0_p1{
            other.m_diff_stencil_coefficients_0_p1},
        m_diff_stencil_coefficients_n1_0{
            other.m_diff_stencil_coefficients_n1_0} {
    for (auto const& [name, prop] : other.m_vertex_properties) {
      m_vertex_properties.emplace(name, prop->clone());
    }
  }
  constexpr grid(grid&& other) noexcept = default;
  //----------------------------------------------------------------------------
  /// The enable if is needed due to gcc bug 80871. See here:
  /// https://stackoverflow.com/questions/46848129/variadic-deduction-guide-not-taken-by-g-taken-by-clang-who-is-correct
  template <typename... _Dimensions>
  requires(sizeof...(_Dimensions) ==
           sizeof...(Dimensions)) constexpr grid(_Dimensions&&... dimensions)
      : m_dimensions{std::forward<_Dimensions>(dimensions)...} {
    static_assert(sizeof...(_Dimensions) == num_dimensions(),
                  "Number of given dimensions does not match number of "
                  "specified dimensions.");
    static_assert(
        (std::is_same_v<std::decay_t<_Dimensions>, Dimensions> && ...),
        "Constructor dimension types differ class dimension types.");
  }
  //----------------------------------------------------------------------------
 private:
  template <typename Real, size_t... Is>
  constexpr grid(axis_aligned_bounding_box<Real, num_dimensions()> const& bb,
                 std::array<size_t, num_dimensions()> const&              res,
                 std::index_sequence<Is...> /*seq*/)
      : m_dimensions{linspace<real_t>{real_t(bb.min(Is)), real_t(bb.max(Is)),
                                      res[Is]}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <typename Real>
  constexpr grid(axis_aligned_bounding_box<Real, num_dimensions()> const& bb,
                 std::array<size_t, num_dimensions()> const&              res)
      : grid{bb, res, seq_t{}} {}
  //----------------------------------------------------------------------------
  ~grid() = default;
  //============================================================================
 private:
  template <size_t... Ds>
  constexpr auto copy_without_properties(
      std::index_sequence<Ds...> /*seq*/) const {
    return this_t{std::get<Ds>(m_dimensions)...};
  }

 public:
  constexpr auto copy_without_properties() const {
    return copy_without_properties(
        std::make_index_sequence<num_dimensions()>{});
  }
  //============================================================================
  constexpr auto operator=(grid const& other) -> grid& = default;
  constexpr auto operator=(grid&& other) noexcept -> grid& = default;
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto dimension() -> auto& {
    static_assert(I < num_dimensions());
    return std::get<I>(m_dimensions);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t I>
  constexpr auto dimension() const -> auto const& {
    static_assert(I < num_dimensions());
    return std::get<I>(m_dimensions);
  }
  //----------------------------------------------------------------------------
  constexpr auto dimensions() -> auto& { return m_dimensions; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto dimensions() const -> auto const& { return m_dimensions; }
  //----------------------------------------------------------------------------
  constexpr auto front_dimension() -> auto& {
    return std::get<0>(m_dimensions);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto front_dimension() const -> auto const& {
    return std::get<0>(m_dimensions);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto min(std::index_sequence<Is...> /*seq*/) const {
    return vec<real_t, num_dimensions()>{static_cast<real_t>(front<Is>())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto min() const { return min(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto max(std::index_sequence<Is...> /*seq*/) const {
    return vec<real_t, num_dimensions()>{static_cast<real_t>(back<Is>())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto max() const { return max(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto resolution(std::index_sequence<Is...> /*seq*/) const {
    return vec<size_t, num_dimensions()>{size<Is>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto resolution() const { return resolution(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto axis_aligned_bounding_box(
      std::index_sequence<Is...> /*seq*/) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return tatooine::axis_aligned_bounding_box<real_t, num_dimensions()>{
        vec<real_t, num_dimensions()>{static_cast<real_t>(front<Is>())...},
        vec<real_t, num_dimensions()>{static_cast<real_t>(back<Is>())...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto axis_aligned_bounding_box() const {
    return axis_aligned_bounding_box(seq_t{});
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto size(std::index_sequence<Is...> /*seq*/) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return std::array{size<Is>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto size() const { return size(seq_t{}); }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto size() const {
    return dimension<I>().size();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  requires std::is_reference_v<
      template_helper::get_t<I, Dimensions...>> constexpr auto
  size() -> auto& {
    return dimension<I>().size();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto front() const {
    return dimension<I>().front();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto front() -> auto& {
    return dimension<I>().front();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto back() const {
    return dimension<I>().back();
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto back() -> auto& {
    return dimension<I>().back();
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto in_domain(std::index_sequence<Is...> /*seq*/,
                           real_number auto const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "number of components does not match number of dimensions");
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return ((front<Is>() <= xs) && ...) && ((xs <= back<Is>()) && ...);
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto in_domain(real_number auto const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "number of components does not match number of dimensions");
    return in_domain(seq_t{}, xs...);
  }

  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto in_domain(std::array<real_t, num_dimensions()> const& x,
                           std::index_sequence<Is...> /*seq*/) const {
    return in_domain(x[Is]...);
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto in_domain(
      std::array<real_t, num_dimensions()> const& x) const {
    return in_domain(x, seq_t{});
  }
  //----------------------------------------------------------------------------
  /// returns cell index and factor for interpolation
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> std::pair<size_t, double> {
    auto const& dim = dimension<DimensionIndex>();
    if constexpr (is_linspace_v<std::decay_t<decltype(dim)>>) {
      // calculate
      auto pos =
          (x - dim.front()) / (dim.back() - dim.front()) * (dim.size() - 1);
      auto quantized_pos = static_cast<size_t>(std::floor(pos));
      auto cell_position = pos - quantized_pos;
      if (quantized_pos == dim.size() - 1) {
        --quantized_pos;
        cell_position = 1;
      }
      return {quantized_pos, cell_position};
    } else {
      // binary search
      size_t left  = 0;
      size_t right = dim.size() - 1;
      while (right - left > 1) {
        auto const center = (right + left) / 2;
        if (x < dim[center]) {
          right = center;
        } else {
          left = center;
        }
      }
      return {left, (x - dim[left]) / (dim[left + 1] - dim[left])};
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// returns cell indices and factors for each dimension for interpolaton
  template <size_t... DimensionIndex>
  auto cell_index(std::index_sequence<DimensionIndex...>,
                  real_number auto const... xs) const
      -> std::array<std::pair<size_t, double>, num_dimensions()> {
    return std::array{cell_index<DimensionIndex>(static_cast<double>(xs))...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto cell_index(real_number auto const... xs) const {
    return cell_index(seq_t{}, xs...);
  }
  //----------------------------------------------------------------------------
  auto diff_stencil_coefficients_n1_0_p1(size_t dim_index, size_t i) const
      -> auto const& {
    return m_diff_stencil_coefficients_n1_0_p1[dim_index][i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto diff_stencil_coefficients_n2_n1_0(size_t dim_index, size_t i) const
      -> auto const& {
    return m_diff_stencil_coefficients_n1_0[dim_index][i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto diff_stencil_coefficients_n1_0(size_t dim_index, size_t i) const
      -> auto const& {
    return m_diff_stencil_coefficients_n1_0[dim_index][i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto diff_stencil_coefficients_0_p1(size_t dim_index, size_t i) const
      -> auto const& {
    return m_diff_stencil_coefficients_0_p1[dim_index][i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto diff_stencil_coefficients_0_p1_p2(size_t dim_index, size_t i) const
      -> auto const& {
    return m_diff_stencil_coefficients_0_p1_p2[dim_index][i];
  }
  //----------------------------------------------------------------------------
  constexpr auto diff_stencil_coefficients_created_once() const {
    return m_diff_stencil_coefficients_created_once;
  }
  //----------------------------------------------------------------------------
  template <size_t... Ds>
  auto update_diff_stencil_coefficients(
      std::index_sequence<Ds...> /*seq*/) const {
    (update_diff_stencil_coefficients_n1_0_p1<Ds>(), ...);
    (update_diff_stencil_coefficients_0_p1_p2<Ds>(), ...);
    (update_diff_stencil_coefficients_n2_n1_0<Ds>(), ...);
    (update_diff_stencil_coefficients_0_p1<Ds>(), ...);
    (update_diff_stencil_coefficients_n1_0<Ds>(), ...);
    m_diff_stencil_coefficients_created_once = true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto update_diff_stencil_coefficients() const {
    update_diff_stencil_coefficients(
        std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t D>
  auto update_diff_stencil_coefficients_n1_0_p1() const {
    auto const& dim = dimension<D>();
    m_diff_stencil_coefficients_n1_0_p1[D].resize(dim.size());

    for (size_t i = 1; i < dim.size() - 1; ++i) {
      vec<double, 3> xs;
      for (size_t j = 0; j < 3; ++j) {
        xs(j) = dim[i - 1 + j] - dim[i];
      }
      auto const cs = finite_differences_coefficients(1, xs);
      m_diff_stencil_coefficients_n1_0_p1[D][i].reserve(3);
      std::copy(begin(cs.data()), end(cs.data()),
                std::back_inserter(m_diff_stencil_coefficients_n1_0_p1[D][i]));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t D>
  auto update_diff_stencil_coefficients_0_p1_p2() const {
    auto const& dim = dimension<D>();
    m_diff_stencil_coefficients_0_p1_p2[D].resize(dim.size());

    for (size_t i = 0; i < dim.size() - 2; ++i) {
      vec<double, 3> xs;
      for (size_t j = 0; j < 3; ++j) {
        xs(j) = dim[i + j] - dim[i];
      }
      auto const cs = finite_differences_coefficients(1, xs);
      m_diff_stencil_coefficients_0_p1_p2[D][i].reserve(3);
      std::copy(begin(cs.data()), end(cs.data()),
                std::back_inserter(m_diff_stencil_coefficients_0_p1_p2[D][i]));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t D>
  auto update_diff_stencil_coefficients_n2_n1_0() const {
    auto const& dim = dimension<D>();
    m_diff_stencil_coefficients_n2_n1_0[D].resize(dim.size());

    for (size_t i = 2; i < dim.size(); ++i) {
      vec<double, 3> xs;
      for (size_t j = 0; j < 3; ++j) {
        xs(j) = dim[i - 2 + j] - dim[i];
      }
      auto const cs = finite_differences_coefficients(1, xs);
      m_diff_stencil_coefficients_n2_n1_0[D][i].reserve(3);
      std::copy(begin(cs.data()), end(cs.data()),
                std::back_inserter(m_diff_stencil_coefficients_n2_n1_0[D][i]));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t D>
  auto update_diff_stencil_coefficients_0_p1() const {
    auto const& dim = dimension<D>();
    m_diff_stencil_coefficients_0_p1[D].resize(dim.size());

    for (size_t i = 0; i < dim.size() - 1; ++i) {
      vec<double, 2> xs;
      for (size_t j = 0; j < 2; ++j) {
        xs(j) = dim[i + j] - dim[i];
      }
      auto const cs = finite_differences_coefficients(1, xs);
      m_diff_stencil_coefficients_0_p1[D][i].reserve(2);
      std::copy(begin(cs.data()), end(cs.data()),
                std::back_inserter(m_diff_stencil_coefficients_0_p1[D][i]));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t D>
  auto update_diff_stencil_coefficients_n1_0() const {
    auto const& dim = dimension<D>();
    m_diff_stencil_coefficients_n1_0[D].resize(dim.size());

    for (size_t i = 1; i < dim.size(); ++i) {
      vec<double, 2> xs;
      for (size_t j = 0; j < 2; ++j) {
        xs(j) = dim[i - 1 + j] - dim[i];
      }
      auto const cs = finite_differences_coefficients(1, xs);
      m_diff_stencil_coefficients_n1_0[D][i].reserve(3);
      std::copy(begin(cs.data()), end(cs.data()),
                std::back_inserter(m_diff_stencil_coefficients_n1_0[D][i]));
    }
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... DIs>
  auto vertex_at(std::index_sequence<DIs...>, integral auto const... is) const
      -> vec<real_t, num_dimensions()> {
    static_assert(sizeof...(DIs) == sizeof...(is));
    static_assert(sizeof...(is) == num_dimensions());
    return pos_t{static_cast<real_t>((std::get<DIs>(m_dimensions)[is]))...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto vertex_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return vertex_at(seq_t{}, is...);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return vertex_at(is...);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto num_vertices(std::index_sequence<Is...> /*seq*/) const {
    return (size<Is>() * ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto num_vertices() const { return num_vertices(seq_t{}); }
  //----------------------------------------------------------------------------
  /// \return number of dimensions for one dimension dim
  // constexpr auto edges() const { return grid_edge_container{this}; }

  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto vertex_begin(std::index_sequence<Is...> /*seq*/) const {
    return vertex_iterator{this, std::array{((void)Is, size_t(0))...}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto vertex_begin() const { return vertex_begin(seq_t{}); }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto vertex_end(std::index_sequence<Is...> /*seq*/) const {
    return vertex_iterator{this, std::array{((void)Is, size_t(0))...,
                                            size<num_dimensions() - 1>()}};
  }
  //----------------------------------------------------------------------------
 public:
  constexpr auto vertex_end() const {
    return vertex_end(std::make_index_sequence<num_dimensions() - 1>());
  }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{*this}; }
  //----------------------------------------------------------------------------
 private:
  template <regular_invocable<decltype(((void)std::declval<Dimensions>(),
                                        size_t{}))...>
                Iteration,
            size_t... Ds>
  auto loop_over_vertex_indices(Iteration&& iteration,
                                std::index_sequence<Ds...>) const
      -> decltype(auto) {
    return for_loop(std::forward<Iteration>(iteration), size<Ds>()...);
  }
  //----------------------------------------------------------------------------
 public:
  template <regular_invocable<decltype(((void)std::declval<Dimensions>(),
                                        size_t{}))...>
                Iteration,
            size_t... Ds>
  auto loop_over_vertex_indices(Iteration&& iteration) const -> decltype(auto) {
    return loop_over_vertex_indices(
        std::forward<Iteration>(iteration),
        std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <regular_invocable<decltype(((void)std::declval<Dimensions>(),
                                        size_t{}))...>
                Iteration,
            size_t... Ds>
  auto parallel_loop_over_vertex_indices(Iteration&& iteration,
                                         std::index_sequence<Ds...>) const
      -> decltype(auto) {
    return parallel_for_loop(std::forward<Iteration>(iteration), size<Ds>()...);
  }
  //----------------------------------------------------------------------------
 public:
  template <regular_invocable<decltype(((void)std::declval<Dimensions>(),
                                        size_t{}))...>
                Iteration,
            size_t... Ds>
  auto parallel_loop_over_vertex_indices(Iteration&& iteration) const
      -> decltype(auto) {
    return parallel_loop_over_vertex_indices(
        std::forward<Iteration>(iteration),
        std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <indexable_space AdditionalDimension, size_t... Is>
  auto add_dimension(AdditionalDimension&& additional_dimension,
                     std::index_sequence<Is...> /*seq*/) const {
    return grid<Dimensions..., std::decay_t<AdditionalDimension>>{
        dimension<Is>()...,
        std::forward<AdditionalDimension>(additional_dimension)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <indexable_space AdditionalDimension>
  auto add_dimension(indexable_space auto&& additional_dimension) const {
    return add_dimension(
        std::forward<AdditionalDimension>(additional_dimension), seq_t{});
  }
  //----------------------------------------------------------------------------
  template <typename Container, typename... Args>
  auto create_vertex_property(std::string const& name, Args&&... args) -> auto& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      auto new_prop = new typed_property_impl_t<Container>{
          *this, std::forward<Args>(args)...};
      m_vertex_properties.emplace(name, std::unique_ptr<property_t>{new_prop});
      if constexpr (sizeof...(Args) == 0) {
        new_prop->resize(size());
      }
      return *new_prop;
    } else {
      if (it->second->container_type() != typeid(Container)) {
        throw std::runtime_error{
            "Queried container type does not match already inserted property "
            "container type."};
      }
      return *dynamic_cast<typed_property_impl_t<Container>*>(it->second.get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T, typename Indexing = x_fastest>
  auto add_vertex_property(std::string const& name) -> auto& {
    return add_contiguous_vertex_property<T, Indexing>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T, typename Indexing = x_fastest>
  auto add_contiguous_vertex_property(std::string const& name) -> auto& {
    return create_vertex_property<dynamic_multidim_array<T, Indexing>>(name,
                                                                    size());
  }
  //----------------------------------------------------------------------------
  template <typename T, typename Indexing = x_fastest>
  auto add_chunked_vertex_property(std::string const&         name,
                                   std::vector<size_t> const& chunk_size)
      -> auto& {
    return create_vertex_property<chunked_multidim_array<T, Indexing>>(
        name, size(), chunk_size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename Indexing = x_fastest>
  auto add_chunked_vertex_property(
      std::string const&                          name,
      std::array<size_t, num_dimensions()> const& chunk_size) -> auto& {
    return create_vertex_property<chunked_multidim_array<T, Indexing>>(
        name, size(), chunk_size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename Indexing = x_fastest, integral... ChunkSize>
  requires(sizeof...(ChunkSize) == num_dimensions())
  auto add_chunked_vertex_property(std::string const& name,
                                   ChunkSize const... chunk_size) -> auto& {
    return create_vertex_property<chunked_multidim_array<T, Indexing>>(
        name, size(), std::vector<size_t>{static_cast<size_t>(chunk_size)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename Indexing = x_fastest>
  auto add_chunked_vertex_property(std::string const& name) -> auto& {
    return create_vertex_property<chunked_multidim_array<T, Indexing>>(
        name, size(), make_array<num_dimensions()>(size_t(10)));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) const -> auto const& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_property_t<T> const*>(it->second.get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto vertex_property(std::string const& name) -> auto& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      throw std::runtime_error{"property \"" + name + "\" not found"};
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<typed_property_t<T>*>(it->second.get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  requires(num_dimensions() == 3)
  void write_amira(std::string const& file_path,
                               std::string const& vertex_property_name) const {
    write_amira(file_path, vertex_property<T>(vertex_property_name));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  requires is_regular && (num_dimensions() == 3)
  void write_amira(std::string const& file_path,
                   typed_property_t<T> const& prop) const {
    std::ofstream     outfile{file_path, std::ofstream::binary};
    std::stringstream header;

    header << "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n\n";
    header << "define Lattice " << size<0>() << " " << size<1>() << " "
           << size<2>() << "\n\n";
    header << "Parameters {\n";
    header << "    BoundingBox " << front<0>() << " " << back<0>() << " "
           << front<1>() << " " << back<1>() << " " << front<2>() << " "
           << back<2>() << ",\n";
    header << "    CoordType \"uniform\"\n";
    header << "}\n";
    if constexpr (num_components_v<T> > 1) {
      header << "Lattice { " << type_name<internal_data_type_t<T>>() << "["
             << num_components_v<T> << "] Data } @1\n\n";
    } else {
      header << "Lattice { " << type_name<internal_data_type_t<T>>()
             << " Data } @1\n\n";
    }
    header << "# Data section follows\n@1\n";
    auto const header_string = header.str();

    std::vector<T> data;
    data.reserve(size<0>() * size<1>() * size<2>());
    auto back_inserter = [&](auto const... is) {
      data.push_back(prop(is...));
    };
    for_loop(back_inserter, size<0>(), size<1>(), size<2>());
    outfile.write((char*)header_string.c_str(),
                  header_string.size() * sizeof(char));
    outfile.write((char*)data.data(), data.size() * sizeof(T));
  }
  //----------------------------------------------------------------------------
 private:
  template <typename T>
  void write_prop_vtk(vtk::legacy_file_writer& writer, std::string const& name,
                      typed_property_t<T> const& prop) const {
    std::vector<T> data;
    loop_over_vertex_indices(
        [&](auto const... is) { data.push_back(prop.data_at(is...)); });
    writer.write_scalars(name, data);
  }

 public:
  template <size_t _N = num_dimensions(),
            std::enable_if_t<(_N == 1 || _N == 2 || _N == 3), bool> = true>
  void write_vtk(std::string const& file_path,
                 std::string const& description = "tatooine grid") const {
    auto writer = [this, &file_path, &description] {
      vtk::legacy_file_writer writer{file_path, vtk::RECTILINEAR_GRID};
      writer.set_title(description);
      writer.write_header();
      if constexpr (is_regular) {
        if constexpr (num_dimensions() == 1) {
          writer.write_dimensions(size<0>(), 1, 1);
          writer.write_origin(front<0>(), 0, 0);
          writer.write_spacing(dimension<0>().spacing(), 0, 0);
        } else if constexpr (num_dimensions() == 2) {
          writer.write_dimensions(size<0>(), size<1>(), 1);
          writer.write_origin(front<0>(), front<1>(), 0);
          writer.write_spacing(dimension<0>().spacing(),
                               dimension<1>().spacing(), 0);
        } else if constexpr (num_dimensions() == 3) {
          writer.write_dimensions(size<0>(), size<1>(), size<2>());
          writer.write_origin(front<0>(), front<1>(), front<2>());
          writer.write_spacing(dimension<0>().spacing(),
                               dimension<1>().spacing(),
                               dimension<2>().spacing());
        }
        return writer;
      } else {
        if constexpr (num_dimensions() == 1) {
          writer.write_dimensions(size<0>(), 1, 1);
          writer.write_x_coordinates(
              std::vector<double>(begin(dimension<0>()), end(dimension<0>())));
          writer.write_y_coordinates(std::vector<double>{0});
          writer.write_z_coordinates(std::vector<double>{0});
        } else if constexpr (num_dimensions() == 2) {
          writer.write_dimensions(size<0>(), size<1>(), 1);
          writer.write_x_coordinates(
              std::vector<double>(begin(dimension<0>()), end(dimension<0>())));
          writer.write_y_coordinates(
              std::vector<double>(begin(dimension<1>()), end(dimension<1>())));
          writer.write_z_coordinates(std::vector<double>{0});
        } else if constexpr (num_dimensions() == 3) {
          writer.write_dimensions(size<0>(), size<1>(), size<2>());
          writer.write_x_coordinates(
              std::vector<double>(begin(dimension<0>()), end(dimension<0>())));
          writer.write_y_coordinates(
              std::vector<double>(begin(dimension<1>()), end(dimension<1>())));
          writer.write_z_coordinates(
              std::vector<double>(begin(dimension<2>()), end(dimension<2>())));
        }
        return writer;
      }
    }();
    // write vertex data
    writer.write_point_data(num_vertices());
    for (const auto& [name, prop] : this->m_vertex_properties) {
      if (prop->type() == typeid(int)) {
        write_prop_vtk(writer, name,
                       *dynamic_cast<const typed_property_t<int>*>(prop.get()));
      } else if (prop->type() == typeid(float)) {
        write_prop_vtk(
            writer, name,
            *dynamic_cast<const typed_property_t<float>*>(prop.get()));
      } else if (prop->type() == typeid(double)) {
        write_prop_vtk(
            writer, name,
            *dynamic_cast<const typed_property_t<double>*>(prop.get()));
      }
    }
  }
};
//==============================================================================
// free functions
//==============================================================================
template <indexable_space... Dimensions>
auto vertices(grid<Dimensions...> const& g) {
  return g.vertices();
}
//==============================================================================
// deduction guides
//==============================================================================
template <typename... Dimensions>
grid(Dimensions&&...) -> grid<std::decay_t<Dimensions>...>;
// additional, for g++
template <typename Dim0, typename... Dims>
grid(Dim0&&, Dims&&...) -> grid<std::decay_t<Dim0>, std::decay_t<Dims>...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, size_t... Is>
grid(axis_aligned_bounding_box<Real, N> const& bb,
     std::array<size_t, N> const&              res, std::index_sequence<Is...>)
    -> grid<decltype(((void)Is, std::declval<linspace<Real>()>))...>;
//==============================================================================
// operators
//==============================================================================
template <indexable_space... Dimensions, indexable_space AdditionalDimension>
auto operator+(grid<Dimensions...> const& grid,
               AdditionalDimension&&      additional_dimension) {
  return grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <indexable_space... Dimensions, indexable_space AdditionalDimension>
auto operator+(AdditionalDimension&&      additional_dimension,
               grid<Dimensions...> const& grid) {
  return grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
