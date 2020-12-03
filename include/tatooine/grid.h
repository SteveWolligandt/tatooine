#ifndef TATOOINE_GRID_H
#define TATOOINE_GRID_H
//==============================================================================
#include <tatooine/amira_file.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/grid_vertex_container.h>
#include <tatooine/grid_vertex_iterator.h>
#include <tatooine/interpolation.h>
#include <tatooine/lazy_netcdf_reader.h>
#include <tatooine/linspace.h>
#include <tatooine/multidim_property.h>
#include <tatooine/netcdf.h>
#include <tatooine/random.h>
#include <tatooine/template_helper.h>
#include <tatooine/vec.h>

#include <filesystem>
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
  static constexpr bool is_uniform =
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
  requires (sizeof...(_Dimensions) == sizeof...(Dimensions)) &&
           (indexable_space<std::decay_t<_Dimensions>> && ...)
  constexpr grid(_Dimensions&&... dimensions)
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr grid(integral auto const... size)
      : grid{linspace{0.0, 1.0, static_cast<size_t>(size)}...} {
    assert(((size >= 0) && ...));
  }
  //----------------------------------------------------------------------------
  grid(std::filesystem::path const& path) { read(path); }
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
  constexpr auto bounding_box(
      std::index_sequence<Is...> /*seq*/) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return axis_aligned_bounding_box<real_t, num_dimensions()>{
        vec<real_t, num_dimensions()>{static_cast<real_t>(front<Is>())...},
        vec<real_t, num_dimensions()>{static_cast<real_t>(back<Is>())...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto bounding_box() const {
    return bounding_box(seq_t{});
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
  constexpr auto size(size_t const i) const {
    if (i == 0) {
      return size<0>();
    } else if constexpr (num_dimensions() > 1) {
      if (i == 1) {
        return size<1>();
      } else if constexpr (num_dimensions() > 2) {
        if (i == 2) {
          return size<2>();
        } else if constexpr (num_dimensions() > 3) {
          if (i == 3) {
            return size<3>();
          } else if constexpr (num_dimensions() > 4) {
            if (i == 4) {
              return size<4>();
            } else if constexpr (num_dimensions() > 5) {
              if (i == 5) {
                return size<5>();
              } else if constexpr (num_dimensions() > 6) {
                if (i == 6) {
                  return size<6>();
                } else if constexpr (num_dimensions() > 7) {
                  if (i == 7) {
                    return size<7>();
                  } else if constexpr (num_dimensions() > 8) {
                    if (i == 8) {
                      return size<8>();
                    } else if constexpr (num_dimensions() > 9) {
                      if (i == 9) {
                        return size<9>();
                      } else if constexpr (num_dimensions() > 10) {
                        if (i == 10) {
                          return size<10>();
                        } else if constexpr (num_dimensions() > 11) {
                          if (i == 11) {
                            return size<11>();
                          } else if constexpr (num_dimensions() > 12) {
                            if (i == 12) {
                              return size<12>();
                            } else if constexpr (num_dimensions() > 13) {
                              if (i == 13) {
                                return size<13>();
                              } else if constexpr (num_dimensions() > 14) {
                                if (i == 14) {
                                  return size<14>();
                                } else if constexpr (num_dimensions() > 15) {
                                  if (i == 15) {
                                    return size<15>();
                                  } else if constexpr (num_dimensions() > 16) {
                                    if (i == 16) {
                                      return size<16>();
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return std::numeric_limits<size_t>::max();
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
  template <real_number... Comps, size_t... Is>
  requires(num_dimensions() == sizeof...(Comps))
  constexpr auto is_inside(std::index_sequence<Is...> /*seq*/,
                           Comps const... comps) const {
    return ((front<Is>() <= comps || comps <= back<Is>()) || ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <real_number... Comps>
  requires(num_dimensions() == sizeof...(Comps)) constexpr auto is_inside(
      Comps const... comps) const {
    return is_inside(std::make_index_sequence<num_dimensions()>{}, comps...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr auto is_inside(pos_t const& p, std::index_sequence<Is...>/*seq*/) const {
    return ((p(Is) < front<Is>() || back<Is>() < p(Is)) && ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto is_inside(pos_t const& p) const {
    return is_inside(p, std::make_index_sequence<num_dimensions()>{});
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
  auto remove_vertex_property(std::string const& name)
      ->void{
    if (auto it = m_vertex_properties.find(name);
        it != end(m_vertex_properties)) {
      m_vertex_properties.erase(it);
    }
  }
  //----------------------------------------------------------------------------
  auto rename_vertex_property(std::string const& current_name,
                              std::string const& new_name) -> void {
    if (auto it = m_vertex_properties.find(current_name);
        it != end(m_vertex_properties)) {
      auto handler = m_vertex_properties.extract(it);
      handler.key() = new_name;
      m_vertex_properties.insert(std::move(handler));
    }
  }
  //----------------------------------------------------------------------------
  template <typename Container, typename... Args>
  auto create_vertex_property(std::string const& name, Args&&... args)
      -> auto& {
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
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  auto vertex_properties() -> auto& { return m_vertex_properties; }
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
  //============================================================================
  template <regular_invocable<pos_t> F>
  auto sample_to_vertex_property(F&& f, std::string const& name) -> auto& {
    using T = std::invoke_result_t<F, pos_t>;
    auto& prop = add_vertex_property<T>(name);
    loop_over_vertex_indices([&](auto const... is) {
      try {
        prop(is...) = f(vertex_at(is...));
      } catch (std::exception&) {
        prop(is...) = T{0.0 / 0.0};
      }
    });
    return prop;
  }
  //============================================================================
  auto read(std::filesystem::path const& path) {
    if (path.extension() == ".nc") {
      read_netcdf(path);
      return;
    }
    if constexpr (num_dimensions() == 2 || num_dimensions() == 3) {
      if (path.extension() == ".vtk") {
        read_vtk(path);
        return;
      }
    }
    if constexpr (is_uniform) {
      if (path.extension() == ".am") {
        read_amira(path);
        return;
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires (num_dimensions() == 2) || (num_dimensions() == 3)
  auto read_vtk(std::filesystem::path const& path) {
    bool is_structured_points = false;
    vec3 spacing;
    struct listener_t : vtk::legacy_file_listener {
      this_t& gr;
      bool&   is_structured_points;
      vec3& spacing;
      listener_t(this_t& gr_, bool& is_structured_points_, vec3& spacing_)
          : gr{gr_}, is_structured_points{is_structured_points_}, spacing{spacing_} {}
      // header data
      auto on_dataset_type(vtk::dataset_type t) -> void override {
        if (t == vtk::dataset_type::structured_points && !is_uniform) {
          is_structured_points = true;
        }
      }

      // coordinate data
      auto on_origin(double x, double y, double z)
          -> void override {
        gr.front<0>() = x;
        gr.front<1>() = y;
        if (num_dimensions() < 3 && z > 1) {
          throw std::runtime_error{
              "[grid::read_vtk] number of dimensions is < 3 but got third "
              "dimension."};
        }
        if constexpr (num_dimensions() > 3) {
          gr.front<2>() = z;
        }
      }
      auto on_spacing(double x, double y, double z) -> void override {
        spacing = {x, y, z};
      }
      auto on_dimensions(size_t x, size_t y, size_t z) -> void override {
        gr.dimension<0>().resize(x);
        gr.dimension<1>().resize(y);
        if (num_dimensions() < 3 && z > 1) {
          throw std::runtime_error{
              "[grid::read_vtk] number of dimensions is < 3 but got third "
              "dimension."};
        }
        if constexpr (num_dimensions() > 3) {
          gr.dimension<2>().resize(z);
        }
      }
      auto on_x_coordinates(std::vector<float> const& /*xs*/) -> void override {
      }
      auto on_x_coordinates(std::vector<double> const& /*xs*/)
          -> void override {}
      auto on_y_coordinates(std::vector<float> const& /*ys*/) -> void override {
      }
      auto on_y_coordinates(std::vector<double> const& /*ys*/)
          -> void override {}
      auto on_z_coordinates(std::vector<float> const& /*zs*/) -> void override {
      }
      auto on_z_coordinates(std::vector<double> const& /*zs*/)
          -> void override {}

      // index data
      auto on_cells(std::vector<int> const&) -> void override {}
      auto on_cell_types(std::vector<vtk::cell_type> const&) -> void override {}
      auto on_vertices(std::vector<int> const&) -> void override {}
      auto on_lines(std::vector<int> const&) -> void override {}
      auto on_polygons(std::vector<int> const&) -> void override {}
      auto on_triangle_strips(std::vector<int> const&) -> void override {}

      // cell- / pointdata
      auto on_vectors(std::string const& /*name*/,
                      std::vector<std::array<float, 3>> const& /*vectors*/,
                      vtk::reader_data) -> void override {}
      auto on_vectors(std::string const& /*name*/,
                      std::vector<std::array<double, 3>> const& /*vectors*/,
                      vtk::reader_data) -> void override {}
      auto on_normals(std::string const& /*name*/,
                      std::vector<std::array<float, 3>> const& /*normals*/,
                      vtk::reader_data) -> void override {}
      auto on_normals(std::string const& /*name*/,
                      std::vector<std::array<double, 3>> const& /*normals*/,
                      vtk::reader_data) -> void override {}
      auto on_texture_coordinates(
          std::string const& /*name*/,
          std::vector<std::array<float, 2>> const& /*texture_coordinates*/,
          vtk::reader_data) -> void override {}
      auto on_texture_coordinates(
          std::string const& /*name*/,
          std::vector<std::array<double, 2>> const& /*texture_coordinates*/,
          vtk::reader_data) -> void override {}
      auto on_tensors(std::string const& /*name*/,
                      std::vector<std::array<float, 9>> const& /*tensors*/,
                      vtk::reader_data) -> void override {}
      auto on_tensors(std::string const& /*name*/,
                      std::vector<std::array<double, 9>> const& /*tensors*/,
                      vtk::reader_data) -> void override {}

      auto on_scalars(std::string const& /*data_name*/,
                      std::string const& /*lookup_table_name*/,
                      size_t const /*num_comps*/,
                      std::vector<float> const& /*scalars*/, vtk::reader_data)
          -> void override {}
      auto on_scalars(std::string const& /*data_name*/,
                      std::string const& /*lookup_table_name*/,
                      size_t const /*num_comps*/,
                      std::vector<double> const& /*scalars*/, vtk::reader_data)
          -> void override {}
      auto on_point_data(size_t) -> void override {}
      auto on_cell_data(size_t) -> void override {}
      auto on_field_array(std::string const /*field_name*/,
                          std::string const /*field_array_name*/,
                          std::vector<int> const& /*data*/,
                          size_t /*num_comps*/, size_t /*num_tuples*/)
          -> void override {}
      auto on_field_array(std::string const /*field_name*/,
                          std::string const /*field_array_name*/,
                          std::vector<float> const& /*data*/,
                          size_t /*num_comps*/, size_t /*num_tuples*/)
          -> void override {}
      auto on_field_array(std::string const /*field_name*/,
                          std::string const /*field_array_name*/,
                          std::vector<double> const& /*data*/,
                          size_t /*num_comps*/, size_t /*num_tuples*/
                          ) -> void override {}
    } listener{*this, is_structured_points, spacing};
    vtk::legacy_file f{path};
    f.add_listener(listener);
    f.read();

    if (is_structured_points) {
      if constexpr (std::is_same_v<std::decay_t<decltype(dimension<0>())>,
                                   linspace<double>>) {
        dimension<0>().back() = front<0>() + (size<0>() - 1) * spacing(0);
      } else {
        size_t i = 0;
        for (auto& d : dimension<0>()) {
          d = front<0>() + i * spacing(0);
        }
      }
      if constexpr (std::is_same_v<std::decay_t<decltype(dimension<1>())>,
                                   linspace<double>>) {
        dimension<1>().back() = front<1>() + (size<1>() - 1) * spacing(1);
      } else {
        size_t i = 0;
        for (auto& d : dimension<1>()) {
          d = front<1>() + i * spacing(1);
        }
      }
      if constexpr (num_dimensions() == 3) {
        if constexpr (std::is_same_v<std::decay_t<decltype(dimension<2>())>,
                                     linspace<double>>) {
          dimension<2>().back() = front<2>() + (size<2>() - 1) * spacing(2);
        } else {
          size_t i = 0;
          for (auto& d : dimension<2>()) {
            d = front<2>() + i * spacing(2);
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires is_uniform auto read_amira(std::filesystem::path const& path) {
    auto const  am        = amira::read(path);
    auto const& data      = std::get<0>(am);
    auto const& dims      = std::get<1>(am);
    auto const& aabb      = std::get<2>(am);
    auto const& num_comps = std::get<3>(am);
    if (dims[2] == 1 && num_dimensions() == 3) {
      throw std::runtime_error{
          "[grid::read_amira] file contains 2-dimensional data. Cannot "
          "read "
          "into 3-dimensional grid"};
    }
    if (dims[2] > 1 && num_dimensions() == 2) {
      throw std::runtime_error{
          "[grid::read_amira] file contains 3-dimensional data. Cannot "
          "read "
          "into 2-dimensional grid"};
    }
    // set dimensions

    if constexpr (std::is_same_v<std::decay_t<decltype(dimension<0>())>,
                                 linspace<double>>) {
      dimension<0>() = linspace<double>{aabb.min(0), aabb.max(0), dims[0]};
    } else if constexpr (std::is_same_v<std::decay_t<decltype(dimension<0>())>,
                                        linspace<float>>) {
      dimension<0>() = linspace<float>{aabb.min(0), aabb.max(0), dims[0]};
    } else {
      linspace<double> d{aabb.min(0), aabb.max(0), dims[0]};
      dimension<0>().resize(dims[0]);
      std::copy(begin(d), end(d), begin(dimension<0>()));
    }
    if constexpr (std::is_same_v<std::decay_t<decltype(dimension<1>())>,
                                 linspace<double>>) {
      dimension<1>() = linspace<double>{aabb.min(1), aabb.max(1), dims[1]};
    } else if constexpr (std::is_same_v<std::decay_t<decltype(dimension<0>())>,
                                        linspace<float>>) {
      dimension<1>() = linspace<float>{aabb.min(1), aabb.max(1), dims[1]};
    } else {
      linspace<double> d{aabb.min(1), aabb.max(1), dims[1]};
      dimension<1>().resize(dims[1]);
      std::copy(begin(d), end(d), begin(dimension<1>()));
    }
    if constexpr (num_dimensions() == 3) {
      if constexpr (std::is_same_v<std::decay_t<decltype(dimension<1>())>,
                                   linspace<double>>) {
        dimension<2>() = linspace<double>{aabb.min(2), aabb.max(2), dims[2]};
      } else if constexpr (std::is_same_v<
                               std::decay_t<decltype(dimension<1>())>,
                               linspace<float>>) {
        dimension<2>() = linspace<float>{aabb.min(2), aabb.max(2), dims[2]};
      } else {
        linspace<double> d{aabb.min(2), aabb.max(2), dims[2]};
        dimension<2>().resize(dims[2]);
        std::copy(begin(d), end(d), begin(dimension<2>()));
      }
    }
    // copy data
    size_t i = 0;
    if (num_comps == 1) {
      auto& prop = add_vertex_property<float>(path.string());
      loop_over_vertex_indices(
          [&](auto const... is) { prop(is...) = data[i++]; });
    } else if (num_comps == 2) {
      auto& prop = add_vertex_property<vec<float, 2>>(path.string());
      loop_over_vertex_indices([&](auto const... is) {
        prop(is...) = {data[i], data[i + 1]};
        i += num_comps;
      });
    } else if (num_comps == 3) {
      auto& prop = add_vertex_property<vec<float, 2>>(path.string());
      loop_over_vertex_indices([&](auto const... is) {
        prop(is...) = {data[i], data[i + 1], data[i + 2]};
        i += num_comps;
      });
    } else if (num_comps == 4) {
      auto& prop = add_vertex_property<vec<float, 2>>(path.string());
      loop_over_vertex_indices([&](auto const... is) {
        prop(is...) = {data[i], data[i + 1], data[i + 2], data[i + 3]};
        i += num_comps;
      });
    }
      }
      //----------------------------------------------------------------------------
      auto read_netcdf(std::filesystem::path const& path) {
        read_netcdf(path, std::make_index_sequence<num_dimensions()>{});
      }
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // - -
      /// this only reads scalar types
      template <size_t... Is>
      auto read_netcdf(std::filesystem::path const& path,
                       std::index_sequence<Is...>) {
        netcdf::file f{path, netCDF::NcFile::read};
        bool         first                 = true;
        auto         add_variables_of_type = [&]<typename T>() {
          for (auto v : f.variables<T>()) {
            if (v.name() == "x" || v.name() == "y" || v.name() == "z" ||
                v.name() == "t" || v.name() == "X" || v.name() == "Y" ||
                v.name() == "Z" || v.name() == "T" || v.name() == "xdim" ||
                v.name() == "ydim" || v.name() == "zdim" ||
                v.name() == "tdim" || v.name() == "Xdim" ||
                v.name() == "Ydim" || v.name() == "Zdim" ||
                v.name() == "Tdim" || v.name() == "XDim" ||
                v.name() == "YDim" || v.name() == "ZDim" ||
                v.name() == "TDim") {
              continue;
            }
            if (v.num_dimensions() != num_dimensions()) {
              throw std::runtime_error{
                  "[grid::read_netcdf] variable's number of dimensions does "
                  "not "
                  "match grid's number of dimensions:\nnumber of grid "
                  "dimensions: " +
                  std::to_string(num_dimensions()) +
                  "\nnumber of data dimensions: " +
                  std::to_string(v.num_dimensions()) +
                  "\nvariable name: " + v.name()};
            }
            if (!first) {
              auto check = [this, &v](size_t i) {
                if (v.size(i) != size(i)) {
                  throw std::runtime_error{
                      "[grid::read_netcdf] variable's size(" +
                      std::to_string(i) +
                      ") does not "
                      "match grid's size(" +
                      std::to_string(i) + ")"};
                }
              };
              (check(Is), ...);
            } else {
              ((f.variable<typename std::decay_t<decltype(
                    dimension<Is>())>::value_type>(v.dimension_name(Is))
                    .read(dimension<Is>())),
               ...);
            }
            create_vertex_property<netcdf::lazy_reader<T>>(
                v.name(), v, std::vector<size_t>{2, 2});
            first = false;
          }
        };
        add_variables_of_type.template operator()<double>();
        add_variables_of_type.template operator()<float>();
        add_variables_of_type.template operator()<int>();
      }
      //----------------------------------------------------------------------------
      template <typename T>
      requires(num_dimensions() == 3) void write_amira(
          std::string const& path, std::string const& vertex_property_name)
          const {
        write_amira(path, vertex_property<T>(vertex_property_name));
      }
      //----------------------------------------------------------------------------
      template <typename T>
          requires is_uniform &&
          (num_dimensions() == 3) void write_amira(
              std::string const& path, typed_property_t<T> const& prop) const {
        std::ofstream     outfile{path, std::ofstream::binary};
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
        if constexpr (num_components_v < T >> 1) {
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
      void write_prop_vtk(vtk::legacy_file_writer & writer,
                          std::string const&         name,
                          typed_property_t<T> const& prop) const {
        std::vector<T> data;
        loop_over_vertex_indices(
            [&](auto const... is) { data.push_back(prop(is...)); });
        writer.write_scalars(name, data);
      }

     public:
      auto write(std::filesystem::path const& path) const {
        auto const ext = path.extension();

        if constexpr (num_dimensions() == 1 || num_dimensions() == 2 ||
                      num_dimensions() == 3) {
          if (ext == ".vtk") {
            write_vtk(path);
            return;
          }
        }
      }
      //----------------------------------------------------------------------------
      template <typename = void>
          requires(num_dimensions() == 1) || (num_dimensions() == 2) ||
          (num_dimensions() == 3) void write_vtk(
              std::filesystem::path const& path,
              std::string const& description = "tatooine grid") const {
        auto writer = [this, &path, &description] {
          if constexpr (is_uniform) {
            vtk::legacy_file_writer writer{
                path, vtk::dataset_type::structured_points};
            writer.set_title(description);
            writer.write_header();
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
            vtk::legacy_file_writer writer{path,
                                           vtk::dataset_type::rectilinear_grid};
            writer.set_title(description);
            writer.write_header();
            if constexpr (num_dimensions() == 1) {
              writer.write_dimensions(size<0>(), 1, 1);
              writer.write_x_coordinates(std::vector<double>(
                  begin(dimension<0>()), end(dimension<0>())));
              writer.write_y_coordinates(std::vector<double>{0});
              writer.write_z_coordinates(std::vector<double>{0});
            } else if constexpr (num_dimensions() == 2) {
              writer.write_dimensions(size<0>(), size<1>(), 1);
              writer.write_x_coordinates(std::vector<double>(
                  begin(dimension<0>()), end(dimension<0>())));
              writer.write_y_coordinates(std::vector<double>(
                  begin(dimension<1>()), end(dimension<1>())));
              writer.write_z_coordinates(std::vector<double>{0});
            } else if constexpr (num_dimensions() == 3) {
              writer.write_dimensions(size<0>(), size<1>(), size<2>());
              writer.write_x_coordinates(std::vector<double>(
                  begin(dimension<0>()), end(dimension<0>())));
              writer.write_y_coordinates(std::vector<double>(
                  begin(dimension<1>()), end(dimension<1>())));
              writer.write_z_coordinates(std::vector<double>(
                  begin(dimension<2>()), end(dimension<2>())));
            }
            return writer;
          }
        }();
        // write vertex data
        writer.write_point_data(num_vertices());
        for (const auto& [name, prop] : this->m_vertex_properties) {
          if (prop->type() == typeid(int)) {
            write_prop_vtk(
                writer, name,
                *dynamic_cast<const typed_property_t<int>*>(prop.get()));
          } else if (prop->type() == typeid(float)) {
            write_prop_vtk(
                writer, name,
                *dynamic_cast<const typed_property_t<float>*>(prop.get()));
          } else if (prop->type() == typeid(double)) {
            write_prop_vtk(
                writer, name,
                *dynamic_cast<const typed_property_t<double>*>(prop.get()));
          } else if (prop->type() == typeid(vec2)) {
            write_prop_vtk(
                writer, name,
                *dynamic_cast<const typed_property_t<vec2>*>(prop.get()));
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
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <integral... Size>
grid(Size const...)
    -> grid<linspace<std::conditional_t<true, double, Size>>...>;
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
// typedefs
//==============================================================================
template <real_number T>
using uniform_grid_2d = grid<linspace<T>, linspace<T>>;
template <real_number T>
using uniform_grid_3d = grid<linspace<T>, linspace<T>, linspace<T>>;
template <real_number T>
using uniform_grid_4d =
    grid<linspace<T>, linspace<T>, linspace<T>, linspace<T>>;
template <real_number T>
//------------------------------------------------------------------------------
using non_uniform_grid_2d = grid<std::vector<T>, std::vector<T>>;
template <real_number T>
using non_uniform_grid_3d =
    grid<std::vector<T>, std::vector<T>, std::vector<T>>;
template <real_number T>
using non_uniform_grid_4d =
    grid<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
