#ifndef TATOOINE_RECTILINEAR_GRID_H
#define TATOOINE_RECTILINEAR_GRID_H
//==============================================================================
#include <tatooine/amira/read.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/cartesian_axis_labels.h>
#include <tatooine/chunked_multidim_array.h>
#include <tatooine/concepts.h>
#include <tatooine/detail/rectilinear_grid/cell_container.h>
#include <tatooine/detail/rectilinear_grid/creator.h>
#include <tatooine/detail/rectilinear_grid/dimension.h>
#include <tatooine/detail/rectilinear_grid/vertex_container.h>
#include <tatooine/detail/rectilinear_grid/vertex_property.h>
#include <tatooine/detail/rectilinear_grid/vtr_writer.h>
#include <tatooine/filesystem.h>
#include <tatooine/for_loop.h>
#include <tatooine/nan.h>
#if TATOOINE_HDF5_AVAILABLE
#include <tatooine/hdf5.h>
#endif
#include <tatooine/interpolation.h>
#include <tatooine/lazy_reader.h>
#include <tatooine/linspace.h>
#include <tatooine/netcdf.h>
#include <tatooine/random.h>
#include <tatooine/tags.h>
#include <tatooine/template_helper.h>
#include <tatooine/tuple.h>
#include <tatooine/vec.h>

#include <map>
#include <memory>
#include <mutex>
//==============================================================================
namespace tatooine {
//==============================================================================
template <detail::rectilinear_grid::dimension... Dimensions>
requires(sizeof...(Dimensions) > 1)
class rectilinear_grid {
 public:
  static constexpr bool is_uniform =
      (is_linspace<std::decay_t<Dimensions>> && ...);
  static constexpr auto num_dimensions() -> std::size_t { return sizeof...(Dimensions); }
  using this_type     = rectilinear_grid<Dimensions...>;
  using real_type     = common_type<typename Dimensions::value_type...>;
  using vec_type      = vec<real_type, num_dimensions()>;
  using pos_type      = vec_type;
  using sequence_type = std::make_index_sequence<num_dimensions()>;

  template <std::size_t I>
  using dimension_type = variadic::ith_type<I, Dimensions...>;
  using dimensions_type = tuple<std::decay_t<Dimensions>...>;

  using vertex_container =
      detail::rectilinear_grid::vertex_container<Dimensions...>;
  using vertex_handle =
      detail::rectilinear_grid::vertex_handle<sizeof...(Dimensions)>;
  using cell_container =
      detail::rectilinear_grid::cell_container<Dimensions...>;

  // general property types
  using vertex_property_type =
      detail::rectilinear_grid::vertex_property<this_type>;
  template <typename ValueType, bool HasNonConstReference = false>
  using typed_vertex_property_interface_type =
      detail::rectilinear_grid::typed_vertex_property_interface<
          this_type, ValueType, HasNonConstReference>;
  template <typename Container>
  using typed_vertex_property_type =
      detail::rectilinear_grid::typed_vertex_property<
          this_type, typename Container::value_type, Container>;
  template <typename F>
  using invoke_result_with_indices =
      std::invoke_result_t<F, decltype(((void)std::declval<Dimensions>(),
                                        std::declval<std::size_t>()))...>;

  using property_ptr_type       = std::unique_ptr<vertex_property_type>;
  using property_container_type = std::map<std::string, property_ptr_type>;
  //============================================================================
  static constexpr std::size_t min_stencil_size = 2;
  static constexpr std::size_t max_stencil_size = 11;

 private:
  static constexpr std::size_t num_stencils =
      max_stencil_size - min_stencil_size + 1;
  mutable std::mutex      m_stencil_mutex;
  dimensions_type         m_dimensions;
  property_container_type m_vertex_properties;
  mutable bool            m_diff_stencil_coefficients_created_once = false;

  using stencil_type      = std::vector<real_type>;
  using stencil_list_type = std::vector<stencil_type>;
  // diff stencils per stencil size per point
  mutable std::array<std::array<std::vector<stencil_list_type>, num_stencils>,
                     num_dimensions()>
              m_diff_stencil_coefficients;
  std::size_t m_chunk_size_for_lazy_properties = 2;
  //============================================================================
 public:
  /// Default CTOR
  constexpr rectilinear_grid() = default;
  //============================================================================
  /// Copy CTOR
  constexpr rectilinear_grid(rectilinear_grid const& other)
      : m_dimensions{other.m_dimensions},
        m_diff_stencil_coefficients{other.m_diff_stencil_coefficients} {
    for (auto const& [name, prop] : other.m_vertex_properties) {
      auto& emplaced_prop =
          m_vertex_properties.emplace(name, prop->clone()).first->second;
      emplaced_prop->set_grid(*this);
    }
  }
  //============================================================================
  /// Move CTOR
  constexpr rectilinear_grid(rectilinear_grid&& other) noexcept
      : m_dimensions{std::move(other.m_dimensions)},
        m_vertex_properties{std::move(other.m_vertex_properties)},
        m_diff_stencil_coefficients{
            std::move(other.m_diff_stencil_coefficients)} {
    for (auto const& [name, prop] : m_vertex_properties) {
      prop->set_grid(*this);
    }
  }
  //----------------------------------------------------------------------------
  /// \param dimensions List of dimensions / axes of the rectilinear grid
  template <typename... Dimensions_>
  requires
    (sizeof...(Dimensions_) == sizeof...(Dimensions)) &&
    (detail::rectilinear_grid::dimension<std::decay_t<Dimensions_>> && ...)
  constexpr rectilinear_grid(Dimensions_&&... dimensions)
      : m_dimensions{std::forward<Dimensions_>(dimensions)...} {}
  //----------------------------------------------------------------------------
 private:
  template <typename Real, integral... Res, std::size_t... Seq>
  constexpr rectilinear_grid(
      axis_aligned_bounding_box<Real, num_dimensions()> const& bb,
      std::index_sequence<Seq...> /*seq*/, Res const... res)
      : m_dimensions{linspace<real_type>{real_type(bb.min(Seq)),
                                         real_type(bb.max(Seq)),
                                         static_cast<std::size_t>(res)}...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Constructs a uniform grid from a tatooine::axis_aligned_bounding_box and a
  /// resolution.
  template <typename Real, integral... Res>
  requires (sizeof...(Res) == num_dimensions())
  constexpr rectilinear_grid(
      axis_aligned_bounding_box<Real, num_dimensions()> const& bb,
      Res const... res)
      : rectilinear_grid{bb, sequence_type{}, res...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Constructs a rectilinear grid in the range of [0 .. 1] with given
  /// resolution.
  /// \param size Resolution of grid.
  constexpr rectilinear_grid(integral auto const... size)
      : rectilinear_grid{
            linspace{0.0, 1.0, static_cast<std::size_t>(size)}...} {
    assert(((size >= 0) && ...));
  }
  //----------------------------------------------------------------------------
  /// Constructs a rectilinear grid by reading a file.
  rectilinear_grid(filesystem::path const& path) { read(path); }
  //----------------------------------------------------------------------------
  ~rectilinear_grid() = default;
  //============================================================================
 private:
  template <std::size_t... Ds>
  constexpr auto copy_without_properties(
      std::index_sequence<Ds...> /*seq*/) const {
    return this_type{m_dimensions.template at<Ds>()...};
  }

 public:
  /// Creates a copy of with any of the give properties
  constexpr auto copy_without_properties() const {
    return copy_without_properties(
        std::make_index_sequence<num_dimensions()>{});
  }
  //============================================================================
  constexpr auto operator=(rectilinear_grid const& other)
      -> rectilinear_grid& = default;
  constexpr auto operator=(rectilinear_grid&& other) noexcept
      -> rectilinear_grid& {
    m_dimensions                = std::move(other.m_dimensions);
    m_vertex_properties         = std::move(other.m_vertex_properties);
    m_diff_stencil_coefficients = std::move(other.m_diff_stencil_coefficients);
    for (auto const& [name, prop] : m_vertex_properties) {
      prop->set_grid(*this);
    }
    for (auto const& [name, prop] : other.m_vertex_properties) {
      prop->set_grid(other);
    }
  }
  //----------------------------------------------------------------------------
  /// Returns a constant reference to the dimension of index I.
  /// \tparam I Index of dimension
  template <std::size_t I>
  requires(I < num_dimensions())
  constexpr auto dimension() const -> auto const& {
    return m_dimensions.template at<I>();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Returns a constant reference to the dimension of index I.
  /// This runtime version is only available if all dimensions are of the same
  /// type.
  /// \param i Index of dimension
  constexpr auto dimension(std::size_t const i) const -> auto const&
  requires (is_same<Dimensions...>) &&
           (num_dimensions() <= 11) {
    if (i == 0) {
      return dimension<0>();
    }
    if constexpr (num_dimensions() > 1) {
      if (i == 1) {
        return dimension<1>();
      }
    }
    if constexpr (num_dimensions() > 2) {
      if (i == 2) {
        return dimension<2>();
      }
    }
    if constexpr (num_dimensions() > 3) {
      if (i == 3) {
        return dimension<3>();
      }
    }
    if constexpr (num_dimensions() > 4) {
      if (i == 4) {
        return dimension<4>();
      }
    }
    if constexpr (num_dimensions() > 5) {
      if (i == 5) {
        return dimension<5>();
      }
    }
    if constexpr (num_dimensions() > 6) {
      if (i == 6) {
        return dimension<6>();
      }
    }
    if constexpr (num_dimensions() > 7) {
      if (i == 7) {
        return dimension<7>();
      }
    }
    if constexpr (num_dimensions() > 8) {
      if (i == 8) {
        return dimension<8>();
      }
    }
    if constexpr (num_dimensions() > 9) {
      if (i == 9) {
        return dimension<9>();
      }
    }
    if constexpr (num_dimensions() > 10) {
      if (i == 10) {
        return dimension<10>();
      }
    }
    return dimension<0>();
  }
  //----------------------------------------------------------------------------
  /// \return Constant reference to all dimensions stored in a tuple.
  constexpr auto dimensions() const -> auto const& { return m_dimensions; }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  constexpr auto min(std::index_sequence<Seq...> /*seq*/) const {
    return vec<real_type, num_dimensions()>{
        static_cast<real_type>(dimension<Seq>().front())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// \return Minimal point in all dimensions.
  constexpr auto min() const { return min(sequence_type{}); }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  constexpr auto max(std::index_sequence<Seq...> /*seq*/) const {
    return vec<real_type, num_dimensions()>{
        static_cast<real_type>(dimension<Seq>().back())...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// \return Maximal point in all dimensions.
  constexpr auto max() const { return max(sequence_type{}); }
  //----------------------------------------------------------------------------
  /// \return Extent of dimension of index I
  template <std::size_t I>
  constexpr auto extent() const {
    return dimension<I>().back() - dimension<I>().front();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  /// \return Extent in all dimesions
  template <std::size_t... Is>
  constexpr auto extent(std::index_sequence<Is...> /*seq*/) const {
    return vec{extent<Is>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// \return Extent in all dimesions
  constexpr auto extent() const {
    return extent(std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  /// \return Extent cell with index cell_index in dimension with Index
  ///         DimensionIndex.
  template <std::size_t DimensionIndex>
  constexpr auto extent(std::size_t const cell_index) const {
    auto const& dim = dimension<DimensionIndex>();
    return dim[cell_index + 1] - dim[cell_index];
  }
  //----------------------------------------------------------------------------
  template <std::size_t I>
  constexpr auto center() const {
    return (dimension<I>().back() + dimension<I>().front()) / 2;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <std::size_t... Is>
  constexpr auto center(std::index_sequence<Is...> /*seq*/) const {
    return vec{center<Is>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto center() const {
    return center(std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  /// \return Axis aligned bounding box of grid
  template <std::size_t... Seq>
  requires(sizeof...(Seq) == num_dimensions())
  constexpr auto bounding_box(std::index_sequence<Seq...> /*seq*/) const {
    return axis_aligned_bounding_box<real_type, num_dimensions()>{
        vec<real_type, num_dimensions()>{
            static_cast<real_type>(dimension<Seq>().front())...},
        vec<real_type, num_dimensions()>{
            static_cast<real_type>(dimension<Seq>().back())...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// \return Axis aligned bounding box of grid
  constexpr auto bounding_box() const { return bounding_box(sequence_type{}); }
  //----------------------------------------------------------------------------
 private:
  /// \return Resolution of grid
  template <std::size_t... Seq>
  requires(sizeof...(Seq) == num_dimensions())
  constexpr auto size(std::index_sequence<Seq...> /*seq*/) const {
    return std::array{size<Seq>()...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// \return Resolution of grid
  constexpr auto size() const { return size(sequence_type{}); }
  //----------------------------------------------------------------------------
  /// \return Resolution of of dimension I
  /// \tparam I Index of dimensions
  template <std::size_t I>
  constexpr auto size() const {
    return dimension<I>().size();
  }
  //----------------------------------------------------------------------------
  /// \return Resolution of of dimension i
  /// Note: i needs to be <= 10
  /// \tparam i Index of dimensions
  constexpr auto size(std::size_t const i) const {
    if (i == 0) {
      return size<0>();
    }
    if constexpr (num_dimensions() > 1) {
      if (i == 1) {
        return size<1>();
      }
    }
    if constexpr (num_dimensions() > 2) {
      if (i == 2) {
        return size<2>();
      }
    }
    if constexpr (num_dimensions() > 3) {
      if (i == 3) {
        return size<3>();
      }
    }
    if constexpr (num_dimensions() > 4) {
      if (i == 4) {
        return size<4>();
      }
    }
    if constexpr (num_dimensions() > 5) {
      if (i == 5) {
        return size<5>();
      }
    }
    if constexpr (num_dimensions() > 6) {
      if (i == 6) {
        return size<6>();
      }
    }
    if constexpr (num_dimensions() > 7) {
      if (i == 7) {
        return size<7>();
      }
    }
    if constexpr (num_dimensions() > 8) {
      if (i == 8) {
        return size<8>();
      }
    }
    if constexpr (num_dimensions() > 9) {
      if (i == 9) {
        return size<9>();
      }
    }
    if constexpr (num_dimensions() > 10) {
      if (i == 10) {
        return size<10>();
      }
    }
    return std::numeric_limits<std::size_t>::max();
  }
  //----------------------------------------------------------------------------
  ///
  template <std::size_t I>
  constexpr auto set_dimension(convertible_to<dimension_type<I>> auto&& dim) {
    // TODO update diff stencils
    m_dimensions.template at<I>() = std::forward<decltype(dim)>(dim);
  }
  //----------------------------------------------------------------------------
  /// Inserts new discrete point in dimension I with extent of last cell.
  template <std::size_t I>
  constexpr auto push_back() {
    // TODO update diff stencils
    auto& dim = m_dimensions.template at<I>();
    if constexpr (is_linspace<std::decay_t<decltype(dim)>>) {
      dim.push_back();
    } else {
      dim.push_back(dimension<I>().back() + extent<I>(size<I>() - 2));
    }
  }
  //----------------------------------------------------------------------------
  /// Removes last discrete point in dimension I.
  template <std::size_t I>
  requires requires(dimension_type<I> dim) { dim.pop_back(); }
  constexpr auto pop_back() {
    // TODO update diff stencils
    m_dimensions.template at<I>().pop_back();
  }
  //----------------------------------------------------------------------------
  /// Removes first discrete point in dimension I.
  template <std::size_t I>
  requires requires(dimension_type<I> dim) { dim.pop_back(); }
  constexpr auto pop_front() {
    // TODO update diff stencils
    m_dimensions.template at<I>().pop_front(); }
  //----------------------------------------------------------------------------
 private:
  /// Checks if point [comps...] is inside of grid.
  template <arithmetic... Comps, std::size_t... Seq>
  requires(num_dimensions() == sizeof...(Comps))
  constexpr auto is_inside(std::index_sequence<Seq...> /*seq*/,
                           Comps const... comps) const {
    return ((dimension<Seq>().front() <= comps &&
             comps <= dimension<Seq>().back()) &&
            ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Checks if point [comps...] is inside of grid.
  template <arithmetic... Comps>
  requires(num_dimensions() == sizeof...(Comps))
  constexpr auto is_inside(Comps const... comps) const {
    return is_inside(std::make_index_sequence<num_dimensions()>{}, comps...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  /// Checks if point p is inside of grid.
  template <std::size_t... Seq>
  constexpr auto is_inside(pos_type const& p,
                           std::index_sequence<Seq...> /*seq*/) const {
    return ((dimension<Seq>().front() <= p(Seq) &&
             p(Seq) <= dimension<Seq>().back()) &&
            ...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  /// Checks if point p is inside of grid.
  constexpr auto is_inside(pos_type const& p) const {
    return is_inside(p, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
 private:
  /// Checks if point p is inside of grid.
  template <std::size_t... Seq>
  constexpr auto is_inside(std::array<real_type, num_dimensions()> const& p,
                           std::index_sequence<Seq...> /*seq*/) const {
    return is_inside(p[Seq]...);
  }
  //----------------------------------------------------------------------------
 public:
  /// Checks if point p is inside of grid.
  constexpr auto is_inside(
      std::array<real_type, num_dimensions()> const& p) const {
    return is_inside(p, sequence_type{});
  }
  //----------------------------------------------------------------------------
  /// returns cell index and factor for interpolation of position x in dimension
  /// DimensionIndex.
  template <std::size_t DimensionIndex>
  auto cell_index(arithmetic auto x) const
      -> std::pair<std::size_t, real_type> {
    auto const& dim = dimension<DimensionIndex>();
    if (std::abs(x - dim.front()) < 1e-10) {
      x = dim.front();
    }
    if (std::abs(x - dim.back()) < 1e-10) {
      x = dim.back();
    }
    if constexpr (is_linspace<std::decay_t<decltype(dim)>>) {
      // calculate
      auto pos =
          (x - dim.front()) / (dim.back() - dim.front()) * (dim.size() - 1);
      auto quantized_pos = static_cast<std::size_t>(std::floor(pos));
      if (quantized_pos == dim.size() - 1) {
        --quantized_pos;
      }
      auto cell_position = pos - quantized_pos;
      if (quantized_pos == dim.size() - 1) {
        --quantized_pos;
        cell_position = 1;
      }
      return {quantized_pos, cell_position};
    } else {
      // binary search
      std::size_t left  = 0;
      std::size_t right = dim.size() - 1;
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
 private:
  template <std::size_t... DimensionIndex>
  auto cell_index(std::index_sequence<DimensionIndex...>,
                  arithmetic auto const... xs) const
      -> std::array<std::pair<std::size_t, double>, num_dimensions()> {
    return std::array{cell_index<DimensionIndex>(static_cast<double>(xs))...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto cell_index(arithmetic auto const... xs) const 
  requires (sizeof...(xs) == num_dimensions()) {
    return cell_index(sequence_type{}, xs...);
  }
  //------------------------------------------------------------------------------
  auto cell_index(fixed_size_vec<num_dimensions()> auto&& xs) const {
    return invoke_unpacked(
        [this](auto const... xs) { return cell_index(xs...); },
        unpack(std::forward<decltype(xs)>(xs)));
  }
  //----------------------------------------------------------------------------
  auto diff_stencil_coefficients(std::size_t const dim_index,
                                 std::size_t const stencil_size,
                                 std::size_t const stencil_center,
                                 std::size_t const i) const -> auto const& {
    return m_diff_stencil_coefficients[dim_index]
                                      [stencil_size - min_stencil_size]
                                      [stencil_center][i];
  }
  //----------------------------------------------------------------------------
  auto diff_stencil_coefficients_created_once() const {
    auto lock = std::lock_guard{m_stencil_mutex};
    return m_diff_stencil_coefficients_created_once;
  }
  //----------------------------------------------------------------------------
  auto update_diff_stencil_coefficients() const {
    auto lock = std::lock_guard{m_stencil_mutex};
    for (std::size_t dim = 0; dim < num_dimensions(); ++dim) {
      for (std::size_t i = 0; i < num_stencils; ++i) {
        m_diff_stencil_coefficients[dim][i].clear();
        m_diff_stencil_coefficients[dim][i].shrink_to_fit();
      }
    }
    update_diff_stencil_coefficients(
        std::make_index_sequence<num_dimensions()>{});
    m_diff_stencil_coefficients_created_once = true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t... DimSeq>
  auto update_diff_stencil_coefficients(
      std::index_sequence<DimSeq...> /*seq*/) const {
    [[maybe_unused]] auto const dim_seqs = std::array{DimSeq...};
    (update_diff_stencil_coefficients_dim<DimSeq>(), ...);
  }
  //----------------------------------------------------------------------------
  template <std::size_t Dim>
  auto update_diff_stencil_coefficients_dim() const {
    auto const& dim                   = dimension<Dim>();
    auto&       stencils_of_dimension = m_diff_stencil_coefficients[Dim];
    for (std::size_t stencil_size = min_stencil_size;
         stencil_size <= std::min(max_stencil_size, dim.size());
         ++stencil_size) {
      auto& stencils_of_cur_size =
          stencils_of_dimension[stencil_size - min_stencil_size];
      stencils_of_cur_size.resize(stencil_size);
      for (std::size_t stencil_center = 0; stencil_center < stencil_size;
           ++stencil_center) {
        update_diff_stencil_coefficients(dim,
                                         stencils_of_cur_size[stencil_center],
                                         stencil_size, stencil_center);
      }
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Dim>
  auto update_diff_stencil_coefficients(
      Dim const& dim, std::vector<std::vector<real_type>>& stencils,
      std::size_t const stencil_size, std::size_t const stencil_center) const {
    stencils.reserve(dim.size());

    for (std::size_t i = 0; i < stencil_center; ++i) {
      stencils.emplace_back();
    }
    for (std::size_t i = stencil_center;
         i < dim.size() - (stencil_size - stencil_center - 1); ++i) {
      std::vector<real_type> xs(stencil_size);
      for (std::size_t j = 0; j < stencil_size; ++j) {
        xs[j] = dim[i - stencil_center + j] - dim[i];
      }
      stencils.push_back(finite_differences_coefficients(1, xs));
    }
  }
  //----------------------------------------------------------------------------
  auto vertices() const { return vertex_container{*this}; }
  //----------------------------------------------------------------------------
  template <std::size_t... DIs, integral Int>
  auto vertex_at(std::index_sequence<DIs...> /*seq*/,
                 std::array<Int, num_dimensions()> const& is) const
      -> vec<real_type, num_dimensions()> {
    return pos_type{static_cast<real_type>(dimension<DIs>()[is[DIs]])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t... DIs>
  auto vertex_at(std::index_sequence<DIs...>, integral auto const... is) const
      -> vec<real_type, num_dimensions()> {
    static_assert(sizeof...(DIs) == sizeof...(is));
    static_assert(sizeof...(is) == num_dimensions());
    return pos_type{static_cast<real_type>(dimension<DIs>()[is])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertex_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions());
    return vertex_at(sequence_type{}, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Int>
  auto vertex_at(std::array<Int, num_dimensions()> const& is) const {
    return vertex_at(sequence_type{}, is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vertex_at(vertex_handle const& h) const {
    return vertex_at(sequence_type{}, h.indices());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](vertex_handle const& h) const {
    return vertex_at(sequence_type{}, h.indices());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto plain_index(integral auto const... is)
    requires (sizeof...(is) == num_dimensions()) {
    auto const arr_is     = std::array{is...};
    auto const size = this->size();
    auto       pi         = std::size_t{};
    auto       multiplier = 1;
    for (std::size_t i = 0; i < sizeof...(is); ++i) {
      pi += arr_is[i] * multiplier;
      multiplier *= size[i];
    }
    return pi;
  }
  //----------------------------------------------------------------------------
  auto cells() const { return cell_container{*this}; }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  auto add_dimension(
      detail::rectilinear_grid::dimension auto&& additional_dimension,
      std::index_sequence<Is...> /*seq*/) const {
    return rectilinear_grid<Dimensions...,
                            std::decay_t<decltype(additional_dimension)>>{
        dimension<Is>()...,
        std::forward<decltype(additional_dimension)>(additional_dimension)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto add_dimension(
      detail::rectilinear_grid::dimension auto&& additional_dimension) const {
    return add_dimension(
        std::forward<decltype(additional_dimension)>(additional_dimension),
        sequence_type{});
  }
  //----------------------------------------------------------------------------
  auto has_vertex_property(std::string const& name) {
    return m_vertex_properties.find(name) != end(m_vertex_properties);
  }
  //----------------------------------------------------------------------------
  auto remove_vertex_property(std::string const& name) -> void {
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
      auto handler  = m_vertex_properties.extract(it);
      handler.key() = new_name;
      m_vertex_properties.insert(std::move(handler));
    }
  }
  //----------------------------------------------------------------------------
  template <typename Container, typename... Args>
  auto create_vertex_property(std::string const& name, Args&&... args)
      -> auto& {
    if (!has_vertex_property(name)) {
      auto new_prop = new typed_vertex_property_type<Container>{
          *this, std::forward<Args>(args)...};
      m_vertex_properties.emplace(
          name, std::unique_ptr<vertex_property_type>{new_prop});
      if constexpr (sizeof...(Args) == 0) {
        new_prop->resize(size());
      }
      return *new_prop;
    }
    throw std::runtime_error{"There is already a vertex property named \"" +
                             name + "\"."};
  }
  //----------------------------------------------------------------------------
  auto vertex_properties() const -> auto const& { return m_vertex_properties; }
  auto vertex_properties() -> auto& { return m_vertex_properties; }
  //----------------------------------------------------------------------------
  template <typename T, typename IndexOrder = x_fastest>
  auto insert_vertex_property(std::string const& name) -> auto& {
    return insert_contiguous_vertex_property<T, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_scalar_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<tatooine::real_number, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_vec2_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<vec2, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_vec3_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<vec3, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_vec4_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<vec4, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_mat2_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<mat2, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_mat3_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<mat3, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest>
  auto insert_mat4_vertex_property(std::string const& name) -> auto& {
    return insert_vertex_property<mat4, IndexOrder>(name);
  }
  //----------------------------------------------------------------------------
  /// @}
  //----------------------------------------------------------------------------
  template <typename T, typename IndexOrder = x_fastest>
  auto insert_contiguous_vertex_property(std::string const& name) -> auto& {
    return create_vertex_property<dynamic_multidim_array<T, IndexOrder>>(
        name, size());
  }
  //----------------------------------------------------------------------------
  template <typename T, typename IndexOrder = x_fastest>
  auto insert_chunked_vertex_property(
      std::string const& name, std::vector<std::size_t> const& chunk_size)
      -> auto& {
    return create_vertex_property<chunked_multidim_array<T, IndexOrder>>(
        name, size(), chunk_size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename IndexOrder = x_fastest>
  auto insert_chunked_vertex_property(
      std::string const&                               name,
      std::array<std::size_t, num_dimensions()> const& chunk_size) -> auto& {
    return create_vertex_property<chunked_multidim_array<T, IndexOrder>>(
        name, size(), chunk_size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename IndexOrder = x_fastest>

  auto insert_chunked_vertex_property(std::string const& name,
                                      integral auto const... chunk_size)
      -> auto& requires(sizeof...(chunk_size) == num_dimensions()) {
    return create_vertex_property<chunked_multidim_array<T, IndexOrder>>(
        name, size(),
        std::vector<std::size_t>{static_cast<std::size_t>(chunk_size)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, typename IndexOrder = x_fastest>
  auto insert_chunked_vertex_property(std::string const& name) -> auto& {
    return create_vertex_property<chunked_multidim_array<T, IndexOrder>>(
        name, size(), make_array<num_dimensions()>(std::size_t(10)));
  }
  //----------------------------------------------------------------------------
  /// \return Reference to a polymorphic vertex property.
  template <typename T, bool HasNonConstReference = true>
  auto vertex_property(std::string const& name)
      -> typed_vertex_property_interface_type<T, HasNonConstReference>& {
    if (auto it = m_vertex_properties.find(name);
        it == end(m_vertex_properties)) {
      return insert_vertex_property<T>(name);
    } else {
      if (typeid(T) != it->second->type()) {
        throw std::runtime_error{
            "type of property \"" + name + "\"(" +
            boost::core::demangle(it->second->type().name()) +
            ") does not match specified type " + type_name<T>() + "."};
      }
      return *dynamic_cast<
          typed_vertex_property_interface_type<T, HasNonConstReference>*>(
          it->second.get());
    }
  }
  //----------------------------------------------------------------------------
  template <typename T, bool HasNonConstReference = true>
  auto vertex_property(std::string const& name) const
      -> typed_vertex_property_interface_type<T, HasNonConstReference> const& {
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
      return *dynamic_cast<
          typed_vertex_property_interface_type<T, HasNonConstReference> const*>(
          it->second.get());
    }
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto scalar_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<tatooine::real_number, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto scalar_vertex_property(std::string const& name) -> auto& {
    return vertex_property<tatooine::real_number, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec2, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec2, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec3, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec3, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<vec4, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto vec4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<vec4, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat2_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat2, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat2_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat2, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat3_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat3, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat3_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat3, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat4_vertex_property(std::string const& name) const -> auto const& {
    return vertex_property<mat4, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <bool HasNonConstReference = true>
  auto mat4_vertex_property(std::string const& name) -> auto& {
    return vertex_property<mat4, HasNonConstReference>(name);
  }
  //----------------------------------------------------------------------------
  template <typename T, typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder = GlobalIndexOrder>
  auto insert_lazy_vertex_property(filesystem::path const& path,
                                   std::string const&      dataset_name)
      -> typed_vertex_property_interface_type<T, false>& {
    auto const ext = path.extension();
#if TATOOINE_HDF5_AVAILABLE
    if (ext == ".h5") {
      return insert_hdf5_lazy_vertex_property<T, GlobalIndexOrder,
                                              LocalIndexOrder>(path,
                                                               dataset_name);
    }
#endif
#ifdef TATOOINE_NETCDF_AVAILABLE
    if (ext == ".nc") {
      return insert_netcdf_lazy_vertex_property<T, GlobalIndexOrder,
                                                LocalIndexOrder>(path,
                                                                 dataset_name);
    }
#endif
    throw std::runtime_error{
        "[rectilinear_grid::insert_lazy_vertex_property] - unknown file "
        "extension"};
  }
  //----------------------------------------------------------------------------
#if TATOOINE_HDF5_AVAILABLE
  template <typename IndexOrder = x_fastest, typename T>
  auto insert_vertex_property(hdf5::dataset<T> const& dataset) -> auto& {
    return insert_vertex_property<IndexOrder>(dataset, dataset.name());
  }
  //----------------------------------------------------------------------------
  template <typename IndexOrder = x_fastest, typename T>
  auto insert_vertex_property(hdf5::dataset<T> const& dataset,
                              std::string const&      name) -> auto& {
    auto num_dims_dataset = dataset.num_dimensions();
    if (num_dimensions() != num_dims_dataset) {
      throw std::runtime_error{
          "Number of dimensions do not match for HDF5 dataset and "
          "rectilinear_grid."};
    }
    auto size_dataset = dataset.size();
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (size_dataset[i] != size(i)) {
        std::stringstream ss;
        ss << "Resolution of rectilinear_grid and HDF5 DataSet do not match. "
              "rectilinear_grid("
           << size(0);
        for (std::size_t i = 1; i < num_dimensions(); ++i) {
          ss << ", " << size(i);
        }
        ss << ") and hdf5 dataset(" << size_dataset[i];
        for (std::size_t i = 1; i < num_dimensions(); ++i) {
          ss << ", " << size_dataset[i];
        }
        ss << ")";
        throw std::runtime_error{ss.str()};
      }
    }
    auto& prop = insert_vertex_property<T, IndexOrder>(name);
    dataset.read(prop);
    return prop;
  }
  //----------------------------------------------------------------------------
  template <typename T, typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder = GlobalIndexOrder>
  auto insert_hdf5_lazy_vertex_property(filesystem::path const& path,
                                        std::string const&      dataset_name)
      -> auto& {
    hdf5::file f{path};
    return insert_lazy_vertex_property<GlobalIndexOrder, LocalIndexOrder>(
        f.dataset<T>(dataset_name));
  }
  //----------------------------------------------------------------------------
  template <typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder  = GlobalIndexOrder, typename T>
  auto insert_lazy_vertex_property(hdf5::dataset<T> const& dataset) -> auto& {
    return insert_lazy_vertex_property<GlobalIndexOrder, LocalIndexOrder>(
        dataset, dataset.name());
  }
  //----------------------------------------------------------------------------
  template <typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder  = GlobalIndexOrder, typename T>
  auto insert_lazy_vertex_property(hdf5::dataset<T> const& dataset,
                                   std::string const&      name) -> auto& {
    auto num_dims_dataset = dataset.num_dimensions();
    if (num_dimensions() != num_dims_dataset) {
      throw std::runtime_error{
          "Number of dimensions do not match for HDF5 dataset and "
          "rectilinear_grid."};
    }
    auto size_dataset = dataset.size();
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (size_dataset[i] != size(i)) {
        std::stringstream ss;
        ss << "Resolution of rectilinear_grid and HDF5 DataSet (\"" << name
           << "\")do not match. rectilinear_grid(" << size(0);
        for (std::size_t i = 1; i < num_dimensions(); ++i) {
          ss << ", " << size(i);
        }
        ss << ") and hdf5 dataset(" << size_dataset[i];
        for (std::size_t i = 1; i < num_dimensions(); ++i) {
          ss << ", " << size_dataset[i];
        }
        ss << ")";
        throw std::runtime_error{ss.str()};
      }
    }
    return create_vertex_property<
        lazy_reader<hdf5::dataset<T>, GlobalIndexOrder, LocalIndexOrder>>(
        name, dataset,
        std::vector<std::size_t>(num_dimensions(),
                                 m_chunk_size_for_lazy_properties));
  }
#endif
#ifdef TATOOINE_NETCDF_AVAILABLE
  //----------------------------------------------------------------------------
  template <typename T, typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder = GlobalIndexOrder>
  auto insert_netcdf_lazy_vertex_property(filesystem::path const& path,
                                          std::string const&      dataset_name)
      -> auto& {
    netcdf::file f{path, netCDF::NcFile::read};
    return insert_lazy_vertex_property<GlobalIndexOrder, LocalIndexOrder, T>(
        f.variable<T>(dataset_name));
  }
  //----------------------------------------------------------------------------
  template <typename GlobalIndexOrder = x_fastest,
            typename LocalIndexOrder  = GlobalIndexOrder, typename T>
  auto insert_lazy_vertex_property(netcdf::variable<T> const& dataset)
      -> auto& {
    return create_vertex_property<
        lazy_reader<netcdf::variable<T>, GlobalIndexOrder, LocalIndexOrder>>(
        dataset.name(), dataset,
        std::vector<std::size_t>(num_dimensions(),
                                 m_chunk_size_for_lazy_properties));
  }
#endif
  //============================================================================
  template <typename F>
  requires invocable_with_n_integrals<F, num_dimensions()> ||
           invocable<F, pos_type>
  auto sample_to_vertex_property(F&& f, std::string const& name) -> auto& {
    return sample_to_vertex_property(std::forward<F>(f), name,
                                     execution_policy::sequential);
  }
  //----------------------------------------------------------------------------
  template <typename F>
  requires invocable_with_n_integrals<F, num_dimensions()> ||
           invocable<F, pos_type>
  auto sample_to_vertex_property(F&& f, std::string const& name,
                                 execution_policy_tag auto tag) -> auto& {
    if constexpr (invocable<F, pos_type>) {
      return sample_to_vertex_property_pos(std::forward<F>(f), name, tag);
    } else {
      return sample_to_vertex_property_indices(
          std::forward<F>(f), name, tag,
          std::make_index_sequence<num_dimensions()>{});
    }
  }
  //----------------------------------------------------------------------------
 private:
  template <invocable_with_n_integrals<num_dimensions()> F, std::size_t... Is>
  auto sample_to_vertex_property_indices(F&& f, std::string const& name,
                                         execution_policy_tag auto tag,
                                         std::index_sequence<Is...> /*seq*/)
      -> auto& {
    using T    = std::invoke_result_t<F, decltype(Is)...>;
    auto& prop = vertex_property<T>(name);
    vertices().iterate_indices(
        [&](auto const... is) {
          try {
            prop(is...) = f(is...);
          } catch (std::exception&) {
            if constexpr (tensor_num_components<T> == 1) {
              prop(is...) = nan<T>();
            } else {
              prop(is...) = T::fill(nan<T>());
            }
          }
        },
        tag);
    return prop;
  }
  //----------------------------------------------------------------------------
  template <invocable<pos_type> F>
  auto sample_to_vertex_property_pos(F&& f, std::string const& name,
                                     execution_policy_tag auto tag) -> auto& {
    using T    = std::invoke_result_t<F, pos_type>;
    auto& prop = vertex_property<T>(name);
    vertices().iterate_indices(
        [&](auto const... is) {
          try {
            prop(is...) = f(vertices()(is...));
          } catch (std::exception&) {
            if constexpr (tensor_num_components<T> == 1) {
              prop(is...) = T{nan<T>()};
            } else {
              prop(is...) = T::fill(nan<tensor_value_type<T>>());
            }
          }
        },
        tag);
    return prop;
  }
  //============================================================================
  public:
  auto read(filesystem::path const& path) {
#ifdef TATOOINE_NETCDF_AVAILABLE
    if constexpr (!is_uniform) {
      if (path.extension() == ".nc") {
        read_netcdf(path);
        return;
      }
    }
#endif
    if constexpr (num_dimensions() == 2 || num_dimensions() == 3) {
      if (path.extension() == ".vtk") {
        read_vtk(path);
        return;
      }
      if constexpr (is_uniform) {
        if (path.extension() == ".am") {
          read_amira(path);
          return;
        }
      }
    }
    throw std::runtime_error{
        "[rectilinear_grid::read] Unknown file extension."};
  }
  //----------------------------------------------------------------------------
  struct vtk_listener : vtk::legacy_file_listener {
    this_type& gr;
    bool&      is_structured_points;
    vec3&      spacing;
    vtk_listener(this_type& gr_, bool& is_structured_points_, vec3& spacing_)
        : gr{gr_},
          is_structured_points{is_structured_points_},
          spacing{spacing_} {}
    // header data
    auto on_dataset_type(vtk::dataset_type t) -> void override {
      if (t == vtk::dataset_type::structured_points && !is_uniform) {
        is_structured_points = true;
      }
    }

    // coordinate data
    auto on_origin(double x, double y, double z) -> void override {
      gr.dimension<0>().front() = x;
      gr.dimension<1>().front() = y;
      if (num_dimensions() < 3 && z > 1) {
        throw std::runtime_error{
            "[rectilinear_grid::read_vtk] number of dimensions is < 3 but got "
            "third "
            "dimension."};
      }
      if constexpr (num_dimensions() > 3) {
        gr.dimension<2>().front() = z;
      }
    }
    auto on_spacing(double x, double y, double z) -> void override {
      spacing = {x, y, z};
    }
    auto on_dimensions(std::size_t x, std::size_t y, std::size_t z)
        -> void override {
      gr.dimension<0>().resize(x);
      gr.dimension<1>().resize(y);
      if (num_dimensions() < 3 && z > 1) {
        throw std::runtime_error{
            "[rectilinear_grid::read_vtk] number of dimensions is < 3 but got "
            "third "
            "dimension."};
      }
      if constexpr (num_dimensions() > 2) {
        gr.dimension<2>().resize(z);
      }
    }
    auto on_x_coordinates(std::vector<float> const& /*xs*/) -> void override {}
    auto on_x_coordinates(std::vector<double> const& /*xs*/) -> void override {}
    auto on_y_coordinates(std::vector<float> const& /*ys*/) -> void override {}
    auto on_y_coordinates(std::vector<double> const& /*ys*/) -> void override {}
    auto on_z_coordinates(std::vector<float> const& /*zs*/) -> void override {}
    auto on_z_coordinates(std::vector<double> const& /*zs*/) -> void override {}

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

    template <typename T>
    auto insert_prop(std::string const& prop_name, std::vector<T> const& data,
                     std::size_t const num_components) {
      std::size_t i = 0;
      if (num_components == 1) {
        auto& prop = gr.insert_vertex_property<T>(prop_name);
        gr.vertices().iterate_indices(
            [&](auto const... is) { prop(is...) = data[i++]; });
      }
      if (num_components == 2) {
        auto& prop = gr.insert_vertex_property<vec<T, 2>>(prop_name);
        gr.vertices().iterate_indices([&](auto const... is) {
          prop(is...) = {data[i], data[i + 1]};
          i += num_components;
        });
      }
      if (num_components == 3) {
        auto& prop = gr.insert_vertex_property<vec<T, 3>>(prop_name);
        gr.vertices().iterate_indices([&](auto const... is) {
          prop(is...) = {data[i], data[i + 1], data[i + 2]};
          i += num_components;
        });
      }
      if (num_components == 4) {
        auto& prop = gr.insert_vertex_property<vec<T, 4>>(prop_name);
        gr.vertices().iterate_indices([&](auto const... is) {
          prop(is...) = {data[i], data[i + 1], data[i + 2], data[i + 3]};
          i += num_components;
        });
      }
    }
    auto on_scalars(std::string const& data_name,
                    std::string const& /*lookup_table_name*/,
                    std::size_t const num_components, std::vector<float> const& data,
                    vtk::reader_data) -> void override {
      insert_prop<float>(data_name, data, num_components);
    }
    auto on_scalars(std::string const& data_name,
                    std::string const& /*lookup_table_name*/,
                    std::size_t const          num_components,
                    std::vector<double> const& data, vtk::reader_data)
        -> void override {
      insert_prop<double>(data_name, data, num_components);
    }
    auto on_point_data(std::size_t) -> void override {}
    auto on_cell_data(std::size_t) -> void override {}
    auto on_field_array(std::string const /*field_name*/,
                        std::string const       field_array_name,
                        std::vector<int> const& data, std::size_t num_components,
                        std::size_t /*num_tuples*/) -> void override {
      insert_prop<int>(field_array_name, data, num_components);
    }
    auto on_field_array(std::string const /*field_name*/,
                        std::string const         field_array_name,
                        std::vector<float> const& data, std::size_t num_components,
                        std::size_t /*num_tuples*/) -> void override {
      insert_prop<float>(field_array_name, data, num_components);
    }
    auto on_field_array(std::string const /*field_name*/,
                        std::string const          field_array_name,
                        std::vector<double> const& data, std::size_t num_components,
                        std::size_t /*num_tuples*/) -> void override {
      insert_prop<double>(field_array_name, data, num_components);
    }
  };
  //============================================================================
  auto read_vtk(filesystem::path const& path) requires(num_dimensions() == 2) ||
      (num_dimensions() == 3) {
    bool             is_structured_points = false;
    vec3             spacing;
    vtk_listener     listener{*this, is_structured_points, spacing};
    vtk::legacy_file f{path};
    f.add_listener(listener);
    f.read();

    if (is_structured_points) {
      if constexpr (is_same<std::decay_t<decltype(dimension<0>())>,
                                   linspace<double>>) {
        dimension<0>().back() = dimension<0>().front() + (size<0>() - 1) * spacing(0);
      } else {
        std::size_t i = 0;
        for (auto& d : dimension<0>()) {
          d = dimension<0>().front() + (i++) * spacing(0);
        }
      }
      if constexpr (is_same<std::decay_t<decltype(dimension<1>())>,
                                   linspace<double>>) {
        dimension<1>().back() = dimension<1>().front() + (size<1>() - 1) * spacing(1);
      } else {
        std::size_t i = 0;
        for (auto& d : dimension<1>()) {
          d = dimension<1>().front() + (i++) * spacing(1);
        }
      }
      if constexpr (num_dimensions() == 3) {
        if constexpr (is_same<std::decay_t<decltype(dimension<2>())>,
                                     linspace<double>>) {
          dimension<2>().back() = dimension<2>().front() + (size<2>() - 1) * spacing(2);
        } else {
          std::size_t i = 0;
          for (auto& d : dimension<2>()) {
            d = dimension<2>().front() + (i++) * spacing(2);
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto read_amira(filesystem::path const& path) requires is_uniform &&
      ((num_dimensions() == 2) || (num_dimensions() == 3)) {
    auto const reader_data = amira::read<real_type>(path);
    auto const& [data, dims, aabb, num_components] = reader_data;
    if (dims[2] == 1 && num_dimensions() == 3) {
      throw std::runtime_error{
          "[rectilinear_grid::read_amira] file contains 2-dimensional data. "
          "Cannot read into 3-dimensional rectilinear_grid"};
    }
    if (dims[2] > 1 && num_dimensions() == 2) {
      throw std::runtime_error{
          "[rectilinear_grid::read_amira] file contains 3-dimensional data. "
          "Cannot read into 2-dimensional rectilinear_grid"};
    }

    // set dimensions
    if constexpr (is_linspace<std::decay_t<decltype(dimension<0>())>>) {
      dimension<0>().front() = aabb.min(0);
      dimension<0>().back()  = aabb.max(0);
      dimension<0>().resize(dims[0]);
    } else {
      auto const uniform_dim =
          linspace<double>{aabb.min(0), aabb.max(0), dims[0]};
      dimension<0>().resize(dims[0]);
      std::ranges::copy(uniform_dim, begin(dimension<0>()));
    }
    if constexpr (is_linspace<std::decay_t<decltype(dimension<1>())>>) {
      dimension<1>().front() = aabb.min(1);
      dimension<1>().back()  = aabb.max(1);
      dimension<1>().resize(dims[1]);
    } else {
      auto const uniform_dim =
          linspace<double>{aabb.min(1), aabb.max(1), dims[1]};
      dimension<1>().resize(dims[1]);
      std::ranges::copy(uniform_dim, begin(dimension<1>()));
    }
    if constexpr (num_dimensions() == 3) {
      if constexpr (is_linspace<std::decay_t<decltype(dimension<2>())>>) {
        dimension<2>().front() = aabb.min(2);
        dimension<2>().back()  = aabb.max(2);
        dimension<2>().resize(dims[2]);
      } else {
        auto const uniform_dim =
            linspace<double>{aabb.min(2), aabb.max(2), dims[2]};
        dimension<2>().resize(dims[2]);
        std::ranges::copy(uniform_dim, begin(dimension<2>()));
      }
    }
    // copy data
    auto i = std::size_t{};
    if (num_components == 1) {
      auto& prop = scalar_vertex_property(path.filename().string());
      vertices().iterate_indices([&](auto const... is) {
        auto const& data = std::get<0>(reader_data);
        prop(is...)      = data[i++];
      });
    } else if (num_components == 2) {
      auto& prop = vertex_property<vec<real_type, 2>>(path.filename().string());
      vertices().iterate_indices([&](auto const... is) {
        auto const& data = std::get<0>(reader_data);
        prop(is...)      = {data[i], data[i + 1]};
        i += num_components;
      });
    } else if (num_components == 3) {
      auto& prop = vertex_property<vec<real_type, 3>>(path.filename().string());
      vertices().iterate_indices([&](auto const... is) {
        auto const& data = std::get<0>(reader_data);
        prop(is...) = {data[i], data[i + 1], data[i + 2]};
        i += num_components;
      });
    }
  }
  //----------------------------------------------------------------------------
#ifdef TATOOINE_NETCDF_AVAILABLE
  auto read_netcdf(filesystem::path const& path) requires (!is_uniform) {
    read_netcdf(path, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename T, std::size_t... Seq>
  auto insert_variables_of_type(netcdf::file& f, bool& first,
                                std::index_sequence<Seq...> /*seq*/) requires (!is_uniform)  {
    for (auto v : f.variables<T>()) {
      if (v.name() == "x" || v.name() == "y" || v.name() == "z" ||
          v.name() == "t" || v.name() == "X" || v.name() == "Y" ||
          v.name() == "Z" || v.name() == "T" || v.name() == "xdim" ||
          v.name() == "ydim" || v.name() == "zdim" || v.name() == "tdim" ||
          v.name() == "Xdim" || v.name() == "Ydim" || v.name() == "Zdim" ||
          v.name() == "Tdim" || v.name() == "XDim" || v.name() == "YDim" ||
          v.name() == "ZDim" || v.name() == "TDim") {
        continue;
      }
      if (v.num_dimensions() != num_dimensions() &&
          v.size()[0] != vertices().size()) {
        throw std::runtime_error{
            "[rectilinear_grid::read_netcdf] variable's number of dimensions "
            "does "
            "not "
            "match rectilinear_grid's number of dimensions:\nnumber of "
            "rectilinear_grid "
            "dimensions: " +
            std::to_string(num_dimensions()) + "\nnumber of data dimensions: " +
            std::to_string(v.num_dimensions()) +
            "\nvariable name: " + v.name()};
      }
      if (!first) {
        auto check = [this, &v](std::size_t i) {
          if (v.size(i) != size(i)) {
            throw std::runtime_error{
                "[rectilinear_grid::read_netcdf] variable's size(" +
                std::to_string(i) +
                ") does not "
                "match rectilinear_grid's size(" +
                std::to_string(i) + ")"};
          }
        };
        (check(Seq), ...);
      } else {
        ((f.variable<
               typename std::decay_t<decltype(dimension<Seq>())>::value_type>(
               v.dimension_name(Seq))
              .read(dimension<Seq>())),
         ...);
        first = false;
      }
      create_vertex_property<lazy_reader<netcdf::variable<T>>>(
          v.name(), v,
          std::vector<std::size_t>(num_dimensions(),
                                   m_chunk_size_for_lazy_properties));
    }
  }
  /// this only reads scalar types
  template <std::size_t... Seq>
  auto read_netcdf(filesystem::path const&     path,
                   std::index_sequence<Seq...> seq) requires (!is_uniform)  {
    netcdf::file f{path, netCDF::NcFile::read};
    bool         first = true;
    insert_variables_of_type<double>(f, first, seq);
    insert_variables_of_type<float>(f, first, seq);
    insert_variables_of_type<int>(f, first, seq);
  }
#endif
  //----------------------------------------------------------------------------
  template <typename T>
  void write_amira(std::string const& path,
                   std::string const& vertex_property_name) const
      requires(num_dimensions() == 3) {
    write_amira(path, vertex_property<T>(vertex_property_name));
  }
  //----------------------------------------------------------------------------
  template <typename T, bool HasNonConstReference>
  void write_amira(
      std::string const&                                                   path,
      typed_vertex_property_interface_type<T, HasNonConstReference> const& prop)
      const requires is_uniform &&(num_dimensions() == 3) {
    std::ofstream     outfile{path, std::ofstream::binary};
    std::stringstream header;

    header << "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n\n";
    header << "define Lattice " << size<0>() << " " << size<1>() << " "
           << size<2>() << "\n\n";
    header << "Parameters {\n";
    header << "    BoundingBox " << dimension<0>().front() << " "
           << dimension<0>().back() << " " << dimension<1>().front() << " "
           << dimension<1>().back() << " " << dimension<2>().front() << " "
           << dimension<2>().back() << ",\n";
    header << "    CoordType \"uniform\"\n";
    header << "}\n";
    if constexpr (tensor_num_components < T >> 1) {
      header << "Lattice { " << type_name<internal_data_type_t<T>>() << "["
             << tensor_num_components<T> << "] Data } @1\n\n";
    } else {
      header << "Lattice { " << type_name<internal_data_type_t<T>>()
             << " Data } @1\n\n";
    }
    header << "# Data section follows\n@1\n";
    auto const header_string = header.str();

    std::vector<T> data;
    data.reserve(size<0>() * size<1>() * size<2>());
    auto back_inserter = [&](auto const... is) { data.push_back(prop(is...)); };
    for_loop(back_inserter, size<0>(), size<1>(), size<2>());
    outfile.write((char*)header_string.c_str(),
                  header_string.size() * sizeof(char));
    outfile.write((char*)data.data(), data.size() * sizeof(T));
  }
  //----------------------------------------------------------------------------

 public:
  auto write(filesystem::path const& path) const {
    auto const ext = path.extension();

    if constexpr (num_dimensions() == 2 || num_dimensions() == 3) {
      if (ext == ".vtk") {
        write_vtk(path);
        return;
      }
    }
    if constexpr (num_dimensions() == 2 || num_dimensions() == 3) {
      if (ext == ".vtr") {
        write_vtr(path);
        return;
      }
    }
    if constexpr (num_dimensions() == 2 || num_dimensions() == 3) {
      if (ext == ".h5") {
        write_visitvs(path);
        return;
      }
    }
    throw std::runtime_error{"Unsupported file extension: \"" + ext.string() +
                             "\"."};
  }
  //----------------------------------------------------------------------------
  auto write_vtk(filesystem::path const& path,
                 std::string const& description = "tatooine rectilinear_grid")
          const -> void requires(num_dimensions() == 2) ||
      (num_dimensions() == 3) {
    auto writer = [this, &path, &description] {
      if constexpr (is_uniform) {
        auto writer =
            vtk::legacy_file_writer{path, vtk::dataset_type::structured_points};
        writer.set_title(description);
        writer.write_header();
        if constexpr (num_dimensions() == 1) {
          writer.write_dimensions(size<0>(), 1, 1);
          writer.write_origin(dimension<0>().front(), 0, 0);
          writer.write_spacing(dimension<0>().spacing(), 0, 0);
        } else if constexpr (num_dimensions() == 2) {
          writer.write_dimensions(size<0>(), size<1>(), 1);
          writer.write_origin(dimension<0>().front(), dimension<1>().front(), 0);
          writer.write_spacing(dimension<0>().spacing(),
                               dimension<1>().spacing(), 0);
        } else if constexpr (num_dimensions() == 3) {
          writer.write_dimensions(size<0>(), size<1>(), size<2>());
          writer.write_origin(dimension<0>().front(), dimension<1>().front(), dimension<2>().front());
          writer.write_spacing(dimension<0>().spacing(),
                               dimension<1>().spacing(),
                               dimension<2>().spacing());
        }
        return writer;
      } else {
        auto writer =
            vtk::legacy_file_writer{path, vtk::dataset_type::rectilinear_grid};
        writer.set_title(description);
        writer.write_header();
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
    writer.write_point_data(vertices().size());
    write_vtk_prop<int, float, double, vec2f, vec3f, vec4f, vec2d, vec3d,
                   vec4d>(writer);
  }
  //----------------------------------------------------------------------------
 private:
  template <typename T, bool HasNonConstReference>
  auto write_vtk_prop(
      vtk::legacy_file_writer& writer, std::string const& name,
      typed_vertex_property_interface_type<T, HasNonConstReference> const& prop)
      const -> void {
    auto data = std::vector<T>{};
    vertices().iterate_indices(
        [&](auto const... is) { data.push_back(prop(is...)); });
    writer.write_scalars(name, data);
  }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto write_vtk_prop(vtk::legacy_file_writer& writer) const -> void {
    for (const auto& [name, prop] : this->m_vertex_properties) {
      (
          [&] {
            if (prop->type() == typeid(Ts)) {
              write_vtk_prop(writer, name, prop->template cast_to_typed<Ts>());
            }
          }(),
          ...);
    }
  }

 public:
  template <typename HeaderType = std::uint64_t>
  auto write_vtr(filesystem::path const& path) const
  requires(num_dimensions() == 2) || (num_dimensions() == 3) {
    detail::rectilinear_grid::vtr_writer<this_type, HeaderType>{*this}.write(
        path);
  }
 public:
  auto write_visitvs(filesystem::path const& path) const -> void {
    write_visitvs(path, std::make_index_sequence<num_dimensions()>{});
  }

 private:
  //----------------------------------------------------------------------------
#if TATOOINE_HDF5_AVAILABLE
  template <typename T, bool HasNonConstReference, std::size_t... Is>
  void write_prop_hdf5(
      hdf5::file& f, std::string const& name,
      typed_vertex_property_interface_type<T, HasNonConstReference> const& prop,
      std::index_sequence<Is...> /*seq*/) const {
    if constexpr (is_arithmetic<T>) {
      auto dataset                = f.create_dataset<T>(name, size<Is>()...);
      dataset.attribute("vsMesh") = "/rectilinear_grid";
      dataset.attribute("vsCentering")  = "nodal";
      dataset.attribute("vsType")       = "variable";
      dataset.attribute("vsIndexOrder") = "compMinorF";
      auto data                         = std::vector<T>{};
      data.reserve(vertices().size());
      vertices().iterate_indices(
          [&](auto const... is) { data.push_back(prop(is...)); });
      dataset.write(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    } else if constexpr (static_vec<T>) {
      using vec_type       = T;
      auto              g  = f.group(name);
      auto              gg = g.sub_group(name);
      std::stringstream ss;
      ss << "{";
      ss << "<" + name + "/" << name << "_0>";
      for (std::size_t i = 1; i < vec_type::dimension(0); ++i) {
        ss << ",<" + name + "/" << name << "_" << i << ">";
      }
      ss << "}";
      gg.attribute(name)     = ss.str();
      gg.attribute("vsType") = "vsVars";

      for (std::size_t i = 0; i < vec_type::dimension(0); ++i) {
        auto dataset = g.create_dataset<typename vec_type::value_type>(
            name + "_" + std::to_string(i), size<Is>()...);
        dataset.attribute("vsMesh")       = "/rectilinear_grid";
        dataset.attribute("vsCentering")  = "nodal";
        dataset.attribute("vsType")       = "variable";
        dataset.attribute("vsIndexOrder") = "compMinorF";
        auto data = std::vector<typename vec_type::value_type>{};
        data.reserve(vertices().size());
        vertices().iterate_indices(
            [&](auto const... is) { data.push_back(prop(is...)(i)); });
        dataset.write(H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
      }
    }
  }
  //----------------------------------------------------------------------------
  template <typename... Ts, std::size_t... Is>
  auto write_prop_hdf5_wrapper(hdf5::file& f, std::string const& name,
                               vertex_property_type const& prop,
                               std::index_sequence<Is...>  seq) const -> void {
    (
        [&] {
          if (prop.type() == typeid(Ts)) {
            write_prop_hdf5(
                f, name,
                *dynamic_cast<
                    const typed_vertex_property_interface_type<Ts, true>*>(
                    &prop),
                seq);
            return;
          }
        }(),
        ...);
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  auto write_visitvs(filesystem::path const&    path,
                     std::index_sequence<Is...> seq) const -> void {
    if (filesystem::exists(path)) {
      filesystem::remove(path);
    }
    auto f     = hdf5::file{path};
    auto group = f.group("rectilinear_grid");

    std::stringstream axis_labels_stream;
    (
        [&] {
          if constexpr (Is == 0) {
            axis_labels_stream << cartesian_axis_label<Is>;
          } else {
            axis_labels_stream << ", " << cartesian_axis_label<Is>;
          }
        }(),
        ...);
    group.attribute("vsAxisLabels") = axis_labels_stream.str();
    group.attribute("vsKind")       = "rectilinear";
    group.attribute("vsType")       = "mesh";
    group.attribute("vsIndexOrder") = "compMinorF";
    (
        [&] {
          using dim_type =
              typename std::decay_t<decltype(dimension<Is>())>::value_type;
          group.attribute("vsAxis" + std::to_string(Is)) =
              "axis" + std::to_string(Is);
          auto dim = f.create_dataset<dim_type>(
              "rectilinear_grid/axis" + std::to_string(Is), size<Is>());
          auto dim_as_vec = std::vector<dim_type>{};
          dim_as_vec.reserve(dimension<Is>().size());
          std::ranges::copy(dimension<Is>(), std::back_inserter(dim_as_vec));
          dim.write(dim_as_vec);
        }(),
        ...);

    for (const auto& [name, prop] : this->m_vertex_properties) {
      write_prop_hdf5_wrapper<std::uint16_t, std::uint32_t, std::int16_t,
                              std::int32_t, float, double, vec2f, vec2d, vec2f,
                              vec2d, vec4f, vec4d, vec5f, vec5d>(f, name, *prop,
                                                                 seq);
    }
  }
#endif

 private:
  template <std::size_t I>
  auto print_dim(std::ostream& out) const {
    auto const& dim = dimension<I>();
    if constexpr (is_linspace<std::decay_t<decltype(dim)>>) {
      out << dim;
    } else {
      out << dim.front() << ", " << dim[1] << ", ..., " << dim.back();
    }
    out << " [" << dim.size() << "]\n";
  }
  template <std::size_t... Seq>
  auto print(std::ostream& out, std::index_sequence<Seq...> /*seq*/) const
      -> auto& {
    (print_dim<Seq>(out), ...);
    return out;
  }

 public:
  auto print(std::ostream& out = std::cout) const -> auto& {
    return print(out, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto chunk_size_for_lazy_properties() {
    return m_chunk_size_for_lazy_properties;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_chunk_size_for_lazy_properties(std::size_t const val) -> void {
    m_chunk_size_for_lazy_properties = val;
  }
};
//==============================================================================
// free functions
//==============================================================================
template <typename... Dimensions>
auto operator<<(std::ostream& out, rectilinear_grid<Dimensions...> const& g)
    -> auto& {
  return g.print(out);
}
template <detail::rectilinear_grid::dimension... Dimensions>
auto vertices(rectilinear_grid<Dimensions...> const& g) {
  return g.vertices();
}
//==============================================================================
// deduction guides
//==============================================================================
template <detail::rectilinear_grid::dimension... Dimensions>
rectilinear_grid(Dimensions&&...)
    -> rectilinear_grid<std::decay_t<Dimensions>...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
rectilinear_grid(axis_aligned_bounding_box<Real, N> const& bb,
                 integral auto const... res)
    -> rectilinear_grid<linspace<std::conditional_t<true, Real, decltype(res)>>...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <integral... Ints>
rectilinear_grid(Ints const... res) -> rectilinear_grid<
    linspace<std::conditional_t<true, double, decltype(res)>>...>;
//==============================================================================
// operators
//==============================================================================
template <detail::rectilinear_grid::dimension... Dimensions,
          detail::rectilinear_grid::dimension AdditionalDimension>
auto operator+(rectilinear_grid<Dimensions...> const& rectilinear_grid,
               AdditionalDimension&&                  additional_dimension) {
  return rectilinear_grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <detail::rectilinear_grid::dimension... Dimensions,
          detail::rectilinear_grid::dimension AdditionalDimension>
auto operator+(AdditionalDimension&&                  additional_dimension,
               rectilinear_grid<Dimensions...> const& rectilinear_grid) {
  return rectilinear_grid.add_dimension(
      std::forward<AdditionalDimension>(additional_dimension));
}
//==============================================================================
// typedefs
//==============================================================================
template <floating_point Real, std::size_t N>
using uniform_rectilinear_grid =
    detail::rectilinear_grid::creator_t<linspace<Real>, N>;
template <std::size_t N>
using UniformRectilinearGrid    = uniform_rectilinear_grid<real_number, N>;
using uniform_rectilinear_grid2 = UniformRectilinearGrid<2>;
using uniform_rectilinear_grid3 = UniformRectilinearGrid<3>;
using uniform_rectilinear_grid4 = UniformRectilinearGrid<4>;
using uniform_rectilinear_grid5 = UniformRectilinearGrid<5>;
template <floating_point Real>
using UniformRectilinearGrid2 = uniform_rectilinear_grid<Real, 2>;
template <floating_point Real>
using UniformRectilinearGrid3 = uniform_rectilinear_grid<Real, 3>;
template <floating_point Real>
using UniformRectilinearGrid4 = uniform_rectilinear_grid<Real, 4>;
template <floating_point Real>
using UniformRectilinearGrid5 = uniform_rectilinear_grid<Real, 5>;
//------------------------------------------------------------------------------
template <arithmetic Real, std::size_t N>
using nonuniform_rectilinear_grid =
    detail::rectilinear_grid::creator_t<std::vector<Real>, N>;
template <std::size_t N>
using NonuniformRectilinearGrid = nonuniform_rectilinear_grid<real_number, N>;
using nonuniform_rectilinear_grid2 = NonuniformRectilinearGrid<2>;
using nonuniform_rectilinear_grid3 = NonuniformRectilinearGrid<3>;
using nonuniform_rectilinear_grid4 = NonuniformRectilinearGrid<4>;
using nonuniform_rectilinear_grid5 = NonuniformRectilinearGrid<5>;
//------------------------------------------------------------------------------
template <arithmetic Real, std::size_t... N>
using static_nonuniform_rectilinear_grid =
    rectilinear_grid<std::array<Real, N>...>;
template <std::size_t N>
using StaticNonUniformGrid = static_nonuniform_rectilinear_grid<real_number, N>;
using static_nonuniform_rectilinear_grid2 = NonuniformRectilinearGrid<2>;
using static_nonuniform_rectilinear_grid3 = NonuniformRectilinearGrid<3>;
using static_nonuniform_rectilinear_grid4 = NonuniformRectilinearGrid<4>;
using static_nonuniform_rectilinear_grid5 = NonuniformRectilinearGrid<5>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/detail/rectilinear_grid/infinite_vertex_property_sampler.h>
#endif
