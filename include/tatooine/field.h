#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/tensor_type.h>
#include <tatooine/type_traits.h>

#include <vector>
//==============================================================================
namespace tatooine::polymorphic {
//==============================================================================
template <typename Real, size_t NumDims, typename Tensor>
struct field {
  //============================================================================
  // typedefs
  //============================================================================
  using real_t   = Real;
  using tensor_t = Tensor;
  using this_t   = field<real_t, NumDims, Tensor>;
  using pos_t    = vec<real_t, NumDims>;
  static auto constexpr ood_tensor() { return tensor_t{tag::fill{0.0 / 0.0}}; }
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto is_field() { return true; }
  static constexpr auto is_scalarfield() { return is_arithmetic<Tensor>; }
  static constexpr auto is_vectorfield() { return tensor_rank() == 1; }
  static constexpr auto is_matrixfield() { return tensor_rank() == 2; }
  static constexpr auto num_dimensions() { return NumDims; }
  //----------------------------------------------------------------------------
  static constexpr auto num_tensor_components() {
    if constexpr (is_scalarfield()) {
      return 1;
    } else {
      return tensor_t::num_components();
    }
  }
  //----------------------------------------------------------------------------
  static constexpr auto tensor_rank() {
    if constexpr (is_scalarfield()) {
      return 0;
    } else {
      return tensor_t::rank();
    }
  }
  //----------------------------------------------------------------------------
  template <size_t tensor_rank           = tensor_rank,
            enable_if<(tensor_rank > 0)> = true>
  static constexpr auto tensor_dimension(size_t i) {
    return tensor_t::dimension(i);
  }
  //============================================================================
  // ctors
  //============================================================================
  constexpr field()                 = default;
  constexpr field(field const&)     = default;
  constexpr field(field&&) noexcept = default;
  //============================================================================
  // assign ops
  //============================================================================
  constexpr auto operator=(field const&) -> field& = default;
  constexpr auto operator=(field&&) noexcept -> field& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~field() = default;
  //============================================================================
  // virtual methods
  //============================================================================
#ifdef __cpp_concepts
  [[nodiscard]] constexpr virtual auto evaluate(pos_t const&,
                                                real_t const) const
      -> tensor_t = 0;
  //[[nodiscard]] constexpr virtual auto in_domain(pos_t const&,
  //                                               real_t const) const
  //    -> bool = 0;
#else
  [[nodiscard]] virtual auto evaluate(pos_t const&, real_t const) const
      -> tensor_t = 0;
  //[[nodiscard]] virtual auto in_domain(pos_t const&, real_t const) const
  //    -> bool = 0;
#endif
  //============================================================================
  // methods
  //============================================================================
  constexpr auto evaluate(pos_t const& x) const -> tensor_t {
    return evaluate(x, 0);
  }
  constexpr auto operator()(pos_t const& x, real_t const t) const -> tensor_t {
    return evaluate(x, t);
  }
  constexpr auto operator()(pos_t const& x) const -> tensor_t {
    return evaluate(x, 0);
  }
  template <typename... Xs, enable_if_arithmetic<Xs...> = true>
  constexpr auto operator()(Xs const... xs) const -> tensor_t {
    static_assert(sizeof...(Xs) == NumDims || sizeof...(Xs) == NumDims + 1);
    if constexpr (sizeof...(Xs) == NumDims) {
      return evaluate(pos_t{xs...});
    } else if constexpr (sizeof...(Xs) == NumDims + 1) {
      auto const data = std::array{static_cast<real_t>(xs)...};
      pos_t      x;
      for (size_t i = 0; i < NumDims; ++i) {
        x(i) = data[i];
      }
      return evaluate(x, data.back());
    }
  }
};  // field
template <typename Real, size_t NumDims, size_t R = NumDims, size_t C = NumDims>
using matrixfield = field<Real, NumDims, mat<Real, R, C>>;
template <typename Real, size_t NumDims, size_t C = NumDims>
using vectorfield = field<Real, NumDims, vec<Real, C>>;
template <typename Real, size_t NumDims>
using scalarfield = field<Real, NumDims, Real>;
//==============================================================================
}  // namespace tatooine::polymorphic
//==============================================================================

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t NumDims, typename Tensor>
using field_list =
    std::vector<std::unique_ptr<polymorphic::field<Real, NumDims, Tensor>>>;
template <typename Real, size_t NumDims, size_t C = NumDims>
using vectorfield_list = field_list<Real, NumDims, vec<Real, C>>;
//==============================================================================
template <typename DerivedField, typename Real, size_t NumDims, typename Tensor>
struct field : polymorphic::field<Real, NumDims, Tensor> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t   = field<DerivedField, Real, NumDims, Tensor>;
  using parent_t = polymorphic::field<Real, NumDims, Tensor>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  //============================================================================
  // ctors
  //============================================================================
  field()                 = default;
  field(field const&)     = default;
  field(field&&) noexcept = default;
  //============================================================================
  // assign ops
  //============================================================================
  auto operator=(field const&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~field() = default;
  //============================================================================
  // methods
  //============================================================================
  auto as_derived() -> auto& { return static_cast<DerivedField&>(*this); }
  auto as_derived() const -> auto const& {
    return static_cast<DerivedField const&>(*this);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t override {
    return as_derived().evaluate(x, t);
  }
  //----------------------------------------------------------------------------
  //[[nodiscard]] auto in_domain(pos_t const& x, real_t const t) const
  //    -> bool override {
  //  return as_derived().in_domain(x, t);
  //}
};
//==============================================================================
template <typename V, typename Real, size_t NumDims, size_t R = NumDims,
          size_t C = NumDims>
using matrixfield = field<V, Real, NumDims, mat<Real, R, C>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t NumDims, size_t C = NumDims>
using vectorfield = field<V, Real, NumDims, vec<Real, NumDims>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t NumDims>
using scalarfield = field<V, Real, NumDims, Real>;
//==============================================================================
// type traits
//==============================================================================
template <typename T, typename = void>
struct is_field_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_field_impl<T> : std::integral_constant<bool, T::is_field()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_field_v = is_field_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_field(T&&) {
  return is_field_v<std::decay_t<T>>;
}
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_field() {
  return is_field_v<T>;
}
//==============================================================================
template <typename T, typename = void>
struct is_scalarfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_scalarfield_impl<T>
    : std::integral_constant<bool, T::is_scalarfield()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_scalarfield_v = is_scalarfield_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_scalarfield(T&&) {
  return is_scalarfield_v<std::decay_t<T>>;
}
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_scalarfield() {
  return is_scalarfield_v<T>;
}
//==============================================================================
template <typename T, typename = void>
struct is_vectorfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_vectorfield_v = is_vectorfield_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_vectorfield_impl<T>
    : std::integral_constant<bool, T::is_vectorfield()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_vectorfield(T&&) {
  return is_vectorfield_v<std::decay_t<T>>;
}
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_vectorfield() {
  return is_vectorfield_v<T>;
}
//==============================================================================
template <typename T, typename = void>
struct is_matrixfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_matrixfield_impl<T>
    : std::integral_constant<bool, T::is_matrixfield()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_matrixfield_v = is_matrixfield_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_matrixfield(T&&) {
  return is_matrixfield_v<std::decay_t<T>>;
}
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_matrixfield() {
  return is_matrixfield_v<T>;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
// free functions
//==============================================================================
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concpets
template <typename V, arithmetic VReal, size_t NumDims, typename Tensor,
          indexable_space... SpatialDimensions, arithmetc T>
#else
template <typename V, typename VReal, size_t NumDims, typename Tensor,
          typename... SpatialDimensions, typename T,
          enable_if_arithmetic<VReal, T> = true>
#endif
auto sample_to_raw(
    field<V, VReal, NumDims, Tensor> const&       f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain, T const t,
    size_t padding = 0, VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  auto               vs = discretized_domain.vertices();
  raw_data.reserve(vs.size() * (f.num_tensor_components() + padding));
  for (auto v : vs) {
    auto const x = vs[v];
    //if (f.in_domain(x, t)) {
      auto sample = f(x, t);
      if constexpr (f.is_scalarfield()) {
        raw_data.push_back(static_cast<VReal>(sample));
      } else {
        for (size_t i = 0; i < f.num_tensor_components(); ++i) {
          raw_data.push_back(static_cast<VReal>(sample[i]));
        }
      }
      for (size_t i = 0; i < padding; ++i) {
        raw_data.push_back(padval);
      }
    //} else {
    //  for (size_t i = 0; i < f.num_tensor_components() + padding; ++i) {
    //    raw_data.push_back(nan);
    //  }
    //}
  }
  return raw_data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <typename V, arithmetic VReal, arithmetic TReal, size_t NumDims,
          typename Tensor, indexable_space... SpatialDimensions,
          indexable_space TemporalDimension>
#else
template <typename V, typename VReal, typename TReal, size_t NumDims,
          typename Tensor, typename... SpatialDimensions,
          typename TemporalDimension, enable_if_arithmetic<VReal, TReal> = true>
#endif
auto sample_to_raw(
    field<V, VReal, NumDims, Tensor> const&       f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain,
    TemporalDimension const& temporal_domain, size_t padding = 0,
    VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  auto               vs = discretized_domain.vertices();
  raw_data.reserve(vs.size() * temporal_domain.size() *
                   (f.num_tensor_components() + padding));
  for (auto t : temporal_domain) {
    for (auto v : vs) {
      auto const x = v.position();
      //if (f.in_domain(x, t)) {
        auto sample = f(x, t);
        if constexpr (f.is_scalarfield()) {
          raw_data.push_back(static_cast<VReal>(sample));
        } else {
          for (size_t i = 0; i < f.num_tensor_components(); ++i) {
            raw_data.push_back(static_cast<VReal>(sample[i]));
          }
        }
        for (size_t i = 0; i < padding; ++i) {
          raw_data.push_back(padval);
        }
      //} else {
      //  for (size_t i = 0; i < f.num_tensor_components() + padding; ++i) {
      //    raw_data.push_back(nan);
      //  }
      //}
    }
  }
  return raw_data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <arithmetic VReal, arithmetic TReal, size_t NumDims, typename Tensor,
          indexable_space... SpatialDimensions>
#else
template <typename VReal, typename TReal, size_t NumDims, typename Tensor,
          typename... SpatialDimensions,
          enable_if_arithmetic<VReal, TReal> = true>
#endif
auto sample_to_vector(
    polymorphic::field<VReal, NumDims, Tensor> const& f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain, TReal t) {
  using V                   = polymorphic::field<VReal, NumDims, Tensor>;
  using tensor_t            = typename V::tensor_t;
  auto const            nan = VReal(0) / VReal(0);
  std::vector<tensor_t> data;
  auto                  vs = discretized_domain.vertices();
  data.reserve(vs.size());
  for (auto v : vs) {
    auto const x = vs[v];
    data.push_back(f(x, t));
  }
  return data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <arithmetic VReal, arithmetic TReal, size_t NumDims, typename Tensor,
          indexable_space... SpatialDimensions, typename ExecutionPolicy>
#else
template <typename VReal, typename TReal, size_t NumDims, typename Tensor,
          typename... SpatialDimensions, typename ExecutionPolicy,
          enable_if_arithmetic<VReal, TReal> = true>
#endif
auto discretize(polymorphic::field<VReal, NumDims, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name, TReal const t, ExecutionPolicy const execution_policy)
    -> auto& {
  auto& discretized_field = [&]() -> decltype(auto) {
    if constexpr (is_arithmetic<Tensor>) {
      return discretized_domain.template insert_chunked_vertex_property<VReal>(
          property_name);
    } else if constexpr (is_vec<Tensor>) {
      return discretized_domain
          .template vertex_property<vec<VReal, Tensor::dimension(0)>>(
              property_name);
    } else if constexpr (is_mat<Tensor>) {
      return discretized_domain.template vertex_property<
          mat<VReal, Tensor::dimension(0), Tensor::dimension(1)>>(
          property_name);
    } else {
      return discretized_domain.template vertex_property<Tensor>(property_name);
    }
  }();
  discretized_domain.vertices().iterate_indices(
      [&](auto const... is) {
        auto const x = discretized_domain.vertex_at(is...);
        discretized_field(is...) = f(x, t);
      },
      execution_policy);
  return discretized_field;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concpets
template <arithmetic VReal, arithmetic TReal, size_t NumDims, typename Tensor,
          indexable_space... SpatialDimensions>
#else
template <typename VReal, typename TReal, size_t NumDims, typename Tensor,
          typename... SpatialDimensions,
          enable_if_arithmetic<VReal, TReal> = true>
#endif
auto discretize(polymorphic::field<VReal, NumDims, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name, TReal const t)
  -> auto& {
  return discretize(f, discretized_domain, property_name, t, tag::sequential);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concpets
template <arithmetic VReal, size_t NumDims, typename Tensor,
          indexable_space... SpatialDimensions>
#else
template <typename VReal, size_t NumDims, typename Tensor,
          typename... SpatialDimensions, enable_if_arithmetic<VReal> = true>
#endif
auto discretize(polymorphic::field<VReal, NumDims, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name) -> auto& {
  return discretize(f, discretized_domain, property_name, 0, tag::sequential);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concpets
template <arithmetic VReal, size_t NumDims, typename Tensor, typename ExecutionPolicy,
          indexable_space... SpatialDimensions>
#else
template <typename VReal, size_t NumDims, typename Tensor, typename ExecutionPolicy,
          typename... SpatialDimensions, enable_if_arithmetic<VReal> = true>
#endif
auto discretize(polymorphic::field<VReal, NumDims, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name, ExecutionPolicy const execution_policy) -> auto& {
  return discretize(f, discretized_domain, property_name, 0, execution_policy);
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, typename VReal, typename TReal, size_t NumDims,
          typename Tensor, typename BasisReal, typename X0Real, typename X1Real,
          enable_if_arithmetic<VReal, TReal> = true>
auto discretize(field<V, VReal, NumDims, Tensor> const& f,
                vec<X0Real, NumDims> const&             x0,
                mat<BasisReal, NumDims, 2> const&       basis,
                vec<X1Real, 2> const& spatial_size, size_t const res0,
                size_t const res1, std::string const& property_name,
                TReal const t) {
  auto const cell_extent =
      vec<VReal, 2>{spatial_size(0) / (res0 - 1), spatial_size(1) / (res1 - 1)};
  std::cerr << x0 << '\n';
  std::cerr << spatial_size << '\n';
  std::cerr << cell_extent << '\n';
  auto discretized_domain = rectilinear_grid{
      linspace<VReal>{0, length(basis * vec{spatial_size(0), 0}), res0},
      linspace<VReal>{0, length(basis * vec{0, spatial_size(1)}), res1}};

  auto& discretized_field = [&]() -> decltype(auto) {
    if constexpr (is_scalarfield<V>()) {
      return discretized_domain.template insert_chunked_vertex_property<VReal>(
          property_name);
    } else if constexpr (is_vectorfield<V>()) {
      return discretized_domain
          .template vertex_property<vec<VReal, V::tensor_t::dimension(0)>>(
              property_name);
    } else if constexpr (is_matrixfield<V>()) {
      return discretized_domain.template vertex_property<
          mat<VReal, V::tensor_t::dimension(0), V::tensor_t::dimension(1)>>(
          property_name);
    } else {
      return discretized_domain.template vertex_property<Tensor>(property_name);
    }
  }();
  for (size_t i1 = 0; i1 < res1; ++i1) {
    for (size_t i0 = 0; i0 < res0; ++i0) {
      auto const x = x0 + basis * vec{cell_extent(0) * i0, cell_extent(1) * i1};
      discretized_field(i0, i1) = f(x, t);
    }
  }
  return discretized_domain;
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, typename VReal, typename TReal, size_t NumDims,
          typename Tensor, typename BasisReal, typename X0Real,
          enable_if_arithmetic<VReal, TReal> = true>
auto discretize(field<V, VReal, NumDims, Tensor> const& f,
                mat<BasisReal, NumDims, 2> const&       basis,
                vec<X0Real, NumDims> const& x0, size_t const res0,
                size_t const res1, std::string const& property_name,
                TReal const t) {
  return discretize(f, x0, basis, vec<BasisReal, 2>::ones(), res0, res1,
                    property_name, t);
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, typename VReal, typename TReal, size_t NumDims,
          typename Tensor, typename BasisReal, typename X0Real,
          enable_if_arithmetic<VReal, TReal> = true>
auto discretize(field<V, VReal, NumDims, Tensor> const& f,
                vec<BasisReal, NumDims> const&          extent0,
                vec<BasisReal, NumDims> const&          extent1,
                vec<X0Real, NumDims> const& /*x0*/, size_t const res0,
                size_t const res1, std::string const& property_name,
                TReal const t) {
  auto basis   = mat<BasisReal, NumDims, 2>{};
  basis.col(0) = extent0;
  basis.col(1) = extent1;
  return discretize(f, basis, res0, res1, property_name, t);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/field_operations.h>
#include <tatooine/field_type_traits.h>
#endif
