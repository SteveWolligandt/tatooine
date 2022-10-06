#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
#include <tatooine/nan.h>

#include <vector>
//==============================================================================
namespace tatooine::polymorphic {
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename Tensor>
struct field {
  //============================================================================
  // typedefs
  //============================================================================
  using real_type   = Real;
  using tensor_type = Tensor;
  using this_type   = field<real_type, NumDimensions, Tensor>;
  using pos_type    = vec<real_type, NumDimensions>;
  static auto constexpr ood_tensor() {
    if constexpr (is_arithmetic<tensor_type>) {
      return nan<Real>();
    } else {
      return tensor_type::fill(nan<Real>());
    }
  }
  static auto constexpr ood_position() {
    return pos_type::fill(nan<Real>());
  }
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto is_field() { return true; }
  static constexpr auto is_scalarfield() { return is_arithmetic<Tensor>; }
  static constexpr auto is_vectorfield() { return tensor_rank() == 1; }
  static constexpr auto is_matrixfield() { return tensor_rank() == 2; }
  static constexpr auto num_dimensions() -> std::size_t { return NumDimensions; }
  //----------------------------------------------------------------------------
  static constexpr auto num_tensor_components() {
    if constexpr (is_scalarfield()) {
      return 1;
    } else {
      return tensor_type::num_components();
    }
  }
  //----------------------------------------------------------------------------
  static constexpr auto tensor_rank() {
    if constexpr (is_scalarfield()) {
      return 0;
    } else {
      return tensor_type::rank();
    }
  }
  //----------------------------------------------------------------------------
  static constexpr auto tensor_dimension(std::size_t i)
  requires(tensor_rank() > 0) {
    return tensor_type::dimension(i);
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
  [[nodiscard]] constexpr virtual auto evaluate(pos_type const&,
                                                real_type const) const
      -> tensor_type = 0;
  //============================================================================
  // methods
  //============================================================================
  constexpr auto evaluate(fixed_size_vec<NumDimensions> auto const& x) const
      -> tensor_type {
    return evaluate(x, 0);
  }
  constexpr auto operator()(fixed_size_vec<NumDimensions> auto const& x,
                            real_type const t) const -> tensor_type {
    return evaluate(x, t);
  }
  constexpr auto operator()(fixed_size_vec<NumDimensions> auto const& x) const
      -> tensor_type {
    return evaluate(x, 0);
  }
  constexpr auto operator()(arithmetic auto const... xs) const -> tensor_type 
  requires (sizeof...(xs) == NumDimensions) ||
           (sizeof...(xs) == NumDimensions + 1) {
    if constexpr (sizeof...(xs) == NumDimensions) {
      return evaluate(pos_type{xs...});
    } else if constexpr (sizeof...(xs) == NumDimensions + 1) {
      auto const data = std::array{static_cast<real_type>(xs)...};
      auto       x    = pos_type{};
      for (std::size_t i = 0; i < NumDimensions; ++i) {
        x(i) = data[i];
      }
      return evaluate(x, data.back());
    }
  }
};  // field
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          std::size_t R = NumDimensions, std::size_t C = NumDimensions>
using matrixfield = field<Real, NumDimensions, mat<Real, R, C>>;
template <typename Real, std::size_t NumDimensions,
          std::size_t C = NumDimensions>
using vectorfield = field<Real, NumDimensions, vec<Real, C>>;
template <typename Real, std::size_t NumDimensions>
using scalarfield = field<Real, NumDimensions, Real>;
//==============================================================================
}  // namespace tatooine::polymorphic
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename Tensor>
using field_list = std::vector<
    std::unique_ptr<polymorphic::field<Real, NumDimensions, Tensor>>>;
template <typename Real, std::size_t NumDimensions,
          std::size_t C = NumDimensions>
using vectorfield_list = field_list<Real, NumDimensions, vec<Real, C>>;
//==============================================================================
template <typename DerivedField, typename Real, std::size_t NumDimensions,
          typename Tensor>
struct field : polymorphic::field<Real, NumDimensions, Tensor> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_type   = field<DerivedField, Real, NumDimensions, Tensor>;
  using parent_type = polymorphic::field<Real, NumDimensions, Tensor>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
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
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type override {
    return as_derived().evaluate(x, t);
  }
};
//==============================================================================
template <typename V, typename Real, std::size_t NumDimensions,
          std::size_t R = NumDimensions, std::size_t C = NumDimensions>
using matrixfield = field<V, Real, NumDimensions, mat<Real, R, C>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t NumDimensions,
          std::size_t C = NumDimensions>
using vectorfield = field<V, Real, NumDimensions, vec<Real, NumDimensions>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t NumDimensions>
using scalarfield = field<V, Real, NumDimensions, Real>;
//==============================================================================
template <std::size_t NumDimensions, typename F>
struct lambda_field
    : field<lambda_field<NumDimensions, F>,
            tensor_value_type<std::invoke_result_t<
                F, vec<tatooine::real_number, NumDimensions>,
                tatooine::real_number>>,
            NumDimensions,
            std::invoke_result_t<F, vec<tatooine::real_number, NumDimensions>,
                                 tatooine::real_number>> {
  F m_f;
  constexpr lambda_field(F&& f) : m_f{std::forward<F>(f)} {}
  ~lambda_field()   = default;
  using parent_type = field<
      lambda_field<NumDimensions, F>,
      tensor_value_type<std::invoke_result_t<
          F, vec<tatooine::real_number, NumDimensions>, tatooine::real_number>>,
      NumDimensions,
      std::invoke_result_t<F, vec<tatooine::real_number, NumDimensions>,
                           tatooine::real_number>>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  [[nodiscard]] constexpr auto evaluate(pos_type const& x,
                                        real_type const t) const
      -> tensor_type {
    return m_f(x, t);
  }
};
//==============================================================================
template <std::size_t NumDimensions, typename F>
constexpr auto make_field(F&& f) {
  return lambda_field<NumDimensions, std::decay_t<F>>{std::forward<F>(f)};
}
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
static constexpr bool is_field = is_field_impl<T>::value;
//==============================================================================
template <typename T, typename = void>
struct is_scalarfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_scalarfield_impl<T>
    : std::integral_constant<bool, T::is_scalarfield()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_scalarfield = is_scalarfield_impl<T>::value;
//==============================================================================
template <typename T, typename = void>
struct is_vectorfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_vectorfield = is_vectorfield_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_vectorfield_impl<T>
    : std::integral_constant<bool, T::is_vectorfield()> {};
//==============================================================================
template <typename T, typename = void>
struct is_matrixfield_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct is_matrixfield_impl<T>
    : std::integral_constant<bool, T::is_matrixfield()> {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_matrixfield = is_matrixfield_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/field_type_traits.h>
//==============================================================================
#endif
