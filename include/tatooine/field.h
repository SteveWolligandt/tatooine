#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include <vector>
#include "crtp.h"
#include "grid.h"
#include "tensor.h"
#include "type_traits.h"
#include "tensor_type.h"
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine::parent {
//╒══════════════════════════════════════════════════════════════════════════╕
template <typename Real, size_t N, size_t... TensorDims>
struct field  {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ typedefs                                                             │
  //├──────────────────────────────────────────────────────────────────────┤
  using real_t   = Real;
  using this_t   = field<Real, N, TensorDims...>;
  using pos_t    = vec<Real, N>;
  using tensor_t = std::conditional_t<sizeof...(TensorDims) == 0, Real,
                                      tensor_type<Real, TensorDims...>>;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ static methods                                                       │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  static constexpr auto num_dimensions() { return N; }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  static constexpr auto num_tensor_dimensions() {
    return sizeof...(TensorDims);
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  template <size_t _num_tensor_dims = sizeof...(TensorDims),
            std::enable_if_t<(_num_tensor_dims > 0)>...>
  static constexpr auto tensor_dimension(size_t i) {
    return tensor_t::dimension(i);
  }
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
  field() = default;
  field(const field&) = default;
  field(field&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator=(const field&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  virtual ~field() = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ virtual methods                                                      │
  //├──────────────────────────────────────────────────────────────────────┤
  [[nodiscard]] virtual auto evaluate(const pos_t& x, Real t = 0) const
      -> tensor_t                                                        = 0;
  [[nodiscard]] virtual auto in_domain(const pos_t&, Real) const -> bool = 0;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator()(const pos_t& x, Real t) const -> tensor_t {
    return evaluate(x, t);
  }
}; // field
//╘══════════════════════════════════════════════════════════════════════════╛
}  // namespace parent
//╚════════════════════════════════════════════════════════════════════════════╝
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {
template <typename Real, size_t N, size_t... TensorDims>
using field_list =
    std::vector<std::unique_ptr<parent::field<Real, N, TensorDims...>>>;
template <typename Real, size_t N, size_t D = N>
using vectorfield_list = field_list<Real, N, D>;
template <typename Derived, typename Real, size_t N, size_t... TensorDims>
//╒══════════════════════════════════════════════════════════════════════════╕
struct field : parent::field<Real, N, TensorDims...>, crtp<Derived> {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ typedefs                                                             │
  //├──────────────────────────────────────────────────────────────────────┤
  using this_t   = field<Derived, Real, N, TensorDims...>;
  using parent_crtp_t = crtp<Derived>;
  using parent_t      = parent::field<Real, N, TensorDims...>;
  using pos_t         = typename parent_t::pos_t;
  using tensor_t      = typename parent_t::tensor_t;
  using parent_crtp_t::as_derived;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
  field() = default;
  field(const field&) = default;
  field(field&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator=(const field&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  ~field() override = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  [[nodiscard]] auto evaluate(const pos_t& x, Real t) const
      -> tensor_t override {
    return as_derived().evaluate(x, t);
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  [[nodiscard]] auto in_domain(const pos_t& x, Real t) const -> bool override {
    return as_derived().in_domain(x, t);
  }
}; // field
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename V, typename Real, size_t N, size_t C = N>
using vectorfield = field<V, Real, N, C>;
//╘══════════════════════════════════════════════════════════════════════════╛
//╒══════════════════════════════════════════════════════════════════════════╕
//│ type traits                                                              │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename T>
struct is_field : std::false_type{};
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename T>
static constexpr bool is_field_v = is_field<T>::value;
template <typename Real, size_t N, size_t... TensorDims>
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
struct is_field<parent::field<Real, N, TensorDims...>> : std::true_type {};
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename Derived, typename Real, size_t N, size_t... TensorDims>
struct is_field<field<Derived, Real, N, TensorDims...>> : std::true_type {};
//╒══════════════════════════════════════════════════════════════════════════╕
//│ free functions                                                           │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename OutReal, typename V, typename FieldReal, typename GridReal,
          typename TReal, size_t N, size_t... TensorDims,
          enable_if_arithmetic<FieldReal, GridReal, TReal> = true>
auto sample_to_raw(const field<V, FieldReal, N, TensorDims...>& f,
                   const grid<GridReal, N>& g, TReal t, size_t padding = 0,
                   OutReal padval = 0) {
  std::vector<OutReal> raw_data;
  raw_data.reserve(g.num_vertices() * V::tensor_t::num_components());
  for (auto v : g.vertices()) {
    const auto x = v.position();
    if (f.in_domain(x, t)) {
      auto sample = f(x, t);
      for (size_t i = 0; i < V::tensor_t::num_components(); ++i) {
        raw_data.push_back(sample[i]);
      }
      for (size_t i = 0; i < padding; ++i) { raw_data.push_back(padval); }
    } else {
      for (size_t i = 0; i < V::tensor_t::num_components(); ++i) {
        raw_data.push_back(0.0 / 0.0);
      }
      for (size_t i = 0; i < padding; ++i) { raw_data.push_back(0.0 / 0.0); }
    }
  }
  return raw_data;
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename OutReal, typename V, typename FieldReal,
          typename GridReal, typename TReal, size_t N, size_t... TensorDims>
auto sample_to_raw(const field<V, FieldReal, N, TensorDims...>& f,
                   const grid<GridReal, N>& g, const linspace<TReal>& ts,
                   size_t padding = 0, OutReal padval = 0) {
  std::vector<OutReal> raw_data;
  raw_data.reserve(g.num_vertices() * V::tensor_t::num_components() *
                   ts.size());
  for (auto t : ts) {
    for (auto v : g.vertices()) {
      auto sample = f(v.position(), t);
      for (size_t i = 0; i < V::tensor_t::num_components(); ++i) {
        raw_data.push_back(sample[i]);
      }
      for (size_t i = 0; i < padding; ++i) { raw_data.push_back(padval); }
    }
  }
  return raw_data;
}
}  // namespace tatooine
//╚════════════════════════════════════════════════════════════════════════════╝
#include "field_operations.h"
#endif
