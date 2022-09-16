#ifndef TATOOINE_DIFFERENTIATED_FLOWMAP_H
#define TATOOINE_DIFFERENTIATED_FLOWMAP_H
//==============================================================================
#include <tatooine/flowmap_concept.h>
#include <tatooine/tags.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Default differentiated flowmap uses central differences for differentiating.
template <flowmap_concept Flowmap>
struct numerically_differentiated_flowmap {
  //============================================================================
 public:
  using flowmap_type = std::decay_t<Flowmap>;
  using this_type    = numerically_differentiated_flowmap<flowmap_type>;
  using real_type    = typename flowmap_type::real_type;
  static constexpr auto num_dimensions() -> std::size_t {
    return flowmap_type::num_dimensions();
  }
  using vec_type      = typename flowmap_type::vec_type;
  using pos_type      = typename flowmap_type::pos_type;
  using mat_type      = mat<real_type, num_dimensions(), num_dimensions()>;
  using value_type = mat_type;
  static constexpr auto default_epsilon = 1e-6;
  //============================================================================
 private:
  flowmap_type m_flowmap;
  vec_type     m_epsilon;
  //============================================================================
 public:
  explicit numerically_differentiated_flowmap(
      convertible_to<Flowmap> auto&& flowmap,
      real_type const                epsilon = default_epsilon)
      : m_flowmap{std::forward<decltype(flowmap)>(flowmap)},
        m_epsilon{tag::fill{epsilon}} {}
  //----------------------------------------------------------------------------
  numerically_differentiated_flowmap(convertible_to<Flowmap> auto&& flowmap,
                                     vec_type const&                epsilon)
      : m_flowmap{std::forward<decltype(flowmap)>(flowmap)},
        m_epsilon{epsilon} {}
  //============================================================================
  auto evaluate(pos_type const& y0, real_type const t0,
                real_type const tau) const {
    auto derivative = value_type{};

    auto offset = pos_type::zeros();
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      offset(i)     = m_epsilon(i);
      auto const dx     = 1 / (2 * m_epsilon(i));
      derivative.col(i) = m_flowmap(y0 + offset, t0, tau) * dx -
                          m_flowmap(y0 - offset, t0, tau) * dx;
      offset(i) = 0;
    }
    return derivative;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(pos_type const& y0, real_type const t0,
                  real_type const tau) const {
    return evaluate(y0, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto epsilon() const -> auto const& { return m_epsilon; }
  auto epsilon() -> auto& { return m_epsilon; }
  auto epsilon(std::size_t i) const { return m_epsilon(i); }
  auto epsilon(std::size_t i) -> auto& { return m_epsilon(i); }
  auto set_epsilon(vec_type const& epsilon) { m_epsilon = epsilon; }
  auto set_epsilon(vec_type&& epsilon) { m_epsilon = std::move(epsilon); }
  auto flowmap() const -> auto const& { return m_flowmap; }
  auto flowmap() -> auto& { return m_flowmap; }
};
//==============================================================================
// copy when having rvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap&&)
    -> numerically_differentiated_flowmap<Flowmap>;
//------------------------------------------------------------------------------
// copy when having rvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap&&, arithmetic auto)
    -> numerically_differentiated_flowmap<Flowmap>;
//------------------------------------------------------------------------------
// copy when having rvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap&&, typename Flowmap::pos_type const&)
    -> numerically_differentiated_flowmap<Flowmap>;
//------------------------------------------------------------------------------
// keep reference when having lvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap const&)
    -> numerically_differentiated_flowmap<Flowmap const&>;
//------------------------------------------------------------------------------
// keep reference when having lvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap const&, arithmetic auto)
    -> numerically_differentiated_flowmap<Flowmap>;
//==============================================================================
// keep reference when having lvalue
template <flowmap_concept Flowmap>
numerically_differentiated_flowmap(Flowmap const&,
                                   typename Flowmap::pos_type const&)
    -> numerically_differentiated_flowmap<Flowmap>;
//==============================================================================
auto diff(flowmap_concept auto&& flowmap, tag::numerical_t /*tag*/) {
  return numerically_differentiated_flowmap{
      std::forward<decltype(flowmap)>(flowmap)};
}
//------------------------------------------------------------------------------
auto diff(flowmap_concept auto&& flowmap, tag::numerical_t /*tag*/,
          arithmetic auto const  epsilon) {
  return numerically_differentiated_flowmap{
      std::forward<decltype(flowmap)>(flowmap), epsilon};
}
//------------------------------------------------------------------------------
auto diff(
    flowmap_concept auto&& flowmap, tag::numerical_t /*tag*/,
    fixed_size_real_vec<
        std::decay_t<decltype(flowmap)>::num_dimensions()> auto&& epsilon) {
  return numerically_differentiated_flowmap{
      std::forward<decltype(flowmap)>(flowmap),
      std::forward<decltype(epsilon)>(epsilon)};
}
//------------------------------------------------------------------------------
auto diff(flowmap_concept auto&& flowmap, arithmetic auto const epsilon) {
  return diff(std::forward<decltype(flowmap)>(flowmap), tag::numerical,
              epsilon);
}
//------------------------------------------------------------------------------
auto diff(
    flowmap_concept auto&& flowmap,
    fixed_size_real_vec<
        std::decay_t<decltype(flowmap)>::num_dimensions()> auto&& epsilon) {
  return numerically_differentiated_flowmap{
      std::forward<decltype(flowmap)>(flowmap),
      std::forward<decltype(epsilon)>(epsilon)};
}
//==============================================================================
/// Default differentiated flowmap uses numerical differences for
/// differentiating.
template <flowmap_concept Flowmap>
struct differentiated_flowmap : numerically_differentiated_flowmap<Flowmap> {
  using parent_type = numerically_differentiated_flowmap<Flowmap>;
  using parent_type::parent_type;
};
//==============================================================================
// copy when having rvalue
template <typename Flowmap>
differentiated_flowmap(Flowmap&&) -> differentiated_flowmap<Flowmap>;
//------------------------------------------------------------------------------
// keep reference when having lvalue
template <typename Flowmap>
differentiated_flowmap(Flowmap const&)
    -> differentiated_flowmap<Flowmap const&>;
//==============================================================================
auto diff(flowmap_concept auto&& flowmap) {
  return differentiated_flowmap{std::forward<decltype(flowmap)>(flowmap)};
}
//==============================================================================
template <typename T>
struct is_differentiated_flowmap_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename Flowmap>
struct is_differentiated_flowmap_impl<differentiated_flowmap<Flowmap>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename Flowmap>
struct is_differentiated_flowmap_impl<
    numerically_differentiated_flowmap<Flowmap>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_differentiated_flowmap =
    is_differentiated_flowmap_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
