#ifndef TATOOINE_CONCPETS_H
#define TATOOINE_CONCPETS_H
//==============================================================================
#include <concepts>
#include <tatooine/type_traits.h>
#include <tatooine/invocable_with_n_types.h>
//==============================================================================
namespace tatooine {
//==============================================================================
// ranges etc.
//==============================================================================
template <typename T>
concept forward_iterator = std::forward_iterator<T>;
//------------------------------------------------------------------------------
template <typename T>
concept bidirectional_iterator = std::bidirectional_iterator<T>;
//------------------------------------------------------------------------------
template <typename Range>
concept range = requires(Range const r) {
  {r.begin()};
  {r.end()};
  {begin(r)};
  {end(r)};
};

//==============================================================================
// typedefs
//==============================================================================
template <typename T>
concept integral = std::integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept signed_integral = std::signed_integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept unsigned_integral = std::unsigned_integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept floating_point = std::floating_point<T>;
//------------------------------------------------------------------------------
template <typename T>
concept arithmetic = integral<T> || floating_point<T>;
//------------------------------------------------------------------------------
template <typename T>
concept arithmetic_or_complex = arithmetic<T> || is_complex<T>;
//------------------------------------------------------------------------------
template <typename From, typename To>
concept convertible_to = std::convertible_to<From, To>;
//------------------------------------------------------------------------------
template <typename From>
concept convertible_to_floating_point =
  convertible_to<From, float> ||
  convertible_to<From, double> ||
  convertible_to<From, long double>;
//------------------------------------------------------------------------------
template <typename From>
concept convertible_to_integral =
  convertible_to<From, bool> ||
  convertible_to<From, char> ||
  convertible_to<From, unsigned char> ||
  convertible_to<From, char8_t> ||
  convertible_to<From, char16_t> ||
  convertible_to<From, char32_t> ||
  convertible_to<From, wchar_t> ||
  convertible_to<From, short> ||
  convertible_to<From, unsigned short> ||
  convertible_to<From, int> ||
  convertible_to<From, unsigned int> ||
  convertible_to<From, long> ||
  convertible_to<From, unsigned long> ||
  convertible_to<From, long long> ||
  convertible_to<From, unsigned long long>;
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_real_t = requires {
  typename T::real_t;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_iterator = requires {
  typename T::iterator;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_this_t = requires {
  typename T::this_t;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_parent_t = requires {
  typename T::parent_t;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_tensor_t = requires {
  typename T::tensor_t;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_pos_t = requires {
  typename T::pos_t;
};
//==============================================================================
// indexable
//==============================================================================
template <typename T>
concept indexable = requires(T const t, std::size_t i) {
  { t[i] };
  { t.at(i) };
};
//------------------------------------------------------------------------------
template <typename T>
concept indexable_space =
  has_defined_iterator<std::decay_t<T>> &&
  requires (T const t, std::size_t i) {
    { t[i] } -> convertible_to_floating_point;
    { t.at(i) } -> convertible_to_floating_point;
  } &&
  requires (T const t) {
    { t.size() } -> convertible_to_integral;
    { size(t)  } -> convertible_to_integral;
    { t.front()  } -> convertible_to_floating_point;
    { t.back()  } -> convertible_to_floating_point;
    { t.begin()  } -> forward_iterator;
    { begin(t)  } -> forward_iterator;
    { t.end()  } -> forward_iterator;
    { end(t)  } -> forward_iterator;
  };
//==============================================================================
// methods
//==============================================================================
template <typename F, typename... Args>
concept invocable = std::invocable<F, Args...>;
//-----------------------------------------------------------------------------
template <typename F, typename... Args>
concept regular_invocable = std::regular_invocable<F, Args...>;
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_num_dimensions_method = requires {
  { T::num_dimensions() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_rank_method = requires {
  { T::rank() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename F, typename... Is>
concept invocable_with_integrals = std::regular_invocable<F, Is...> &&
                                   (std::is_integral_v<Is> && ...);
//==============================================================================
// types
//==============================================================================
template <typename Tensor, size_t... Dims>
concept tensor_c =
  has_static_rank_method<Tensor> &&
  invocable_with_n_integrals_v<Tensor, Tensor::rank()> &&
  has_defined_real_t<Tensor>;
//-----------------------------------------------------------------------------
template <typename Tensor, size_t N>
concept vec_c = tensor_c<Tensor, N>;
//-----------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
concept mat_c = tensor_c<Tensor, M, N>;
//-----------------------------------------------------------------------------
template <typename Tensor, typename... Is>
concept field_c =
    invocable_with_integrals<Tensor, Is...> &&
    has_static_rank_method<Tensor>                 &&
    has_defined_real_t<Tensor>&& requires(Tensor const t, Is const... is) {
      { t(is...) } -> std::convertible_to<typename Tensor::real_t>;
    }                                       &&
    sizeof...(Is) == Tensor::rank();
//-----------------------------------------------------------------------------
template <typename Flowmap>
concept flowmap_c =
  has_defined_real_t<Flowmap> &&
  has_defined_pos_t<Flowmap>&&
  has_static_num_dimensions_method<Flowmap> &&
  requires(Flowmap const flowmap,
           typename Flowmap::pos_t const& x,
           typename Flowmap::real_t const t,
           typename Flowmap::real_t const tau) {
    { flowmap(x, t, tau) }
      -> std::convertible_to<typename Flowmap::pos_t>;
    { flowmap.evaluate(x, t, tau) }
      -> std::convertible_to<typename Flowmap::pos_t>;
  };
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Flowmap, size_t N>
concept fixed_dims_flowmap_c =
  flowmap_c<Flowmap> &&
  Flowmap::num_dimensions() == N;
//==============================================================================
// stuff
//==============================================================================
template <typename Reader, typename Readable>
concept can_read = requires(Reader reader, Readable readable) {
  reader.read(readable);
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
