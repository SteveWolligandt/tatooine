#ifndef TATOOINE_INDEX_ORDER_H
#define TATOOINE_INDEX_ORDER_H
//==============================================================================
#include <array>
#include <boost/range/adaptor/reversed.hpp>
#include <numeric>
#include <vector>

#include <tatooine/concepts.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// converts multi-dimensional index to a one dimensional index where first
/// dimensions grows fastest
struct x_fastest {
#ifdef __cpp_concepts
  template <std::forward_iterator ForwardIterator, integral... Is>
#else
  template <typename ForwardIterator, typename... Is,
            enable_if<is_forward_iterator<ForwardIterator>> = true,
            enable_if<is_integral<Is...>>                   = true>
#endif
  static constexpr auto plain_index(ForwardIterator resolution_it,
                                    Is const... is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for_each(
        [&](auto i) {
          idx += i * multiplier;
          multiplier *= *(resolution_it++);
        },
        is...);
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <std::forward_iterator ForwardIterator, range Is>
#else
  template <typename ForwardIterator, typename Is,
            enable_if<is_forward_iterator<ForwardIterator>> = true,
            enable_if<is_range<Is>>                         = true>
#endif
  static constexpr auto plain_index(ForwardIterator resolution_it,
                                    Is const&  is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for (auto i : is) {
      idx += i * multiplier;
      multiplier *= *(resolution_it++);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if<is_range<Resolution>> = true,
            enable_if<is_integral<Is...>>   = true>
#endif
  static constexpr auto plain_index(Resolution const& resolution,
                                    Is const... is) {
    static_assert(std::is_integral_v<typename Resolution::value_type>,
                  "resolution range must hold integral type");
    return plain_index(begin(resolution), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, range IndexRange>
#else
  template <typename Resolution, typename IndexRange,
            enable_if<is_range<Resolution, IndexRange>> = true>
#endif
  static constexpr auto plain_index(Resolution const& resolution,
                                    IndexRange const& is) {
    static_assert(std::is_integral_v<typename Resolution::value_type>,
                  "resolution range must hold integral type");
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(resolution.size() == is.size());
    return plain_index(begin(resolution), is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <range Resolution>
#else
  template <typename Resolution, enable_if<is_range<Resolution>> = true>
#endif
  static auto multi_index(Resolution const& resolution, size_t plain_index) {
    std::vector<size_t> is(resolution.size());
    size_t              multiplier =
        std::accumulate(begin(resolution), std::prev(end(resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto resolution_it = std::prev(end(resolution), 2);
    for (size_t j = 0; j < resolution.size(); ++j, --resolution_it) {
      size_t i = resolution.size() - 1 - j;
      is[i]    = plain_index / multiplier;
      plain_index -= is[i] * multiplier;
      if (resolution_it >= begin(resolution)) {
        multiplier /= *resolution_it;
      }
    }
    return is;
  }
};
//==============================================================================
/// converts multi-dimensional index to a one dimensional index where first
/// dimensions grows slowest
struct x_slowest {
 private:
#ifdef __cpp_concepts
  template <range Indices, std::forward_iterator ResolutionIterator>
#else
  template <typename Indices, typename ResolutionIterator,
            enable_if<is_range<Indices>,
                      is_forward_iterator<ResolutionIterator> > = true>
#endif
  static constexpr auto internal_plain_index(ResolutionIterator resolution_it,
                                             Indices const& is)
      -> size_t {
    size_t multiplier = 1;
    size_t idx        = 0;

    for (auto i : is | boost::adaptors::reversed) {
      idx += i * multiplier;
      multiplier *= *(resolution_it--);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if<is_range<Resolution>> = true,
            enable_if<is_integral<Is...>>       = true>
#endif
  static constexpr auto internal_plain_index(Resolution resolution,
                                             Is const... p_is) -> size_t {
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < size(is); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      //idx += is[i] * multiplier;
      multiplier *= resolution[is.size() - 1 - i];
    }
    return idx;
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if<is_range<Resolution>, is_integral<Is...> > = true>
#endif
  static constexpr auto plain_index(Resolution const& resolution,
                                    Is const... is) -> size_t {
    assert(sizeof...(is) == resolution.size());
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, range Indices>
#else
  template <typename Resolution, typename Indices,
            enable_if<is_range<Resolution, Indices>,
                      is_integral<typename Indices::value_type> > = true>
#endif
  static auto plain_index(const Resolution& resolution,
                          const Indices&         is) -> size_t {
    assert(is.size() == resolution.size());
    return internal_plain_index(prev(end(resolution)), is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Resolution>
#else
  template <typename Resolution, enable_if<is_integral<Resolution>> = true>
#endif
  static auto multi_index(const std::vector<Resolution>& resolution,
                          size_t /*plain_index*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::vector<size_t>&, size_t) not "
        "implemented"};
    std::vector<Resolution> is(resolution.size());
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Resolution>
#else
  template <size_t N, typename Resolution,
            enable_if<is_integral<Resolution>> = true>
#endif
  static constexpr auto multi_index(
      const std::array<Resolution, N>& /*resolution*/, size_t /*plain_index*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::array<size_t, N>&, size_t) not "
        "implemented"};
    auto is = make_array<Resolution, N>();
    return is;
  }
};
//==============================================================================
/// converts multi-dimensional index to a one dimensional index using a
/// space-filling curve algorithm
//struct hilbert_curve {
//  static constexpr auto plain_index(const std::vector<size_t>& [>resolution<],
//                                    integral auto... [>is<]) -> size_t {
//    throw std::runtime_error{
//        "hilbert_curve::plain_index(const std::vector<size_t>&, Is... is) not "
//        "implemented"};
//    return 0;
//  }
//  template <size_t N>
//  static constexpr auto plain_index(const std::array<size_t, N>& [>resolution<],
//                                    integral auto... [>is<]) -> size_t {
//    throw std::runtime_error{
//        "hilbert_curve::plain_index(const std::array<size_t, N>&, Is... is) "
//        "not "
//        "implemented"};
//    return 0;
//  }
//  //----------------------------------------------------------------------------
//  static auto multi_index(const std::vector<size_t>& resolution,
//                          size_t [>plain_index<]) {
//    std::vector<size_t> is(resolution.size());
//    throw std::runtime_error{
//        "hilbert_curve::multi_index(const std::vector<size_t>&, size_t) not "
//        "implemented"};
//    return is;
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  template <size_t N>
//  static constexpr auto multi_index(const std::array<size_t, N>& [>resolution<],
//                                    size_t [>plain_index<]) {
//    auto is = make_array<size_t, N>();
//    throw std::runtime_error{
//        "hilbert_curve::multi_index(const std::array<size_t, N>&, size_t) not "
//        "implemented"};
//    return is;
//  }
//};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
