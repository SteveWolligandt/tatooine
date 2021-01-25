#ifndef TATOOINE_INDEX_ORDERING_H
#define TATOOINE_INDEX_ORDERING_H
//==============================================================================
#include <array>
#include <boost/range/adaptors.hpp>
#include <numeric>
#include <vector>

#include "concepts.h"
#include "type_traits.h"
#include "utility.h"
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
            enable_if_forward_iterator<ForwardIterator> = true,
            enable_if_integral<Is...>                   = true>
#endif
  static constexpr auto plain_index(ForwardIterator resoltion_it,
                                    Is const... is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for_each(
        [&](auto i) {
          idx += i * multiplier;
          multiplier *= *(resoltion_it++);
        },
        is...);
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <std::forward_iterator ForwardIterator, range Is>
#else
  template <typename ForwardIterator, typename Is,
            enable_if_forward_iterator<ForwardIterator> = true,
            enable_if_range<Is>                         = true>
#endif
  static constexpr auto plain_index(ForwardIterator resoltion_it,
                                    Is const&  is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for (auto i : is) {
      idx += i * multiplier;
      multiplier *= *(resoltion_it++);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if_range<Resolution> = true,
            enable_if_integral<Is...>   = true>
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
            enable_if_range<Resolution, IndexRange> = true>
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
  template <typename Resolution, enable_if_range<Resolution> = true>
#endif
  static auto multi_index(Resolution const& resolution, size_t plain_index) {
    std::vector<size_t> is(resolution.size());
    size_t              multiplier =
        std::accumulate(begin(resolution), std::prev(end(resolution)),
                        size_t(1), std::multiplies<size_t>{});

    auto resoltion_it = std::prev(end(resolution), 2);
    for (size_t j = 0; j < resolution.size(); ++j, --resoltion_it) {
      size_t i = resolution.size() - 1 - j;
      is[i]    = plain_index / multiplier;
      plain_index -= is[i] * multiplier;
      if (resoltion_it >= begin(resolution)) {
        multiplier /= *resoltion_it;
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
  template <integral Is, std::forward_iterator ResolutionIterator>
#else
  template <typename Is, typename ResolutionIterator,
            enable_if_integral<Is> = true,
            enable_if_forward_iterator<ResolutionIterator> = true >
#endif
  static constexpr auto internal_plain_index(ResolutionIterator resoltion_it,
                                             std::vector<Is> const& is)
      -> size_t {
    size_t multiplier = 1;
    size_t idx        = 0;

    for (auto i : is | boost::adaptors::reversed) {
      idx += i * multiplier;
      multiplier *= *(resoltion_it--);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <indexable Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if_indexable<Resolution> = true,
            enable_if_integral<Is...>       = true>
#endif
  static constexpr auto internal_plain_index(Resolution resolution,
                                             Is const... p_is) -> size_t {
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < is.size(); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      multiplier *= resolution[is.size() - 1 - i];
    }
    return idx;
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Resolution, integral... Is>
#else
  template <typename Resolution, typename... Is,
            enable_if_integral<Resolution, Is...> = true>
#endif
  static constexpr auto plain_index(const std::vector<Resolution>& resolution,
                                    Is const... is) -> size_t {
    assert(sizeof...(is) == resolution.size());
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Resolution, integral Is>
#else
  template <typename Resolution, typename Is,
            enable_if_integral<Resolution, Is> = true>
#endif
  static auto plain_index(const std::vector<Resolution>& resolution,
                          const std::vector<Is>&         is) -> size_t {
    assert(is.size() == resolution.size());
    return internal_plain_index(prev(end(resolution)), is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Resolution, integral... Is>
#else
  template <size_t N, typename Resolution, typename... Is,
            enable_if_integral<Resolution> = true>
#endif
  static constexpr auto plain_index(const std::array<Resolution, N>& resolution,
                                    Is const... is) -> size_t {
    static_assert(sizeof...(is) == N);
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Resolution, integral Is>
#else
  template <size_t N, typename Resolution, typename Is,
            enable_if_integral<Resolution, Is> = true >
#endif
  static constexpr auto plain_index(const std::array<Resolution, N>& resolution,
                                    const std::vector<Is>& is) -> size_t {
    assert(is.size() == N);
    return internal_plain_index(prev(end(resolution)), is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Resolution>
#else
  template <typename Resolution, enable_if_integral<Resolution> = true>
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
            enable_if_integral<Resolution> = true>
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
