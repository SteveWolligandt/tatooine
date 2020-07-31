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
  static constexpr auto plain_index(std::forward_iterator auto resoltion_it,
                                    integral auto... is) {
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
  static constexpr auto plain_index(std::forward_iterator auto resoltion_it,
                                    range auto const&          indices) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for (auto i : indices) {
      idx += i * multiplier;
      multiplier *= *(resoltion_it++);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range ResolutionRange>
  static constexpr auto plain_index(ResolutionRange const& resolution_range,
                                    integral auto... is) {
    static_assert(std::is_integral_v<typename ResolutionRange::value_type>,
                  "resolution range must hold integral type");
    return plain_index(begin(resolution_range), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <range ResolutionRange, range IndexRange>
  static constexpr auto plain_index(ResolutionRange const& resolution_range,
                                    IndexRange const& indices) {
    static_assert(std::is_integral_v<typename ResolutionRange::value_type>,
                  "resolution range must hold integral type");
    static_assert(std::is_integral_v<typename IndexRange::value_type>,
                  "index range must hold integral type");
    assert(resolution_range.size() == indices.size());
    return plain_index(begin(resolution_range), indices);
  }
  //----------------------------------------------------------------------------
  template <range ResolutionRange>
  static auto multi_index(ResolutionRange const& resolution,
                          size_t                 plain_index) {
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
  template <integral IsType>
  static constexpr auto internal_plain_index(
      std::forward_iterator auto resoltion_it, const std::vector<IsType>& is)
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
  static constexpr auto internal_plain_index(indexable auto resolution,
                                             integral auto... p_is) -> size_t {
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
  template <integral ResType>
  static constexpr auto plain_index(const std::vector<ResType>& resolution,
                                    integral auto... is) -> size_t {
    assert(sizeof...(is) == resolution.size());
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral ResType, integral IsType>
  static auto plain_index(const std::vector<ResType>& resolution,
                          const std::vector<IsType>&  is) -> size_t {
    assert(is.size() == resolution.size());
    return internal_plain_index(prev(end(resolution)), is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral ResType>
  static constexpr auto plain_index(const std::array<ResType, N>& resolution,
                                    integral auto... is) -> size_t {
    static_assert(sizeof...(is) == N);
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral ResType, integral IsType>
  static constexpr auto plain_index(const std::array<ResType, N>& resolution,
                                    const std::vector<IsType>& is) -> size_t {
    assert(is.size() == N);
    return internal_plain_index(prev(end(resolution)), is);
  }
  //----------------------------------------------------------------------------
  template <integral ResType>
  static auto multi_index(const std::vector<ResType>& resolution,
                          size_t /*plain_index*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::vector<size_t>&, size_t) not "
        "implemented"};
    std::vector<ResType> is(resolution.size());
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral ResType>
  static constexpr auto multi_index(
      const std::array<ResType, N>& /*resolution*/, size_t /*plain_index*/) {
    throw std::runtime_error{
        "x_slowest::multi_index(const std::array<size_t, N>&, size_t) not "
        "implemented"};
    auto is = make_array<ResType, N>();
    return is;
  }
};
//==============================================================================
/// converts multi-dimensional index to a one dimensional index using a
/// space-filling curve algorithm
struct hilbert_curve {
  static constexpr auto plain_index(const std::vector<size_t>& /*resolution*/,
                                    integral auto... /*is*/) -> size_t {
    throw std::runtime_error{
        "hilbert_curve::plain_index(const std::vector<size_t>&, Is... is) not "
        "implemented"};
    return 0;
  }
  template <size_t N>
  static constexpr auto plain_index(const std::array<size_t, N>& /*resolution*/,
                                    integral auto... /*is*/) -> size_t {
    throw std::runtime_error{
        "hilbert_curve::plain_index(const std::array<size_t, N>&, Is... is) "
        "not "
        "implemented"};
    return 0;
  }
  //----------------------------------------------------------------------------
  static auto multi_index(const std::vector<size_t>& resolution,
                          size_t /*plain_index*/) {
    std::vector<size_t> is(resolution.size());
    throw std::runtime_error{
        "hilbert_curve::multi_index(const std::vector<size_t>&, size_t) not "
        "implemented"};
    return is;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N>
  static constexpr auto multi_index(const std::array<size_t, N>& /*resolution*/,
                                    size_t /*plain_index*/) {
    auto is = make_array<size_t, N>();
    throw std::runtime_error{
        "hilbert_curve::multi_index(const std::array<size_t, N>&, size_t) not "
        "implemented"};
    return is;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
