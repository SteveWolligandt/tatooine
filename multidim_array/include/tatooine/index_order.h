#ifndef TATOOINE_INDEX_ORDER_H
#define TATOOINE_INDEX_ORDER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

#include <array>
#include <boost/range/adaptor/reversed.hpp>
#include <numeric>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
/// converts multi-dimensional index to a one dimensional index where first
/// dimensions grows fastest
struct x_fastest {
  static constexpr auto plain_index(std::forward_iterator auto resolution_it,
                                    integral auto const... is) {
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
  static constexpr auto plain_index(std::forward_iterator auto resolution_it,
                                    range auto const&          is) {
    size_t multiplier = 1;
    size_t idx        = 0;
    for (auto i : is) {
      idx += i * multiplier;
      multiplier *= *(resolution_it++);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(range auto const& resolution,
                                    integral auto const... is) {
    using Resolution = std::decay_t<decltype(resolution)>;
    static_assert(is_integral<typename Resolution::value_type>,
                  "resolution range must hold integral type");
    return plain_index(begin(resolution), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(range auto const& resolution,
                                    range auto const& indices) {
    using Resolution = std::decay_t<decltype(resolution)>;
    using Indices    = std::decay_t<decltype(resolution)>;
    static_assert(is_integral<typename Resolution::value_type>,
                  "resolution range must hold integral type");
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(resolution.size() == indices.size());
    return plain_index(begin(resolution), indices);
  }
  //----------------------------------------------------------------------------
  static auto multi_index(range auto const& resolution, size_t plain_index) {
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
  static constexpr auto internal_plain_index(
      std::forward_iterator auto resolution_it, range auto const& is)
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
  static constexpr auto internal_plain_index(range auto const& resolution,
                                             integral auto const... p_is)
      -> size_t {
    std::array is{p_is...};

    size_t multiplier = 1;
    size_t idx        = 0;

    for (size_t i = 0; i < size(is); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      // idx += is[i] * multiplier;
      multiplier *= resolution[is.size() - 1 - i];
    }
    return idx;
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(range auto const& resolution,
                                    integral auto const... is) -> size_t {
    assert(sizeof...(is) == resolution.size());
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto plain_index(range auto const& resolution, range auto const& is)
      -> size_t {
    assert(is.size() == resolution.size());
    return internal_plain_index(prev(end(resolution)), is);
  }
  //----------------------------------------------------------------------------
  static auto multi_index(range auto const& resolution, size_t plain_index) {
    std::vector<size_t> is(resolution.size());
    size_t              multiplier = 1;

    auto resolution_it = std::prev(end(resolution));
    auto is_it         = std::prev(end(is));
    for (; resolution_it != begin(resolution); --resolution_it, --is_it) {
      *is_it = plain_index * multiplier;
      plain_index -= *is_it;
      multiplier *= *resolution_it;
    }
    return is;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
