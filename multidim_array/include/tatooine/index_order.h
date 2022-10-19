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
  template <std::forward_iterator Iterator>
  static constexpr auto plain_index(Iterator resolution_it,
                                    integral auto const... is) {
    using int_t     = typename std::iterator_traits<Iterator>::value_type;
    auto multiplier = int_t(1);
    auto idx        = int_t(0);
    auto it         = [&](int_t const i) {
      idx += i * multiplier;
      multiplier *= *(resolution_it++);
    };
    for_each(it, static_cast<int_t>(is)...);
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::forward_iterator Iterator>
  static constexpr auto plain_index(Iterator                  resolution_it,
                                    integral_range auto const& is) {
    using int_t     = typename std::iterator_traits<Iterator>::value_type;
    auto multiplier = int_t(1);
    auto idx        = int_t(0);
    for (auto const i : is) {
      idx += static_cast<int_t>(i) * multiplier;
      multiplier *= *(resolution_it++);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(integral_range auto const& resolution,
                                    integral auto const... is) {
    return plain_index(begin(resolution), is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(integral_range auto const& resolution,
                                    integral_range auto const& indices) {
    assert(resolution.size() == indices.size());
    return plain_index(begin(resolution), indices);
  }
  //----------------------------------------------------------------------------
  template <integral_range Resolution>
  static auto multi_index(Resolution const& resolution,
                          integral auto     plain_index) {
    using int_t     = std::ranges::range_value_t<Resolution>;
    auto is         = std::vector<int_t>(resolution.size());
    auto multiplier = std::accumulate(std::ranges::begin(resolution), std::ranges::prev(end(resolution)),
                                      int_t(1), std::multiplies<int_t>{});

    auto resolution_it = std::ranges::prev(end(resolution), 2);
    for (std::size_t j = 0; j < resolution.size(); ++j, --resolution_it) {
      auto i = resolution.size() - 1 - j;
      is[i]  = plain_index / multiplier;
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
      -> std::size_t {
    auto multiplier = std::size_t(1);
    auto idx        = std::size_t(0);

    for (auto i : is | boost::adaptors::reversed) {
      idx += i * multiplier;
      multiplier *= *(resolution_it--);
    }
    return idx;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto internal_plain_index(range auto const& resolution,
                                             integral auto const... p_is)
      -> std::size_t {
    std::array is{p_is...};

    std::size_t multiplier = 1;
    std::size_t idx        = 0;

    for (std::size_t i = 0; i < size(is); ++i) {
      idx += is[is.size() - 1 - i] * multiplier;
      // idx += is[i] * multiplier;
      multiplier *= resolution[is.size() - 1 - i];
    }
    return idx;
  }

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static constexpr auto plain_index(range auto const& resolution,
                                    integral auto const... is) -> std::size_t {
    assert(sizeof...(is) == resolution.size());
    return internal_plain_index(resolution, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto plain_index(range auto const& resolution, range auto const& is)
      -> std::size_t {
    assert(is.size() == resolution.size());
    return internal_plain_index(prev(end(resolution)), is);
  }
  //----------------------------------------------------------------------------
  static auto multi_index(range auto const& resolution, std::size_t plain_index) {
    auto is = std::vector<std::size_t> (resolution.size());
    std::size_t multiplier = 1;

    auto resolution_it = prev(end(resolution));
    auto is_it         = prev(end(is));
    for (; resolution_it != begin(resolution); --resolution_it, --is_it) {
      *is_it = plain_index * multiplier;
      plain_index -= *is_it;
      multiplier *= *resolution_it;
    }
    return is;
  }
};
template <typename T>
concept index_order = std::same_as<T, x_fastest> || std::same_as<T, x_slowest>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
