#ifndef TATOOINE_RBC_H
#define TATOOINE_RBC_H

#include "field.h"
#include "grid_sampler.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct rbc : field<rbc, double, 2, 2> {
  using this_type   = rbc;
  using parent_type = field<this_type, real_type, 2, 2>;
  using parent_type::pos_type;
  using parent_type::real_type;
  using parent_type::tensor_type;
  using grid_t = grid_sampler<real_type, 2, vec<real_type, 2>, interpolation::linear,
                              interpolation::linear>;
  static constexpr std::array dim{512ul, 128ul, 201ul};
  static constexpr grid       domain{linspace{0.00390625, 3.99609375, dim[0]},
                               linspace{0.00390625, 0.99609375, dim[1]},
                               linspace{2000.0, 2020.0, dim[2]}};
  //============================================================================
 private:
  std::vector<grid_t> grids;

  //============================================================================
 public:
  rbc(const std::string& rbc_binary_dir) { read_from_binary(rbc_binary_dir); }
  //----------------------------------------------------------------------------
  void read_from_binary(const std::string& rbc_binary_dir) {
    grids.reserve(dim[2]);
    for (size_t ti = 0; ti < dim[2]; ++ti) {
      std::stringstream ss;
      ss << domain.dimension(2)[ti];
      const std::string filename = rbc_binary_dir + "/rbc_" + ss.str() + ".bin";
      grids.emplace_back(domain.dimension(0), domain.dimension(1));

      std::ifstream file(filename, std::ifstream::binary);
      if (file.is_open()) {
        std::vector<vec<double, 2>> data(dim[0] * dim[1]);
        // std::cout << "reading: " << filename <<'\n';
        constexpr auto num_bytes = sizeof(double) * dim[0] * dim[1] * 2;
        file.read((char*)(data.data()), num_bytes);
        file.close();

        grids.back().data() = data;
      } else {
        throw std::runtime_error{"could not open " + filename};
      }
    }
  }
  //----------------------------------------------------------------------------
  tensor_type evaluate(const pos_type& pos, real_type t) const {
    const auto& times = domain.dimension(2);
    for (size_t i = 0; i < grids.size() - 1; ++i)
      if (times[i] <= t && t <= times[i + 1]) {
        real_type f = (t - times[i]) / (times[i + 1] - times[i]);
        return (1 - f) * grids[i](pos(0), pos(1)) +
               f * grids[i + 1](pos(0), pos(1));
      }
    return {0, 0};
  }
  //----------------------------------------------------------------------------
  bool in_domain(const pos_type& p, real_type t) const {
    auto& times = domain.dimension(2);
    return times.front() <= t && t <= times.back() &&
           grids.front().in_domain(p(0), p(1));
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
