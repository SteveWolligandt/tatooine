#include <tatooine/analytical/fields/frankes_test.h>
#include <tatooine/pointset.h>
#include <tatooine/rectilinear_grid.h>

#include <boost/program_options.hpp>
#include <cstdint>
//==============================================================================
using namespace tatooine;
//==============================================================================
struct options_t {
  real_number radius;
  size_t      output_res_x, output_res_y, num_datapoints;
};
//==============================================================================
auto parse_args(int argc, char const** argv) -> std::optional<options_t>;
//==============================================================================
auto main(int argc, char const** argv) -> int {
  auto const options_opt = parse_args(argc, argv);
  if (!options_opt.has_value()) {
    return 1;
  }
  auto const options = *options_opt;
  auto f = analytical::fields::numerical::frankes_test{};
  auto nabla_f = diff(f);
  auto resample_grid = uniform_rectilinear_grid2{linspace{0.0, 1.0, options.output_res_x},
                                      linspace{0.0, 1.0, options.output_res_y}};
  auto       rand    = random::uniform{0.0, 1.0, std::mt19937_64{1234}};
  auto       scattered_data      = pointset2{};
  auto&      scattered_franke = scattered_data.scalar_vertex_property("franke");

  // sample scattered ground truth
  for (size_t i = 0; i < options.num_datapoints; ++i) {
    auto v = scattered_data.insert_vertex(rand(), rand());
    scattered_franke.at(v) = f(scattered_data[v]);
  }
  // sample ground truth
  resample_grid.sample_to_vertex_property(f, "ground_truth", execution_policy::parallel);

  // sample constant inverse distance
  resample_grid.sample_to_vertex_property(
      [&](auto const& q) {
        auto [indices, squared_distances] =
            scattered_data.nearest_neighbors_radius_raw(q, options.radius);
        if (indices.empty()) {
          return 0.0 / 0.0;
        }
        auto accumulated_prop_val = real_number{};
        auto accumulated_weight   = real_number{};

        auto index_it        = begin(indices);
        auto squared_dist_it = begin(squared_distances);
        for (; index_it != end(indices); ++index_it, ++squared_dist_it) {
          auto const& val = scattered_franke.at(*index_it);

          if (*squared_dist_it == 0) {
            return val;
          };
          auto const weight = 1 / *squared_dist_it;
          accumulated_prop_val += val * weight;
          accumulated_weight += weight;
        }
        return accumulated_prop_val / accumulated_weight;
      },
      "inverse_distance_weighting_constant", execution_policy::parallel);

  // sample linear inverse distance
  resample_grid.sample_to_vertex_property(
      [&](auto const& q) {
        auto [indices, squared_distances] =
            scattered_data.nearest_neighbors_radius_raw(q, options.radius);
        if (indices.empty()) {
          return 0.0 / 0.0;
        }
        auto accumulated_prop_val = real_number{};
        auto accumulated_weight   = real_number{};

        auto index_it        = begin(indices);
        auto squared_dist_it = begin(squared_distances);
        for (; index_it != end(indices); ++index_it, ++squared_dist_it) {
        auto const& x_i = scattered_data.vertex_at(*index_it);
        auto const  val =
            dot(nabla_f(x_i), q - x_i) + scattered_franke.at(*index_it);

        if (*squared_dist_it == 0) {
          return val;
          };
          auto const weight = 1 / *squared_dist_it;
          accumulated_prop_val += val * weight;
          accumulated_weight += weight;
        }
        return accumulated_prop_val / accumulated_weight;
      },
      "inverse_distance_weighting_linear", execution_policy::parallel);

  // sample full radial basis functions
  resample_grid.sample_to_vertex_property(
      scattered_data
          .radial_basis_functions_sampler_with_thin_plate_spline_kernel(
              scattered_franke),
      "full_radial_bases", execution_policy::parallel);

  // sample full radial basis functions with polynomial
  resample_grid.sample_to_vertex_property(
      scattered_data
          .radial_basis_functions_sampler_with_polynomial_and_thin_plate_spline_kernel(
              scattered_franke),
      "full_radial_bases_with_polynomial", execution_policy::parallel);


  // sample full radial basis functions with polynomial
  resample_grid.sample_to_vertex_property(
      scattered_data
          .radial_basis_functions_sampler_with_polynomial_and_thin_plate_spline_kernel(
              scattered_franke),
      "full_radial_bases_with_polynomial", execution_policy::parallel);

  // local radial basis functions constant
  resample_grid.sample_to_vertex_property(
      [&](auto const& q) {
        auto [indices, squared_distances] =
            scattered_data.nearest_neighbors_radius_raw(q, options.radius);
        if (indices.empty()) {
          return 0.0 / 0.0;
        }

        auto const N = indices.size();
        auto const NumDimensions = 2;

        // construct lower part of symmetric matrix A
        auto A = tensor<real_number>::zeros(N + NumDimensions + 1,
                                            N + NumDimensions + 1);
        auto weights_and_coeffs =
            tensor<real_number>::zeros(N + NumDimensions + 1);
        //auto A = tensor<real_number>::zeros(N, N);
        //auto weights_and_coeffs = tensor<real_number>::zeros(N);
        for (std::size_t c = 0; c < N; ++c) {
          for (std::size_t r = c + 1; r < N; ++r) {
            A(r, c) = thin_plate_spline(squared_euclidean_distance(
                scattered_data.vertex_at(indices[c]),
                scattered_data.vertex_at(indices[r])));
          }
        }
        // construct polynomial requirements
        for (std::size_t c = 0; c < N; ++c) {
          auto const& p = scattered_data.vertex_at(indices[c]);
          // constant part
          A(N, c) = 1;

          // linear part
          for (std::size_t i = 0; i < NumDimensions; ++i) {
            A(N + i + 1, c) = p(i);
          }
        }

        for (std::size_t i = 0; i < N; ++i) {
          weights_and_coeffs(i) = scattered_franke[indices[i]];
        }
        // do not copy by moving A and weights_and_coeffs into solver
        weights_and_coeffs = *solve_symmetric_lapack(
            std::move(A), std::move(weights_and_coeffs), tatooine::lapack::Uplo::Lower);

        auto       acc = real_number{};
        // radial bases
        for (std::size_t i = 0; i < N; ++i) {
          auto const v = indices[i];
          if (squared_distances[i] == 0) {
            return scattered_franke[v];
          }
          acc += weights_and_coeffs(i) * thin_plate_spline(squared_distances[i]);
        }
        // monomial bases
        acc += weights_and_coeffs(N);
        for (std::size_t k = 0; k < NumDimensions; ++k) {
          acc += weights_and_coeffs(N + 1 + k) * q(k);
        }
        return acc;
      },
      "local_radial_bases_with_polynomial_constant", execution_policy::parallel);

  // local radial basis functions with gradients in system
  resample_grid.sample_to_vertex_property(
      [&](auto const& q) {
        auto [indices, squared_distances] =
            scattered_data.nearest_neighbors_radius_raw(q, options.radius);
        if (indices.empty()) {
          return 0.0 / 0.0;
        }

        auto const N = indices.size();
        auto const NumDimensions = 2;

        // construct lower part of symmetric matrix A
        auto A = tensor<real_number>::zeros(N + NumDimensions + 1,
                                            N + NumDimensions + 1);
        auto weights_and_coeffs =
            tensor<real_number>::zeros(N + NumDimensions + 1);
        //auto A = tensor<real_number>::zeros(N, N);
        //auto weights_and_coeffs = tensor<real_number>::zeros(N);
        for (std::size_t c = 0; c < N; ++c) {
          for (std::size_t r = c + 1; r < N; ++r) {
            A(r, c) = thin_plate_spline(squared_euclidean_distance(
                scattered_data.vertex_at(indices[c]),
                scattered_data.vertex_at(indices[r])));
          }
        }
        // construct polynomial requirements
        for (std::size_t c = 0; c < N; ++c) {
          auto const& p = scattered_data.vertex_at(indices[c]);
          // constant part
          A(N, c) = 1;

          // linear part
          for (std::size_t i = 0; i < NumDimensions; ++i) {
            A(N + i + 1, c) = p(i);
          }
        }

        for (std::size_t i = 0; i < N; ++i) {
          weights_and_coeffs(i) = scattered_franke[indices[i]];
        }
        // do not copy by moving A and weights_and_coeffs into solver
        weights_and_coeffs = *solve_symmetric_lapack(
            std::move(A), std::move(weights_and_coeffs), tatooine::lapack::Uplo::Lower);

        auto       acc = real_number{};
        // radial basis functions
        for (std::size_t i = 0; i < N; ++i) {
          auto const v = indices[i];
          if (squared_distances[i] == 0) {
            return scattered_franke[v];
          }
          acc += weights_and_coeffs(i) * thin_plate_spline_from_squared(squared_distances[i]);
        }
        // polynomial part
        acc += weights_and_coeffs(N);
        for (std::size_t k = 0; k < NumDimensions; ++k) {
          acc += weights_and_coeffs(N + 1 + k) * q(k);
        }
        return acc;
      },
      "local_radial_bases_with_polynomial_gradient_in_system",
      execution_policy::parallel);

  // local radial basis functions linear
  resample_grid.sample_to_vertex_property(
      [&](auto const& q) {
        auto [indices, squared_distances] =
            scattered_data.nearest_neighbors_radius_raw(q, options.radius);
        if (indices.empty()) {
          return 0.0 / 0.0;
        }

        auto const N = indices.size();
        auto const NumDimensions = 2;

        // construct lower part of symmetric matrix A
        //auto A = tensor<real_number>::zeros(N + NumDimensions + 1,
        //                                    N + NumDimensions + 1);
        //auto weights_and_coeffs =
        //    tensor<real_number>::zeros(N + NumDimensions + 1);
        auto A = tensor<real_number>::zeros(N, N);
        auto weights_and_coeffs = tensor<real_number>::zeros(N);
        for (std::size_t c = 0; c < N; ++c) {
          for (std::size_t r = c + 1; r < N; ++r) {
            A(r, c) = thin_plate_spline_from_squared(squared_euclidean_distance(
                scattered_data.vertex_at(indices[c]),
                scattered_data.vertex_at(indices[r])));
          }
        }
        // construct polynomial requirements
        for (std::size_t c = 0; c < N; ++c) {
          auto const& p = scattered_data.vertex_at(indices[c]);
          // constant part
          A(N, c) = 1;

          // linear part
          for (std::size_t i = 0; i < NumDimensions; ++i) {
            A(N + i + 1, c) = p(i);
          }
        }

        for (std::size_t i = 0; i < N; ++i) {
          auto const& x_i = scattered_data.vertex_at(indices[i]);
          weights_and_coeffs(i) =
              dot(nabla_f(x_i), q - x_i) + scattered_franke.at(indices[i]);
        }
        // do not copy by moving A and weights_and_coeffs into solver
        weights_and_coeffs = *solve_symmetric_lapack(
            std::move(A), std::move(weights_and_coeffs), tatooine::lapack::Uplo::Lower);

        auto       acc = real_number{};
        // radial basis functions
        for (std::size_t i = 0; i < N; ++i) {
          auto const v = indices[i];
          if (squared_distances[i] == 0) {
            return scattered_franke[v];
          }
          acc += weights_and_coeffs(i) * thin_plate_spline(squared_distances[i]);
        }
        // polynomial part
        acc += weights_and_coeffs(N);
        for (std::size_t k = 0; k < NumDimensions; ++k) {
          acc += weights_and_coeffs(N + 1 + k) * q(k);
        }
        return acc;
      },
      "local_radial_bases_with_polynomial_linear", execution_policy::parallel);

  resample_grid.write("scattered_data_comparison.vtr");
  scattered_data.write("scattered_data_comparison.vtp");
}
//==============================================================================
auto parse_args(int const argc, char const** argv) -> std::optional<options_t> {
  namespace po = boost::program_options;
  size_t output_res_x, output_res_y, num_datapoints;

  auto desc = po::options_description{"Allowed options"};
  auto vm   = po::variables_map{};

  // Declare supported options.
  desc.add_options()("help", "produce help message")(
      "num_datapoints", po::value<size_t>(), "number of data points")(
      "output_res_x", po::value<size_t>(), "set outputresolution width")(
      "output_res_y", po::value<size_t>(), "set outputresolution height");

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") > 0) {
    std::cout << desc;
    return std::nullopt;
  }
  if (vm.count("radius") > 0) {
    radius = vm["radius"].as<real_number>();
  } else {
    std::cerr << "--radius not specified!\n";
    return std::nullopt;
  }
  if (vm.count("output_res_x") > 0) {
    output_res_x = vm["output_res_x"].as<size_t>();
  } else {
    std::cerr << "--output_res_x not specified!\n";
    return std::nullopt;
  }
  if (vm.count("output_res_y") > 0) {
    output_res_y = vm["output_res_y"].as<size_t>();
  } else {
    std::cerr << "--output_res_y not specified!\n";
    return std::nullopt;
  }
  if (vm.count("num_datapoints") > 0) {
    num_datapoints = vm["num_datapoints"].as<size_t>();
  } else {
    std::cerr << "--num_datapoints not specified!\n";
    return std::nullopt;
  }
  return options_t{radius, output_res_x, output_res_y, num_datapoints};
}
