#include <tatooine/analytical/fields/numerical/center.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/analytical/fields/numerical/sinuscosinus.h>
#include <tatooine/filesystem.h>
#include <tatooine/gl/context.h>
#include <tatooine/gpu/lic.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <istream>
#include <stdexcept>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto gpu_lic_janos(filesystem::path const& path, real_number const t,
                        std::size_t const num_samples,
                        real_number const step_size) {
  struct velocity_janos : vectorfield<velocity_janos, real_number, 2> {
    constexpr auto evaluate(vec2 const& p, real_number const t) const
        -> vectorfield<velocity_janos, real_number, 2>::tensor_type {
      return vec2{-p.x() * (2 * p.x() * p.x() - 2) / 2, -p.y()};
    }
  } v;
  gpu::lic(v, linspace{-1.0, 1.0, 2000}, linspace{-1.5, 1.5, 1000}, t,
           {1000, 1500}, num_samples, step_size, {256, 256}, random::uniform{})
      .write_png(path);
}
//==============================================================================
auto gpu_lic_doublegyre(filesystem::path const& path, real_number const t,
                        std::size_t const num_samples,
                        real_number const step_size) {
  auto v = analytical::fields::numerical::doublegyre{};
  gpu::lic(v, linspace{0.0, 2.0, 2000}, linspace{0.0, 1.0, 1000}, t,
           {1000, 500}, num_samples, step_size, {256, 256}, random::uniform{})
      .write_png(path);
}
//==============================================================================
auto gpu_lic_saddle(filesystem::path const& path, real_number const t,
                    std::size_t const num_samples,
                    real_number const step_size) {
  auto v = analytical::fields::numerical::saddle{};
  gpu::lic(v, linspace{-1.0, 1.0, 500}, linspace{-1.0, 1.0, 500}, t,
           Vec2<size_t>{1000, 1000}, num_samples, step_size, {256, 256},
           random::uniform{})
      .write_png(path);
}
//==============================================================================
auto gpu_lic_center(filesystem::path const& path, real_number const t,
                    std::size_t const num_samples,
                    real_number const step_size) {
  auto v = analytical::fields::numerical::center{};
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, t,
           Vec2<size_t>{1000, 1000}, num_samples, step_size, {256, 256},
           random::uniform{})
      .write_png(path);
}
//==============================================================================
auto gpu_lic_sinuscosinus(filesystem::path const& path, real_number const t,
                          std::size_t const num_samples,
                          real_number const step_size) {
  auto v = analytical::fields::numerical::sinuscosinus{};
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, t,
           Vec2<size_t>{1000, 1000}, num_samples, step_size, {256, 256},
           random::uniform{})
      .write_png(path);
}
//==============================================================================
auto gpu_lic_cosinussinus(filesystem::path const& path, real_number const t,
                          std::size_t const num_samples,
                          real_number const step_size) {
  auto v = analytical::fields::numerical::cosinussinus{};
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, t,
           Vec2<size_t>{1000, 1000}, num_samples, step_size, {256, 256},
           random::uniform{})
      .write_png(path);
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
enum class field {
  janos,
  doublegyre,
  saddle,
  center,
  sinuscosinus,
  cosinussinus,
  unknown
};

auto operator>>(std::istream& in, field& f) -> auto& {
  auto token = std::string{};
  in >> token;
  if (token == "janos") {
    f = field::janos;
    return in;
  } else if (token == "doublegyre") {
    f = field::doublegyre;
    return in;
  } else if (token == "saddle") {
    f = field::saddle;
    return in;
  } else if (token == "center") {
    f = field::center;
    return in;
  } else if (token == "sinuscosinus") {
    f = field::sinuscosinus;
    return in;
  } else if (token == "cosinussinus") {
    f = field::cosinussinus;
    return in;
  }
  f = field::unknown;
  return in;
}
auto main(int argc, char** argv) -> int {
  auto ctx     = tatooine::gl::context{};
  namespace po = boost::program_options;
  auto desc    = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "time", po::value<tatooine::real_number>())(
      "out", po::value<tatooine::filesystem::path>())("num_samples",
                                                      po::value<std::size_t>())(
      "step_size", po::value<tatooine::real_number>())("field",
                                                       po::value<field>());

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return 0;
  }

  auto const num_samples = [&]() -> std::size_t {
    if (variables_map.count("num_samples") > 0) {
      return variables_map["num_samples"].as<std::size_t>();
    }
    return 30;
  }();
  auto const step_size = [&]() -> tatooine::real_number {
    if (variables_map.count("step_size") > 0) {
      return variables_map["step_size"].as<tatooine::real_number>();
    }
    return 1e-3;
  }();
  auto const t = [&]() -> tatooine::real_number {
    if (variables_map.count("time") > 0) {
      return variables_map["time"].as<tatooine::real_number>();
    } else {
      throw std::runtime_error{"Flag --time not specified."};
    }
    return 0;
  }();
  auto const path = [&]() -> tatooine::filesystem::path {
    if (variables_map.count("out") > 0) {
      return variables_map["out"].as<tatooine::filesystem::path>();
    } else {
      throw std::runtime_error{"Flag --out not specified."};
    }
    return {};
  }();
  auto const f = [&]() -> field {
    if (variables_map.count("field") > 0) {
      return variables_map["field"].as<field>();
    } else {
      throw std::runtime_error{"Flag --field not specified."};
    }
    return field::unknown;
  }();
  if (f == field::unknown) {
    throw std::runtime_error{"Specified unknown field."};
  }
  switch (f) {
    case field::janos:
      tatooine::examples::gpu_lic_janos(path, t, num_samples, step_size);
      break;
    case field::doublegyre:
      tatooine::examples::gpu_lic_doublegyre(path, t, num_samples, step_size);
      break;
    case field::saddle:
      tatooine::examples::gpu_lic_saddle(path, t, num_samples, step_size);
      break;
    case field::center:
      tatooine::examples::gpu_lic_center(path, t, num_samples, step_size);
      break;
    case field::sinuscosinus:
      tatooine::examples::gpu_lic_sinuscosinus(path, t, num_samples, step_size);
      break;
    case field::cosinussinus:
      tatooine::examples::gpu_lic_cosinussinus(path, t, num_samples, step_size);
      break;
    default:
      break;
  }
}
