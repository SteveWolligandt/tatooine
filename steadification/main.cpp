#include "start.h"
auto main(int argc, char** argv) -> int {
  using namespace tatooine;
  using namespace steadification;
  using namespace numerical;
  const std::string v = argv[1];
  const size_t      griddim = atoi(argv[2]);
  if (griddim < 2 || griddim > 3) {
    throw std::runtime_error{"grid dimension must be 2 or 3"};
  }
  if (v == "dg" || v == "doublegyre") {
    if (griddim == 2) {
      calc2(numerical::doublegyre<double>{}, argc, argv);
    } else if (griddim == 3) {
      calc3(numerical::doublegyre<double>{}, argc, argv);
    }
  } else if (v == "la" || v == "laminar") {
    if (griddim == 2) {
      calc2(laminar<double>{}, argc, argv);
    } else if (griddim == 3) {
      calc3(laminar<double>{}, argc, argv);
    }
  //} else if (v == "fdg") {
  //  calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  } else if (v == "sc" || v== "sinuscosinus") {
    if (griddim == 2) {
      calc2(numerical::sinuscosinus<double>{}, argc, argv);
    } else if (griddim == 3) {
      calc3(numerical::sinuscosinus<double>{}, argc, argv);
    }
     //} else if (v == "cy")  { calc            (cylinder{}, argc, argv);
    // } else if (v == "fw")  { calc        (FlappingWing{}, argc, argv);
  //} else if (v == "mg") {
  //  calc(movinggyre<double>{}, argc, argv);
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
  } else if (v == "bou") {
    std::cerr << "reading boussinesq... ";
    boussinesq v{dataset_dir + "/boussinesq.am"};
    std::cerr << "done!\n";
    if (griddim == 2) {
      calc2(v, argc, argv);
    } else if (griddim == 3) {
      calc3(v, argc, argv);
    }
  } else if (v == "cav") {
    std::cerr << "reading cavity... ";
    cavity v{};
    std::cerr << "done!\n";
    if (griddim == 2) {
      calc2(v, argc, argv);
    } else if (griddim == 3) {
      calc3(v, argc, argv);
    }
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
