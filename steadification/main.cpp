#include "start.h"
auto main(int argc, char** argv) -> int {
  yavin::context context{4, 5};
  using namespace tatooine;
  using namespace steadification;
  using namespace numerical;
  const std::string v = argv[1];
  if (v == "dg" || v == "doublegyre") {
    numerical::doublegyre<double> v;
    v.set_infinite_domain(false);
    calc(v, argc, argv);
  } else if (v == "la" || v == "laminar") {
    calc(laminar<double>{}, argc, argv);
  //} else if (v == "fdg") {
  //    calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  //} else if (v == "sc" || v == "sinuscosinus") {
  //  calc(numerical::sinuscosinus<double>{}, argc, argv);
    //} else if (v == "cy")  { calc            (cylinder{}, argc, argv);
    // } else if (v == "fw")  { calc        (FlappingWing{}, argc, argv);
    //} else if (v == "mg") {
    //  calc(movinggyre<double>{}, argc, argv);
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
  //} else if (v == "bou") {
  //  std::cerr << "reading boussinesq... ";
  //  boussinesq v{dataset_dir + "/boussinesq.am"};
  //  std::cerr << "done!\n";
  //  calc(v, argc, argv);
  //} else if (v == "cav") {
  //  std::cerr << "reading cavity... ";
  //  cavity v;
  //  std::cerr << "done!\n";
  //  calc(v, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
