#include <H5Cpp.h>

#include <iostream>
#include <string>
#include <tatooine/hdf5.h>
//==============================================================================
std::string const file_name    = "SDS.h5";
std::string const dataset_name = "Array";
constexpr int     nx           = 100;  // data set dimensions
constexpr int     ny           = 100;

using namespace tatooine;
auto main() -> int {
  // Data initialization.
  double cnt = 1;
  double data[nx * ny];  // buffer for data to write
  for (size_t i = 0; i < nx * ny; ++i) {
    data[i] = cnt;
    cnt += 0.1;
  }

  // Try block to detect exceptions raised by any of the calls inside it
  try {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    H5::Exception::dontPrint();

    // Create a new file using H5F_ACC_TRUNC access,
    // default file creation properties, and default file
    // access properties.
    hdf5::file file{file_name, H5F_ACC_TRUNC};

    // Define the size of the array and create the data space for fixed
    // size data set.
    auto data_set = file.add_dataset<double>(dataset_name, nx, ny);
    //std::cerr << data_set.num_dimensions() << '\n';
    //std::cerr << data_set.size(0) << '\n';
    //std::cerr << data_set.size(1) << '\n';

    // Write the data to the data set using default memory space, file
    // space, and transfer properties.
    linspace data_src{0.0, 1.0, nx * ny};
    data_set.write(data_src);

    // read back data
    auto const r = data_set.read();
    std::cerr << r.num_dimensions() << '\n';
    std::cerr << r.size(0) << '\n';
    std::cerr << r.size(1) << '\n';
    std::cerr << r(0, 0) << ", " << r(1, 0) << ", ..., " << r(99, 99) << '\n';
    std::cerr << data_src[0] << ", " << data_src[1] << ", ..., "
              << data_src.back() << '\n';

    // read chunk
    auto const chunk = data_set.read_chunk(std::vector<size_t>{1, 1},
                                           std::vector<size_t>{3, 3});
    std::cerr << chunk.num_dimensions() << '\n';
    std::cerr << chunk.size(0) << '\n';
    std::cerr << chunk.size(1) << '\n';
    std::cerr << r(1, 1) << ", " << r(2, 1) << ", ..., " << r(3, 3) << '\n';
    std::cerr << chunk(0, 0) << ", " << chunk(1, 0) << ", ..., " << chunk(2, 2) << '\n';

    // read lazy
    auto lazy = data_set.read_lazy({2, 2});
    std::cerr << lazy(0, 0) << ", " << lazy(1, 0) << ", ..., " << lazy(99, 99) << '\n';
  }  // end of try block

  // catch failure caused by the H5File operations
  catch (H5::FileIException error) {
    error.printErrorStack();
    return -1;
  }

  // catch failure caused by the DataSet operations
  catch (H5::DataSetIException error) {
    std::cerr << "DataSetIException\n";
    error.printErrorStack();
    return -1;
  }

  // catch failure caused by the DataSpace operations
  catch (H5::DataSpaceIException error) {
    error.printErrorStack();
    return -1;
  }

  // catch failure caused by the DataSpace operations
  catch (H5::DataTypeIException error) {
    error.printErrorStack();
    return -1;
  }

}
