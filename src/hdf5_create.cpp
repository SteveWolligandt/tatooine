/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 *  This example writes a dataset to a new HDF5 file.
 */
#include <iostream>
#include <tatooine/hdf5.h>
const std::string  FILE_NAME( "SDS.h5" );
const std::string  DATASET_NAME( "IntArray" );
const int   NX = 4;                    // dataset dimensions
const int   NY = 3;
const int   RANK = 2;

int main() {
  /*
   * Data initialization.
   */
  int i, j, k = 1;
  tatooine::dynamic_multidim_array<int> data{NX, NY};
  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {
      data(i, j) = k++;
    }
  }
  for (j = 0; j < NY; j++) {
    for (i = 0; i < NX; i++) {
      std::cout << data(i, NY - 1 - j) << ' ';
    }
    std::cout << '\n';
  }
  /*
   * 0 1 2 3 4 5
   * 1 2 3 4 5 6
   * 2 3 4 5 6 7
   * 3 4 5 6 7 8
   * 4 5 6 7 8 9
   */
  // Try block to detect exceptions raised by any of the calls inside it
  /*
   * Create a new file using H5F_ACC_TRUNC access,
   * default file creation properties, and default file
   * access properties.
   */
  tatooine::hdf5::file file{FILE_NAME, H5F_ACC_TRUNC};

  auto dataset = file.add_dataset<int>(DATASET_NAME, NX, NY);
  /*
   * Write the data to the dataset using default memory space, file
   * space, and transfer properties.
   */
  dataset.write(data.data_ptr());

  auto const chunk = dataset.read_chunk(std::vector<size_t>{1,1},std::vector<size_t> {2,2});

  for (j = 0; j < 2; j++) {
    for (i = 0; i < 2; i++) {
      std::cout << chunk(i, 2 - 1 - j) << ' ';
    }
    std::cout << '\n';
  }
  return 0;  // successfully terminated
}

