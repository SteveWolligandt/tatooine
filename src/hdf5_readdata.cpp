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
//
//      This example reads hyperslab from the SDS.h5 file into
//      two-dimensional plane of a three-dimensional array.  Various
//      information about the dataset in the SDS.h5 file is obtained.
//
#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;
#include <string>

#include "H5Cpp.h"
using namespace H5;
const H5std_string FILE_NAME("SDS.h5");
const H5std_string DATASET_NAME("IntArray");
const int          NX_SUB   = 3;  // hyperslab dimensions
const int          NY_SUB   = 2;
const int          RANK_OUT = 2;

int main() {
  /*
   * Output buffer initialization.
   */
  int i, j;
  int data_out[NY_SUB][NX_SUB]; /* output buffer */
  for (j = 0; j < NY_SUB; j++) {
    for (i = 0; i < NX_SUB; i++) {
      data_out[j][i] = 0;
    }
  }
  H5File  file(FILE_NAME, H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet(DATASET_NAME);

  DataSpace dataspace = dataset.getSpace();
  int       rank      = dataspace.getSimpleExtentNdims();
  hsize_t   dims_out[2];
  dataspace.getSimpleExtentDims(dims_out, NULL);
  cout << "rank " << rank << ", dimensions " << (unsigned long)(dims_out[0])
       << " x " << (unsigned long)(dims_out[1]) << endl;
  /*
   * Define hyperslab in the dataset; implicitly giving strike and
   * block NULL.
   */
  hsize_t offset[2];  // hyperslab offset in the file
  hsize_t count[2];   // size of the hyperslab in the file
  offset[0] = 0;
  offset[1] = 0;
  count[0]  = NY_SUB;
  count[1]  = NX_SUB;
  dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
  DataSpace memspace(RANK_OUT, count);
  dataset.read(data_out, PredType::NATIVE_INT, memspace, dataspace);
  for (j = 0; j < NY_SUB; j++) {
    for (i = 0; i < NX_SUB; i++) {
      cout << data_out[NY_SUB - 1 - j][i] << " ";
    }
    cout << endl;
    }
    /*
     * 0 0 0 0 0 0 0
     * 0 0 0 0 0 0 0
     * 0 0 0 0 0 0 0
     * 3 4 5 6 0 0 0
     * 4 5 6 7 0 0 0
     * 5 6 7 8 0 0 0
     * 0 0 0 0 0 0 0
     */
  //}  // end of try block
  //// catch failure caused by the H5File operations
  //catch (FileIException error) {
  //  std::cerr << "FileIException\n";
  //  return -1;
  //}
  //// catch failure caused by the DataSet operations
  //catch (DataSetIException error) {
  //  std::cerr << "DataIException\n";
  //  return -1;
  //}
  //// catch failure caused by the DataSpace operations
  //catch (DataSpaceIException error) {
  //  return -1;
  //}
  //// catch failure caused by the DataSpace operations
  //catch (DataTypeIException error) {
  //  std::cerr << "DataTypeIException\n";
  //  // error.printError();
  //  return -1;
  //}
  return 0;  // successfully terminated
}
