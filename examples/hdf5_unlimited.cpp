/************************************************************

  This example shows how to create and extend an unlimited
  dataset.  The program first writes integers to a dataset
  with dataspace dimensions of DIM0, then closes the
  file.  Next, it reopens the file, reads back the data,
  outputs it to the screen, extends the dataset, and writes
  new data to the extended portions of the dataset.  Finally
  it reopens the file again, reads back the data, and
  outputs it to the screen.

  This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <array>
#include <vector>
#include <iostream>

#include "hdf5.h"

#define FILE    "h5ex_d_unlimadd.h5"
#define DATASET "DS1"
#define DIM0    4
#define EDIM0   6
#define CHUNK0  4

int main(void) {
  hid_t   file, space, dset; /* Handles */
  herr_t  status;
  hsize_t dims[1] = {DIM0}, extdims[1] = {EDIM0}, maxdims[1], start[1],
          count[1];
  std::array<int, DIM0>  wdata;  /* Write buffer */
  std::array<int, EDIM0> wdata2; /* Write buffer for
                            extension */
  int              ndims, i;
  std::vector<int> rdata;

  /*
   * Initialize data.
   */
  for (i = 0; i < DIM0; i++)
    wdata[i] = i * i - i;

  /*
   * Create a new file using the default properties.
   */
  file = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Create dataspace with unlimited dimensions.
   */
  maxdims[0] = H5S_UNLIMITED;
  space      = H5Screate_simple(1, dims, maxdims);

  /*
   * Create the unlimited dataset.
   */
  dset = H5Dcreate(file, DATASET, H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT,
                   H5P_DEFAULT);

  /*
   * Write the data to the dataset.
   */
  status =
      H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wdata.data());

  /*
   * Close and release resources.
   */
  status = H5Dclose(dset);
  status = H5Sclose(space);
  status = H5Fclose(file);
  /**/
  /*
   * In this next section we read back the data, extend the dataset,
   * and write new data to the extended portions.
   */
  /**/
  /*
   * Open file and dataset using the default properties.
   */
  /*file = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);*/
  /*dset = H5Dopen(file, DATASET, H5P_DEFAULT);*/
  /**/
  /*
   * Get dataspace and allocate memory for read buffer.  This is a
   * two dimensional dataset so the dynamic allocation must be done
   * in steps.
   */
  /*space = H5Dget_space(dset);*/
  /*ndims = H5Sget_simple_extent_dims(space, dims, NULL);*/
  /**/
  /*rdata.resize(dims[0]);*/
  /**/
  /*
   * Read the data using the default properties.
   */
  /*status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,*/
  /*                 rdata.data());*/
  /**/
  /*
   * Output the data to the screen.
   */
  /*std::cout << "Dataset before extension:\n";*/
  /*for (i = 0; i < dims[0]; i++) {*/
  /*  std::cout << ' ' << rdata[i];*/
  /*}*/
  /*std::cout << std::endl;*/
  /**/
  /*status = H5Sclose(space);*/
  /**/
  /*
   * Extend the dataset.
   */
  /*status = H5Dset_extent(dset, extdims);*/
  /**/
  /*
   * Retrieve the dataspace for the newly extended dataset.
   */
  /*space = H5Dget_space(dset);*/
  /**/
  /*
   * Initialize data for writing to the extended dataset.
   */
  /*for (i = 0; i < EDIM0; i++)*/
  /*    wdata2[i] = i;*/
  /**/
  /*
   * Select the entire dataspace.
   */
  /*status = H5Sselect_all(space);*/
  /**/
  /*
   * Subtract a hyperslab reflecting the original dimensions from the
   * selection.  The selection now contains only the newly extended
   * portions of the dataset.
   */
  /*start[0] = 0;*/
  /*start[1] = 0;*/
  /*count[0] = dims[0];*/
  /*count[1] = dims[1];*/
  /*status =*/
  /*    H5Sselect_hyperslab(space, H5S_SELECT_NOTB, start, NULL, count, NULL);*/
  /**/
  /*
   * Write the data to the selected portion of the dataset.
   */
  /*status =*/
  /*    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, space, H5P_DEFAULT, wdata2.data());*/
  /**/
  /*
   * Close and release resources.
   */
  /*status = H5Dclose(dset);*/
  /*status = H5Sclose(space);*/
  /*status = H5Fclose(file);*/
  /**/
  /*
   * Now we simply read back the data and output it to the screen.
   */
  /**/
  /*
   * Open file and dataset using the default properties.
   */
  /*file = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);*/
  /*dset = H5Dopen(file, DATASET, H5P_DEFAULT);*/
  /**/
  /*
   * Get dataspace and allocate memory for the read buffer as before.
   */
  /*space    = H5Dget_space(dset);*/
  /*ndims    = H5Sget_simple_extent_dims(space, dims, NULL);*/
  /*rdata.resize(dims[0]);*/
  /**/
  /*
   * Read the data using the default properties.
   */
  /*status =*/
  /*    H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata.data());*/
  /**/
  /*
   * Output the data to the screen.
   */
  /*printf("\nDataset after extension:\n");*/
  /*for (i = 0; i < dims[0]; i++) {*/
  /*  printf(" %3d", rdata[i]);*/
  /*}*/
  /**/
  /*
   * Close and release resources.
   */
  /*status = H5Dclose(dset);*/
  /*status = H5Sclose(space);*/
  /*status = H5Fclose(file);*/

  return 0;
}
