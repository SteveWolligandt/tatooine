#!/bin/bash
#Intel
#CXX_COMPILER=mpiicpc
#C_COMPILER=mpiicc
#Fortran_COMPILER=mpiifort

#GCC
CXX_COMPILER=mpicxx
C_COMPILER=mpicc
Fortran_COMPILER=mpif90

#  -DCMAKE_CXX_COMPILER=$CXX_COMPILER\
#  -DCMAKE_C_COMPILER=$C_COMPILER\
#  -DCMAKE_Fortran_COMPILER=$Fortran_COMPILER\
cmake \
  -Bbuild\
  -DTATOOINE_INCLUDE_MKL_LAPACKE=ON\
  -DBLAS_LIBRARIES=$EBROOTIMKL/mkl/lib/intel64/libmkl_rt.so\
  -DLAPACK_LIBRARIES=$EBROOTIMKL/mkl/lib/intel64/libmkl_rt.so\
  -DLAPACKE_INCLUDE_DIRS=$EBROOTIMKL/mkl/include\
  -Dvcode_DIR=/p/home/jusers/wolligandt1/juwels/vcode/build\
  -DTATOOINE_INCLUDE_MKL_LAPACKE=ON\
  .
