program hello
  use, intrinsic:: iso_fortran_env, only: stderr => error_unit
  use iso_c_binding
  use c_interface
  use, intrinsic:: ieee_arithmetic
  !==============================================================================
  integer(c_int), dimension (:,:), allocatable, target:: A, B    
  integer(c_int), dimension (2, 2, 2, 2), target:: arr
  integer:: c, s1, s2     
  integer:: dino_err

  character(len = 6)  :: file_path = 'nml_in'
  integer           :: x, y
  real              :: r(2)
  integer                        :: fu, rc
  logical:: nml_logical
  real(kind = c_double), dimension(10):: nml_double_arr

  namelist/NML_HELLO_WORLD/ &
    nml_logical, nml_double_arr
  !==============================================================================
  do i = 1, 10
    nml_double_arr(i)=ieee_value(nml_double_arr(i), ieee_quiet_nan)
  end do
  inquire (file = file_path, iostat = rc)

  if (rc /= 0) then
      write (stderr, '(3a)') 'Error: input file "', trim(file_path), '" does not exist.'
      return
  end if

  ! Open and read Namelist file.
  open (action='read', file = file_path, iostat = rc, newunit = fu)
  read (nml = nml_hello_world, iostat = rc, unit = fu)

  if (rc /= 0) then
      write (stderr, '(a)') 'Error: invalid Namelist format.'
  end if

  close (fu)
  print *, nml_logical
  do i = 1, 10
    write(*,*) 'i =', nml_double_arr(i)
  end do
  !==============================================================================
  call hello_world
  !------------------------------------------------------------------------------
  c = add(1, 2)
  print *,c
  !------------------------------------------------------------------------------
  s1 = 3
  s2 = 4

  allocate(A(s1, s2))
  allocate(B(s1, s2))
  A(:,:) = 1
  B(:,:) = 2
  A(1, 2) = 5
  call add_matrix(c_loc(A), c_loc(B), s1, s2)
  deallocate (A, B)
  !------------------------------------------------------------------------------
  arr(1, 1, 1, 1) =  1; 
  arr(2, 1, 1, 1) =  2; 
  arr(1, 2, 1, 1) =  3; 
  arr(2, 2, 1, 1) =  4; 
  arr(1, 1, 2, 1) =  5; 
  arr(2, 1, 2, 1) =  6; 
  arr(1, 2, 2, 1) =  7; 
  arr(2, 2, 2, 1) =  8; 
  arr(1, 1, 1, 2) =  9; 
  arr(2, 1, 1, 2) = 10; 
  arr(1, 2, 1, 2) = 11; 
  arr(2, 2, 1, 2) = 12; 
  arr(1, 1, 2, 2) = 13; 
  arr(2, 1, 2, 2) = 14; 
  arr(1, 2, 2, 2) = 15; 
  arr(2, 2, 2, 2) = 16; 
  call multi_arr(c_loc(arr))
  !------------------------------------------------------------------------------
  call print_str(c_char_"abc"//c_null_char)
end program
