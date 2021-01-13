program hello
  use iso_c_binding, only: c_int
  implicit none
  integer(c_int), dimension (:,:), allocatable:: A, B    
  integer:: c, s1, s2     

  ! These are the interfaces to the c functions
  interface
    subroutine hello_world() bind(C, name="hello_world")
      implicit none
    end subroutine hello_world

    function add(a, b) result(c) bind(C, name="add")
      use iso_c_binding, only: c_int
      implicit none
      integer(c_int), VALUE:: a
      integer(c_int), VALUE:: b
      integer(c_int):: c
    end function add

    subroutine add_matrix(a, b, s1, s2)  bind(C, name="add_matrix")
      use iso_c_binding, only: c_int
      implicit none
      integer(c_int), dimension(5, 5):: a
      integer(c_int), dimension(5, 5):: b
      integer(c_int), value:: s1
      integer(c_int), value:: s2
    end subroutine add_matrix
  end interface

  ! now the functions can be called
  call hello_world()

  c = add(1, 2)
  print *,c

  print*, "Enter the size of the array:"     
  read*, s1, s2      
  
  allocate(A(s1, s2))
  allocate(B(s1, s2))
  A(:,:) = 1
  B(:,:) = 2 
  A(1, 2) = 5
  call add_matrix(A, B, s1, s2)
  deallocate (A)
  deallocate (B)
end program
