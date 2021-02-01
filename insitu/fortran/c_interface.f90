!===============================================================================
! These are the interfaces to the c functions
!===============================================================================
module c_interface
  interface
    !---------------------------------------------------------------------------
    subroutine hello_world() bind(C, name="hello_world")
      implicit none
    end subroutine
    !---------------------------------------------------------------------------
    function add(a, b) result(c) bind(C, name="add")
      use iso_c_binding
      implicit none
      integer(c_int), intent(in), value:: a, b
      integer(c_int):: c
    end function
    !---------------------------------------------------------------------------
    subroutine add_matrix(A, B, s1, s2)  bind(C, name="add_matrix")
      use iso_c_binding
      implicit none
      type(c_ptr), value:: A, B
      integer(c_int), intent(in), value:: s1, s2
    end subroutine
    !---------------------------------------------------------------------------
    subroutine multi_arr(arr) bind(C, name="multi_arr")
      use iso_c_binding
      implicit none
      type(c_ptr), value:: arr
    end subroutine
    !---------------------------------------------------------------------------
    subroutine print_str(str) bind(C, name="print_str")
      use iso_c_binding
      implicit none
      character(kind = c_char):: str(*)
    end subroutine
    !---------------------------------------------------------------------------
  end interface
end module
