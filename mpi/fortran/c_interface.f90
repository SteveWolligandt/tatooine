module c_interface
  !==============================================================================
  ! These are the interfaces to the c functions
  !==============================================================================
  interface
    subroutine hello_world() bind(C, name="hello_world")
      implicit none
    end subroutine hello_world
    !----------------------------------------------------------------------------
    function add(a, b) result(c) bind(C, name="add")
      use iso_c_binding
      implicit none
      integer(c_int), intent(in), value:: a, b
      integer(c_int):: c
    end function add
    !----------------------------------------------------------------------------
    subroutine add_matrix(A, B, s1, s2)  bind(C, name="add_matrix")
      use iso_c_binding
      implicit none
      integer(c_int), intent(in), dimension(:,:):: A, B
      integer(c_int), intent(in), value:: s1, s2
    end subroutine add_matrix
    !----------------------------------------------------------------------------
    subroutine multi_arr(arr) bind(C, name="multi_arr")
      use iso_c_binding
      implicit none
      integer(c_int), intent(in), dimension(:, :, :, :):: arr
    end subroutine multi_arr
  end interface
end module c_interface
