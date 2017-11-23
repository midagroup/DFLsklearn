!f2py   sd.f90 wrap_mixed.f90 problem1.f90 main_box.f90 -m dfl  -h dfl.pyf
!f2py -m dfl  -c sd.f90 wrap_mixed.f90 problem1.f90 main_box.f90
!============================================================================================
!    DFL - Derivative-Free Linesearch program for Mixed Integer Nonlinear Programming 
!    Copyright (C) 2011  G.Liuzzi, S.Lucidi, F.Rinaldi
!
!    This program is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    This program is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
!    G.Liuzzi, S.Lucidi, F.Rinaldi. Derivative-free methods for bound constrained 
!    mixed-integer optimization, Computational Optimization and Applications, 2011.
!    DOI: 10.1007/s10589-011-9405-3
!============================================================================================

!============================================================================================
!This Program has been modified by Vittorio Latorre and Federico Benvenuto and has been
!downloaded by github
!For the license plase check the license file
!============================================================================================


!-----------------------------------------------------------------------------!
!          Program DFL for bound-constrained mixed integer problems           !
!-----------------------------------------------------------------------------! 
!
!
!We solve the following problem:
!                                
!                            min f(x)
!                             l<=x<=u            
!                            
!                            l>-\infty
!                            u<\infty
!
!The file "parameter.f" contains the number of variables n
!
!
!-----------------------------------------------------------------------------!

!module mainvar
!end module mainvar

!subroutine py_interface()
!    integer :: ext_n

!    common /in/ext_n
!    call main_box_discr(ext_n)
!end subroutine py_interface


subroutine main_box_discr(funct_obj, ext_x, ext_bl, ext_bu, step, init_int_step, is_integer, ext_n,r)
!      use mainvar
      implicit none

!parameter.f: file containing the number of variables of the problem
!include 'parameter.f'
    EXTERNAL :: funct_obj
    REAL     :: funct_obj

!-----------------------------------------------------------------------------
    integer             :: ext_n, ext_nf_max,ext_iprint
    intent(in)          :: ext_n
    real*8              :: ext_x(ext_n),ext_bl(ext_n),ext_bu(ext_n)
    real*8              :: ext_alfa_stop

!      INTEGER, PARAMETER::  N=5
      integer   :: i, istop, icheck
      integer:: index_int(ext_n)
      real      :: tbegin, tend
      logical :: is_integer(ext_n)

      real*8 :: x(ext_n),bl(ext_n),bu(ext_n),scale_int(ext_n),step(ext_n),r(ext_n),init_int_step(ext_n)
      intent(in) :: ext_x,ext_bl,ext_bu
      intent(out) :: r


!-----------------------------------------------------------------------------

      integer ::            n,num_funct,num_iter
      real*8             :: f,alfamax,delta
      real*8               :: fob,fapp
      real*8               :: violiniz, violint, finiz, fex
      real*8             :: alfa_stop
      integer            :: nf_max,iprint



!    common /ext/ ext_nf_max,ext_iprint, ext_is_integer, ext_x,ext_bl,ext_bu, ext_alfa_stop
    common /ext/ ext_nf_max,ext_iprint, ext_alfa_stop




    n=ext_n
!     common /num/f
!     common /calfamax/alfamax
!------------------------------------------------------------------------------

!    allocate(x(n),bl(n),bu(n),scale_int(n),step(n),is_integer(n),index_int(n))
 	  call cpu_time(tbegin)




!-----------------------------------------------------------------------
!      Starting point and bound calculation
!-----------------------------------------------------------------------

      write(1,*) ' '

      write(*,*) "n:",n

!VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
!	  call inizpar(n,x,bl,bu)
!VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

        x=ext_x
        bl=ext_bl
        bu=ext_bu


!        write(*,*) ext_nf_max,ext_iprint, ext_alfa_stop
        write(*,*) x,bl,bu

 
	  do i=1,n
		write(1,*) ' x(',i,')=',x(i)
	  enddo


	  do i=1,n

         if((x(i).lt.bl(i)).or.(x(i).gt.bu(i))) then
		   write(*,*) ' Initial point not in box'
		   stop
		 endif
     
	  enddo

2002  format(2d20.10)






!-----------------------------------------------------------------------
!     show starting point info
!-----------------------------------------------------------------------


      index_int=0
	  scale_int= 0.d0
	 
      do i = 1,n
		if(is_integer(i)) then
		  index_int(i)=1
		  scale_int(i) = step(i) 
		endif
      enddo


!-----------------------------------------------------------------------
!     calculate starting point violation 
!-----------------------------------------------------------------------

    violiniz=0.0d0
    
	do i=1,n
       violiniz=max(violiniz,x(i)-bu(i),bl(i)-x(i)) 
	end do
    




	  	call funct(funct_obj,x,n,fob)


        finiz=fob
   
        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- '

        write(*,*) ' objective function at xo = ',fob
        write(1,*) ' objective function at xo = ',fob

!       ---- objective function value ----

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- '


!      ---- x(i) =  i-th variable value----

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- ' 		      
	    do i=1,n
		   write(*,*) 'xo(',i,') =',x(i)
		   write(1,*) 'xo(',i,') =',x(i)
        enddo








!------------------------------------

      
      call funct(funct_obj,x,n,fob)
      

	  write(*,*) 'fob = ',fob

	   num_funct   = 0 
	   num_iter    = 0



!      set minimum stepsize to 0.001
!	   alfa_stop=1.d-3
       alfa_stop=ext_alfa_stop
!      set maximum number of function evaluations
!  	   nf_max=20000
       nf_max=ext_nf_max
!      set output verbosity
!       iprint=0
       iprint=ext_iprint
!       write(*,*) 'step', step
	   open(78,file='solution.dat',status='replace')

	   
       call sd_box(n,index_int,scale_int,init_int_step,x,f,bl,bu,alfa_stop,nf_max,num_iter,num_funct,iprint,istop,funct_obj)

       
    
	 
     
	   call funct(funct_obj,x,n,fob)


      !-----------------------------------------------------------------------
      !     integrality constraints violation for x* (last solution found) 
      !-----------------------------------------------------------------------

		violint=0.0d0
    
		do i=1,n

		   if(is_integer(i)) then
		   		if((bl(i) > -1.d+10).and.(bu(i)) < 1.d+10) then
					     violint=max( violint,abs(x(i)-bl(i)-( floor( ( x(i)-bl(i))/step(i)+0.5d0 )*step(i) ) )  )
		  
				elseif(bl(i) > -1.d+10) then
					 violint=max( violint,abs(x(i)-bl(i)-( floor( ( x(i)-bl(i))/step(i)+0.5d0 )*step(i) ) )  )
		  
				elseif(bu(i) <  1.d+10) then
					violint=max( violint,abs(bu(i)-x(i)-( floor( (bu(i)-x(i))/step(i)+0.5d0 )*step(i) ) )  )
		  
				else
				   violint=max( violint,abs(x(i)-( floor( x(i)/step(i)+0.5d0 )*step(i) ) )  )
		  
				endif
			   
			endif   	   
		  
		end do


      !-----------------------------------------------------------------------

	   call cpu_time(tend)

       write(2,987) n,violiniz,finiz,fob,violint,num_funct,num_iter
 
 987 format(' & ', i3,' & ',es9.2,' & ',es14.7,' & ',es14.7,' & ',es9.2,' & ',i5,' & ',i5,'\\')

!	   close(77)
      
	   write(*,*) '------------------------------------------------------------------------------'     
	   if(istop.eq.1) then
         write(*,*)  ' END - stopping criterion satisfied '
	   endif
       if(istop.eq.2) then
         write(*,*)  ' END - maximum number of function calculation reached  =',nf_max
	   endif

       if(istop.eq.3) then
         write(*,*)  ' END -  maximum number of iterations reached =',nf_max
	   endif

       write(78,*) 'objective function=', fob

	   write(78,*) ''

	   write(78,*) 'variables'
	   do i=1,n
         write(78,*)' x(',i,')=',x(i)
       enddo
	   do i=1,n
         if((x(i)-bl(i)).le.1.d-24) write(78,*)' la variabile x(',i,') al suo limite inferiore'
		 if((bu(i)-x(i)).le.1.d-24) write(78,*)' la variabile x(',i,') al suo limite superiore'
       enddo
	   write(78,*) ''
       write(78,*) 'CPU time=', tend-tbegin

	   close(78)

	   write(*,*) ' total time:',tend-tbegin
	   write(*,*) '------------------------------------------------------------------------------'  

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- '

!       ---- fo = objective function value ----

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- '

        write(*,*) ' objective function = ',fob
        write(1,*) ' objective function = ',fob


!      ---- x(i) = i-th variable ----

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- ' 		      
	    do i=1,n
		   write(*,*) 'x(',i,') =',x(i)
		   write(1,*) 'x(',i,') =',x(i)
        enddo

!      ---- nftot = number of function evaluations ----

        write(*,*) ' ------------------------------------------------- '
        write(1,*) ' ------------------------------------------------- '

        write(*,*) ' number of function evaluations = ',num_funct 
        write(1,*) ' number of function evaluations = ',num_funct     		    
	   
	   do i=1,n
         if((x(i)-bl(i)).le.1.d-24) write(*,*)' variable x(',i,') is at lower bound'
		 if((bu(i)-x(i)).le.1.d-24) write(*,*)' variable x(',i,') is at upper bound'
         if((x(i)-bl(i)).le.1.d-24) write(1,*)' variable x(',i,') is at lower bound'
		 if((bu(i)-x(i)).le.1.d-24) write(1,*)' variable x(',i,') is at upper bound'
       enddo

       r=x
       return
!        deallocate(x,bl,bu,scale_int,step,is_integer,index_int)
end subroutine main_box_discr



subroutine funct(funct_obj,x,n,f)
    implicit none
    EXTERNAL :: funct_obj
    REAL     :: funct_obj

    integer         :: n
    intent(in)      :: n
    real*8          :: x(n), f,q
    intent(in)      :: x
    intent(out)     :: f

    f=0
    !write(*,*) funct_obj(n,x)
    f= funct_obj(n,x)
    !write(*,*) f
    return

end subroutine funct
