!    -*- f90 -*-
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

      subroutine sd_box(n,index_int,scale_int,init_int_step,x,f,bl,bu,alfa_stop,&
      nf_max,ni,nf,iprint,istop,funct_obj)
      implicit none
      EXTERNAL :: funct_obj
      REAL     :: funct_obj

	  logical :: cambio_eps
      integer :: n,i,j,i_corr,nf,ni,nf_max,index_int(n)
      integer :: num_fal,istop
      integer :: iprint,i_corr_fall
	  integer :: flag_fail(n)

      real*8 :: x(n),z(n),d(n)
      real*8 :: alfa_d(n),alfa,alfa_max, alfa_d_old
      real*8 :: f,fz , eta
	  real*8 :: bl(n),bu(n),alfa_stop,maxeps,scale_int(n),init_int_step(n)
	  logical:: discr_change





!     values of f calculated on a n+1 simplex

      real*8 :: fstop(n+1)


!     num_fal number of failures

!     i_corr is the index of the current direction


!     initialization

	  discr_change = .false. 

	  eta = 1.d-6

      flag_fail=0

	  num_fal=0

      istop = 0

      fstop=0.d0

!     ---- choice of the starting stepsizes along the directions --------

      do i=1,n

        if(index_int(i).eq.0) then
        
           alfa_d(i)=dmax1(1.d-3,dmin1(1.d0,dabs(x(i))))
      
           if(iprint.ge.1) then
              write(*,*) ' alpha_begin(',i,')=',alfa_d(i)
              write(1,*) ' alpha_begin(',i,')=',alfa_d(i)
           endif
		else

!		   alfa_d(i)=dmax1(scale_int(i),dmin1(2.d0*scale_int(i),&
!		   dabs(x(i))))
!VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
!This change in the original code is done to have an initial stepsize for integer variables
!different from the minimum step
           alfa_d(i)=dmax1(scale_int(i),dmin1(2.d0*scale_int(i),&
           dabs(x(i))),init_int_step(i))
!VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
           if(iprint.ge.1) then
              write(*,*) ' alpha_begin(',i,')=',alfa_d(i)
              write(1,*) ' alpha_begin(',i,')=',alfa_d(i)
           endif
		end if
      end do

      do i=1,n      
        d(i)=1.d0 
      end do
!     ---------------------------------------------------------------------  
     
      
      call funct(funct_obj,x,n,f)
      

	  nf=nf+1

	  i_corr=1

      fstop(i_corr)=f

      do i=1,n
	    z(i)=x(i)
      end do

      if(iprint.ge.2) then
        write(*,*) ' ----------------------------------'
        write(1,*) ' ----------------------------------'
        write(*,*) ' f_begin =',f
        write(1,*) ' f_begin =',f
        do i=1,n
          write(*,*) ' x_begin(',i,')=',x(i)
          write(1,*) ' x_begin(',i,')=',x(i)
        enddo
      endif

!---------------------------   
!     main loop
!---------------------------

      do 

         if(iprint.ge.0) then
         
           write(*,100) ni,nf,f,alfa_max
           write(1,100) ni,nf,f,alfa_max
100        format(' ni=',i4,'  nf=',i5,'   f=',d12.5,'  &
         alfamax=',d12.5)
         endif
         if(iprint.ge.2) then
	       do i=1,n
                write(*,*) ' x(',i,')=',x(i)
                write(1,*) ' x(',i,')=',x(i)
            enddo
         endif
!-------------------------------------
!    sampling along coordinate i_corr
!-------------------------------------
         if(index_int(i_corr).eq.0) then 
 
                call linesearchbox_cont(n,scale_int,x,f,d,alfa,alfa_d,&
           z,fz,i_corr,num_fal,alfa_max,i_corr_fall,iprint,bl,bu,ni,nf,funct_obj)

         else
                alfa_d_old=alfa_d(i_corr)
                call linesearchbox_discr(n,eta,index_int,scale_int,x,f,&
                d,alfa,alfa_d,z,fz,i_corr,num_fal,alfa_max,i_corr_fall,&
                iprint,bl,bu,ni,nf,discr_change,flag_fail,funct_obj)

         endif

         if(dabs(alfa).ge.1.d-12) then
		    
			flag_fail(i_corr)=0
		               
            x(i_corr) = x(i_corr)+alfa*d(i_corr)
            f=fz
 	        fstop(i_corr)=f
			     
            num_fal=0
            ni=ni+1
      
         else

			flag_fail(i_corr)=1

			if ((index_int(i_corr).eq.1).and.(alfa_d_old.gt.&
			scale_int(i_corr)))  flag_fail(i_corr)=0

	        if(i_corr_fall.lt.2) then 

		      fstop(i_corr)=fz         

              num_fal=num_fal+1
              ni=ni+1

	        endif

	     end if

		 z(i_corr) = x(i_corr)

         if(i_corr.lt.n) then
            i_corr=i_corr+1
         else
		    if(.not.discr_change) then
				do i = 1,n
					if((index_int(i).eq.1).and.(alfa_d(i) > 1)) then
						discr_change = .true.
						exit
					endif
				enddo
				if(.not.discr_change) then
					eta = eta/2.d0
				endif
			endif
            i_corr=1
	   	    discr_change = .false. 
         end if 

         call stop(n,index_int,scale_int,alfa_d,istop,alfa_max,nf,ni,&
         fstop,f,alfa_stop,nf_max,flag_fail)

         if (istop.ge.1) exit


      enddo
      return
    


      end
        


!     #######################################################

      subroutine stop(n,index_int,scale_int, alfa_d,istop,alfa_max,nf,&
      ni,fstop,f,alfa_stop,nf_max, flag_fail)
      implicit none
      
      integer :: n,istop,i,nf,ni,nf_max
	  integer :: index_int(n), flag_fail(n)

      real*8 :: alfa_d(n),alfa_max,fstop(n+1),ffstop,ffm,f,alfa_stop
	  real*8 :: scale_int(n) 

	  logical :: test

      istop=0

      alfa_max=0.0d0


      do i=1,n				 
	    if (index_int(i).eq.0) then 
          if(alfa_d(i).gt.alfa_max) then
            alfa_max=alfa_d(i)
          end if
		end if
      end do
     

      if(ni.ge.(n+1)) then
        ffm=f
        do i=1,n
          ffm=ffm+fstop(i)
        enddo
        ffm=ffm/dfloat((n+1))

        ffstop=(f-ffm)*(f-ffm)
        do i=1,n
           ffstop=ffstop+(fstop(i)-ffm)*(fstop(i)-ffm)
        enddo
 
        ffstop=dsqrt(ffstop/dfloat(n+1))



	  endif


      !write(*,*) "alfa_max", alfa_max, alfa_stop
      if(alfa_max.le.alfa_stop) then
	    test=.true.
		do i=1,n
		!write(*,*) test, index_int
         if (index_int(i).eq.1) then 
        
		  if((alfa_d(i).ne.scale_int(i)).or.(flag_fail(i).ne.1)) then
		    test=.false.
	      end if
        
		 end if
	  

		end do
        if (test.eqv..true.) then
		   istop = 1
		end if
        
	  end if
      


      if(nf.gt.nf_max) then
        istop = 2
      end if

      if(ni.gt.nf_max) then
        istop = 3
      end if

      return

      end




!     *********************************************************
!     *         
!     *                 Continuous Linesearch
!     *
!     *********************************************************
           
 
      subroutine linesearchbox_cont(n,scale_int,x,f,d,alfa,alfa_d,z,fz,&
      i_corr,num_fal,alfa_max,i_corr_fall,iprint,bl,bu,ni,nf,funct_obj)
      
      implicit none
      EXTERNAL :: funct_obj
      REAL     :: funct_obj

      integer :: n,i_corr,nf
      integer :: i,j
      integer :: ni,num_fal
      integer :: iprint,i_corr_fall
	  integer :: ifront,ielle
      real*8 :: x(n),d(n),alfa_d(n),z(n),bl(n),bu(n),scale_int(n)
      real*8 :: f,alfa,alfa_max,alfaex, fz,gamma, gamma_int
      real*8 :: delta,delta1,fpar,fzdelta

	  
	  gamma=1.d-6      !-6

      delta =0.5d0
      delta1 =0.5d0

      i_corr_fall=0

	  ifront=0

!     index of current direction

      j=i_corr

	  if(iprint.ge.1) then
			write(*,*) 'continuous variable  j =',j,'    d(j) =',d(j),' alfa=',alfa_d(j)
			write(1,*) 'continuous variable  j =',j,'    d(j) =',d(j),'&
			 alfa=',alfa_d(j)
	  endif


	  if(dabs(alfa_d(j)).le.1.d-3*dmin1(1.d0,alfa_max)) then
			alfa=0.d0
			if(iprint.ge.1) then
				 write(*,*) '  small alpha'
				 write(1,*) '  small alpha'
				 write(*,*) ' alfa_d(j)=',alfa_d(j),'    alfamax=',alfa_max
				 write(1,*) ' alfa_d(j)=',alfa_d(j),'  &
				   alfamax=',alfa_max
			endif
			return
	  endif
      
!     choice of the direction

	  do ielle=1,2

		 if(d(j).gt.0.d0) then

		     if((alfa_d(j)-(bu(j)-x(j))).lt.(-1.d-6)) then                 
   			    alfa=dmax1(1.d-24,alfa_d(j))
			 else
			    alfa=bu(j)-x(j)
				ifront=1
				if(iprint.ge.1) then
					   write(*,*) ' point on the boundary. *'
					   write(1,*) ' point on the boundary. *'
				endif
			 endif

		  else

			 if((alfa_d(j)-(x(j)-bl(j))).lt.(-1.d-6)) then
			    alfa=dmax1(1.d-24,alfa_d(j))
			 else
				alfa=x(j)-bl(j)
				ifront=1
				if(iprint.ge.1) then
					   write(*,*) ' expansion point on the boundary *'
					   write(1,*) ' expansion point on the boundary *'
				endif
			 endif

		  endif

		  if(dabs(alfa).le.1.d-3*dmin1(1.d0,alfa_max)) then
  
			 d(j)=-d(j)
			 i_corr_fall=i_corr_fall+1
			 ifront=0

			 if(iprint.ge.1) then
				   write(*,*) ' opposite direction for small alpha'
				   write(1,*) ' opposite direction for small alpha'
				   write(*,*) ' j =',j,'    d(j) =',d(j)
				   write(1,*) ' j =',j,'    d(j) =',d(j)
				   write(*,*) ' alfa=',alfa,'    alfamax=',alfa_max
				   write(1,*) ' alfa=',alfa,'    alfamax=',alfa_max
			  endif
			  alfa=0.d0
			  cycle

		  endif

		  alfaex=alfa

		  z(j) = x(j)+alfa*d(j)
    
		 
	      call funct(funct_obj,z,n,fz)
		  

		  nf=nf+1

		  if(iprint.ge.1) then
				write(*,*) ' fz =',fz,'   alfa =',alfa
				write(1,*) ' fz =',fz,'   alfa =',alfa
		  endif
		  if(iprint.ge.2) then
			  do i=1,n
				  write(*,*) ' z(',i,')=',z(i)
				  write(1,*) ' z(',i,')=',z(i)
			  enddo
		  endif

		  fpar= f-gamma*alfa*alfa

!         test on the direction

		  if(fz.lt.fpar) then

!         expansion step

			 do

		   	   if((ifront.eq.1)) then

			         if(iprint.ge.1) then
				         write(*,*) ' accept point on the boundary fz =',fz,'   alpha =',alfa
				         write(1,*) ' accept point on the boundary &
				         fz =',fz,'   alpha =',alfa
			         endif
				     alfa_d(j)=delta*alfa

				     return

				 end if

				 if(d(j).gt.0.d0) then
							
					 if((alfa/delta1-(bu(j)-x(j))).lt.(-1.d-6)) then
						 alfaex=alfa/delta1
					 else
						 alfaex=bu(j)-x(j)
						 ifront=1
						 if(iprint.ge.1) then
							write(*,*) ' expansion point on the boundary'
							write(1,*) ' expansion point on the boundary'
						 endif
					 end if

				 else

					 if((alfa/delta1-(x(j)-bl(j))).lt.(-1.d-6)) then
						 alfaex=alfa/delta1
					 else
						 alfaex=x(j)-bl(j)
						 ifront=1
						 if(iprint.ge.1) then
							write(*,*) ' expansion point on the boundary'
							write(1,*) ' expansion point on the boundary'
						 endif
					 end if

				 endif
						 
				 z(j) = x(j)+alfaex*d(j) 
				   
     
				
			     call funct(funct_obj,z,n,fzdelta)
							      
				
				 nf=nf+1

				 if(iprint.ge.1) then
					  write(*,*) ' fzex=',fzdelta,'  alfaex=',alfaex
					  write(1,*) ' fzex=',fzdelta,'  alfaex=',alfaex
				 endif
				 if(iprint.ge.2) then
					  do i=1,n
						 write(*,*) ' z(',i,')=',z(i)
						 write(1,*) ' z(',i,')=',z(i)
					  enddo
				 endif

				 fpar= f-gamma*alfaex*alfaex

				 if(fzdelta.lt.fpar) then

					 fz=fzdelta
					 alfa=alfaex

				 else               
					 alfa_d(j)=delta*alfa
			         if(iprint.ge.1) then
				         write(*,*) ' accept point fz =',fz,'   alfa =',alfa
				         write(1,*) ' accept point fz =',fz,'&
				            alfa =',alfa
			         endif
					 return
				 end if

		     enddo 

		  else   !opposite direction    

			 d(j)=-d(j)
			 ifront=0

			 if(iprint.ge.1) then
				   write(*,*) ' opposite direction'
				   write(1,*) ' opposite direction'
				   write(*,*) ' j =',j,'    d(j) =',d(j)
				   write(1,*) ' j =',j,'    d(j) =',d(j)
			 endif

		  endif       ! test on the direction
			  
	  enddo       

	  if(i_corr_fall.eq.2) then
			 alfa_d(j)=alfa_d(j)
	  else
			 alfa_d(j)=delta*alfa_d(j)
	  end if

	  alfa=0.d0

	  if(iprint.ge.1) then
			write(*,*) ' failure along the direction'
			write(1,*) ' failure along the direction'
	  endif

	  return      
	  
      end

!     *********************************************************
!     *         
!     *         Linesearch along the discrete variables
!     *
!     ********************************************************
           
 
      subroutine linesearchbox_discr(n,eta,index_int,scale_int,x,f,d,&
      alfa,alfa_d,z,fz,i_corr,num_fal,alfa_max,i_corr_fall,iprint,bl,&
      bu,ni,nf,discr_change,flag_fail,funct_obj)
      
      implicit none

      EXTERNAL :: funct_obj
      REAL     :: funct_obj

      integer :: n,i_corr,nf
      integer :: i,j
      integer :: ni,num_fal
      integer :: iprint,i_corr_fall
	  integer :: ifront,ielle
	  integer :: index_int(n),flag_fail(n)
      real*8 :: x(n),d(n),alfa_d(n),z(n),bl(n),bu(n),scale_int(n)
      real*8 :: f,alfa,alfa_max,alfaex, fz,gamma, gamma_int,eta
      real*8 :: delta,delta1,fpar,fzdelta
	  logical:: discr_change, test

	  
      gamma_int=1.d-0 

      delta =0.5d0
      delta1 =0.5d0

      i_corr_fall=0

	  ifront=0

!     index of the current direction

      j=i_corr


	  if(iprint.ge.1) then
			   write(*,*) 'discrete variable  j =',j,'    d(j) =',d(j),' alpha=',alfa_d(j)
			   write(1,*) 'discrete variable  j =',j,'    &
			   d(j) =',d(j),' alpha=',alfa_d(j)
	  endif

	  test=.true.

      if(alfa_d(i_corr).eq.scale_int(i_corr)) then  
        
		  do i=1,n
		  
		    if((flag_fail(i).eq.0)) then

		      test=.false.
			  exit

	        end if

		  enddo

          if(test) then

		     alfa=0.d0

		     if(iprint.ge.1) then
			    write(*,*) ' direction already analyzed'
			    write(1,*) ' direction already analyzed'
		     endif

             return
          endif
                  
	   end if
      
	   do ielle=1,2

		   if(d(j).gt.0.d0) then

				if( ( ( bu(j)-x(j)-alfa_d(j) ) ).lt.0.d0 ) then  
				   alfa=      bu(j)-x(j)
				   ifront=1
				   if (alfa.eq.0.d0) then 
				      d(j)=-d(j)
					  ifront=0
				      cycle
				   endif
                else
				   alfa=alfa_d(j)
				end if		

		   else

				if( ((x(j)-alfa_d(j)-bl(j))).lt.0.0d0 ) then
				   alfa=      x(j)-bl(j)
				   if(iprint .gt. 1) then
					   !write(*,*) 'alfa =',alfa
				   endif
				   ifront=1
				   if (alfa.eq.0.d0) then
				      d(j)=-d(j)
					  ifront=0
				      cycle
				   endif
				 else
				   alfa=alfa_d(j)
				endif

		   endif

           alfaex=alfa

		   z(j) = x(j)+alfa*d(j)
    
		   
		   call funct(funct_obj,z,n,fz)
		   

		   nf=nf+1

		   if(iprint.ge.1) then
				write(*,*) ' fz =',fz,'   alpha =',alfa
				write(1,*) ' fz =',fz,'   alpha =',alfa
		   endif
		   if(iprint.ge.2) then
			   do i=1,n
				  write(*,*) ' z(',i,')=',z(i)
				  write(1,*) ' z(',i,')=',z(i)
			   enddo
		   endif

		   fpar= f-gamma_int*eta

!          test on the direction

		   if(fz.lt.fpar) then
			  
			  discr_change = .true.

!             expansion step

			  do 
                  if(ifront.eq.1) then 

                     if(iprint.ge.1) then
				              write(*,*) ' accept point on the boundary fz =',fz,'   alpha =',alfa
				              write(1,*) ' accept point on the boundary fz =',fz,'   alpha =',alfa
			         endif

				     return

				  endif

				  if(d(j).gt.0.d0) then
							
					 if((bu(j)-x(j)-2.0d0*alfa ).lt.(0.0d0)) then

					    alfaex=      bu(j)-x(j)
				        ifront=1

				        if (alfaex.le.alfa) then
						 
                           alfa_d(j)=max(scale_int(j),max(dble&
           (floor((alfa/2.0d0)/scale_int(j)+0.5d0)),1.d0)*scale_int(j))

			               if(iprint.ge.1) then
				              write(*,*) ' accept point on the boundary fz =',fz,'   alpha =',alfa
				              write(1,*) ' accept point on the boundary fz =',fz,'   alpha =',alfa
			               endif

						   return

					    endif
						 
					  else

					     alfaex=alfa*2.0d0						
					
					  end if

				   else

					  if(( x(j)-2.0d0*alfa-bl(j) ).lt.(0.0d0)) then

					      alfaex=      x(j)-bl(j)
						  ifront=1

						  if (alfaex.le.alfa) then
						 
						     alfa_d(j)=max(scale_int(j),max(dble&
         (floor((alfa/2.0d0)/scale_int(j)+0.5d0)),1.d0)*scale_int(j))

			                 if(iprint.ge.1) then
				               write(*,*) ' accept point on the boundary fz =',fz,'   alfa =',alfa
				               write(1,*) ' accept point on the boundary fz =',fz,'   alfa =',alfa
			                 endif

						     return

						  endif
						 
					  else

					      alfaex=alfa*2.0d0						
					
					  end if

				   endif
						 
				   z(j) = x(j)+alfaex*d(j) 
				   
     
				  
				   call funct(funct_obj,z,n,fzdelta)
				   			      
				
				   nf=nf+1

				   if(iprint.ge.1) then
					  write(*,*) ' fzex=',fzdelta,'  alphaex=',alfaex
					  write(1,*) ' fzex=',fzdelta,'  alphaex=',alfaex
				   endif
				   if(iprint.ge.2) then
					  do i=1,n
						 write(*,*) ' z(',i,')=',z(i)
						 write(1,*) ' z(',i,')=',z(i)
					  enddo
				   endif

				   fpar= f-gamma_int*eta

				   if(fzdelta.lt.fpar) then

					  fz=fzdelta
					  alfa=alfaex

				   else               
					   alfa_d(j)=max(scale_int(j),max(dble(&
           floor((alfa/2.0d0)/scale_int(j)+0.5d0)),1.d0)*scale_int(j))
			           if(iprint.ge.1) then
				         write(*,*) ' accept point  fz =',fz,'   alpha =',alfa
				         write(1,*) ' accept point  fz =',fz,' &
				           alpha =',alfa
			           endif

					  return
				   end if

				enddo

			 else 

				d(j)=-d(j)
				ifront=0

				if(iprint.ge.1) then
				   write(*,*) ' opposite direction'
				   write(1,*) ' opposite direction'
				   write(*,*) ' j =',j,'    d(j) =',d(j)
				   write(1,*) ' j =',j,'    d(j) =',d(j)
				endif

			 endif
			  
		  enddo

		  alfa_d(j)=max(scale_int(j),max(dble(&
		  floor((alfa/2.0d0)/scale_int(j)+0.5d0)),1.d0)*scale_int(j))

		  alfa=0.d0
		  if(iprint.ge.1) then
			  write(*,*) ' direction failure'
			  write(1,*) ' direction failure'
		  endif

		  return
	  	     
      end


