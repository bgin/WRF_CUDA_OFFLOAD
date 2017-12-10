#ifndef _CUDA_COLUMN_IMPL_CUH_
#define _CUDA_COLUMN_IMPL_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuwrf_config.cuh"
#include <cassert>

// File version info.
#if !defined (CUDA_COLUMN_IMPL_VER_MAJOR)
#define CUDA_COLUMN_IMPL_VER_MAJOR 1
#endif

#if !defined (CUDA_COLUMN_IMPL_VER_MINOR)
#define CUDA_COLUMN_IMPL_VER_MINOR 0
#endif

#if !defined (CUDA_COLUMN_IMPL_VER_MICRO)
#define CUDA_COLUMN_IMPL_VER_MICRO 0
#endif

#if !defined (CUDA_COLUMN_IMPL_VER_FULL)
#define CUDA_COLUMN_IMPL_VER_FULL 1000
#endif

#if !defined (CUDA_COLUMN_IMPL_CREATION_DATE)
#define CUDA_COLUMN_IMPL_CREATION_DATE "24-06-2017 09:07 AM -00200, (Sun 24 Jun 2017 09:07 AM -00200)"
#endif

#if !defined (CUDA_COLUMN_IMPL_BUILD_DATE)
#define CUDA_COLUMN_IMPL_BUILD_DATE " "
#endif

#if !defined (CUDA_COLUMN_IMPL_AUTHOR)
#define CUDA_COLUMN_IMPL_AUTHOR "Programmer: Bernard Gingold, e-mail: beniekg@gmail.com"
#endif

#if !defined (CUDA_COLUMN_IMPL_SYNOPSIS)
#define CUDA_COLUMN_IMPL_SYNOPSIS "C-CUDA implementation of F90 ra_goddard:column subroutine."
#endif





 /*
	For convenience:
					static functions describing extended file info.
					Not callable from Fortran.
 */
	
    // Version major
	int cuda_column_impl_major();

	// Version minor
	int cuda_column_impl_minor();

	// Version micro
	int cuda_column_impl_micro();

	// Version full
	int cuda_column_impl_verfull();

	// Creation date
	const char * cuda_column_impl_createdate();

	// Build date
	const char * cuda_column_impl_buildate();

	// Author info
	const char * cuda_column_impl_author();

	// File synopsis
	const char * cuda_column_impl_synopsis();

#if FORTRAN_CALLABLE == 1

#if defined __cplusplus

extern "C" {

	 // File version major
	 void f90cuda_column_impl_major(int );

	 // File version minor
	 void f90cuda_column_impl_minor(int );

	 // File version micro
	 void f90cuda_column_impl_micro(int );

	 // File full version
	 void f90cuda_column_impl_fullver(int );

	 // File creation date
	 void f90cuda_column_impl_createdate( char *  );

	 // File build date
	 void f90cuda_column_impl_buildate( char * );

	 // File author info
	 void f90cuda_column_impl_author( char * );

	 // File synopsis
	 void f90cuda_column_impl_synopsis( char * );
}

#endif
#endif



#if defined __cplusplus
 extern "C" {
	
/*
!***********************************************************************
!-----compute column-integrated (from top of the model atmosphere)
!     absorber amount (sabs), absorber-weighted pressure (spre) and
!     temperature (stem).
!     computations follow eqs. (8.24) - (8.26).
!
!--- input parameters
!   number of soundings (m)
!   number of atmospheric layers (np)
!   layer pressure (pa)
!   layer temperature minus 250k (dt)
!   layer absorber amount (sabs0)
!
!--- output parameters
!   column-integrated absorber amount (sabs)
!   column absorber-weighted pressure (spre)
!   column absorber-weighted temperature (stem)
!
!--- units of pa and dt are mb and k, respectively.
!    units of sabs are g/cm**2 for water vapor and (cm-atm)stp
!    for co2 and o3
!***********************************************************************
*/
	void column(const int,const int,const double *,
			    const double *, const double *,
				double *, double *, double * );
}
#endif

    __global__ void column_kernel(const int, const int, const double * __restrict,
								  const double * __restrict, const double * __restrict,
								  double * __restrict, double * __restrict, double * __restrict);


#endif /*_CUDA_COLUMN_IMPL_CUH_*/