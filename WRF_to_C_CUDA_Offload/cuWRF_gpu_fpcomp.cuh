
#ifndef __CUWRF_GPU_FPCOMP_CUH__
#define __CUWRF_GPU_FPCOMP_CUH__

#if !defined (CUWRF_GPU_FPCOMP_MAJOR)
#define CUWRF_GPU_FPCOMP_MAJOR 1
#endif

#if !defined (CUWRF_GPU_FPCOMP_MINOR)
#define CUWRF_GPU_FPCOMP_MINOR 0 
#endif

#if !defined (CUWRF_GPU_FPCOMP_MICRO)
#define CUWRF_GPU_FPCOMP_MICRO 0
#endif

#if !defined (CUWRF_GPU_FPCOMP_FULLVER)
#define CUWRF_GPU_FPCOMP_FULLVER 1000
#endif

#if !defined (CUWRF_GPU_FPCOMP_CREATE_DATE)
#define CUWRF_GPU_FPCOMP_CREATE_DATE "15-11-2017 16:10 +00200 (WED 15 NOV 2017 GMT+2)"
#endif

#if !defined (CUWRF_GPU_FPCOMP_BUILD_DATE)
#define CUWRF_GPU_FPCOMP_BUILD_DATE " "
#endif

#if !defined (CUWRF_GPU_FPCOMP_AUTHOR)
#define CUWRF_GPU_FPCOMP_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_GPU_FPCOMP_DESCRIPT)
#define CUWRF_GPU_FPCOMP_DESCRIPT "Floating-point safe comparison operations gpu version __device__"
#endif

#include <cstdint>
#include "cuda_runtime.h"
#include "cuwrf_config.cuh"

//
// Locals static constants
//

static const float gRADIX2f32   = 2.f;

static const double gRADIX2F64  = 2.0;

static const float gZEROF32     = 0.F;

static const float gONEF32      = 1.F;

static const float gTENF32      = 10.F;

static const float gHUNDREDF32  = 100.F;

static const float gEPSF32      = 1.0E-15F;

static const float  gSRSF32      = 5.9604645E-8F;

static const float  gLRSF32      = 1.1920929E-7F;

static const double gZEROF64    = 0.0;

static const double gONEF64     = 1.0;

static const double gTENF64     = 10.0; 

static const double gHUNDREDF64  = 100.0;

static const double gEPSF64      = 1.0E-15;

static const float  gSRSF32      = 5.9604645E-8F;

static const float  gLRSF32      = 1.1920929E-7F;

static const double gSRSF64      = 1.110223024625157E-16;

static const double gLRSF64      = 2.220446049250313E-16;

static const uint32_t gEXPF32    = 24;

static const uint32_t gEXPF64    = 53;


// helper function for spacaing computation
// Spacing (single precision value) = b^e-p

__device__ float gpu_float_spacing(const float, int32_t * ); 

// Helper function for spacing computation
// Spacing (double precision) computed by
// frexp (single precision) possibly unsafe.



// Scalar functions for equality computation.

__device__ bool   gpu_equal_to_float(const float, const float);



// Scalar functions for greater than '>' computation

__device__ bool   gpu_greater_than_float(const float, const float);



// Scalar functions for less than '<' computation

__device__ bool   gpu_less_than_float(const float, const float);



// Functions for floating-point number comparison.
 // Varying adjustable tolerance is used.

__device__ bool  gpu_compare_to_float(const float, const float,
								   const uint32_t  );

#if 0 // To be implemented later

// Computes tolerance value for single precision
 // and double precision values (numbers)

__device__ float gpu_float_tolerance(const float, int32_t);



// Compare if value lies within specific tolerance.

__device__ bool gpu_float_within_tol(const float, const float,
								  const int32_t
#if FORTRAN_OPTIONAL == 1
											 ,
								  const float
#endif
								  );


#endif





#endif /*__CUWRF_GPU_FPCOMP_CUH__*/