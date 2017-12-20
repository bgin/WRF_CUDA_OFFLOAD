#ifndef _CUWRF_FPCOMP_H_
#define _CUWRF_FPCOM_H_

#if !defined (CUWRF_FPCOMP_MAJOR)
#define CUWRF_FPCOMP_MAJOR 1
#endif

#if !defined (CUWRF_FPCOMP_MINOR)
#define CUWRF_FPCOMP_MINOR 0
#endif

#if !defined (CUWRF_FPCOMP_MICRO)
#define CUWRF_FPCOMP_MICRO 0
#endif

#if !defined (CUWRF_FPCOMP_FULLVER)
#define CUWRF_FPCOMP_FULLVER 1000
#endif

#if !defined (CUWRF_FPCOMP_CREATE_DATE)
#define CUWRF_FPCOMP_CREATE_DATE "09-11-2017 12:37 +00200 (Thr, 09 NOV 2017 GMT+2) "
#endif
//
//	Set this value after successful compilation
//
#if !defined (CUWRF_FPCOMP_BUILD_DATE)
#define CUWRF_FPCOMP_BUILD_DATE " "
#endif

#if !defined (CUWRF_FPCOMP_AUTHOR)
#define CUWRF_FPCOMP_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_FPCOMP_DESCRIPT)
#define CUWRF_FPCOMP_DESCRIPT "Floating-point safe comparison operations."
#endif

/*
	Based on Fortran module from comGSIv3.5_EnKFv1.1 library
	Original author:
					  Paul van Delst, 01-Apr-2003
					  paul.vandelst@noaa.gov
	Modified and ported to C by Bernard Gingold, 09-11-2017

*/

#include "cuwrf_config.cuh"
#include <cstdint>

typedef struct FP4_COMP_METADATA {

	char cfname[256]; // name of tested cpu function
	char gfname[256]; // name of tested gpu(kernel) function
	
	uint32_t * __restrict cloop_idx; // array of loop indices
								  // of cpu-running code.
								  // This array stores indices
								  // floating-point failed comparision
							      // CPU and GPU,
	uint32_t * __restrict gloop_idx; // Same as above stores indices
								  // of GPU failed values.
#if (GPU_LARGE_MEM_SPACE) == 1

	uint64_t * __restrict cloopl_idx;

	uint64_t * __restrict gloopl_idx;

#endif

	REAL4 * __restrict cpu_valid;	// CPU-based code valid values

	REAL4* __restrict gpu_invalid;   // GPU-based code invalid values

	bool is_allocated;
};

typedef struct FP8_COMP_METADATA {

	char cfname[256];// name of tested cpu function
	char gfname[256];// name of tested gpu(kernel) function
	uint32_t * __restrict cloop_idx; // array of loop indices
								  // of cpu-running code.
								  // This array stores indices
								  // floating-point failed comparision
							      // CPU and GPU,


	uint32_t * __restrict gloop_idx; // Same as above stores indices
								  // of GPU failed values.

#if (GPU_LARGE_MEM_SPACE) == 1

	uint64_t * __restrict cloopl_idx;

	uint64_t * __restrict gloopl_idx;

#endif

	REAL8 * __restrict cpu_valid; // CPU-based code valid values

	REAL8 * __restrict gpu_invalid; // GPU-based code invalid values

	bool is_allocated;
};


//
// Locals static constants
//

static const float RADIX2f32   = 2.f;

static const double RADIX2F64  = 2.0;

static const float ZEROF32     = 0.F;

static const float ONEF32      = 1.F;

static const float TENF32      = 10.F;

static const float HUNDREDF32  = 100.F;

static const float EPSF32      = 1.0E-15F;

static const float  SRSF32      = 5.9604645E-8F;

static const float  LRSF32      = 1.1920929E-7F;

static const double ZEROF64    = 0.0;

static const double ONEF64     = 1.0;

static const double TENF64     = 10.0; 

static const double HUNDREDF64  = 100.0;

static const double EPSF64      = 1.0E-15;

static const float  SRSF32      = 5.9604645E-8F;

static const float  LRSF32      = 1.1920929E-7F;

static const double SRSF64      = 1.110223024625157E-16;

static const double LRSF64      = 2.220446049250313E-16;

static const uint32_t EXPF32    = 24;

static const uint32_t EXPF64    = 53;

#if !defined (MAX)
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

#if !defined (MIN)
#define MIN(x,y) (((x < (y)) ? (y) : (x))
#endif




//
// Functions decalarations
//

// helper function for spcaing computation
// Spacing (single precision value) = b^e-p
 static float float_spacing(const float,int32_t *);

 static double double_spacing(const double, int32_t *); 


// Scalar functions for equality computation.

 bool is_equal_to_float(const float , const float );

 bool is_equal_to_double(const double , const double);

// Scalar functions for greater than '>' computation

 bool is_greater_than_float(const float, const float);

 bool is_greater_than_double(const double , const double);

// Scalar functions for less than '<' computation

 bool is_less_than_float(const float, const float);

 bool is_less_than_double(const double, const double);

 // Functions for floating-point number comparison.
 // Varying adjustable tolerance is used.

 bool compare_float(const float, const float, 
				  const uint32_t);

 bool compare_double(const double, const double,

				   const uint32_t);

 // Computes tolerance value for single precision
 // and double precision values (numbers)

 float float_tolerance(const float, int32_t);

 double double_tolerance(const double, int32_t);

 // Compare if value lies within specific tolerance

 bool compare_float_within_tol(const float, const float,
						    const int32_t
#if FORTRAN_OPTIONAL == 1
							, 

							const float
#endif
							);

 bool compare_double_within_tol(const double, const double,
							 const int32_t
#if FORTRAN_OPTIONAL == 1
							 , 

							 const double
#endif
							 );


//
//	Medata-data struct init and destroy routines
//

void init_fp4_comp_metadata(struct FP4_COMP_METADATA *,
						 const size_t,
						 int32_t *);

void destroy_fp4_comp_metadata(struct FP4_COMP_METADATA *,
							 int32_t * );

void init_fp8_comp_metadata(struct FP8_COMP_METADATA *,
						 const size_t,
						 int32_t *);

void destroy_fp8_comp_metadata(struct FP8_COMP_METADATA *,
							int32_t * );

// Compare between GPU and CPU.

// Comparing arrays 1D.
void compare_gpu_cpu_real4(const REAL4 * __restrict,
						 const REAL4 * __restrict,
						 const int32_t,
						 struct FP4_COMP_METADATA *,
						 const uint32_t,
						  int32_t * ,
						  const char[256],
						  const char[256],
						  const uint32_t,
						  const int32_t,
						  const REAL4);

//
// Comparing arrays 2D
//
void compare_gpu_cpu_real4(const REAL4 * __restrict,
						 const REAL4 * __restrict,
						 const int32_t,
						 const int32_t,
						 struct FP4_COMP_METADATA * ,
						 const uint32_t,
						 int32_t * ,
						 const char[256],
						 const char[256],
						 const uint32_t,
						 const int32_t,
						 const REAL4);

//
// Comparing arrays 3D
//
void compare_gpu_cpu_real4(const REAL4 * __restrict,
						 const REAL4 * __restrict,
						 const int32_t ,
						 const int32_t,
						 const int32_t,
						 struct FP4_COMP_METADATA * ,
						 const uint32_t,
						 int32_t * ,
						 const char[256],
						 const char[256],
						 const uint32_t,
						 const int32_t,
						 const REAL4);

//
// Comparing arrays 4D
//
void compare_gpu_cpu_real4(const REAL4 * __restrict,
						 const REAL4 * __restrict,
						 const int32_t,
						 const int32_t,
						 const int32_t,
						 const int32_t,
						 struct FP4_COMP_METADATA *,
						 const uint32_t,
						 int32_t * ,
						 const char[256],
						 const char[256],
						 const uint32_t,
						 const int32_t,
						 const REAL4 );


				// To be implemented later
//void compare_gpu_cpu_real8(const REAL8 * __restrict,
//						 const REAL8 * __restrict,
//						 const int32_t,
//						 struct FP8_COMP_METADATA * );
//




#endif /*CUWRF_FPCOMP_H_*/