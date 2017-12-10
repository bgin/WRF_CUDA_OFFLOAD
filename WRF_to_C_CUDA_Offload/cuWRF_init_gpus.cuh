
#ifndef __CUWRF_INIT_GPUS_CUH__
#define __CUWRF_INIT_GPUS_CUH__

#if !defined (CUWRF_INIT_GPUS_MAJOR)
#define CUWRF_INIT_GPUS_MAJOR 1
#endif

#if !defined (CUWRF_INIT_GPU_MINOR)
#define CUWRF_INIT_GPUS_MINOR 0
#endif

#if !defined (CUWRF_INIT_GPUS_MICRO)
#define CUWRF_INIT_GPUS_MICRO 0
#endif

#if !defined (CUWRF_INIT_GPUS_FULLVER)
#define CUWRF_INIT_GPUS_FULLVER 1000
#endif

#if !defined (CUWRF_INIT_GPUS_CREATE_DATE)
#define CUWRF_INIT_GPUS_CREATE_DATE "04-12-2017 13:01 +00200 (MON 04 DEC 2017 GMT+2)"
#endif
//
// Set this value after successful compilation.
//
#if !defined (CUWRF_INIT_GPUS_BUILD_DATE)
#define CUWRF_INIT_GPUS_BUILD_DATE " "
#endif

#if !defined (CUWRF_INIT_GPUS_AUTHOR)
#define CUWRF_INIT_GPUS_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_INIT_GPUS_DESCRIPT)
#define CUWRF_INIT_GPUS_DESCRIPT "Various routines related to GPU initialization."
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuwrf_config.cuh"
#include "cuWRF_gpu.cuh"

//
//	Declarations
//

//
//	Print device capabilities to screen.
//
void devcaps_to_screen(const int, int * , const bool);


//
//	Print device capabilities to file.
//
void devcaps_to_file(const int, const char *, 
				   int * , const bool );


//
// Run integer and real arithmentic test.
// Host code.
//
cudaError_t  gpu_vec_add_tests(const int);

// 
// Device code.
// simple vector addition int32_t,REAL4 and REAL8.
//
__global__ void kvec_add_int32(int32_t * __restrict,
							const int32_t * __restrict,
							const int32_t * __restrict);

__global__ void kvec_add_real4(REAL4 * __restrict,
						   const REAL4 * __restrict ,
						   const REAL4 * __restrict);

__global__ void kvec_add_real8(REAL8 * __restrict,
						    const REAL8 * __restrict,
							const REAL8 * __restrict);




#endif /*__CUWRF_INIT_GPUS_CUH__*/