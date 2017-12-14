
#ifndef  __CUWRF_CPU_MALLOC_CUH__
#define  __CUWRF_CPU_MALLOC_CUH__

#if !defined (CUWRF_CPU_MALLOC_MAJOR) 
#define CUWRF_CPU_MALLOC_MAJOR 1
#endif

#if !defined (CUWRF_CPU_MALLOC_MINOR)
#define CUWRF_CPU_MALLOC_MINOR 0
#endif

#if !defined (CUWRF_CPU_MALLOC_MICRO)
#define CUWRF_CPU_MALLOC_MICRO 0
#endif

#if !defined (CUWRF_CPU_MALLOC_FULLVER)
#define CUWRF_CPU_MALLOC_FULLVER 1000
#endif

#if !defined (CUWRF_CPU_MALLOC_CREATE_DATE)
#define CUWRF_CPU_MALLOC_CREATE_DATE "14-12-2017 12:41 +00200 (THR 14 DEC 2017 GMT+2)"
#endif
//
// Set this value after successsful compilation.
//
#if !defined (CUWRF_CPU_MALLOC_BUILD_DATE)
#define CUWRF_CPU_MALLOC_BUILD_DATE ""
#endif

#if !defined (CUWRF_CPU_MALLOC_AUTHOR)
#define CUWRF_CPU_MALLOC_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_CPU_MALLOC_DESCRIPT)
#define CUWRF_CPU_MALLOC_DESCRIPT "CPU memory allocation and initialization routines."
#endif

#include <cstdint>
#include "cuwrf_config.cuh"

//
// Host memory allocation subroutines
//


//
// Allocate and zero-initialize array 1D of type int32_t
// Aligned allocation in use.
//
void alloc1D_int32_host(int32_t * __restrict,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 2D of type int32_t
// Aligned allocation in use.
//
void alloc2D_int32_host(int32_t * __restrict,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 3D of type int32_t
// Aligned allocation in use.
//
void alloc3D_int32_host(int32_t * __restrict,
					  const size_t,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 1D of type REAL4.
// Aligned allocation in use of size 64-bytes.
//
void alloc1D_real4_host(REAL4 * __restrict ,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 2D of type REAL4.
// Aligned allocation in use of size 64-bytes.
//
void alloc2D_real4_host(REAL4 * __restrict,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 3D of type REAL4.
// Aligned allocation in use of size 64-bytes.
//
void alloc3D_real4_host(REAL4 * __restrict,
					  const size_t,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 4D of type REAL4.
// Aligned allocation in use of size 64-bytes.
//
void alloc4D_real4_host(REAL4 * __restrict,
					  const size_t,
					  const size_t,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 1D of type REAL8.
// Aligned allocation in use of size 64-bytes.
//
void alloc1D_real8_host(REAL8 * __restrict,
					 const size_t,
					 int32_t * );

//
// Allocate and zero-initialize array 2D of type REAL8.
// Aligned allocation in use of size 64-bytes.
//
void alloc2D_real8_host(REAL8 * __restrict,
					 const size_t,
					 const size_t,
					 int32_t * );

//
// Allocate and zero-initialize array 3D of type REAL8.
// Aligned allocation in use of size 64-bytes.
//
void alloc3D_real8_host(REAL8 * __restrict,
					  const size_t,
					  const size_t,
					  const size_t,
					  int32_t * );

//
// Allocate and zero-initialize array 3D of type REAL8.
// Aligned allocation in use of size 64-bytes.
//
void alloc4D_real8_host(REAL8 * __restrict,
						const size_t,
						const size_t,
						const size_t,
						const size_t,
						int32_t * );



#endif /*__CUWRF_CPU_MALLOC_CUH__*/