
#ifndef __CUWRF_GPU_CUH__
#define __CUWRF_GPU_CUH__

#if !defined (CUWRF_GPU_MAJOR)
#define CUWRF_GPU_MAJOR 1
#endif

#if !defined (CUWRF_GPU_MINOR)
#define CUWRF_GPU_MINOR 0
#endif

#if !defined (CUWRF_GPU_MICRO)
#define CUWRF_GPU_MICRO 0
#endif

#if !defined (CUWRF_GPU_FULLVER)
#define CUWRF_GPU_FULLVER 1000
#endif

#if !defined (CUWRF_GPU_CREATE_DATE)
#define CUWRF_GPU_CREATE_DATE "12-11-2017 14:37 +00200 (SUN, 12 NOV 2017 GMT+2) "
#endif
//
//	Set this value after successful build.
//
#if !defined (CUWRF_GPU_BUILD_DATE)
#define CUWRF_GPU_BUILD_DATE " "
#endif

#if !defined (CUWRF_GPU_AUTHOR)
#define CUWRF_GPU_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_GPU_DESCRIPT)
#define CUWRF_GPU_DESCRIPT "Helper GPU routines."
#endif


#include "cuwrf_config.cuh"
#include <cstdint>

typedef float REAL4;

typedef double REAL8;

//
// CPU-to-GPU and GPU-to-CPU memory copy routines
// Small memory space <= 4 GiB
//

//
// Copy int32_t array 1D (linearized) from CPU to GPU. 
//
void copy1D_int32_cpu_to_gpu(int32_t * __restrict, const int32_t * __restrict,
						   const int32_t, int32_t * );

//
// Copy int32_t array 2D (linearized) from CPU to GPU.
//
void copy2D_int32_cpu_to_gpu(int32_t * __restrict, const int32_t * __restrict,
						   const int32_t, const int32_t, int32_t *);

//
// Copy int32_t array 3D (linearized) from CPU to GPU.
//
void copy3D_int32_cpu_to_gpu(int32_t * __restrict, const int32_t * __restrict,
						   const int32_t, const int32_t, const int32_t, int32_t *);

//
// Copy int32_t array 4D (linearized) from CPU to GPU.
//
void copy4D_int32_cpu_to_gpu(int32_t * __restrict, const int32_t * __restrict, const int32_t,
						  const int32_t, const int32_t, const int32_t, int32_t *);

//
// Copy real4(float) array 1D (linearized) from CPU to GPU.
//
void copy1D_real4_cpu_to_gpu(REAL4 * __restrict, const REAL4 * __restrict, 
						  const int32_t, int32_t *);

//
// Copy real4(float) array 2D (linearized) from CPUT to GPU.
//
void copy2D_real4_cpu_to_gpu(REAL4 * __restrict, const REAL4 * __restrict, 
						   const int32_t, const int32_t, int32_t *);

//
// Copy real4(float) array 3D (linearized) from CPU to GPU.
//
void copy3D_real4_cpu_to_gpu(REAL4 * __restrict, const REAL4 * __restrict, const int32_t,
						  const int32_t, const int32_t, int32_t * );

//
// Copy real4(float) array 4D (linearized) from CPU to GPU.
//
void copy4D_real4_cpu_to_gpu(REAL4 * __restrict, const REAL4 * __restrict, const int32_t,
						  const int32_t, const int32_t, const int32_t, int32_t * );


//
// Copy real8(double) array 1D (linearized) from CPU to GPU.
//
void copy1D_real8_cpu_to_gpu(REAL8* __restrict, REAL8 * __restrict, 
						  const int32_t, int32_t * );

//
// Copy real8(double) array 2D (linearized) from CPU to GPU.
//
void copy2D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int32_t, const int32_t, int32_t * );

//
// Copy real8(double) array 3D (linearized) from CPU to GPU.
//
void copy3D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 *  __restrict, const int32_t, 
						  const int32_t, const int32_t, int32_t * );

//
// Copy real8(double) array 4D (linearized) from CPU to GPU.
//
void copy4D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict, const int32_t,
						   const int32_t, const int32_t, const int32_t, int32_t * );


#if GPU_LARGE_MEM_SPACE == 1

//
// CPU-to-GPU and GPU-to-CPU memory copy routines
// Large memory space >= 4GiB device memory.
//

//
// Copy int32_t array 1D (linearized) from CPU to GPU.
//
void copy1D_int32_cpu_to_gpu(int32_t * __restrict, int32_t * __restrict, 
						   const int64_t, int32_t * );

//
// Copy int32_t array 2D (linearized) from CPU to GPU.
//
void copy2D_int32_cpu_to_gpu(int32_t * __restrict, int32_t * __restrict,
						   const int64_t, const int64_t, int32_t * );

//
// Copy int32_t array 3D (linearized) from CPU to GPU.
//
void copy3D_int32_cpu_to_gpu(int32_t * __restrict, int32_t * restrict, const int64_t,
						   const int64_t, const int64_t, int32_t * );

//
// Copy int32_t array 4D (linearized) from CPU to GPU.
//
void copy4D_int32_cpu_to_gpu(int32_t * __restrict, int32_t * __restrict, const int64_t,
						   const int64_t, const int64_t, const int64_t, int32_t * );

//
// Copy real4 array 1D (linearized) from CPU to GPU
//
void copy1D_real4_cpu_to_gpu(REAL4 * __restrict, REAL4 * __restrict, 
						   const int64_t, int32_t * );

//
// Copy real4 array 2D (linearized) from CPU to GPU.
//
void copy2D_real4_cpu_to_gpu(REAL4 * __restrict, REAL4 * __restrict,
						  const int64_t, const int64_t, int32_t * );

//
// Copy real4 array 3D (linearize) from CPU to GPU.
//
void copy3D_real4_cpu_to_gpu(REAL4 * __restrict, REAL4 * __restrict, const int64_t,
						  const int64_t, const int64_t, int32_t * );

//
// Copy real4 array 4D (linearize) from CPU to GPU.
//
void copy4D_real4_cpu_to_gpu(REAL4 * __restrict, REAL4 * __restrict, const int64_t,
						   const int64_t, const int64_t, const int64_t, int32_t * );

//
// Copy real8 array 1D (linearized) from CPU to GPU.
//
void copy1D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int64_t, int32_t * );

//
// Copy real8 array 2D (linearized) from CPU to GPU.
//
void copy2D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict, 
						   const int64_t, const int64_t, int32_t * );

//
// Copy real8 array 3D (linearized) from CPU to GPU.
//
void copy3D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict, const int64_t,
						   const int64_t, const int64_t, int32_t * );

//
// Copy real8 array 4D (linearized) from CPU to GPU.
//
void copy4D_real8_cpu_to_gpu(REAL8 * __restrict, REAL8 * __restrict, const int64_t,
						   const int64_t, const int64_t, const int64_t, int32_t * );

#endif

//
//	Allocate memory on GPU.
//   Small memory space <= 4GiB device memory.
//

//
// Allocate array 1D of type int32_t on GPU.
//
void alloc1D_int32_gpu(int32_t * __restrict, const int32_t, int32_t * );

//
// Allocate array 2D (linearized) of type int32_t on GPU.
//
void alloc2D_int32_gpu(int32_t * __restrict, const int32_t,
					 const int32_t, int32_t * );

//
// Allocate array 3D (linearized) of type int32_t on GPU.
//
void alloc3D_int32_gpu(int32_t * __restrict, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
// Allocate array 4D (linearized) of type int32_t on GPU.
//
void alloc4D_int32_gpu(int32_t * __restrict, const int32_t, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
// Allocate array 1D of type real4 on GPU.
//
void alloc1D_real4_gpu(REAL4 * __restrict, const int32_t, int32_t * );

//
// Allocate array 2D (linearized) of type real4 on GPU.
//
void alloc2D_real4_gpu(REAL4 * __restrict, const int32_t, 
					 const int32_t, int32_t * );

//
// Allocate array 3D (linearized) of type real4 on GPU.
//
void alloc3D_real4_gpu(REAL4 * __restrict, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
// Allocate array 4D (linearized) of type real4 on GPU.
//
void alloc4D_real4_gpu(REAL4 * __restrict, const int32_t, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
// Allocate array 1D of type real8 on GPU.
//
void alloc1D_real8_gpu(REAL8 * __restrict, const int32_t, int32_t * );

//
// Allocate array 2D (linearized) of type real8 on GPU.
//
void alloc2D_real8_gpu(REAL8 * __restrict, const int32_t, 
					 const int32_t, int32_t * );

//
// Allocate array 3D (linearized) of type real8 on GPU.
//
void alloc3D_real8_gpu(REAL8 * __restrict, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
// Allocate array 4D (linearize) of type real8 on GPU.
//
void alloc4D_real8_gpu(REAL8 * __restrict, const int32_t, const int32_t,
					 const int32_t, const int32_t, int32_t * );

//
//	Allocate memory on GPU.
//   Large memory space >= 4GiB device memory.
//

#if (GPU_LARGE_MEM_SPACE) == 1

//
// Allocate array 1D of type int32_t on GPU.
//
void alloc1D_int32_gpu(int32_t * __restrict, const int64_t, int32_t * );

//
//
// Allocate array 2D (linearized) of type int32_t on GPU.
void alloc2D_int32_gpu(int32_t * __restrict, const int64_t, 
					 const int64_t, int32_t * );

//
// Allocate array 3D (linearized) of type int32_t on GPU.
//
void alloc3D_int32_gpu(int32_t * __restrict, const int64_t,
					 const int64_t, const int64_t, int32_t * );

//
// Allocate array 4D (linearized) of type int32_t on GPU.
//
void alloc4D_int32_gpu(int32_t * __restrict, const int64_t, const int64_t,
					 const int64_t, const int64_t, int32_t * );

//
// Allocate array 1D of type real4 on GPU.
//
void alloc1D_real4_gpu(REAL4 * __restrict, const int64_t, int32_t * );

//
// Allocate array 2D (linearized) of type real4 on GPU.
//
void alloc2D_real4_gpu(REAL4 * __restrict, const int64_t,
					 const int64_t, int32_t * );

//
// Allocate array 3D (linearized) of type real4 on GPU.
//
void alloc3D_real4_gpu(REAL * __restrict, const int64_t,
					 const int64_t, const int64_t, int32_t * );

//
// Allocate array 4D (linearized) of type real4 on GPU.
//
void alloc4D_real4_gpu(REAL * __restrict, const int64_t, const int64_t,
					 const int64_t, const int64_t, int32_t * );

//
// Allocate array 1D of type real8 on GPU.
//
void alloc1D_real8_gpu(REAL * __restrict, const int64_t, int32_t * );

//
// Allocate array 2D (linearized) of type real8 on GPU.
//
void alloc2D_real8_gpu(REAL * __restrict, const int64_t,  
					 const int64_t, int32_t * );

//
// Allocate array 3D (linearized) of type real8 on GPU.
//
void alloc3D_real8_gpu(REAL8 * __restrict, const int64_t,
					 const int64_t, const int64_t, int32_t * );

//
// Allocate array 4D (linearized) of type real8 on GPU.
//
void alloc4D_real8_gpu(REAL8 * __restrict, const int64_t, const int64_t,
					 const int64_t, const int64_t, int32_t * );

#endif

//
// GPU to CPU memory copy routines
//

//
// Small GPU memory space <= 4GiB
//

//
// Copy array 1D of int32_t from GPU to CPU.
//
void copy1D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict,
						   const int32_t, int32_t * );

//
// Copy array 2D (linearized) of int32_t from GPU to CPU.
//
void copy2D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, 
						   const int32_t, const int32_t, int32_t * );

//
// Copy array 3D (linearized) of int32_t from GPU to CPU.
//
void copy3D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, const int32_t,
						   const int32_t, const int32_t, int32_t * );

//
// Copy array 4D (linearized) of int32_t from GPU to CPU.
//
void copy4D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, const int32_t, 
						   const int32_t, const int32_t, const int32_t, int32_t * );

//
// Copy array 1D of type real4 from GPU to CPU.
//
void copy1D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, 
						  const int32_t, int32_t * );

//
// Copy array 2D (linearized) of type real4 from GPU to CPU.
//
void copy2D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict,
						   const int32_t, const int32_t, int32_t * );

//
// Copy array 3D (linearized) of type real4 from GPU to CPU.
//
void copy3D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, const int32_t,
						  const int32_t, const int32_t, int32_t * );

//
// Copy array 4D (linearized) of type real4 from GPU to CPU.
//
void copy4D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, const int32_t,
						  const int32_t, const int32_t, const int32_t, int32_t * );

//
// Copy array 1D of type real8 from GPU to CPU.
//
void copy1D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int32_t, int32_t * );

//
// Copy array 2D (linearized) of type real8 from GPU to CPU.
//
void copy2D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int32_t, const int32_t, int32_t * );

//
// Copy array 3D (linearized) of type real8 from GPU to CPU.
//
void copy3D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict, const int32_t, 
						   const int32_t, const int32_t, int32_t * );

//
// Copy array 4D (linearized) of type real8 from GPU to CPU
//
void copy4D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict, const int32_t,
						  const int32_t, const int32_t, const int32_t, int32_t * );

//
// Large GPU memory space >= 4GiB
//
#if (GPU_LARGE_MEM_SPACE) == 1

//
// Copy array 1D of int32_t from GPU to CPU.
//
void copy1D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict,
						   const int64_t, int32_t * );

//
// Copy array 2D (linearized) of int32_t from GPU to CPU.
//
void copy2D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, 
						   const int64_t, const int64_t, int32_t * );

//
// Copy array 3D (linearized) of int32_t from GPU to CPU.
//
void copy3D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, const int64_t,
						   const int64_t, const int64_t, int32_t * );

//
// Copy array 4D (linearized) of int32_t from GPU to CPU.
//
void copy4D_int32_gpu_to_cpu(int32_t * __restrict, int32_t * __restrict, const int64_t, 
						   const int64_t, const int64_t, const int64_t, int32_t * );

//
// Copy array 1D of type real4 from GPU to CPU.
//
void copy1D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, 
						  const int64_t, int32_t * );

//
// Copy array 2D (linearized) of type real4 from GPU to CPU.
//
void copy2D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict,
						   const int64_t, const int64_t, int32_t * );

//
// Copy array 3D (linearized) of type real4 from GPU to CPU.
//
void copy3D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, const int64_t,
						  const int64_t, const int64_t, int32_t * );

//
// Copy array 4D (linearized) of type real4 from GPU to CPU.
//
void copy4D_real4_gpu_to_cpu(REAL4 * __restrict, REAL4 * __restrict, const int64_t,
						  const int64_t, const int64_t, const int64_t, int32_t * );

//
// Copy array 1D of type real8 from GPU to CPU.
//
void copy1D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int64_t, int32_t * );

//
// Copy array 2D (linearized) of type real8 from GPU to CPU.
//
void copy2D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict,
						  const int64_t, const int64_t, int32_t * );

//
// Copy array 3D (linearized) of type real8 from GPU to CPU.
//
void copy3D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict, const int64_t, 
						   const int64_t, const int64_t, int32_t * );

//
// Copy array 4D (linearized) of type real8 from GPU to CPU
//
void copy4D_real8_gpu_to_cpu(REAL8 * __restrict, REAL8 * __restrict, const int64_t,
						  const int64_t, const int64_t, const int64_t, int32_t * );


#endif

//
//	Allocate memory on host by calling
//	cudaMallocHost.
//   This routine prepares a memory on page-size
//   boundary to be directly accessible by device.
//







#endif /*__CUWRF_GPU_CUH__*/