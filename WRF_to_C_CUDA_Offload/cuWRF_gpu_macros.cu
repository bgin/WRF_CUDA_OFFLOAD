
#include "cuWRF_gpu.cuh"

//
// Parametrized macros for convenience
//

// Expands to call to any of 1D copy Host-to-Device functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (C1D_X_HD)
#define C1D_X_HD(fx) (fx)
#endif

// Expands to call to any of 2D copy Host-to-Device functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (C2D_X_HD)
#define C2D_X_HD(fx) (fx)
#endif

// Expands to call to any of 3D copy Host-to-Device functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (C3D_X_HD)
#define C3D_X_HD(fx) (fx)
#endif

// Expands to call to any of 4D copy Host-to-Device functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (C4D_X_HD)
#define C4D_X_HD(fx) (fx)
#endif

// Small memory <= 4GiB GPU allocations

// Expands to call to any of 1D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A1D_X_GPU)
#define A1D_X_GPU(fx) (fx)
#endif

// Expands to call to any of 2D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A2D_X_GPU)
#define A2D_X_GPU(fx) (fx)
#endif

// Expands to call to any of 3D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A3D_X_GPU)
#define A3D_X_GPU(fx) (fx)
#endif

// Expands to call to any of 4D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A4D_X_GPU)
#define A4D_X_GPU(fx) (fx)
#endif

// Large memory >= 4GiB GPU allocations

// Expands to call to any of 1D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A1DL_X_GPU)
#define A1DL_X_GPU(fx) (fx)
#endif

// Expands to call to any of 2D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A2DL_X_GPU)
#define A2DL_X_GPU(fx) (fx)
#endif

// Expands to call to any of 3D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A3DL_X_GPU)
#define A3DL_X_GPU(fx) (fx)
#endif

// Expands to call to any of 4D GPU allocation functions for 
// following types:
// 1) int32_t
// 2) REAL4
// 3) REAL8
#if !defined (A4DL_X_GPU)
#define A4DL_X_GPU(fx) (fx)
#endif



#if !defined (C1D_I32_HD)
#define C1D_I32_HD(d_ptr,h_ptr,nx,ierr)  \
	copy1D_int32_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_I32_HD)
#define C2D_I32_HD(d_ptr,h_ptr,nx,ny,ierr)  \
	 copy2D_int32_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_I32_HD)
#define C3D_I32_HD(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_int32_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_I32_HD)
#define C4D_I32_HD(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_int32_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (C1D_R4_HD)
#define C1D_R4_HD(d_ptr,h_ptr,nx,ierr) \
	copy1D_real4_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_R4_HD)
#define C2D_R4_HD(d_ptr,h_ptr,nx,ny,ierr) \
	copy2D_real4_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_R4_HD)
#define C3D_R4_HD(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_real4_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_R4_HD)
#define C4D_R4_HD(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_real4_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (C1D_R8_HD)
#define C1D_R8_HD(d_ptr,h_ptr,nx,ierr) \
	copy1D_real8_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_R8_HD)
#define C2D_R8_HD(d_ptr,h_ptr,nx,ny,ierr) \
	copy2D_real8_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_R8_HD)
#define C3D_R8_HD(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_real8_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_R8_HD)
#define C4D_R8_HD(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_real8_cpu_to_gpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif

// GPU memory allocation parametrized macros

#if !defined (A1DGPU_I32)
#define A1DGPU_I32(d_ptr,nx,ierr) \
	alloc1D_int32_gpu(&d_ptr[0],nx,&ierr);
#endif

#if !defined (A2DGPU_I32)
#define A2DGPU_I32(d_ptr,nx,ny,ierr) \
	alloc2D_int32_gpu(&d_ptr[0],nx,ny,&ierr);
#endif

#if !defined (A3DGPU_I32)
#define A3DGPU_I32(d_ptr,nx,ny,nz,ierr) \
	alloc3D_int32_gpu(&d_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (A4DGPU_I32)
#define A4DGPU_I32(d_ptr,nx,ny,nz,nw,ierr) \
	alloc4D_int32_gpu(&d_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (A1DGPU_R4)
#define A1DGPU_R4(d_ptr,nx,ierr) \
	alloc1D_real4_gpu(&d_ptr[0],nx,&ierr);
#endif

#if !defined (A2DGPU_R4)
#define A2DGPU_R4(d_ptr,nx,ny,ierr) \
	alloc2D_real4_gpu(&d_ptr[0],nx,ny,&ierr);
#endif

#if !defined (A3DGPU_R4)
#define A3DGPU_R4(d_ptr,nx,ny,nz,ierr) \
	alloc3D_real4_gpu(&d_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (A4DGPU_R4)
#define A4DGPU_R4(d_ptr,nx,ny,nz,nw,ierr) \
	alloc4D_real4_gpu(&d_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (A1DGPU_R8)
#define A1DGPU_R8(d_ptr,nx,ierr) \
	alloc1D_real8_gpu(&d_ptr[0],nx,&ierr);
#endif

#if !defined (A2DGPU_R8)
#define A2DGPU_R8(d_ptr,nx,ny,ierr) \
	alloc2D_real8_gpu(&d_ptr[0],nx,ny,&ierr);
#endif

#if !defined (A3DGPU_R8)
#define A3DGPU_R8(d_ptr,nx,ny,nz,ierr) \
	alloc3D_real8_gpu(&d_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (A4DGPU_R8)
#define A4DGPU_R8(d_ptr,nx,ny,nz,nw,ierr) \
	alloc4D_real8_gpu(&d_ptr[0],nx,ny,nz,nw,&ierr);
#endif

// Copy GPU-to-CPU parametrized macros

#if !defined (C1D_I32_DH)
#define C1D_I32_DH(d_ptr,h_ptr,nx,ierr) \
	copy1D_int32_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_I32_DH)
#define C2D_I32_DH(d_ptr,h_ptr,nx,ny,ierr) \
	copy2D_int32_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_I32_DH)
#define C3D_I32_DH(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_int32_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_I32_DH)
#define C4D_I32_DH(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_int32_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (C1D_R4_DH)
#define C1D_R4_DH(d_ptr,h_ptr,nx,ierr) \
	copy1D_real4_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_R4_DH)
#define C2D_R4_DH(d_ptr,h_ptr,nx,ny,ierr) \
	copy2D_real4_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_R4_DH)
#define C3D_R4_DH(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_real4_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_R4_DH)
#define C4D_R4_DH(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_real4_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif

#if !defined (C1D_R8_DH)
#define C1D_R8_DH(d_ptr,h_ptr,nx,ierr) \
	copy1D_real8_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,&ierr);
#endif

#if !defined (C2D_R8_DH)
#define C2D_R8_DH(d_ptr,h_ptr,nx,ny,ierr) \
	copy2D_real8_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,&ierr);
#endif

#if !defined (C3D_R8_DH)
#define C3D_R8_DH(d_ptr,h_ptr,nx,ny,nz,ierr) \
	copy3D_real8_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,&ierr);
#endif

#if !defined (C4D_R8_DH)
#define C3D_R8_DH(d_ptr,h_ptr,nx,ny,nz,nw,ierr) \
	copy4D_real8_gpu_to_cpu(&d_ptr[0],&h_ptr[0],nx,ny,nz,nw,&ierr);
#endif


