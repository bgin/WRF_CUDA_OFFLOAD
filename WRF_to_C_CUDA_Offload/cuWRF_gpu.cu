
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuWRF_gpu.cuh"


//
//	Implementation
//

//
// CPU-to-GPU and GPU-to-CPU memory copy routines
// Small memory space <= 4 GiB
//

void copy1D_int32_cpu_to_gpu(int32_t * __restrict d_ptr, 
						   const int32_t * __restrict h_ptr,
						   const int32_t nx,
						   int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if( NULL == h_ptr || nx <= 0){ //  Host error handling
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy1D_int32_cpu_to_gpu!!");
#endif
	   *ierr = -1;
	   return; 
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx * sizeof(int32_t),cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(int32_t),cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree(d_ptr);

	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_int32_cpu_to_gpu: -- cudaMalloc/cudaMemcpy failure!!",
					status);
}

void copy2D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int32_t nx, 
						  const int32_t ny,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || nx <= 0 || ny <= 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_int32_cpu_to_gpu");
#endif
	   *ierr = -1;
	   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(int32_t),cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(int32_t),cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree(d_ptr);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_int32_cpu_to_gpu: -- cudaMemcopy/CudaMalloc failure!!",
					status);
}

void copy3D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						   const int32_t * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   int32_t * ierr) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || nx <= 0 ||
	   ny <= 0      || ny <= 0) {
#if (DEBUG_VERBOSE) == 1
	REPORT_ERROR("Invalid argument(s) in copy3D_int32_cpu_to_gpu");
#endif
	   *ierr = -1;
	   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_int32_cpu_to_gpu: -- cudaMalloc/cudaMemcpy failed",
				    status);
}

void copy4D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  const int32_t nw,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || nx <= 0 ||
	   ny <= 0  || nz <= 0 || nw <= 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_int32_cpu_to_gpu");
#endif
	   *ierr = -1;
	   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_int32_cpu_to_gpu: -- cudaMalloc/cudaMemcpy failure!!",
					status);
}

void copy1D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int32_t nx,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || nx <= 0){
#if (DEBUG_VERBOSE) == 1
		 REPORT_ERROR("Invalid argument(s) in copy1D_real4_cpu_to_gpu");
#endif
	   *ierr = -1;
	   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy2D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	  nx <= 0       ||
	  ny <= 0       ) {
#if (DEBUG_VERBOSE) == 1
		 REPORT_ERROR("Invalid argument(s) in copy2D_real4_cpu_to_gpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy3D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	  nx <= 0       ||
	  ny <= 0	    ||
	  nz <= 0          ) {
#if (DEBUG_VERBOSE) == 1
		 REPORT_ERROR("Invalid argument(s) in copy3D_real4_cpu_to_gpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy4D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  const int32_t nw,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   nx <= 0      ||
	   ny <= 0      ||
	   nz <= 0      ||
	   nw <= 0       ) {
#if (DEBUG_VERBOSE) == 1
		 REPORT_ERROR("Invalid argument(s) in copy4D_real4_cpu_to_gpu");
#endif
	     *ierr = -1;
		 return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

//
// Copy real8(double) array 1D (linearized) from CPU to GPU.
//

void copy1D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int32_t nx,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0 >= nx     ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy1D_real8_cpu_to_gpu");
#endif
	*ierr = -1;
	return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy2D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || 
	   0 >= nx      ||
	   0 >= ny     ) {
#if (DEBUG_VERBOSE) == 1
		  REPORT_ERROR("Invalid argument(s) in copy2D_real8_cpu_to_gpu");
#endif
    *ierr = -1;
	return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy3D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	  0 >= nx       ||
	  0 >= ny       ||
	  0 >= nz       ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real8_cpu_to_gpu");
#endif
	 *ierr = -1;
	 return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy4D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  const int32_t nw,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0 >= nx      ||
	   0 >= ny      ||
	   0 >= nz      ||
	   0 >= nw      ) {
#if (DEBUG_VERBOSE) == 1
			REPORT_ERROR("Invalid argument(s) in copy4D_real8_cpu_to_gpu");
#endif
	*ierr = -1;
	return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr,nx*ny*nz*nw*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr,nx*ny*nz*nw*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

#if (GPU_LARGE_MEM_SPACE) == 1

//
// CPU-to-GPU and GPU-to-CPU memory copy routines
// Large memory space >= 4GiB device memory.
//

void copy1D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int64_t nx,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx     ) {
#if (DEBUG_VERBOSE) == 1
		  REPORT_ERROR("Invalid argument(s) in copy1D_int32_cpu_to_gpu (large memory)");
#endif
	*ierr = -1;
	return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_int32_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy2D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  int32_t * ierr  ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny     ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_int32_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
		CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
		CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(int32_t),
					    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(int32_t),
					    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_int32_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy3D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  int32_t * ierr   ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz     ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_int32_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(int32_t),
				    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(int32_t),
				    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_int32_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy4D_int32_cpu_to_gpu(int32_t * __restrict d_ptr,
						  const int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  const int64_t nw,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz    ||
	   0LL >= nw      ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy4D_int32_cpu_to_gpu (large memory)");
#endif
		   *ierr = -1;
		   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
		            cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
		            cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_int32_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy1D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  int32_t * ierr  ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx     ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy1D_real4_cpu_to_gpu (large memory)");
#endif
		   *ierr = -1;
		   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy2D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  int32_t * ierr   ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny     ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy2D_real4_cpu_to_gpu (large memory)");
#endif
		   *ierr = -1;
		   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy3D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  int32_t * ierr   ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr || 
	  0LL >= nx     ||
	  0LL >= ny     ||
	  0LL >= nz      ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real4_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL4),
				    cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy4D_real4_cpu_to_gpu(REAL4 * __restrict d_ptr,
						  const REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  const int64_t nw,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz    ||
	   0LL >= nw      ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy4D_real4_cpu_to_gpu (large memory)");
#endif
		   *ierr = -1;
		   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real4_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy1D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int64_t nx,
						  int32_t * ierr ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx     ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy1D_real8_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy2D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  int32_t * ierr   ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny      ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argumemt(s) in copy2D_real8_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy3D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  int32_t * ierr  ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	  0LL >= nx     ||
	  0LL >= ny     ||
	  0LL >= nz     ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real8_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

void copy4D_real8_cpu_to_gpu(REAL8 * __restrict d_ptr,
						  const REAL8 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  const int64_t nw,
						  int32_t * ierr  ) {
	if(*ierr <= 0) *ierr = 0;
	if(NULL == h_ptr ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz    ||
	   0LL >= nw   ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_real8_cpu_to_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
	CuWRF_DEBUG_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
	CuWRF_CHECK(cudaMemcpy((void*)&d_ptr[0],(void*)&h_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
					cudaMemcpyHostToDevice));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real8_cpu_to_gpu: cudaMalloc/cudaMemcpy failure!!",
		            status);
}

#endif

//
//	Allocate memory on GPU.
//   Small memory space <= 4GiB device memory.
//

void alloc1D_int32_gpu(int32_t * __restrict d_ptr,
					 const int32_t nx,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_int32_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_int32_gpu(int32_t * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc2D_int32_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_int32_gpu(int32_t * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ||
	   0 >= ny ||
	   0 >= nz  ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_int_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_int32_gpu(int32_t * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 const int32_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ||
	   0 >= ny ||
	   0 >= nz ||
	   0 >= nw   ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc4D_int32_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc4D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc1D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int32_t nx,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc1D_real4_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc2D_real4_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 int32_t * ierr   ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ||
	   0 >= ny ||
	   0 >= nz   ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_real4_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 const int32_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ||
	   0 >= ny ||
	   0 >= nz ||
	   0 >= nw   ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc4D_real4_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc4D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc1D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int32_t nx,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc1D_real8_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid arguments in alloc2D_real8_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx ||
	   0 >= ny ||
	   0 >= nz   )  {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_real8_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int32_t nx,
					 const int32_t ny,
					 const int32_t nz,
					 const int32_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0  >=  nx || 
	   0  >=  ny ||
       0  >=  nz ||
	   0  >=  nw   ) {
#if (DEBUG_VERBOSE) == 1
	    REPORT_ERROR("Invalid argument(s) in alloc3D_real8_gpu.");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc4D_real8_gpu: cudaMalloc failure!!",
		            status);
}

#if (GPU_LARGE_MEM_SPACE) == 1

//
// Allocate array 1D of type int32_t on GPU.
//

void alloc1D_int32_gpu(int32_t * __restrict d_ptr,
					 const int64_t nx,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_int32_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_int32_gpu(int32_t * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc2D_int32_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_int32_gpu(int32_t * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx ||
	   0LL >= ny ||
	   0LL >= nz    ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_int32_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_int32_gpu(int32_t * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 const int64_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx  ||
	   0LL >= ny  ||
	   0LL >= nz  ||
	   0LL >= nw    ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc4D_int32_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(int32_t)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	 fatal_gpu_error("cuWRF_gpu.cu/alloc4D_int32_gpu: cudaMalloc failure!!",
		            status);
}

void alloc1D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int64_t nx,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_real4_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc2D_real4_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz             ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_real4_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)d_ptr,nx*ny*nz*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)d_ptr,nx*ny*nz*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_real4_gpu(REAL4 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 const int64_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz || 0LL >= nw ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc4D_real4_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL4)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc4D_real4_gpu: cudaMalloc failure!!",
		            status);
}

void alloc1D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int64_t nx,
					 int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_real8_gpu (large memory)");
		
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc1D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc2D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc2D_real8_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc2D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc3D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz )   {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc3D_real8_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc3D_real8_gpu: cudaMalloc failure!!",
		            status);
}

void alloc4D_real8_gpu(REAL8 * __restrict d_ptr,
					 const int64_t nx,
					 const int64_t ny,
					 const int64_t nz,
					 const int64_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx  || 0LL >= ny ||
	   0LL >= nz  || 0LL >= nw ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in alloc4D_real8_gpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
#else
	CuWRF_CHECK(cudaMalloc((void**)&d_ptr,nx*ny*nz*nw*sizeof(REAL8)));
#endif
	*ierr = 0;
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/alloc4D_real8_gpu: cudaMalloc failure!!",
		            status);
}



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

void copy1D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						   int32_t * __restrict h_ptr,
						   const int32_t nx,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in copy1D_int32_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						   int32_t * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_int32_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						  int32_t * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  const int32_t nz,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz     ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_int32_gpu_to_cpu");		
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(int32_t),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(int32_t),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						   int32_t * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   const int32_t nw,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz || 0 >= nw ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_int32_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
			    cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy1D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						   REAL4 * __restrict h_ptr,
						   const int32_t nx,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in copy1D_real4_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						  REAL4 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_real4_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL4),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						   REAL4 * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz      ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real4_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL4),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL4),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						   REAL4 * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   const int32_t nw,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz || 0 >= nw  ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_int32_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy1D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int32_t nx,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in copy1D_real8_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL8),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						  REAL8 * __restrict h_ptr,
						  const int32_t nx,
						  const int32_t ny,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_real8_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL8),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL8),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz  ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real8_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL8),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL8),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int32_t nx,
						   const int32_t ny,
						   const int32_t nz,
						   const int32_t nw,
						   int32_t  * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0 >= nx || 0 >= ny ||
	   0 >= nz || 0 >= nw)  {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_real8_gpu_to_cpu");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
				    cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
				    cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

#if (GPU_LARGE_MEM_SPACE) == 1

void copy1D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						   int32_t * __restrict h_ptr,
						   const int64_t nx,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy1D_int32_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						  int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_int32_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						  int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz   ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_int32_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_int32_gpu_to_cpu(const int32_t * __restrict d_ptr,
						  int32_t * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  const int64_t nw,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz || 0LL >= nw) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_int32_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(int32_t),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_int32_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy1D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						  REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy1D_real4_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL4),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						  REAL4 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_real4_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL4),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL4),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						   REAL4 * __restrict h_ptr,
						   const int64_t nx,
						   const int64_t ny,
						   const int64_t nz,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz  ) {
#if (DEBUG_VERBOSE) == 1
		   REPORT_ERROR("Invalid argument(s) in copy3D_real4_gpu_to_cpu (large memory)");
#endif
		   *ierr = -1;
		   return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_real4_gpu_to_cpu(const REAL4 * __restrict d_ptr,
						   REAL4 * __restrict h_ptr,
						   const int64_t nx,
						   const int64_t ny,
						   const int64_t nz,
						   const int64_t nw,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz || 0LL >= nw) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_real4_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	 CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	 *ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL4),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real4_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy1D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int64_t nx,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("invalid argument(s) in copy1D_real8_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*sizeof(REAL8),
					cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy1D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy2D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int64_t nx,
						   const int64_t ny,
						   int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy2D_real8_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL8),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*sizeof(REAL8),
				     cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy2D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy3D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						  REAL8 * __restrict h_ptr,
						  const int64_t nx,
						  const int64_t ny,
						  const int64_t nz,
						  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	   0LL >= nz  ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy3D_real8_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL8),
				    cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*sizeof(REAL8),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy3D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}

void copy4D_real8_gpu_to_cpu(const REAL8 * __restrict d_ptr,
						   REAL8 * __restrict h_ptr,
						   const int64_t nx,
						   const int64_t ny,
						   const int64_t nz,
						   const int64_t nw,
						   int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if(0LL >= nx || 0LL >= ny ||
	  0LL >= nz || 0LL >= nw) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument(s) in copy4D_real8_gpu_to_cpu (large memory)");
#endif
		*ierr = -1;
		return;
	}
	cudaError status;
#if (CuWRF_DEBUG_ON) == 1
	CuWRF_DEBUG_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
					 cudaMemcpyDeviceToHost));
	*ierr = 0;
#else
	CuWRF_CHECK(cudaMemcpy(&h_ptr[0],&d_ptr[0],nx*ny*nz*nw*sizeof(REAL8),
			   cudaMemcpyDeviceToHost));
	*ierr = 0;
#endif
Error:
	cudaFree((void*)&d_ptr[0]);
	*ierr = -2;
	fatal_gpu_error("cuWRF_gpu.cu/copy4D_real8_gpu_to_cpu: cudaMemcpy failure!!",
		            status);
}


#endif

