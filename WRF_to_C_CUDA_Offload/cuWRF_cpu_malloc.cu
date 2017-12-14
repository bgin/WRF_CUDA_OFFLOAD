
#include <intrin.h>
#include "cuWRF_cpu_malloc.cuh"


//
// Implementation
//

#if !defined (CPU_MALLOC_INIT_INT32)
#define CPU_MALLOC_INIT_IN32(len) \
	for(size_t i = 0ULL; i != (len); ++i) { \
		h_ptr[i] = 0;                  \
	}
#endif

#if !defined (CPU_MALLOC_INIT_REAL4)
#define CPU_MALLOC_INIT_REAL4(len)   \
	for(size_t i = 0ULL; i != (len); ++i) { \
		h_ptr[i] = 0.F;				     \
	}
#endif

#if !defined (CPU_MALLOC_INIT_REAL8)  
#define CPU_MALLOC_INIT_REAL8(len)   \
	for(size_t i = 0ULL; i != (len); ++i) { \
		h_ptr[i] = 0.0;					 \
	}
#endif

#if !defined (CPU_MALLOC_CHECK_FOR_NULL)
#define CPU_MALLOC_CHECK_FOR_NULL(ptr,msg,line) \
	do { \
	     if(NULL == ptr) {  \
		    *ierr = -2;    \
			fatal_runtime_error(msg,line);  \
		 }\
	} while(0); 
#endif
	


void alloc1D_int32_host(int32_t * __restrict h_ptr,
					  const size_t nx,
					  int32_t * ierr  ) {
	if(*ierr < 0) *ierr = 0;
	if (0ULL >= nx ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_int32_host\n");
#endif
		*ierr = -1;
		return;
	}
	h_ptr = (int32_t*)_aligned_malloc(nx*sizeof(uint32_t),HOST_ALIGN64);
	if(NULL == h_ptr){
		*ierr = -2; // Maybe return to Fortran process.
		fatal_runtime_error("alloc1D_int32_host: !! Failed to allocate memory !! ..Terminating execution!",
		                  __LINE__);
	}
	CPU_MALLOC_INIT_IN32(nx)
	*ierr = 0;
}

void alloc2D_int32_host(int32_t * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  int32_t * ierr) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc2D_int32_host\n");
#endif
		*ierr = -1;
		return;
	}
	const size_t len = nx*ny;
	h_ptr = (int32_t *)_aligned_malloc(len*sizeof(int32_t),HOST_ALIGN64);
	if(NULL == h_ptr) {
		*ierr = -2;
		fatal_runtime_error("alloc2D_int32_host: !! Failed to allocate memory !! ..Terminating execution!",
						  __LINE__);
	}
	CPU_MALLOC_INIT_IN32(len)
	*ierr = 0;
}

void alloc3D_int32_host(int32_t * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  const size_t nz,
					  int32_t * ierr ) {
	if(*ierr < 0) ierr = 0;
	if(0ULL >= nx || 0ULL >= ny ||
	   0ULL >= nz  ) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc3D_int32_host!");
#endif
		*ierr = -1;
		return;
	}
	const size_t len = nx*ny*nz;
	h_ptr = (int32_t*)_aligned_malloc(len*sizeof(int32_t),HOST_ALIGN64);
	if(NULL == h_ptr) {
		*ierr = -2;
		fatal_runtime_error("alloc3D_int32_host: !! Failed to allocate memort !! .. Terminating execution!",
						  __LINE__);
	}
	CPU_MALLOC_INIT_IN32(len)
	*ierr = 0;
}

void alloc1D_real4_host(REAL4 * __restrict h_ptr,
					  const size_t nx,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_real4_host!")
#endif
		*ierr = -1;
		return;
	}
	h_ptr = (REAL4*)_aligned_malloc(nx*sizeof(REAL4),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc1D_real4_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL4(nx)
	*ierr = 0;
}

void alloc2D_real4_host(REAL4 * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc2D_real4_host!")
#endif
	    *ierr -1;
		return;
	}
	const size_t len = nx*ny;
	h_ptr = (REAL4*)_aligned_malloc(len*sizeof(REAL4),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc2D_real4_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
    CPU_MALLOC_INIT_REAL4(len)
	*ierr = 0;
}

void alloc3D_real4_host(REAL4 * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  const size_t nz,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny ||
	   0ULL >= nz ) {
#if (DEBUG_VERBOSE) == 1
	   REPORT_ERROR("Invalid argument in alloc3D_real4_host!")
#endif
		 *ierr = -1;
	      return;
	}
	const size_t len = nx*ny*nz;
	h_ptr = (REAL4*)_aligned_malloc(len*sizeof(REAL4),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc3D_real4_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL4(len)
	*ierr = 0;
}

void alloc4D_real4_host(REAL4 * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  const size_t nz,
					  const size_t nw,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny ||
	   0ULL >= nz || 0ULL >= nw) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc4D_real4_host!")
#endif
		*ierr = -1;
		 return;
	}
	const size_t len = nx*ny*nz*nw;
	h_ptr = (REAL4*)_aligned_malloc(len*sizeof(REAL4),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc4D_real4_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL4(len)
	*ierr = 0;
}

void alloc1D_real8_host(REAL8 * __restrict h_ptr,
					  const size_t nx,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc1D_real8_host!")
#endif
			*ierr = -1;
		    return;
	}
	h_ptr = (REAL8*)_aligned_malloc(nx*sizeof(REAL8),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc1D_real8_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL8(nx)
	*ierr = 0;
}

void alloc2D_real8_host(REAL8 * __restrict h_ptr,
					  const size_t nx,
					  const size_t ny,
					  int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc2D_real8_host!")
#endif
			*ierr = -1;
		    return;
	}
	const size_t len = nx*ny;
	h_ptr = (REAL8*)_aligned_malloc(len*sizeof(REAL8),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc2D_real8_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL8(len)
	*ierr = 0;
}

void alloc3D_real8_host(REAL8 * __restrict h_ptr,
					 const size_t nx,
					 const size_t ny,
					 const size_t nz,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny ||
	   0ULL >= nz) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc3D_real8_host!")
#endif
			*ierr = -1;
		     return;
	}
	const size_t len = nx*ny*nz;
	h_ptr = (REAL8*)_aligned_malloc(len*sizeof(REAL8),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc3D_real8_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL8(len)
	*ierr = 0;
}

void alloc4D_real8_host(REAL8 * __restrict h_ptr,
					 const size_t nx,
					 const size_t ny,
					 const size_t nz,
					 const size_t nw,
					 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
	if(0ULL >= nx || 0ULL >= ny ||
	   0ULL >= nz || 0ULL >= nw) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("Invalid argument in alloc3D_real8_host!")
#endif
			*ierr = -1;
		     return;
	}
	const size_t len = nx*ny*nz*nw;
	h_ptr = (REAL8*)_aligned_malloc(len*sizeof(REAL8),HOST_ALIGN64);
	CPU_MALLOC_CHECK_FOR_NULL(h_ptr,"alloc3D_real8_host: !! Failed to allocate memory !! .. Terminating execution!",
						    __LINE__)
	CPU_MALLOC_INIT_REAL8(len)
	*ierr = 0;
}