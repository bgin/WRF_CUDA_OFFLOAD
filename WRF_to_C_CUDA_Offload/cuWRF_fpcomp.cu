

#include <math.h>
#include <intrin.h>
#include "cuWRF_fpcomp.cuh"
#include "cuWRF_cpu_malloc.cuh"
//
//	Implementation
//

static float float_spacing(const float num, int32_t * exponent) {

	float mantissa = frexpf(num,exponent);
	return (powf(RADIX2f32,(float)(*exponent-EXPF32)));
}

static double double_spacing(const double num, int32_t * exponent) {
	
	double mantissa = frexp(num,exponent);
	return (pow(RADIX2F64,(double)(*exponent-EXPF64)));
	
}

bool is_equal_to_float(const float x, const float y) {

	bool bres = false;
	int32_t exponent = 0;
	bres = fabsf(x-y) < float_spacing(MAX(fabsf(x),fabsf(y)),&exponent);
	return (bres);
}

bool is_equal_to_double(const double x, const double y) {

	bool bres = false;
	int32_t exponent = 0;
	bres = fabs(x-y) < double_spacing(MAX(fabs(x),fabs(y)),&exponent);
	return (bres);
}

bool is_greater_than_float(const float x, const float y) {
	
	bool bres = false;
	int32_t exponent = 0;
	if((x-y) >= float_spacing(MAX(fabsf(x),fabsf(y)),&exponent))
	   bres = true;
	else
	   bres = false;

	return (bres);
}

bool is_greater_than_double(const double x, const double y) {

	bool bres = false;
	int32_t exponent = 0;
	if((x-y) >= double_spacing(MAX(fabs(x),fabs(y)),&exponent))
		bres = true;
	else
		bres = false;

	return (bres);
}

bool is_less_than_float(const float x, const float y) {

	bool bres = false;
	int32_t exponent = 0;
	if((y-x) >= float_spacing(MAX(fabsf(x),fabsf(y)),&exponent))
		bres = true;
	else
		bres = false;

	return (bres);
}

bool is_less_than_double(const double x, const double y) {

	bool bres = false;
	int32_t exponent = 0;
	if((y-x) >= double_spacing(MAX(fabs(x),fabs(y)),&exponent))
		bres = true;
	else
		bres = false;

	return (bres);
}

bool compare_float(const float x, const float y,
				 const uint32_t ulp) {
	
	bool bres = false;
	
	float relative = 0.F;
	int32_t exponent = 0;
	
	relative = fabsf((float)ulp);
	bres = fabsf(x-y) < (relative * float_spacing(MAX(fabsf(x),fabsf(y)),&exponent));
	return (bres);
}

bool compare_double(const double x, const double y,
				  const uint32_t ulp ) {
	
	bool bres = false;
	double relative = 0.0;
	int32_t exponent = 0;
	relative = fabs((double)ulp);
	bres = fabs(x-y) < (relative * double_spacing(MAX(fabs(x),fabs(y)),&exponent));
	return (bres);
}

float float_tolerance(const float x, int32_t n) {

	float tol  = 0.F;
	float pval = 0.F;
	if(fabsf(x) > ZEROF32){
		pval = floorf(log10f(fabsf(x))) - (float)n;
	    tol = powf(TENF32,pval);
	}
	 else {
		 tol = ONEF32;
	 }
	return (tol);
}

double double_tolerance(const double x, const int32_t n) {

	double tol  = 0.0;
	double pval = 0.0;
	if(fabs(x) > ZEROF64) {
		pval = floor(log10(fabs(x))) - (double)n;
		tol = pow(TENF64,pval);
	}
	else {
		tol = ONEF64;
	}
	return (tol);
}

bool compare_float_within_tol(const float x, const float y,
						   const int32_t n
#if FORTRAN_OPTIONAL == 1
						               ,
						   const float eps
#endif
						   ) {
	
    bool bres = false;
	float val = 0.F;
#if FORTRAN_OPTIONAL == 1
	val = eps;
#else
	val = EPSF32;
#endif
	if(fabsf(x) > val || fabsf(y) > val) bres = fabsf(x-y) < float_tolerance(x,n);
	return (bres);
}

bool compare_double_within_tol(const double x, const double y,
							const int32_t n
#if FORTRAN_OPTIONAL == 1
									   ,
							const double eps
#endif
							) {
	
	bool bres = false;
	double val = 0.0;
#if FORTRAN_OPTIONAL == 1
	val = eps;
#else
	val = EPSF64;
#endif
	if(fabs(x) > val || fabs(y) > val) bres = fabs(x-y) < double_tolerance(x,n);
	return (bres);
}

#if !defined (FPCOMP_CHECK_IERR)
#define FPCOMP_CHECK_IERR(ierr,msg)  \
	do {                          \
		if((*ierr) < 0){           \
		                         \
		REPORT_ERROR(msg)         \
		return;					\
		}                       \
	}while(0);
#endif

#if !defined (FPCOMP_PRINT)
#define FPCOMP_PRINT(msg)  \
	printf(" !!! <%s> !!!\n",msg);
#endif

void init_fp4_comp_metadata(struct FP4_COMP_METADATA * data,
						 const size_t buf_len,
						 int32_t * ierr) {

	
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data && 0ULL < buf_len);
#else
	if(NULL == data || 0ULL >= buf_len) {
		REPORT_ERROR("Invalid argument(s) in init_fp4_comp_metadata");
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == true) {
#if   (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- FP4_COMP_METADATA already allocated!!")
#endif
			*ierr = -2;
			return;
	}
	if(*ierr < 0) *ierr = 0;
	// Begin allocation of member arrays
	alloc1D_uint32_host(&data->cloop_idx[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_uint32_host: failed!!") // If fatal would occurre then this block
#endif															   // will not be executed.
			return;
	}
	alloc1D_uint32_host(&data->gloop_idx[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_uint32_host: failed!!")
#endif
			return;
	}
	alloc1D_real4_host(&data->cpu_valid[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_real4_host: failed!!")
#endif
			return;
	}
	alloc1D_real4_host(&data->gpu_invalid[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_real4_host: failed!!")
#endif
			return;
	}
	data->is_allocated = true;
	*ierr = 0;
}

void destroy_fp4_comp_metadata(struct FP4_COMP_METADATA * data,
							int32_t * ierr) {
    if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data);
#else
	if(NULL == data) {
		REPORT_ERROR("Invalid argument in destroy_fp4_comp_metadata!!")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == false) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- FP8_COMP_METADATA already deallocated!!")
#endif
			*ierr = -2;
			 return;
	}

	
	// Deallocation begun.
	if(data->gpu_invalid != NULL) {
		_aligned_free(&data->gpu_invalid);
		data->gpu_invalid = NULL;
	}
	if(data->cpu_valid != NULL) {
		_aligned_free(&data->cpu_valid);
		data->cpu_valid = NULL;
	}
	if(data->gloop_idx != NULL) {
		_aligned_free(&data->gloop_idx);
		data->gloop_idx = NULL;
	}
	if(data->cloop_idx != NULL) {
		_aligned_free(&data->cloop_idx);
		data->cloop_idx = NULL;
	}
	data->is_allocated = false;
	*ierr = 0;
}

void init_fp8_comp_metadata(struct FP8_COMP_METADATA * data,
						 const size_t buf_len,
						 int32_t * ierr ) {
	if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data && 0ULL < buf_len);
#else
	if(NULL == data || 0ULL >= buf_len) {
		REPORT_ERROR("Invalid argument(s) in init_fp8_comp_metadata!!")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == true) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- FP8_COMP_METADATA already allocated!!")
#endif
	    *ierr = -2;
		 return;
	}
	
	// Begin allocation of member arrays.
	alloc1D_uint32_host(&data->cloop_idx[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_uint32_host: failed!!")
#endif
		return;
	}
	alloc1D_uint32_host(&data->gloop_idx[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_uint32_host: failed!!")
#endif
			return;
	}
	alloc1D_real8_host(&data->cpu_valid[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("****Non-Fatal*** -- alloc1D_real8_host: failed!!")
#endif
		return;
	}
	alloc1D_real8_host(&data->gpu_invalid[0],buf_len,ierr);
	if(*ierr < 0) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- alloc1D_real8_host: failed!!")
#endif
			return;
	}
	data->is_allocated = true;
	*ierr = 0;
}

void destroy_fp8_comp_metadata(struct FP8_COMP_METADATA * data,
							int32_t * ierr ) {
    if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
	 _ASSERTE(NULL != data);
#else
	if(NULL == data) {
		REPORT_ERROR("Invalid argument(s) in destroy_fp8_comp_metadata!!")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == false) {
#if (DEBUG_VERBOSE) == 1
		REPORT_ERROR("***Non-Fatal*** -- FP8_COMP_METADATA already destroyed!!")
#endif
		*ierr = -2;
		return;
	}
	// Begin deallocation of arrays.
	if(data->gpu_invalid != NULL) {
		_aligned_free(data->gpu_invalid);
		data->gpu_invalid = NULL;
	}
	if(data->cpu_valid != NULL) {
		_aligned_free(data->cpu_valid);
		data->cpu_valid = NULL;
	}
	if(data->gloop_idx != NULL) {
		_aligned_free(data->gloop_idx);
		data->gloop_idx = NULL;
	}
	if(data->cloop_idx != NULL) {
		_aligned_free(data->cloop_idx);
		data->cloop_idx = NULL;
	}
	data->is_allocated = false;
	*ierr = 0;
}

void compare_gpu_cpu_real4(const REAL4 * __restrict d_ptr,
						 const REAL4 * __restrict h_ptr,
						 const int32_t len,
						 struct FP4_COMP_METADATA * data,
						 const uint32_t method,
						 int32_t * ierr,
						 const char cfname[256],
						 const char gfname[256],
						 const uint32_t  ulp,
						 const int32_t n,
						 const REAL4 eps) {

	if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
	  _ASSERTE(NULL != d_ptr &&
		      NULL != h_ptr &&
			  0 < len       &&
			  NULL != data);
#else
	if(NULL == h_ptr || NULL == d_ptr ||
	   0 > len      || NULL == data ) {
		REPORT_ERROR("***Non-Fatal*** -- Invalid argument(s) in compare_gpu_cpu_real4!!")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == false)
		init_fp4_comp_metadata(data,len,ierr);
	if(*ierr < 0) {
		REPORT_ERROR("***Non-Fatal*** -- init_fp4_comp_metadata: failed!!")
		return;
	}
	// Begin switch statements
	// Various fp comparison methods are employed.
	printf("Begin comparison of GPU and CPU results.\n");
	for(int32_t i = 0; i != 256; ++i) {
		data->cfname[i] = cfname[i];
		data->gfname[i] = gfname[i];
	}
	uint32_t bres = 9999;
	switch(method) {
	
	case 0: {
		      printf("Using naive floating-point comparison (to epsilon). \n");
			  for(int32_t i = 0; i != len; ++i) {
				  if(_isnanf(d_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					  printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				  if(_isnanf(h_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				  if(d_ptr[i] != 0.F && h_ptr[i] != 0.F) {
				     if((fabsf(d_ptr[i])-fabsf(h_ptr[i])) > EPSF32) {
					     data->cloop_idx[i] = i;
					     data->gloop_idx[i] = i;
					     data->cpu_valid[i] = h_ptr[i]; // Only true branch is coded
					     data->gpu_invalid[i] = d_ptr[i]; // valid values will not alter the arrays contents.
				     }
				  }
			  }
			  *ierr = 0;
		}
		break;

	case 1: {
			 printf("Using floating-point comparison based on relative spacing. \n");
			 for(int32_t i = 0; i != len; ++i) {
				  if(_isnanf(d_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					  printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				  if(_isnanf(h_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				 if(d_ptr[i] != 0.F && h_ptr[i] != 0.F) {
				    bres = (uint32_t)is_equal_to_float(d_ptr[i],h_ptr[i]);
				    if(bres == 0) {
					    data->cloop_idx[i] = i;
					    data->gloop_idx[i] = i;
					    data->cpu_valid[i] = h_ptr[i];
					    data->gpu_invalid[i] = d_ptr[i];
				   }
				 }
		     }
            *ierr = 0;
	   }
		break;

	case 2: {
			 printf("Using floating-point comparison based on ulp distance. \n");
			 for(int32_t i = 0; i != len; ++i) {
				  if(_isnanf(d_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					  printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				  if(_isnanf(h_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				 if(d_ptr[i] != 0.F && h_ptr[i] != 0.F) {
				    bres = (uint32_t)compare_float(d_ptr[i],h_ptr[i],ulp);
				    if(bres == 0) {
					    data->cloop_idx[i] = i;
					    data->gloop_idx[i] = i;
					    data->cpu_valid[i] = h_ptr[i];
					    data->gpu_invalid[i] = d_ptr[i];
				   }
				 }
			 }
			 *ierr = 0;
	    }
		 break;

	case 3: {
			 printf("Using floating-point comparison within specific tolerance. \n");
			 for(int32_t i = 0; i != len; ++i) {
				  if(_isnanf(d_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					  printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				  if(_isnanf(h_ptr[i]) != 0) {
					  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i],i);
					  *ierr = -3;
					  return;
				  }
				 if(d_ptr[i] != 0.F && h_ptr[i] != 0.F) {
				    bres = (uint32_t)compare_float_within_tol(d_ptr[i],h_ptr[i],n,eps);
				    if(bres == 0) {
					    data->cloop_idx[i] = i;
					    data->gloop_idx[i] = i;
					    data->cpu_valid[i] = h_ptr[i];
					    data->gpu_invalid[i] = d_ptr[i];
				   }
				 }
			 }
			 *ierr = 0;
	    }
	   break;

	default : {
				REPORT_ERROR("***Non-Fatal*** -- Invalid argument to switch statement!!\n");
				*ierr = -2;
				return;
		  }
	}
 }

void compare_gpu_cpu_real4(const REAL4 * __restrict d_ptr,
						 const REAL4 * __restrict h_ptr,
						 const int32_t nx,
						 const int32_t ny,
						 struct FP4_COMP_METADATA * data,
						 const uint32_t method,
						 int32_t * ierr,
						 const char cfname[256],
						 const char gfname[256],
						 const uint32_t ulp,
						 const int32_t n,
						 const REAL4 eps)  {
	
	if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
		_ASSERTE(NULL != d_ptr &&
			    NULL != h_ptr &&
				0 < nx && 0 < ny &&
				NULL != data);
#else
	if(NULL == d_ptr || NULL == h_ptr ||
	   0 >= nx || 0 >= ny || NULL == data) {
		REPORT_ERROR("Invalid argument(s) in compare_gpu_cpu_real4 (array 2D)!")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == false)
		init_fp4_comp_metadata(data,(nx*ny),ierr);
	if(*ierr < 0) {
		REPORT_ERROR("***Non-Fatal*** -- init_fp4_comp_metadata: failed!!")
		return;
	}
	for(int32_t i = 0; i != 256; ++i) {
	    data->cfname[i] = cfname[i];
		data->gfname[i] = gfname[i];
	}
	uint32_t bres = 9999;
	// Begin switch statements
	// Various fp comparison methods are employed.
	printf("Begin comparison of GPU and CPU results.\n");

	switch(method) {

	case 0: {
			  FPCOMP_PRINT("Using naive floating-point comparison (to epsilon).")
			  for(int32_t i = 0; i != nx; ++i) {
				  for(int32_t j = 0; j != ny; ++j) {
					  if(_isnanf(d_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					       printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(_isnanf(h_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					       printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(d_ptr[i+nx*j] != 0.F && h_ptr[i+nx*j] != 0.F) {
					     if((fabsf(d_ptr[i+nx*j])-fabsf(h_ptr[i+nx*j])) > EPSF32) {
						     data->cloop_idx[i+nx*j] = i+nx*j;
						     data->gloop_idx[i+nx*j] = i+nx*j;
						     data->cpu_valid[i+nx*j] = h_ptr[i+nx*j];
						     data->gpu_invalid[i+nx*j] = d_ptr[i+nx*j];
					     }
					  }
				  }
			  }
			  *ierr = 0;
		}
		break;

	case 1: {
			  FPCOMP_PRINT("Using floating-point comparison based on relative spacing.")
			  for(int32_t i = 0; i != nx; ++i) {
				  for(int32_t j = 0; j != ny; ++j) {
					  if(_isnanf(d_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					       printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(_isnanf(h_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					       printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(d_ptr[i+nx*j] != 0.F && h_ptr[i+nx*j]) {
					     bres = (uint32_t)is_equal_to_float(d_ptr[i+nx*j],h_ptr[i+nx*j]);
					     if(bres == 0) {
						     data->cloop_idx[i+nx*j] = i+nx*j;
						     data->gloop_idx[i+nx*j] = i+nx*j;
						     data->cpu_valid[i+nx*j] = h_ptr[i+nx*j];
						     data->gpu_invalid[i+nx*j] = d_ptr[i+nx*j];
					    }
					  }
				  }
			  }
			*ierr = 0;
	    }
		break;

	case 2: {
			  FPCOMP_PRINT("Using floating-point comparison based on ulp distance.")
			  for(int32_t i = 0; i != nx; ++i) {
				  for(int32_t j = 0; j != ny; ++j) {
					  if(_isnanf(d_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					       printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(_isnanf(h_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					       printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(d_ptr[i+nx*j] != 0.F && h_ptr[i+nx*j] != 0.F) {
					     bres = (uint32_t)compare_float(d_ptr[i+nx*j],h_ptr[i+nx*j],ulp);
					     if(bres == 0) {
						     data->cloop_idx[i+nx*j] = i+nx*j;
						     data->gloop_idx[i+nx*j] = i+nx*j;
						     data->cpu_valid[i+nx*j] = h_ptr[i+nx*j];
						     data->gpu_invalid[i+nx*j] = d_ptr[i+nx*j];
					     }
					  }
				  }
			  }
			*ierr = 0;
	      }
		break;

	case 3:  {
				FPCOMP_PRINT("Using floating-point comparison within specific tolerance.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j  != ny; ++j) {
					 if(_isnanf(d_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					       printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(_isnanf(h_ptr[i+nx*j]) != 0) {
						   REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					       printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[i+nx*j],i+nx*j);
					       *ierr = -3;
					       return;
					  }
					  if(d_ptr[i+nx*j] != 0.F && h_ptr[i+nx*j] != 0.F) {
						  bres = (uint32_t)compare_float_within_tol(d_ptr[i+nx*j],h_ptr[i+nx*j],n,eps);
						  if(bres == 0) {
							   data->cloop_idx[i+nx*j] = i+nx*j;
							   data->gloop_idx[i+nx*j] = i+nx*j;
							   data->cpu_valid[i+nx*j] = h_ptr[i+nx*j];
							   data->gpu_invalid[i+nx*j] = d_ptr[i+nx*j];
						}
					  }
					}
				}
				*ierr = 0;
	    }
	    break;

	default : {
				REPORT_ERROR("***Non-Fatal*** -- Invalid switch statment parameter!!")
				*ierr = -2;
				return;
		  }
	}
}

#include "common.cuh"

void compare_gpu_cpu_real4(const REAL4 * __restrict d_ptr,
						 const REAL4 * __restrict h_ptr,
						 const int32_t nx,
						 const int32_t ny,
						 const int32_t nz,
						 struct FP4_COMP_METADATA * data,
						 const uint32_t method,
						 int32_t * ierr,
						 const char cfname[256],
						 const char gfname[256],
						 const uint32_t ulp,
						 const int32_t n,
						 const REAL4 eps) {

	if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != d_ptr  &&
		    NULL != h_ptr  &&
			0 < nx && 0 < ny &&
			0 < nz && NULL != data);
#else
	if(NULL == d_ptr || NULL == h_ptr ||
	   nx > 0 || ny > 0 || nz > 0 || 
	   NULL == data   )         {
		REPORT_ERROR("***Non-Fatal*** -- Invalid argument(s) in compare_gpu_cpu_real4 (array 3D)")
		*ierr = -1;
		return;
	}
#endif
	if(data->is_allocated == false)
		init_fp4_comp_metadata(data,(nx*ny*nz),ierr);
	if(*ierr < 0) {
		REPORT_ERROR("***Non-Fatal*** -- init_fp4_comp_metadata: failed!!")
		*ierr = -2;
		return;
	}
	for(int32_t i = 0; i != 256; ++i) {
		data->cfname[i] = cfname[i];
		data->gfname[i] = gfname[i];
	}
	uint32_t bres = 9999;
	// Begin switch statements
	// Various fp comparison methods are employed.
	printf("Begin comparison of GPU and CPU results.\n");
	switch(method) {

	case 0: {
				FPCOMP_PRINT("Using naive floating-point comparison (to epsilon).")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							if(_isnanf(d_ptr[I3D(i,j,k)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					             printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(_isnanf(h_ptr[I3D(i,k,j)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					             printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(d_ptr[I3D(i,j,k)] != 0.F && h_ptr[I3D(i,j,k)] != 0.F) {
							   if((fabsf(d_ptr[I3D(i,j,k)])-fabsf(h_ptr[I3D(i,j,k)])) > EPSF32) {
								    data->cloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								    data->gloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								    data->cpu_valid[I3D(i,j,k)] = h_ptr[I3D(i,j,k)];
								    data->gpu_invalid[I3D(i,j,k)] = d_ptr[I3D(i,j,k)];
							    }
							}
						}
					}
				}
				*ierr = 0;
	       }
		    break;

	case 1:  {
				FPCOMP_PRINT("Using floating-point comparison based on relative spacing.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							if(_isnanf(d_ptr[I3D(i,j,k)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					             printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(_isnanf(h_ptr[I3D(i,k,j)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					             printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(d_ptr[I3D(i,j,k)] != 0.F && h_ptr[I3D(i,j,k)] != 0.F) {
							    bres = (uint32_t)is_equal_to_float(d_ptr[I3D(i,j,k)],h_ptr[I3D(i,j,k)]);
							      if(bres == 0) {
								      data->cloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								      data->gloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								      data->cpu_valid[I3D(i,j,k)] = h_ptr[I3D(i,j,k)];
								      data->gpu_invalid[I3D(i,j,k)] = d_ptr[I3D(i,j,k)];
							   }
							}
						}
					}
				}
				*ierr = 0;
		 }
		 break;

	case 2: {
				FPCOMP_PRINT("Using floating-point comparison based on ulp distance.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
						   if(_isnanf(d_ptr[I3D(i,j,k)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					             printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(_isnanf(h_ptr[I3D(i,k,j)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					             printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(d_ptr[I3D(i,j,k)] != 0.F && h_ptr[I3D(i,k,j)] != 0.F) {
							    bres = (uint32_t)compare_float(d_ptr[I3D(i,j,k)],h_ptr[I3D(i,j,k)],ulp);
							       if(bres == 0) {
								       data->cloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								       data->gloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								       data->cpu_valid[I3D(i,j,k)] = h_ptr[I3D(i,j,k)];
								       data->gpu_invalid[I3D(i,j,k)] = d_ptr[I3D(i,j,k)];
							    }
							}
						}
					}
				}
				*ierr = 0;
	      }
		  break;

	case 3: {
				FPCOMP_PRINT("Using floating-point comparison within specific tolerance.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							if(_isnanf(d_ptr[I3D(i,j,k)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					             printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(_isnanf(h_ptr[I3D(i,k,j)]) != 0) {
								 REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
					             printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I3D(i,k,j)],I3D(i,k,j));
					             *ierr = -3;
					             return;
							}
							if(d_ptr[I3D(i,j,k)] != 0.F && h_ptr[I3D(i,k,j)] != 0.F) {
							    bres = (uint32_t)compare_float_within_tol(d_ptr[I3D(i,j,k)],h_ptr[I3D(i,j,k)],n,eps);
							       if(bres == 0) {
								        data->cloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								        data->gloop_idx[I3D(i,j,k)] = I3D(i,j,k);
								        data->cpu_valid[I3D(i,j,k)] = h_ptr[I3D(i,j,k)];
								        data->gpu_invalid[I3D(i,j,k)] = d_ptr[I3D(i,j,k)];
							     }
							}
						}
					}
				}
				*ierr = 0;
	 	   }
		   break;

	default : {
				REPORT_ERROR("***Non-Fatal*** -- Invalid switch statment parameter!!")
				*ierr = -2;
				return;
		 }
	}
}

#if !defined (FPCOMP_COPY_NAMES)
#define FPCOMP_COPY_NAMES    \
	for(int32_t i = 0; i != 256; ++i) { \
		data->cfname[i] = cfname[i];    \
		data->gfname[i] = gfname[i];    \
	}
#endif

void compare_gpu_cpu_real4(const REAL4 * __restrict d_ptr,
						 const REAL4 * __restrict h_ptr,
						 const int32_t nx,
						 const int32_t ny,
						 const int32_t nz,
						 const int32_t nw,
						 struct FP4_COMP_METADATA * data,
						 const uint32_t method,
						 int32_t * ierr,
						 const char cfname[256],
						 const char gfname[256],
						 const uint32_t ulp,
						 const int32_t n,
						 const REAL4 eps)  {
	if(*ierr < 0) *ierr = 0;
#if (CuWRF_DEBUG_ON) == 1
		_ASSERTE(NULL != d_ptr &&
				NULL != h_ptr &&
				0 < nx && 0 < ny &&
				0 < nz && 0 < nw &&
				NULL != data);
#else
	if(NULL == d_ptr ||
	   NULL == h_ptr ||
	   0 >= nx || 0 >= ny ||
	   0 >= nz || 0 >= nw ||
	   NULL == data ) {
		   REPORT_ERROR("***Non-Fatal*** -- Invalid argument(s) in compare_gpu_cpu_real4 (arrays 4D).")
		   *ierr = -1;
		   return;
	}
#endif
	if(data->is_allocated == false)
		init_fp4_comp_metadata(data,(nx*ny*nz*nw),ierr);
	if(*ierr < 0) {
		REPORT_ERROR("***Non-Fatal -- init_fp4_comp_metadata: failed!!")
		*ierr = -2;
		return;
	}
	FPCOMP_COPY_NAMES
	uint32_t bres = 9999;
	// Begin switch statements
	// Various fp comparison methods are employed.
	printf("Begin comparison of GPU and CPU results.\n");
	switch(method) {

	case 0: {
				FPCOMP_PRINT("Using naive floating-point comparison (to epsilon).")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							for(int32_t l = 0; l != nw; ++l) {
								if(_isnanf(d_ptr[I4D(i,k,j,l)]) != 0) {
									 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					                 printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                 *ierr = -3;
					                 return;
								}
								if(_isnanf(h_ptr[I4D(i,k,j,l)]) != 0) {
									  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
									  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                  *ierr = -3;
					                  return;
								}
								if(d_ptr[I4D(i,j,k,l)] != 0.F && h_ptr[I4D(i,j,k,l)] != 0.F) {
									if((fabsf(d_ptr[I4D(i,j,k,l)])-fabsf(h_ptr[I4D(i,j,k,l)])) > EPSF32) {
										data->cloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
										data->gloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
										data->cpu_valid[I4D(i,j,k,l)] = h_ptr[I4D(i,j,k,l)];
										data->gpu_invalid[I4D(i,j,k,l)] = d_ptr[I4D(i,j,k,l)];
									}
								}
							}
						}
					}
				}
				*ierr = 0;
	       }
		   break;

	case 1: {
				FPCOMP_PRINT("Using floating-point comparison based on relative spacing.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							for(int32_t l = 0; l != nw; ++l) {
								if(_isnanf(d_ptr[I4D(i,k,j,l)]) != 0) {
									 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					                 printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                 *ierr = -3;
					                 return;
								}
								if(_isnanf(h_ptr[I4D(i,k,j,l)]) != 0) {
									  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
									  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                  *ierr = -3;
					                  return;
								}
								if(d_ptr[I4D(i,j,k,l)] != 0.F && h_ptr[I4D(i,j,k,l)] != 0.F) {
									bres = (uint32_t)is_equal_to_float(d_ptr[I4D(i,j,k,l)],h_ptr[I4D(i,j,k,l)]);
									   if(bres == 0) {
										    data->cloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->gloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->cpu_valid[I4D(i,j,k,l)] = h_ptr[I4D(i,j,k,l)];
											data->gpu_invalid[I4D(i,j,k,l)] = d_ptr[I4D(i,j,k,l)];
									}
								}
							}
						}
					}
				}
				*ierr = 0;
		   }
		  break;

	case 2: {
				FPCOMP_PRINT("Using floating-point comparison based on ulp distance.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							for(int32_t l = 0; l != nw; ++l) {
							    if(_isnanf(d_ptr[I4D(i,k,j,l)]) != 0) {
									 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					                 printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                 *ierr = -3;
					                 return;
								}
								if(_isnanf(h_ptr[I4D(i,k,j,l)]) != 0) {
									  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
									  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                  *ierr = -3;
					                  return;
								}
								if(d_ptr[I4D(i,j,k,l)] != 0.F && h_ptr[I4D(i,j,k,l)] != 0.F) {
									bres = (uint32_t)compare_float(d_ptr[I4D(i,j,k,l)],h_ptr[I4D(i,j,k,l)],ulp);
									   if(bres == 0) {
										    data->cloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->gloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->cpu_valid[I4D(i,j,k,l)] = h_ptr[I4D(i,j,k,l)];
											data->gpu_invalid[I4D(i,j,k,l)] = d_ptr[I4D(i,j,k,l)];
									} 
								}
							}
						}
					}
				}
				*ierr = 0;
		}
		break;

	case 3: {
				FPCOMP_PRINT("Using floating-point comparison within specific tolerance.")
				for(int32_t i = 0; i != nx; ++i) {
					for(int32_t j = 0; j != ny; ++j) {
						for(int32_t k = 0; k != nz; ++k) {
							for(int32_t l = 0; l != nw; ++l) {
							   if(_isnanf(d_ptr[I4D(i,k,j,l)]) != 0) {
									 REPORT_ERROR("Detected invalid number -- NAN in d_ptr array!!")
					                 printf("NAN: <%.6f> at index: d_ptr[%d].\n",d_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                 *ierr = -3;
					                 return;
								}
								if(_isnanf(h_ptr[I4D(i,k,j,l)]) != 0) {
									  REPORT_ERROR("Detected invalid number -- NAN in h_ptr array!!")
									  printf("NAN: <%.6f> at index: h_ptr[%d].\n",h_ptr[I4D(i,k,j,l)],I4D(i,k,j,l));
					                  *ierr = -3;
					                  return;
								}
								if(d_ptr[I4D(i,j,k,l)] != 0.F && h_ptr[I4D(i,j,k,l)] != 0.F) {
									bres = (uint32_t)compare_float_within_tol(d_ptr[I4D(i,k,j,l)],h_ptr[I4D(i,k,j,l)],n,eps);
									   if(bres == 0) {
										    data->cloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->gloop_idx[I4D(i,j,k,l)] = I4D(i,j,k,l);
											data->cpu_valid[I4D(i,j,k,l)] = h_ptr[I4D(i,j,k,l)];
											data->gpu_invalid[I4D(i,j,k,l)] = d_ptr[I4D(i,j,k,l)];
									   }
								}
							}
						}
					}
				}
			    *ierr = 0;
		}
		break;

	default : {
				REPORT_ERROR("***Non-Fatal*** -- Invalid switch statment parameter!!")
				*ierr = -2;
				return;
		 }
	}
}