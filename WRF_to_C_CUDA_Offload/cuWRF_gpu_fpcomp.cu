
#include "cuWRF_gpu_fpcomp.cuh"
#include "math_functions.h"


//
//	Implementation
//

__device__ float gpu_float_spacing(const float num, int32_t * exponent) {
	float mantissa = frexpf(num,exponent);
	return (powf(gRADIX2f32,(float)(*exponent-gEXPF32)));
}

__device__ bool gpu_equal_to_float(const float x, const float y) {
	bool bres = false;
	int exponent = 0;
	bres = fabsf(x-y) < gpu_float_spacing(fmaxf(fabsf(x),fabsf(y)),&exponent);
	return (bres);
}

__device__ bool gpu_greater_than_float(const float x, const float y) {
	bool bres = false;
	int exponent = 0;
	bres = fabsf(x-y) >= gpu_float_spacing(fmaxf(fabsf(x),fabsf(y)),&exponent);
	return (bres);
}

__device__ bool gpu_less_than_float(const float x, const float y) {
	bool bres = false;
	int exponent = 0;
	bres = fabsf(y-x) >= gpu_float_spacing(fmaxf(fabsf(x),fabsf(y)),&exponent);
	return (bres);
}

__device__ bool gpu_compare_to_float(const float x, const float y,
								   const  uint32_t ulp ) {
	bool bres = false;
	float relative = 0.F;
	int32_t exponent = 0;
	relative = fabsf((float)ulp);
	bres = fabsf(x-y) < (relative * gpu_float_spacing(fmaxf(fabsf(x),fabsf(y)),&exponent));
	return (bres);
}





