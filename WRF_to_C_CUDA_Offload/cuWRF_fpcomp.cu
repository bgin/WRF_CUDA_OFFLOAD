
#include "cuWRF_fpcomp.cuh"
#include <math.h>
#include <intrin.h>

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