
#ifndef __CUWRF_FPEXCEPT_CUH__
#define __CUWRF_FPEXCEPT_CUH__


#if !defined (CUWRF_FPEXCEPT_MAJOR)
#define CUWRF_FPEXCEPT_MAJOR 1
#endif

#if !defined (CUWRF_FPEXCEPT_MINOR)
#define CUWRF_FPEXCPET_MINOR 0
#endif

#if !defined (CUWRF_FPEXCEPT_MICRO)
#define CUWRF_FPEXCEPT_MICRO 0
#endif

#if !defined (CUWRF_FPEXCEPT_FULLVER)
#define CUWRF_FPEXCEPT_FULLVER 1000
#endif

#if !defined (CUWRF_FPEXCEPT_CREATE_DATE)
#define CUWRF_FPEXCEPT_CREATE_DATE "24-12-2017 13:12 +00200 (SUN 24 DEC 2017 GMT+2)"
#endif
//
//	Set this value after successful build.
//
#if !defined (CUWRF_FPEXCEPT_BUILD_DATE)
#define CUWRF_FPEXCEPT_BUILD_DATE " "
#endif

#if !defined (CUWRF_FPEXCEPT_AUTHOR)
#define CUWRF_FPEXCEPT_AUTHOR "Programmer: Bernard Gingold e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_FPEXCEPT_DESCRIPT)
#define CUWRF_FPEXCEPT_DESCRIPT "Various floating-point exception handling routines."
#endif

#include <cstdint>
#include "cuWRF_typedefs.cuh"


  /*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 1D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
      
        @Calls:   Nothing
     */
void is_domain1D_ltz(const REAL4 * __restrict,
					    const int64_t,
					    uint64_t *,
						int32_t *,
						const bool );

/*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 2D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
     */
void is_domain2D_ltz(const REAL4 * __restrict,
						const int64_t,
						const int64_t,
					    uint64_t *,
						int32_t *,
						const bool );

/*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 3D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
     */
void is_domain3D_ltz(const REAL4 * __restrict,
						const int64_t,
						const int64_t,
						const int64_t,
						uint64_t *,
						int32_t *,
						const bool);

/*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 4D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
     */
void is_domain4D_ltz(const REAL4 * __restrict,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   int32_t * ,
				   const bool );

 /*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 1D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
      
        @Calls:   Nothing
     */
void is_domain1D_ltz(const REAL8 * __restrict,
				   const int64_t,
				   uint64_t *,
				   int32_t * ,
				   const bool);

 /*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 2D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
      
        @Calls:   Nothing
     */
void is_domain2D_ltz(const REAL8 * __restrict,
				   const int64_t,
				   const int64_t,
				   uint64_t *,
				   int32_t * ,
				   const bool);

 /*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 3D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
      
        @Calls:   Nothing
     */
void is_domain3D_ltz(const REAL8 * __restrict,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t  *,
				   int32_t * ,
				   const bool);

 /*
        @Description: Checks for input domain of x < 0.F
					Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < 0.F
					  
        @Params:  source array 4D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
      
        @Calls:   Nothing
     */
void is_domain4D_ltz(const REAL8 * __restrict,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t *,
				   int32_t *,
				   const bool);


	 /*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.F and > 1.F
                     
        @Params:  source array 1D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain1D_bt1(const REAL4 * __restrict ,
					const int64_t,
					uint64_t * ,
					int32_t *,
					const bool );

 /*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.F and > 1.F
                     
        @Params:  source array 2D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain2D_bt1(const REAL4 * __restrict,
					const int64_t ,
					const int64_t ,
					uint64_t * ,
					int32_t *,
					const bool );

	
 /*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.F and > 1.F
                     
        @Params:  source array 3D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain3D_bt1(const REAL4 * __restrict,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   int32_t *,
				   const bool );

/*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.F and > 1.F
                     
        @Params:  source array 3D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain4D_bt1(const REAL4 * __restrict,
					const int64_t,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t *,
					const bool );

 /*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.F and > 1.F
                     
        @Params:  source array 1D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain1D_bt1(const REAL8 * __restrict ,
				   const int64_t,
				   uint64_t * ,
				   int32_t * ,
				   const bool );

/*
        @Description: Checks for input domain of -1.0 <= x <= 1.0
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.0 and > 1.0
                     
        @Params:  source array 2D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain2D_bt1(const REAL8 * __restrict,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   int32_t * ,
				   const bool );

/*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.0 and > 1.0
                     
        @Params:  source array 3D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain3D_bt1(const REAL8 * __restrict ,
				   const int64_t ,
				   const int64_t ,
				   const int64_t ,
				   uint64_t * ,
				   int32_t * ,
				   const bool );

/*
        @Description: Checks for input domain of -1.0F <= x <= 1.0F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are < -1.0 and > 1.0
                     
        @Params:  source array 4D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
void is_domain4D_bt1(const REAL8 * __restrict,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   int32_t * ,
				   const bool  );

/*
        @Description: Checks for input domain of x[i] != 0.F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.F
                     
        @Params:  source array 1D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain1D_nez(const REAL4 * __restrict,
					const int64_t,
					uint64_t * ,
					int32_t *,
					const bool );

 /*
        @Description: Checks for input domain of x[i] != 0.F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.F.
                     
        @Params:  source array 2D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain2D_nez(const REAL4 * __restrict,
				    const int64_t ,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool );

 /*
        @Description: Checks for input domain of x[i] != 0.F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.F.
                     
        @Params:  source array 3D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain3D_nez(const REAL4 * __restrict,
				    const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool );

 /*
        @Description: Checks for input domain of x[i] != 0.F
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.F.
                     
        @Params:  source array 4D holding floats (32-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain4D_nez(const REAL4 * __restrict,
					const int64_t,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool );

 /*
        @Description: Checks for input domain of x[i] != 0.0
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.0
                     
        @Params:  source array 1D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain1D_nez(const REAL8 * __restrict,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool);

 /*
        @Description: Checks for input domain of x[i] != 0.0
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.0
                     
        @Params:  source array 2D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain2D_nez(const REAL8 * __restrict,
				    const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t *,
					const bool );

/*
        @Description: Checks for input domain of x[i] != 0.0
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.0
                     
        @Params:  source array 3D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain3D_nez(const REAL8 * __restrict ,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool  );

 /*
        @Description: Checks for input domain of x[i] != 0.0
                      Upon detecting error condition 'domain range'
                      indicator is accumulating the number of values
					which are == 0.0
                     
        @Params:  source array 4D holding doubles (64-bit)
        @Params:  lenght of source array
        @Params:  pointer to variable holding 'Domain Range'error value.
        @Params:  none
        @Params:  none
        @Returns: Nothing
        
        @Calls:   Nothing
	 */
 void is_domain4D_nez(const REAL8 * __restrict ,
					const int64_t,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					int32_t * ,
					const bool );


 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 1D
						  Upon detecting error condition 'domain range'
						  third argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 1D holding floats (32-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf32(const REAL4 * __restrict,
					const int64_t,
					uint64_t * ,
					const bool,
					const uint32_t );

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 2D
						  Upon detecting error condition 'domain range'
						  fourth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 2D holding floats (32-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf32(const REAL4 * __restrict,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   const bool,
				   const uint32_t );

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 3D
						  Upon detecting error condition 'domain range'
						  fifth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 3D holding floats (32-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf32(const REAL4 * __restrict data,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint32_t * ,
				   const bool,
				   const uint32_t );

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 4D
						  Upon detecting error condition 'domain range'
						  sixth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 4D holding floats (32-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf32(const REAL4 * __restrict ,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   const int64_t,
				   uint64_t * ,
				   const bool,
				   const uint32_t);

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 1D
						  Upon detecting error condition 'domain range'
						  third argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 1D holding doubles (64-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf64(const REAL8 * __restrict,
				   const int64_t ,
				   uint64_t * ,
				   const bool,
				   const uint32_t );

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 2D
						  Upon detecting error condition 'domain range'
						  fourth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 2D holding doubles (64-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf64(const REAL8 * __restrict,
				    const int64_t,
					const int64_t,
					uint64_t * ,
					const bool,
					const uint32_t );

 /*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 3D
						  Upon detecting error condition 'domain range'
						  fifth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 3D holding doubles (64-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf64(const REAL8 * __restrict,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					const bool,
					const uint32_t );

/*
			@Description: Checks for existance of invalid values
						  in domain range of dimension 4D
						  Upon detecting error condition 'domain range'
						  sixth argument accumulates number of denormal
						  occurrences in array.
						  Checks for:
						  INF
						  DENORMAL
						  NAN
						  Fast version without input checking
		@Params:  1st source array 4D holding doubles (64-bit)
		@Params:  none
		@Params:  length of  source array.
		@Params:  none
		@Params:  pointer to variable holding 'Domain Range'error value.
		@Returns: Nothing
		@Throws:  Nothing
		@Calls:   fpclassify
		*/
 void is_abnormalf64(const REAL8 * __restrict,
				    const int64_t,
					const int64_t,
					const int64_t,
					const int64_t,
					uint64_t * ,
					const bool,
					const uint32_t );

  /*
        @Description: Clears all floating-point state exceptions
                      Exception cleared:
					  FE_DENORMAL, FE_INVALID, FE_INEXACT, FE_UNDERFLOW
					  FE_OVERFLOW.
                      Scalar version
        @Params:  none
        @Params:  none
        @Params:  none
        @Params:  none
        @Params:  none
        @Returns: integer 'err' which indicates success or error as a return
				  value from library 'feclearexcept' function
				  Non-zero value means error.
        
        @Calls:   'feclearexcept'
     */
 int32_t clear_fexcepts(void);

  /*
        @Description: Clears only FE_DENORMAL exception
                      Exception cleared:
                      FE_DENORMAL
        @Params:  none
        @Params:  none
        @Params:  none
        @Params:  none
        @Params:  none
        @Returns: integer 'err' which indicates success or error as a return
                  value from library 'feclearexcept' function
                  Non-zero value means error.
        
        @Calls:   'feclearexcept'
     */
 int32_t clear_fedenormal(void);

 /*
		@Description: Clears only FE_INEXACT exception
		Exception cleared:
		FE_INEXACT
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'err' which indicates success or error as a return
		value from library 'feclearexcept' function
		Non-zero value means error.
		
		@Calls:   'feclearexcept'
		*/
 int32_t clear_feinexact(void);

 /*
		@Description: Clears only FE_INVALID exception
		Exception cleared:
		FE_INVALID
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'err' which indicates success or error as a return
		value from library 'feclearexcept' function
		Non-zero value means error.
		
		@Calls:   'feclearexcept'
		*/
 int32_t clear_feinvalid(void);

 /*
		@Description: Clears only FE_DIVBYZERO exception
		Exception cleared:
		FE_DIVBYZERO
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'err' which indicates success or error as a return
		value from library 'feclearexcept' function
		Non-zero value means error.
		
		@Calls:   'feclearexcept'
		*/
 int32_t clear_fedivbyzero(void);

 /*
		@Description: Clears only FE_OVERFLOW exception
		Exception cleared:
		FE_OVERFLOW
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'err' which indicates success or error as a return
		value from library 'feclearexcept' function
		Non-zero value means error.
		
		@Calls:   'feclearexcept'
		*/
 int32_t clear_feoverflow(void);

 /*
		@Description: Clears only FE_UNDERFLOW exception
		Exception cleared:
		FE_UNDERFLOW
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'err' which indicates success or error as a return
		value from library 'feclearexcept' function
		Non-zero value means error.
		
		@Calls:   'feclearexcept'
		*/
 int32_t clear_feunderflow(void);

 /*
		@Description: Tests if all floating-point exceptions have been set.

		@Params:  All 7 floating-point exception types (exception values must be or'ed).
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_allfpexcepts(const int32_t);

 /*
		@Description: Tests for existance of FE_INVALID exception.

		@Params:  argument FE_INVALID macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_feinvalid(const int32_t);

 /*
		@Description: Tests for existance of FE_INEXACT exception.

		@Params:  argument FE_INEXACT macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_feinexact(const int32_t);

 /*
		@Description: Tests for existance of FE_DIVBYZERO exception.

		@Params:  argument FE_DIVBYZERO macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_fedivbyzero(const int32_t);

 /*
		@Description: Tests for existance of FE_UNNORMAL exception.

		@Params:  argument FE_UNNORMAL macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_fedenormal(const int32_t);

 /*
		@Description: Tests for existance of FE_OVERFLOW exception.

		@Params:  argument FE_OVERFLOW macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_feoverflow(const int32_t);

 /*
		@Description: Tests for existance of FE_UNDERFLOW exception.

		@Params:  argument FE_UNDERFLOW macro.
		@Params:  none
		@Params:  none
		@Params:  none
		@Params:  none
		@Returns: integer 'val' which indicates success or error as a return
		value from library 'fetestexcept' function

		
		@Calls:   'fetestexcept'
		*/
 int32_t test_feunderflow(const int32_t);

 /*
		Clean previous exception if has been set and raise an exception.
		Raise FE_DENORMAL exception.
 */
 void raise_fedenormal(const bool,
					 const int32_t,
					 int32_t * );

 /*
		Clean previous exception if has been set and raise an exception.
		Raise FE_INVALID exception.
 */
 void raise_feinvalid(const bool,
				     const int32_t,
					 int32_t * );

 /*
	    Clean previous exception if has been set and raise an exception.
		Raise FE_INEXACT exception.

 */
 void raise_feinexact(const bool,
					const int32_t,
					int32_t * );

 /*
		Clean previous exception if has been set and raise an exception.
		Raise FE_DIVBYZERO exception.
	*/
 void raise_fedivbyzero(const bool,
				      const int32_t,
					  int32_t * );

 /*
		Clean previous exception if has been set and raise an exception.
		Raise FE_OVERFLOW exception.
 */
 void raise_feoverflow(const bool,
					 const int32_t,
					 int32_t * );

 /*
		Clean previous exception if has been set and raise an exception.
		Raise FE_UNDERFLOW exception
 */
 void raise_feunderflow(const bool,
					  const int32_t,
					  int32_t * );

 
 /*
	Set specific rounding mode.
 */
 void set_round_mode(const int32_t,
				   int32_t * ierr);

 /*
	Get current rounding mode.
 */
 int32_t get_round_mode(void);

 /*
	Show rounding mode.
 */
 void show_round_mode(const int32_t ,
					 int32_t * );


 

#endif   /*__CUWRF_FPEXCEPT_CUH__*/