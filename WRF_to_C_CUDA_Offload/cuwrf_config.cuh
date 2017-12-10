
#ifndef _CUWRF_CONFIG_CUH_
#define _CUWRF_CONFIG_CUH_

#if !defined (CUWRF_CONFIG_MAJOR)
#define CUWRF_CONFIG_MAJOR 1
#endif

#if !defined (CUWRF_CONFIG_MINOR)
#define CUWRF_CONFIG_MINOR 0
#endif

#if !defined (CUWRF_CONFIG_MICRO)
#define CUWRF_CONFIG_MICRO 0
#endif

#if !defined (CUWRF_CONFIG_VER_FULL)
#define CUWRF_CONFIG_VER_FULL 1000
#endif

#if !defined (CUWRF_CONFIG_CREATION_DATE)
#define CUWRF_CONFIG_CREATION_DATE "24-06-2017 13:03 PM -00200 (Sat 24 Jun 2017 13:03 PM -00200)"
#endif

#if !defined (CUWRF_CONFIG_BUILD_DATE)
#define CUWRF_CONFIG_BUILD_DATE " "
#endif

#if !defined (CUWRF_CONFIG_AUTHOR)
#define CUWRF_CONFIG_AUTHOR "Programmer: Bernard Gingold, e-mail: beniekg@gmail.com"
#endif

#if !defined (CUWRF_CONFIG_SYNOPSIS)
#define CUWRF_CONFIG_SYNOPSIS "CuWRF global configuration file"
#endif
	
#include <stdio.h>


#if !defined (FORTRAN_CALLABLE)
#define FORTRAN_CALLABLE 1
#else
#define FORTRAN_CALLABLE 0
#endif


/*
	Short convenience functions for querying
	file basic information
*/
	// File version major
	int cuwrf_config_major() {
		return (CUWRF_CONFIG_MAJOR);
	}

	// File version minor
	int cuwrf_config_minor() {
		return (CUWRF_CONFIG_MINOR);
	}

	// File version micro
	int cuwrf_config_micro() {
		return (CUWRF_CONFIG_MICRO);
	}

	// File version full
	int cuwrf_config_verfull() {
		return (CUWRF_CONFIG_VER_FULL);
	}

	// File creation date
	const char * cuwrf_config_createdate() {
		return (CUWRF_CONFIG_CREATION_DATE);
	}

	// File build date
	const char * cuwrf_config_buildate() {
		return (CUWRF_CONFIG_BUILD_DATE);
	}

	// File author info
	const char * cuwrf_config_author() {
		return (CUWRF_CONFIG_AUTHOR);
	}

	// File synopsis
	const char * cuwrf_config_synopsis() {
		return (CUWRF_CONFIG_SYNOPSIS);
	}



#if !defined (CHECK_FORTRAN_ARRAYS)
#define CHECK_FORTRAN_ARRAYS 1
#endif

#if !defined (CHECK_FORTRAN_SCALAR_ARGS)
#define CHECK_FORTRAN_SCALAR_ARGS 1
#endif

#if defined _DEBUG
#define CuWRF_DEBUG_ON 1
#else
#define CuWRF_DEBUG_OFF 0
#endif

// Error handling macro
#if (CuWRF_DEBUG_ON) == 1
#define CuWRF_DEBUG_CHECK(func) do { \
 (status) = (func);  \
 if(cudaSuccess != (status)) { \
 fprintf(stderr, "CUDA Runtime Failure: (line %d of file %s) : \n\t" \
 "%s returned 0x%x (%s)\n", \
 __LINE__ , __FILE__ , #func,status, cudaGetErrorString(status));  \
 goto Error; \
 }    \
	} while(0) ;
#else
#define CuWRF_CHECK(func) do { \
  status = (func); \
  if(cudaSuccess != (status))  { \
     goto Error;
  } \
	} while(0);
#endif

#if defined(_WIN64) && CuWRF_DEBUG_ON == 1
#include <crtdbg.h>
#endif

// Workaround for Fortran optional argument on C-side.
#if !defined (FORTRAN_OPTIONAL)
#define FORTRAN_OPTIONAL 1
#endif

#if !defined (PRINT_ERROR_TO_SCREEN)
#define PRINT_ERROR_TO_SCREEN 1
#endif

#if !defined (PRINT_ERROR_TO_FILE)
#define PRINT_ERROR_TO_FILE 1
#endif

// cuWRF Assertions based on CUDA Handbook code
#if CuWRF_DEBUG_ON == 1
#define CuWRF_ASSERT(predicate) if ( ! (predicate)) __debugbreak();
#else
#define CuWRF_ASSERT(predicate) if ( ! (predicate)) _asm int 3
// helper option (not used)
#if 0
do { if(!(predicate)) {fprintf(stderr, "Asserion failed: %s at line %d in file %s\n", \
		#predicate, __LINE__,__FILE__); \
		_asm int 3; \
	      } \
 } while(0); 

#endif
#endif

//
// Impotrant:
//             Set this value to '1' 
//			  if your GPU memory has more then 4 GiB.
//
#if !defined (GPU_LARGE_MEM_SPACE)
#define GPU_LARGE_MEM_SPACE 0
#endif

#if !defined (VERBOSE) && (CuWRF_DEBUG_ON) == 1
#define DEBUG_VERBOSE 1
#endif


#if !defined (REPORT_ERROR)
#define REPORT_ERROR(msg) fprintf(stderr,"%s at line %d in file %s\n", \
	msg,__LINE__,__FILE__);
#endif

#if !defined (LOG_ACTIVITY)
#define LOG_ACTIVITY 1
#endif

#if !defined (CuWRF_LOG)
#define CuWRF_LOG(msg) printf("Logger: %s %s %s at line %d in file %s\n", \
	__DATE__,__TIME__ ,msg, __LINE__,__FILE__);
#endif

#if !defined (HOST_ALIGN32)
#define HOST_ALIGN32 32
#endif

#if !defined (HOST_ALIGN64) // Cache aware alignment.
#define HOST_ALIGN64 64
#endif


#include <stdlib.h>

//
//	Log error message to file and call exit
//

void exit_fatal(const char * msg,const char * fname) {

	_ASSERT(NULL != msg && NULL != fname);
	printf("***FATAL-ERROR***\n");
	printf(" %s\n",msg);

	FILE * fp = NULL;
	fp = fopen(fname,"a+");
	if(NULL != fp) {
		fprintf(fp, "FATAL ERROR: %s\n",msg);
		fclose(fp);
	}
	exit(EXIT_FAILURE);
}

//
// Log error message to file and call exit
// FATAL CUDA runtime error
//
void fatal_gpu_error(const char *msg, cudaError cuerr) {
	_ASSERT(NULL != msg);
	printf("***CUDA-RUNTIME***: FATAL-ERROR\n");
	printf("%s\n",msg);
	FILE* fp = NULL;
	fp = fopen("CUDA_RUNTIME_FILES/error_messages.txt","a+");
	if(NULL != fp) {
		cudaDeviceProp dp;
		cudaError stat1,stat2;
		int dev = -1;
		stat1 = cudaGetDevice(&dev);
		stat2 = cudaGetDeviceProperties(&dp,dev);
		if(stat1 == cudaSuccess && stat2 == cudaSuccess){
		   fprintf(fp,"\tCUDA-ERROR: !!!!-- [%s] --!!!! returned by device: [%s]\n",
				                 cudaGetErrorString(cuerr),dp.name);
		   fprintf(fp, " %s \n",msg);
		}
		else {
			 fprintf(fp,"\tCUDA-ERROR: !!!-- [%s] --!!! \n",msg);
		}
		fclose(fp);
	}
	exit(EXIT_FAILURE);
}


#endif /*_CUWRF_CONFIG_CUH_*/