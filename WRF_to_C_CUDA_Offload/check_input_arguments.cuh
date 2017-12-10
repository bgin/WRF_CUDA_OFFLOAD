
#ifndef _CHECK_INPUT_ARGUMENTS_CUH_
#define _CHECK_INPUT_ARGUMENTS_CUH_

#include "cuwrf_config.cuh"
#include <stdlib.h>
#include <Windows.h> // For GetLastError call

#if !defined (CHECK_INPUT_ARGUMENTS_MAJOR)
#define CHECK_INPUT_ARGUMENTS_MAJOR 1
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_MINOR)
#define CHECK_INPUT_ARGUMENTS_MINOR 0
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_MICRO)
#define CHECK_INPUT_ARGUMENTS_MICRO 0
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_FULLVER)
#define CHECK_INPUT_ARGUMENTS_FULLVER 1000
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_CREATION_DATE)
#define CHECK_INPUT_ARGUMENTS_CREATION_DATE "24-06-2017 16:33 PM -00200 (Sat Jun 24 2017 16:33 PM -00200)"
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_BUILD_DATE)
#define CHECK_INPUT_ARGUMENTS_BUILD_DATE " "
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_AUTHOR)
#define CHECK_INPUT_ARGUMENTS_AUTHOR "Programmer: Bernard Gingold, e-mail: beniekg@gmail.com"
#endif

#if !defined (CHECK_INPUT_ARGUMENTS_SYNOPSIS)
#define CHECK_INPUT_ARGUMENTS_SYNOPSIS "Checks host-side functions input arguments."
#endif

/*
	File info getters
*/

	// File version major
	int check_input_arguments_major();

	// File version minor
	int check_input_arguments_minor();

	// File version micro
	int check_input_arguments_micro();

	// File version full
	int check_input_arguments_fullver();

	// File creation date
	const char * check_input_arguments_createdate();

	// File build date
	const char * check_input_arguments_buildate();

	// File author info
	const char * check_input_arguments_author();

	// File synopsis
	const char * check_input_arguments_synopsis();

	
	/*
		Declaration of various args checking helpers
		Scalar arguments checked against specific value.
		Upon detecting an error either:
		1) In debug build assertion is executed
		2) In release build exit is called
	*/

	void checkint32_exit_failure(const int,
							   const int,
							   const char);

	void checkint64_exit_failure(const long long, 
		                       const long long,
							   const char);

	void checkuint32_exit_failure(const unsigned int,
								const unsigned int,
								const char );

	void checkuint64_exit_failure(const unsigned long long, 
								const unsigned long long,
								const char );

	void checkf32_exit_failure(const float, 
						      const float, 
							  float,
							  const char);
							  

	void checkf64_exit_failure(const double, 
							const double, 
							const double,
							const char);
							

	/*
		Declaration of various args checking helpers
		Array (pointer) arguments checked for NULL.
		Upon detecting an error either:
		1) In debug build assertion is executed
		2) In release build exit is called
	*/
	void check_nullptr_int32_exit_failure(int *);

	void check_nullptr_int64_exit_failure(long long *);

	void check_nullptr_uint32_exit_failure(unsigned int *);

	void check_nullptr_uint64_exit_failure(unsigned long long *);

	void check_nullptr_f32_exit_failure(float *);

	void check_nullptr_f64_exit_failure(double *);

	/*
		Declaration of 'lightweight' error checking
		helpers.
		No error handling code is executed.
		Scalar arguments.
	*/
	bool check_int32(const int,  const int);

	bool check_int64(const long long, const long long);

	bool check_uint32(const unsigned int , const unsigned int);

	bool check_uint64(const unsigned long long, const unsigned long long);

	bool check_f32(const float, const float, const float);

	bool check_f64(const double, const double, const double);

	/*
		Declaration of 'lightweight' error checking
		helpers.
		No error handling code is executed.
		Array (pointer) arguments checked for NULL.
		Upon detecting an error bool = true is returned.
		
	*/

	bool check_nullptri32(const int *);

	bool check_nullptri64(const long long *);

	bool check_nullptrui32(const unsigned int *);

	bool check_nullptrui64(const unsigned long long *);

	bool check_nullptrf32(const float *);

	bool check_nullptrf64(const double *);


#endif /*_CHECK_INPUT_ARGUMENTS_CUH_*/