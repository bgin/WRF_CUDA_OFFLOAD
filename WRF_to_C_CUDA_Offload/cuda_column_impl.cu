
#include "cuda_column_impl.cuh"

int cuda_column_impl_major() {
	return (CUDA_COLUMN_IMPL_VER_MAJOR);
}

int cuda_column_impl_minor() {
	return (CUDA_COLUMN_IMPL_VER_MINOR);
}

int cuda_column_impl_micro() {
	return (CUDA_COLUMN_IMPL_VER_MICRO);
}

int cuda_column_impl_verfull() {
	return (CUDA_COLUMN_IMPL_VER_FULL);
}

const char * cuda_column_impl_createdate() {
	return (CUDA_COLUMN_IMPL_CREATION_DATE);
}

const char * cuda_column_impl_buildate() {
	return (CUDA_COLUMN_IMPL_BUILD_DATE);
}

const char * cuda_column_impl_author() {
	return (CUDA_COLUMN_IMPL_AUTHOR);
}

const char * cuda_column_impl_synopsis() {
	return (CUDA_COLUMN_IMPL_SYNOPSIS);
}

#if FORTRAN_CALLABLE == 1

void f90cuda_column_impl_major(int major) {
	major = cuda_column_impl_major();
}

void f90cuda_column_impl_minor(int minor) {
	minor = cuda_column_impl_minor();
}

void f90cuda_column_impl_micro(int micro) {
	micro = cuda_column_impl_micro();
}

void f90cuda_column_impl_fullver(int fullver) {
	fullver = cuda_column_impl_verfull();
}

void f90cuda_column_impl_createdate(char * cdate ) {
	cdate = (char *)cuda_column_impl_createdate();
}

void f90cuda_column_impl_buildate(char * bdate) {
	bdate = (char *)cuda_column_impl_buildate();
}

void f90cuda_column_impl_author(char * author) {
	author = (char*)cuda_column_impl_author();
}

void f90cuda_column_impl_synopsis(char * synopsis) {
	synopsis = (char *)cuda_column_impl_synopsis();
}

#endif

/*
	Implementation of Host routine column
	Callable from Fortran.
*/
void column(const int m, const int np, const double * __restrict pa,
		   const double * __restrict dt, const double * __restrict sabs0,
		   double * __restrict sabs, double * __restrict spre, double * __restrict stem) {

#if CHECK_FORTRAN_SCALAR_ARGS == 1
	assert(m  != 0);
	assert(np != 0);
#endif

#if CHECK_FORTRAN_ARRAYS == 1
	 // Sanity check on Fortran arrays input.
     assert(pa    != NULL);
	 assert(dt    != NULL);
	 assert(sabs0 != NULL);
	 assert(sabs  != NULL);
	 assert(spre  != NULL);
	 assert(stem  != NULL);
#endif

	// 2D Input
	double *dev_pa    = NULL; 
	double *dev_dt    = NULL;
	double *dev_sabs0 = NULL;
	// 2D Output
	double *dev_sabs  = NULL;
	double *dev_spre  = NULL;
	double *dev_stem  = NULL;


}
		   

