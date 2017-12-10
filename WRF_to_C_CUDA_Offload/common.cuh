
#ifndef _COMMON_CUH_
#define _COMMON_CUH_

/*
	Flat multidimensional array indices
*/

#define Idx2D(i,j)       ((i) + idim * (j))

#define Idx3D(i,j,k)     ((i) + idim * ((j) + jdim * (k)))

#define Idx4D(i,j,k,l)   ((i) + idim * ((j) + jdim * (k) + kdim * (l)))

#define Idx5D(i,j,k,l,m) ((i) + idim * ((j) + jdim * (k) + kdim * (l) + ldim * (m)))

// Different indexing scheme.

#define IS2D(i,idim,j)   i + idim * j

#define IS3D(i,idim,j,jdim,k) i + idim * (j + jdim * k)

#define IS4D(i,idim,j,jdim,k,kdim,l) i + idim * (j + jdim * (k + kdim * l))

#define IS5D(i,idim,j,jdim,k,kdim,l,ldim,m) i + idim * (j + jdim * (k + kdim + (l + ldim * m)))

typedef float REAL4;

typedef double REAL8;



#endif /*_COMMON_CUH_*/