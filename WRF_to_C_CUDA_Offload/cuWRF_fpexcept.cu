
#include <../../../Microsoft Visual Studio 12.0/VC/include/math.h> // for fpclassify
#include "cuWRF_fpexcpt.cuh"
#include "cuwrf_config.cuh"
#include "common.cuh"

//
//	Implementation
//

#if !defined (FPEXCEPT_CHECK_IERR)
#define FPEXCPET_CHECK_IERR(val)    \
	do {                           \
	     if(*(val) < 0) *(val) = 0;  \
	}while(0);  
#endif

#if !defined (FPEXCEPT_CHECK_LTZ)     
#define FPEXCEPT_CHECK_LTZ(val)       \
   do {                             \
		if(*(val) > 0) *(val) = 0;   \
   }while(0);
#endif

#if !defined (CuWRF_FPEXCEPT_FENV_RETURN)
#define CuWRF_FPEXCEPT_FENV_RETURN(msg,err) \
	do{									\
		if((err) != 0) {                 \
		    REPORT_ERROR_VALUE((msg),(err)) \
			*ierr = -1;                   \
             return;						 \
		}                                \
		                                 \
	}while(0);
#endif




void is_domain1D_ltz(const REAL4 * __restrict data,
				   const int64_t nx,
				   uint64_t * ltz ,
				   int32_t * ierr,
				   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data && 0LL < nx);
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_ltz!")
		ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
	   for(int64_t i = 0LL; i != nx; ++i) {
		     if(data[i] < 0.F)
			      *ltz += 1;
	    }
	*ierr = 0;
	}
	else {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] < 0.F) {
				*ltz = 1;
				*ierr = 0;
				return;
			}
		}
		*ierr = 0;
	}
}

void is_domain2D_ltz(const REAL4 * __restrict data,
				   const int64_t nx,
				   const int64_t ny,
				   uint64_t * ltz,
				   int32_t * ierr,
				   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data  &&
			0LL < nx      &&
			0LL < ny      );
#else
	if(NULL == data || 0LL >= nx || 
	   0LL >= ny  ) {
		REPORT_ERROR("Invalid argument(s) in is_domain2D_ltz!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
	     for(int64_t i = 0LL; i != nx; ++i) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
		     for(int64_t j = 0LL; j != ny; ++j) {
			      if(data[I2D(i,j)] < 0.F)
				       *ltz += 1;
		}
	}
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] < 0.F) {
					*ltz = 1;
					*ierr = 0;
					return;
				}
			}
		}
		*ierr = 0;
	}
}

void is_domain3D_ltz(const REAL4 * __restrict data,
				   const int64_t nx,
				   const int64_t ny,
				   const int64_t nz,
				   uint64_t * ltz,
				   int32_t * ierr,
				   const bool process_slowly)  {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
	        0LL < nx     &&
			0LL < ny     &&
			0LL < nz   );
#else
	if(NULL == data ||
	   0LL >=  nx   ||
	   0LL >=  ny   ||
	   0LL >=  nz   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain3D_ltz! ")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
	    for(int64_t i = 0LL; i != nx; ++i) {
		    for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch:0:8
//#pragma prefetch:1:8
			    for(int64_t k = 0LL; k != nz; ++k) {
				     if(data[I3D(i,j,k)] < 0.F)
					     *ltz += 1;
			}
		}
	}
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					  if(data[I3D(i,j,k)] < 0.F) {
						  *ltz = 1;
						  *ierr = 0;
						  return;
					  }
				}
			}
		}
		*ierr = 0;
	}
}

void is_domain4D_ltz(const REAL4 * __restrict data,
					const int64_t nx,
					const int64_t ny,
					const int64_t nz,
					const int64_t nw,
					uint64_t * ltz,
					int32_t  * ierr,
					const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx    &&
			0LL < ny    &&
			0LL < nz    &&
			0LL < nw    );
#else
	if(NULL == data ||
	   0LL >=  nx   ||
	   0LL >=  ny   ||
	   0LL >=  nz   ||
	   0LL >=  nw  ) {
		REPORT_ERROR("Invalid argument(s) in is_domain4D_ltz!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i){
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
//#pragma prefetch:0:8
//#pragma prefetch:1:8
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,k,j,l)] < 0.F)
							*ltz += 1;
				}
			}
		}
	}
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < 0.F) {
							*ltz = 1;
							*ierr = 0;
							return;
						}
					}
				}
			}
		}
		*ierr = 0;
	}
}

void	  is_domain1D_ltz(const REAL8 * __restrict data,
					const int64_t nx,
					uint64_t * ltz,
					int32_t * ierr,
					const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
		    0LL < nx );
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_ltz")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
	  for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] < 0.0)
				*ltz += 1;
		}
	*ierr = 0;
	}
	else {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] < 0.0) {
				*ltz = 1;
				*ierr = 0;
				return;
			}
		}
		*ierr = 0;
	}
}

void		is_domain2D_ltz(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   uint64_t * ltz,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx && 0LL < ny);
#else
	if(NULL == data || 
	   0LL >= nx   ||
	   0LL >= ny   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain2D_ltz")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0; i != nx; ++i) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
			for(int64_t j = 0; j != ny; ++j) {
				if(data[I2D(i,j)] < 0.0 )
					*ltz += 1;
		}
	}
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] < 0.0) {
					*ltz = 1;
					*ierr = 0;
					return;
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain3D_ltz(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   uint64_t * ltz,
					   int32_t * ierr ,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx     &&
			0LL < ny     &&
			0LL < nz    );
#else
	if(NULL == data || 
	  0LL >= nx     ||
	  0LL >= ny     ||
	  0LL >= nz    ) {
		REPORT_ERROR("Invalid argument(s) in is_domain3D_ltz")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0; i != nx; ++i) {
			for(int64_t j = 0; j != ny; ++j) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
				for(int64_t k = 0; k != nz; ++k) {
					if(data[I3D(i,j,k)] < 0.0 )
						*ltz += 1;
			  }
		  }
	   }
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] < 0.0) {
						*ltz = 1;
						*ierr = 0;
						return;
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain4D_ltz(const REAL8 * __restrict data,
					  const int64_t nx,
					  const int64_t ny,
					  const int64_t nz,
					  const int64_t nw,
					  uint64_t * ltz,
					  int32_t  * ierr,
					  const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(ltz)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
	        0LL < nx     &&
			0LL < ny     &&
			0LL < nz     &&
			0LL < nw     );
#else
	if(NULL == data ||
	   0LL >= nx   ||
	   0LL >= ny   ||
	   0LL >= nz   ||
	   0LL >= nw )  {
		REPORT_ERROR("Invalid argument(s) in is_domain4D_ltz")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0; i != nx; ++i) {
			for(int64_t j = 0; j != ny; ++j) {
				for(int64_t k = 0; k != nz; ++k) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
					for(int64_t l = 0; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < 0.0)
							*ltz += 1;
					}
				}
			}
		}
	*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
//#pragma prefetch data:0:8
//#pragma prefetch data:1:8
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < 0.0) {
							*ltz = 1;
							*ierr = 0;
							return;
						}
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain1D_bt1(const REAL4 * __restrict data,
					  const int64_t nx,
					  uint64_t * bt1,
					  int32_t * ierr,
					  const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx);
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_bt1")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
	    for(int64_t i = 0; i != nx; ++i) {
		     if(data[i] < -1.F || data[i] > 1.F)
			     *bt1 += 1;
	}
	*ierr = 0;
  } 
	else {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
		 for(int64_t i = 0LL; i != nx; ++i) {
			 if(data[i] < -1.F || data[i] > 1.F){
				 *bt1 = 1;
				 *ierr = 0;
				 return;
			 }
		 }
		 *ierr = 0;
	}
}

void		is_domain2D_bt1(const REAL4 * __restrict data,
					  const int64_t nx,
					  const int64_t ny,
					  uint64_t * bt1,
					  int32_t * ierr,
					  const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx   &&
			0LL < ny);
#else
	if(NULL == data ||
	  0LL >= nx    ||
	  0LL >= ny ) {
		  REPORT_ERROR("Invalid argument(s) in is_domain2D_bt1")
		  *ierr = -1;
		  return;
	}
#endif
	if(process_slowly == true) {
	      for(int64_t i = 0LL; i != nx; ++i) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
		      for(int64_t j = 0LL; j != ny; ++j) {
			       if(data[I2D(i,j)] < -1.F || data[I2D(i,j)] > 1.F)
				        *bt1 += 1;
		     }
	     }
	   *ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] < -1.F || data[I2D(i,j)] > 1.F) {
					*bt1 = 1;
					*ierr = 0;
					return;
				}

			}
		}
		*ierr = 0;
	}
}

void		is_domain3D_bt1(const REAL4 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   uint64_t * bt1,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx     &&
			0LL < ny     &&
			0LL < nz     );
#else
	if(NULL == data ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain3D_bt1")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] < -1.F || data[I3D(i,j,k)] > 1.F)
						*bt1 += 1;
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] < -1.F || data[I3D(i,j,k)] > 1.F) {
						*bt1 = 1;
						 *ierr = 0;
						 return;
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain4D_bt1(const REAL4 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   const int64_t nw,
					   uint64_t * bt1,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx     &&
			0LL < ny     &&
			0LL < nw     &&
			0LL < nz     );
#else
	if(NULL == data  ||
	   0LL >=  nx    ||
	   0LL >=  ny    ||
	   0LL >=  nz    ||
	   0LL >=  nw     ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain4D_bt1")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < -1.F || data[I4D(i,j,k,l)] > 1.F)
							*bt1 += 1;
					}
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < 1.F || data[I4D(i,j,k,l)] > 1.F) {
							*bt1 = 1;
							*ierr = 0;
							return;
						}
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain1D_bt1(const REAL8 * __restrict data,
					   const int64_t nx,
					   uint64_t * bt1,
					   int32_t * ierr,
					   const bool process_slowly ) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx);
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_bt1")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
	    for(int64_t i = 0; i != nx; ++i) {
		     if(data[i] < -1.0 || data[i] > 1.0)
			     *bt1 += 1;
	}
	*ierr = 0;
  } 
	else {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
		 for(int64_t i = 0LL; i != nx; ++i) {
			 if(data[i] < -1.0 || data[i] > 1.0){
				 *bt1 = 1;
				 *ierr = 0;
				 return;
			 }
		 }
		 *ierr = 0;
	}
}

void		is_domain2D_bt1(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   uint64_t * bt1,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx   &&
			0LL < ny);
#else
	if(NULL == data ||
	  0LL >= nx    ||
	  0LL >= ny ) {
		  REPORT_ERROR("Invalid argument(s) in is_domain2D_bt1")
		  *ierr = -1;
		  return;
	}
#endif
	if(process_slowly == true) {
	      for(int64_t i = 0LL; i != nx; ++i) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
		      for(int64_t j = 0LL; j != ny; ++j) {
			       if(data[I2D(i,j)] < -1.0 || data[I2D(i,j)] > 1.0)
				        *bt1 += 1;
		     }
	     }
	   *ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
//#//pragma prefetch data:0:4
//#pragma prefetch data:1:8
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] < -1.0 || data[I2D(i,j)] > 1.0) {
					*bt1 = 1;
					*ierr = 0;
					return;
				}

			}
		}
		*ierr = 0;
	}
}

void		is_domain3D_bt1(const REAL8 * __restrict data,
					  const int64_t nx,
					  const int64_t ny,
					  const int64_t nz,
					  uint64_t * bt1,
					  int32_t * ierr,
					  const bool process_slowly ) {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx     &&
			0LL < ny     &&
			0LL < nz     );
#else
	if(NULL == data ||
	   0LL >= nx    ||
	   0LL >= ny    ||
	   0LL >= nz   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain3D_bt1")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] < -1.0 || data[I3D(i,j,k)] > 1.0)
						*bt1 += 1;
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
//#pragma prefetch data:0:4
//#pragma prefetch data:1:8
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] < -1.0 || data[I3D(i,j,k)] > 1.0) {
						*bt1 = 1;
						 *ierr = 0;
						 return;
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain4D_bt1(const REAL8 * __restrict data,
					  const int64_t nx,
					  const int64_t ny,
					  const int64_t nz,
					  const int64_t nw,
					  uint64_t  * bt1,
					  int32_t * ierr,
					  const bool process_slowly)  {
	FPEXCEPT_CHECK_LTZ(bt1)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	 _ASSERTE(NULL != data  &&
			 0LL < nx      &&
			 0LL < ny      &&
			 0LL < nz      &&
			 0LL < nw      );
#else
	if(NULL == data  ||
	   0LL >=  nx   ||
	   0LL >=  ny   ||
	   0LL >=  nz   ||
	   0LL >=  nw  ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain4D_bt1")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t  i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < -1.0 || data[I4D(i,j,k,l)] > 1.0)
							*bt1 += 1;
					}
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] < -1.0 || data[I4D(i,j,k,l)] > 1.0) {
							*bt1 = 1;
							*ierr = 0;
							return;
						}
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain1D_nez(const REAL4 * __restrict data,
					  const int64_t nx,
					  uint64_t * nez,
					  int32_t * ierr,
					  const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx );
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_nez!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] == 0.F)
				*nez += 1;
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] == 0.F) {
				*nez = 1;
				*ierr = 0;
				return;
			}
		}
		*ierr = 0;
	}
}

void		is_domain2D_nez(const REAL4 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	 _ASSERTE(NULL != data  &&
			 0LL < nx      &&
			 0LL < ny  );
#else
	if(NULL == data || 
	   0LL >=  nx   ||
	   0LL >=  ny   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain2D_nez!!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] == 0.F)
					*nez += 1;
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] == 0.F) {
					*nez = 1;
					*ierr = 0;
					return;
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain3D_nez(const REAL4 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly ) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL <   nx   &&
			0LL <   ny   &&
			0LL <   nz    );
#else
	if(NULL == data   ||
	   0LL >=  nx    ||
	   0LL >=  ny    ||
	   0LL >=  nz   ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain3D_nez!")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] == 0.F)
						*nez += 1;
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] == 0.F) {
						*nez = 1;
						*ierr = 0;
						return;
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain4D_nez(const REAL4 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   const int64_t nw,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly)  {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data  &&
			0LL <   nx    &&
			0LL <   ny    &&
			0LL <   nz    &&
			0LL <   nw    );
#else
	if(NULL == data  ||
	   0LL  >= nx   ||
	   0LL  >= ny   ||
	   0LL  >= nz   ||
	   0LL  >= nw   ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain4D_nez (float)")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] == 0.F)
							*nez += 1;
					}
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] == 0.F) {
							*nez = 1;
							*ierr = 0;
							return;
						}
							
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain1D_nez(const REAL8 * __restrict data,
					  const int64_t nx,
					  uint64_t * nez,
					  int32_t * ierr,
					  const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL < nx );
#else
	if(NULL == data || 0LL >= nx) {
		REPORT_ERROR("Invalid argument(s) in is_domain1D_nez (double prec.)!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] == 0.0)
				*nez += 1;
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			if(data[i] == 0.0) {
				*nez = 1;
				*ierr = 0;
				return;
			}
		}
		*ierr = 0;
	}
}

void		is_domain2D_nez(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	 _ASSERTE(NULL != data  &&
			 0LL < nx      &&
			 0LL < ny  );
#else
	if(NULL == data || 
	   0LL >=  nx   ||
	   0LL >=  ny   ) {
		REPORT_ERROR("Invalid argument(s) in is_domain2D_nez!!")
		*ierr = -1;
		return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] == 0.0)
					*nez += 1;
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				if(data[I2D(i,j)] == 0.0) {
					*nez = 1;
					*ierr = 0;
					return;
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain3D_nez(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly) {
		FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data &&
			0LL <   nx   &&
			0LL <   ny   &&
			0LL <   nz    );
#else
	if(NULL == data   ||
	   0LL >=  nx    ||
	   0LL >=  ny    ||
	   0LL >=  nz   ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain3D_nez (double prec.!")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] == 0.0)
						*nez += 1;
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					if(data[I3D(i,j,k)] == 0.0) {
						*nez = 1;
						*ierr = 0;
						return;
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_domain4D_nez(const REAL8 * __restrict data,
					   const int64_t nx,
					   const int64_t ny,
					   const int64_t nz,
					   const int64_t nw,
					   uint64_t * nez,
					   int32_t * ierr,
					   const bool process_slowly) {
	FPEXCEPT_CHECK_LTZ(nez)
	FPEXCPET_CHECK_IERR(ierr)
#if (CuWRF_DEBUG_ON) == 1
	_ASSERTE(NULL != data  &&
			0LL <   nx    &&
			0LL <   ny    &&
			0LL <   nz    &&
			0LL <   nw    );
#else
	if(NULL == data  ||
	   0LL  >= nx   ||
	   0LL  >= ny   ||
	   0LL  >= nz   ||
	   0LL  >= nw   ) {
		   REPORT_ERROR("Invalid argument(s) in is_domain4D_nez (double prec.)")
		   *ierr = -1;
		   return;
	}
#endif
	if(process_slowly == true) {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] == 0.0)
							*nez += 1;
					}
				}
			}
		}
		*ierr = 0;
	}
	else {
		for(int64_t i = 0LL; i != nx; ++i) {
			for(int64_t j = 0LL; j != ny; ++j) {
				for(int64_t k = 0LL; k != nz; ++k) {
					for(int64_t l = 0LL; l != nw; ++l) {
						if(data[I4D(i,j,k,l)] == 0.0) {
							*nez = 1;
							*ierr = 0;
							return;
						}
							
					}
				}
			}
		}
		*ierr = 0;
	}
}

void		is_abnormalf32(const REAL4 * __restrict data,
					  const int64_t nx,
					  uint64_t * count,
					  const bool process_slowly,
					  const uint32_t option ) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
			if(process_slowly == true) {
				for(int64_t i = 0LL; i != nx; ++i) {
					if(fpclassify(data[i]) == FP_SUBNORMAL)
						*count += 1;
				}
			}
			else {
				for(int64_t i = 0LL; i != nx; ++i) {
					if(fpclassify(data[i]) == FP_SUBNORMAL) {
						*count = 1;
						return;
					}
				}
			}
	   }
		break;

	case 1: {
				// NAN
			 if(process_slowly == true) {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_NAN)
						 *count += 1;
				 }
			 }
			 else {
				  for(int64_t i = 0LL; i != nx; ++i) {
					  if(fpclassify(data[i]) == FP_NAN) {
						  *count = 1;
						  return;
					  }
				  }
			 }
		}
		break;

	case 2: {
				// INF
			 if(process_slowly == true) {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_INFINITE)
						 *count += 1;
				 }
			 }
			 else {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_INFINITE) {
						 *count = 1;
						 return;
					 }
				 }
			 }
	   }
		break;

	default : {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf32 (domain 1D)")
		 }

	}
}

void is_abnormalf32(const REAL4 * __restrict data,
				  const int64_t nx,
				  const int64_t ny,
				  uint64_t *count,
				  const bool process_slowly,
				  const uint32_t option  )  {
	FPEXCEPT_CHECK_LTZ(count)
    switch(option) {

	case 0: {
				// DENORMAL
		       if(process_slowly == true) {
				   for(int64_t i = 0LL; i != nx; ++i) {
					   for(int64_t j = 0LL; j != ny; ++j) {
						   if(fpclassify(data[I2D(i,j)]) == FP_SUBNORMAL)
							   *count += 1;
					   }
				   }
			   }
			   else {
				    for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_SUBNORMAL) {
								*count = 1;
								return;
							}
						}
					}
			   }
		   }
			break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_NAN)
								*count += 1;
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_NAN) {
								*count = 1;
								return;
							}
						}
					}
			}
		}
		 break;

	case 2: {
				//INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_INFINITE)
								*count += 1;
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_INFINITE) {
								*count = 1;
								return;
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid argument to switch in: is_abnormalf32 (domain 2D)")
		}
	}
}

void is_abnormalf32(const REAL4 * __restrict data,
				  const int64_t nx,
				  const int64_t ny,
				  const int64_t nz,
				  uint64_t * count,
				  const bool process_slowly,
				  const uint32_t option ) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_SUBNORMAL)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_SUBNORMAL) {
									*count = 1;
									return;
								}
							}
						}
					}
				}
		 }
		 break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_NAN)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_NAN) {
									*count = 1;
									return;
								}
							}
						}
					}
			}
		}
		break;

	case 2: {
				//	INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_INFINITE)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_INFINITE) {
									*count = 1;
									return;
								}
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf32 (domain 3D)")
		}
	}
}

void is_abnormalf32(const REAL4 * __restrict data,
				  const int64_t nx,
				  const int64_t ny,
				  const int64_t nz,
				  const int64_t nw,
				  uint64_t * count,
				  const bool process_slowly,
				  const uint32_t option) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_SUBNORMAL)
										*count += 1;
								}
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_SUBNORMAL) {
										*count = 1;
										return;
									}
										
								}
							}
						}
					}
			}
		}
		break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_NAN)
										*count += 1;
								}
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_NAN) {
										*count = 1;
										return;
									}
								}
							}
						}
					}
			}
		}
		break;

	case 2: {
				// INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_INFINITE)
										*count += 1;
								}
							}
						}
					}
			    }
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_INFINITE) {
										*count = 1;
										return;
									}
								}
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf32 (domain 4D)")
		}
	}
}

void is_abnormalf64(const REAL8 * __restrict data,
				  const int64_t nx,
				  uint64_t * count,
				  const bool process_slowly,
				  const uint32_t option ) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
			if(process_slowly == true) {
				for(int64_t i = 0LL; i != nx; ++i) {
					if(fpclassify(data[i]) == FP_SUBNORMAL)
						*count += 1;
				}
			}
			else {
				for(int64_t i = 0LL; i != nx; ++i) {
					if(fpclassify(data[i]) == FP_SUBNORMAL) {
						*count = 1;
						return;
					}
				}
			}
	   }
		break;

	case 1: {
				// NAN
			 if(process_slowly == true) {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_NAN)
						 *count += 1;
				 }
			 }
			 else {
				  for(int64_t i = 0LL; i != nx; ++i) {
					  if(fpclassify(data[i]) == FP_NAN) {
						  *count = 1;
						  return;
					  }
				  }
			 }
		}
		break;

	case 2: {
				// INF
			 if(process_slowly == true) {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_INFINITE)
						 *count += 1;
				 }
			 }
			 else {
				 for(int64_t i = 0LL; i != nx; ++i) {
					 if(fpclassify(data[i]) == FP_INFINITE) {
						 *count = 1;
						 return;
					 }
				 }
			 }
	   }
		break;

	default : {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf64 (domain 1D)")
		 }

	}
}

void is_abnormalf64(const REAL8 * __restrict data,
				  const int64_t nx,
				  const int64_t ny,
				  uint64_t * count,
				  const bool process_slowly,
				  const uint32_t option) {
	FPEXCEPT_CHECK_LTZ(count)
	 switch(option) {

	case 0: {
				// DENORMAL
		       if(process_slowly == true) {
				   for(int64_t i = 0LL; i != nx; ++i) {
					   for(int64_t j = 0LL; j != ny; ++j) {
						   if(fpclassify(data[I2D(i,j)]) == FP_SUBNORMAL)
							   *count += 1;
					   }
				   }
			   }
			   else {
				    for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_SUBNORMAL) {
								*count = 1;
								return;
							}
						}
					}
			   }
		   }
			break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_NAN)
								*count += 1;
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_NAN) {
								*count = 1;
								return;
							}
						}
					}
			}
		}
		 break;

	case 2: {
				//INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_INFINITE)
								*count += 1;
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							if(fpclassify(data[I2D(i,j)]) == FP_INFINITE) {
								*count = 1;
								return;
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid argument to switch in: is_abnormalf64 (domain 2D)")
		}
	}
}

 void is_abnormalf64(const REAL8 * __restrict data,
				    const int64_t nx,
					const int64_t ny,
					const int64_t nz,
					uint64_t * count,
					const bool process_slowly,
					const uint32_t option ) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_SUBNORMAL)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_SUBNORMAL) {
									*count = 1;
									return;
								}
							}
						}
					}
				}
		 }
		 break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_NAN)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_NAN) {
									*count = 1;
									return;
								}
							}
						}
					}
			}
		}
		break;

	case 2: {
				//	INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_INFINITE)
									*count += 1;
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								if(fpclassify(data[I3D(i,j,k)]) == FP_INFINITE) {
									*count = 1;
									return;
								}
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf64 (domain 3D)")
		}
	}
 }

 void is_abnormalf64(const REAL8 * __restrict data,
				   const int64_t nx,
				   const int64_t ny,
				   const int64_t nz,
				   const int64_t nw,
				   uint64_t * count,
				   const bool process_slowly,
				   const uint32_t option ) {
	FPEXCEPT_CHECK_LTZ(count)
	switch(option) {

	case 0: {
				// DENORMAL
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_SUBNORMAL)
										*count += 1;
								}
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_SUBNORMAL) {
										*count = 1;
										return;
									}
										
								}
							}
						}
					}
			}
		}
		break;

	case 1: {
				// NAN
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_NAN)
										*count += 1;
								}
							}
						}
					}
				}
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_NAN) {
										*count = 1;
										return;
									}
								}
							}
						}
					}
			}
		}
		break;

	case 2: {
				// INFINITE
				if(process_slowly == true) {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_INFINITE)
										*count += 1;
								}
							}
						}
					}
			    }
				else {
					for(int64_t i = 0LL; i != nx; ++i) {
						for(int64_t j = 0LL; j != ny; ++j) {
							for(int64_t k = 0LL; k != nz; ++k) {
								for(int64_t l = 0LL; l != nw; ++l) {
									if(fpclassify(data[I4D(i,j,k,l)]) == FP_INFINITE) {
										*count = 1;
										return;
									}
								}
							}
						}
					}
			}
		}
		break;

	default: {
				REPORT_ERROR("Invalid parameter to switch in: is_abnormalf64 (domain 4D)")
		}
	}
 }

 int32_t clear_fexcepts(void) {

	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_ALL_EXCEPT);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_fexcepts: feclearexcept failed with an error: ",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_fedenormal(void) {

	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_DENORMAL);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_fedenormal: feclearexcept failed with an error: ",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_feinexact(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_INEXACT);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_feinexact: feclearexcept failed with an error:",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_feinvalid(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_INVALID);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_feinvalid: feclearexcept failed with an error:",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_fedivbyzero(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_DIVBYZERO);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_fedivbyzero: feclearexcept failed with an error:",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_feoverflow(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_OVERFLOW);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_fedivbyzero: feclearexcept failed with an error:",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t clear_feunderflow(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = feclearexcept(FE_UNDERFLOW);
		 if(err != 0) {
			 REPORT_ERROR_VALUE("clear_feunderflow: feclearexcept failed with an error: ",err)
			 return (err);
		 }
		 return (err);
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_allfpexcepts(const int32_t fpexcs) {
 _ASSERT(FE_ALL_EXCEPT == fpexcs);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(FE_ALL_EXCEPT);
		 if(FE_ALL_EXCEPT == err) {
			 PRINT_MESSAGE("test_allfpexcept: All FP Exceptions are set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_allfpexcepts: FP Exceptions are not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_feinvalid(const int32_t feinv) {
 _ASSERT(FE_INVALID == feinv);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(feinv);
		 if(FE_INVALID == err) {
			 PRINT_MESSAGE("test_feinvalid:  FP_INVALID is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_feinvalid: FP_INVALID is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_feinexact(const int32_t feinex) {
	 _ASSERT(FE_INEXACT == feinex);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(feinex);
		 if(FE_INEXACT == err) {
			 PRINT_MESSAGE("test_feinexact:  FP_INEXACT is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_feinexact: FP_INEXACT is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_fedivbyzero(const int32_t fedbz) {
	 _ASSERT(FE_DIVBYZERO == fedbz);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(fedbz);
		 if(FE_DIVBYZERO == err) {
			 PRINT_MESSAGE("test_fedivbyzero:  FP_DIVBYZERO is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_fedivbyzero: FP_DIVBYZERO is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_fedenormal(const int32_t feden) {
	 _ASSERT(FE_DENORMAL == feden);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(feden);
		 if(FE_DENORMAL == err) {
			 PRINT_MESSAGE("test_fedenormal:  FP_DENORMAL is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_fedenormal:   FP_DENORMAL is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_feoverflow(const int32_t feov) {
	 _ASSERT(FE_OVERFLOW == feov);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(feov);
		 if(FE_OVERFLOW == err) {
			 PRINT_MESSAGE("test_feoverflow:  FP_OVERFLOW is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_feoverflow: FP_OVERFLOWL is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 int32_t test_feunderflow(const int32_t feund) {
	 _ASSERT(FE_UNDERFLOW == feund);
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
#if defined (math_errhandling) && defined (MATH_ERREXCEPT)
	 if(math_errhandling & MATH_ERREXCEPT) {
		 err = fetestexcept(feund);
		 if(FE_UNDERFLOW == err) {
			 PRINT_MESSAGE("test_feunderflow:  FP_UNDERFLOW is set, numeric value: ",err)
			 return (err);
		 }
		 else {
			 PRINT_MESSAGE("test_feunderflow:  FP_UNDERFLOW is not set, numeric value: ",err)
			 return (err);
		 }
	 }
#if (CuWRF_SILENCE_COMPILER) == 1
	 return (err);
#endif
#else
#error "Undefined: 'math_errhandling and MATH_ERREXCEPT' "
#endif
 }

 void raise_fedenormal(const bool clear_prev,
					 const int32_t feden,
					 int32_t * ierr ) {
	_ASSERT(FE_DENORMAL == feden);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_fedenormal(feden) == FE_DENORMAL) {
			err = clear_fedenormal();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_fedenormal' failed with a value: ",err)
			
			err = feraiseexcept(feden);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ", err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_fedenormal -- failed!!'")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_fedenormal(feden) == FE_DENORMAL) {
			err = feraiseexcept(feden);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_fedenormal -- failed!!'")
			*ierr = -2;
			return;
		}
	}
 }

 void raise_feinvalid(const bool clear_prev,
					const int32_t feinv,
					int32_t * ierr ) {
	_ASSERT(FE_INVALID == feinv);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_feinvalid(feinv) == FE_INVALID) {
			err = clear_feinvalid();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_feinvalid' failed with a value: ",err)
			
			err = feraiseexcept(feinv);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feinvalid' -- failed")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_feinvalid(feinv) == FE_INVALID) {
			err = feraiseexcept(feinv);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feinvalid' -- failed!!")
			*ierr = -2;
			return;
		}
	}
 }

 void raise_feinexact(const bool clear_prev,
					const int32_t feinex,
					int32_t * ierr ) {
	_ASSERT(FE_INEXACT == feinex);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_feinvalid(feinex) == FE_INEXACT) {
			err = clear_feinexact();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_feinexact' failed with a value: ",err)
			
			err = feraiseexcept(feinex);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feinexact' -- failed")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_feinvalid(feinex) == FE_INEXACT) {
			err = feraiseexcept(feinex);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feinexact' -- failed!!")
			*ierr = -2;
			return;
		}
	}

 }

 void raise_fedivbyzero(const bool clear_prev,
					  const int32_t fdbz,
					  int32_t * ierr) {
	_ASSERT(FE_DIVBYZERO == fdbz);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_fedivbyzero(fdbz) == FE_DIVBYZERO) {
			err = clear_fedivbyzero();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_fedivbyzero' failed with a value: ",err)
			
			err = feraiseexcept(fdbz);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_fedivbyzero' -- failed")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_feinvalid(fdbz) == FE_DIVBYZERO) {
			err = feraiseexcept(fdbz);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_fedivbyzero' -- failed!!")
			*ierr = -2;
			return;
		}
	}

 }

 void raise_feoverflow(const bool clear_prev,
					 const int32_t feov,
					 int32_t * ierr ) {
	_ASSERT(FE_OVERFLOW == feov);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_feoverflow(feov) == FE_OVERFLOW) {
			err = clear_feoverflow();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_feoverflow' failed with a value: ",err)
			
			err = feraiseexcept(feov);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feoverflow' -- failed")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_feoverflow(feov) == FE_OVERFLOW) {
			err = feraiseexcept(feov);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feoverflow' -- failed!!")
			*ierr = -2;
			return;
		}
	}

 }

 void raise_feunderflow(const bool clear_prev,
					  const int32_t feun,
					  int32_t * ierr ) {
	_ASSERT(FE_UNDERFLOW == feun);
	FPEXCPET_CHECK_IERR(ierr)
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	if(clear_prev == true) {
		if(test_feunderflow(feun) == FE_UNDERFLOW) {
			err = clear_feunderflow();
			CuWRF_FPEXCEPT_FENV_RETURN(" 'clear_feunderflow' failed with a value: ",err)
			
			err = feraiseexcept(feun);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feunderflow' -- failed")
			*ierr = -2;
			return;
		}
	}
	else {
		if(test_feunderflow(feun) == FE_UNDERFLOW) {
			err = feraiseexcept(feun);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'feraiseexcept' failed with a value: ",err)
			
			*ierr = 0;
		}
		else {
			REPORT_ERROR(" 'test_feunderflow' -- failed!!")
			*ierr = -2;
			return;
		}
	}
 }

 void set_round_mode(const int32_t mode,
				    int32_t * ierr ) {

	FPEXCPET_CHECK_IERR(ierr)
	_ASSERT(FE_DOWNWARD    == mode ||
		   FE_UPWARD      == mode ||
		   FE_TONEAREST   == mode ||
		   FE_TOWARDZERO  == mode);
	int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	err = fesetround(mode);
	CuWRF_FPEXCEPT_FENV_RETURN(" 'set_round_mode: ' failed with a value: ",err)
	*ierr = 0;
 }

 int32_t get_round_mode(void) {
	 int32_t err = -9999;
#pragma STD FENV_ACCESS ON
	 err = fegetround();
	 if(err < 0) {
		REPORT_ERROR_VALUE(" 'get_round_mode: ' fegetround failed with a value: ",err)
		return (err);
	 }
	 return (err);
 }

 void show_round_mode(const int32_t mode,
					int32_t * ierr ) {
	FPEXCPET_CHECK_IERR(ierr)
	_ASSERT(FE_DOWNWARD    == mode ||
		   FE_UPWARD      == mode ||
		   FE_TONEAREST   == mode ||
		   FE_TOWARDZERO  == mode);
	int32_t err = -9999;
	switch(mode) {

	case FE_TONEAREST : {
			set_round_mode(mode,ierr);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'show_round_mode: ' set_round_mode failed with a value: ",err)
			err = get_round_mode();
			if(err < 0) {
				REPORT_ERROR_VALUE("'get_round_mode: ' failed with a value: ", err)
				*ierr = -2;
				return;
			}
			printf("case FE_TONEAREST: Successfuly set FP rounding mode: 0x%x \n",err);
			*ierr = 0;
		}
		break;

	case FE_UPWARD : {
			set_round_mode(mode,ierr);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'show_round_mode: ' set_round_mode failed with a value: ",err)
		    err = get_round_mode();
			if(err < 0) {
				REPORT_ERROR_VALUE(" 'get_round_mode: ' failed with a value: ", err)
				*ierr = -2;
				return;
			}
			printf("case FE_UPWARD: Successfuly set FP rounding mode: 0x%x \n", err);
			*ierr = 0;
		 }
		 break;

	case FE_DOWNWARD : {
			set_round_mode(mode,ierr);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'show_round_mode: ' set_round_mode failed with a value: ",err)
			err = get_round_mode();
			if(err < 0) {
				REPORT_ERROR_VALUE(" 'get_round_mode: ' failed with a value: ", err)
				*ierr = -2;
				return;
			}
			printf("case FP_UPWARD: Successfuly set FP rounding mode: 0x%x \n",err);
			*ierr = 0;
		  }
		  break;

	case FE_TOWARDZERO : {
			set_round_mode(mode,ierr);
			CuWRF_FPEXCEPT_FENV_RETURN(" 'show_round_mode: ' set_round_mode failed with a value: ",err)
			err = get_round_mode();
			if(err < 0) {
				REPORT_ERROR_VALUE(" 'get_round_mode: ' failed with a value: ",err)
				*ierr = -2;
				return;
			}
			printf("case FP_TOWARDZERO: Successfuly set FP rounding mode: 0x%x \n", err);
			*ierr = 0;
		 }
		 break;

	default : {
				REPORT_ERROR_VALUE(" 'show_round_mode: ' Invalid switch parameter!!",err)
				*ierr = -3;
				return;
		 }
	}
 }
