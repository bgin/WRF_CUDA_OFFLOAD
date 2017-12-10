
#include "check_input_arguments.cuh"

int check_input_arguments_major() {
	return (CHECK_INPUT_ARGUMENTS_MAJOR);
}

int check_input_arguments_minor() {
	return (CHECK_INPUT_ARGUMENTS_MINOR);
}

int check_input_arguments_micro() {
	return (CHECK_INPUT_ARGUMENTS_MICRO);
}

int check_input_arguments_fullver() {
	return (CHECK_INPUT_ARGUMENTS_FULLVER);
}

const char * check_input_arguments_createdate() {
	return (CHECK_INPUT_ARGUMENTS_CREATION_DATE);
}

const char * check_input_arguments_buildate() {
	return (CHECK_INPUT_ARGUMENTS_BUILD_DATE);
}

const char * check_input_arguments_author() {
	return (CHECK_INPUT_ARGUMENTS_AUTHOR);
}

const char * check_input_arguments_synopsis() {
	return (CHECK_INPUT_ARGUMENTS_SYNOPSIS);
}


/*
	!===============================================50
	!  Definition of error checking helper routines.
	!===============================================50
*/

/*
	Checks if first argument(value) against specific
	value(lim), which constitues a second argument.
	Argument option: L,G,E,N executes specific
	branch of switch statements:
	Valid options: L (less), G (greater), E (equal), N (not equal)
	Upon detecting an error either hard assert executes (Debug build) or
	exit(EXIT_FAILURE) (Release build) is called.
*/
void checkint32_exit_failure(const int value, 
						   const int lim, 
						   const char option) {
	
	switch(option) {
		
	case 'L' : 
		{
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value < lim);
	
#else
	if(value > lim) {
		fprintf(stderr,"[check_int32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
		fprintf("value=%d, lim=%d\n",value,lim);
		exit(EXIT_FAILURE);			 
	}
#endif
		}
	case 'G' : 
		{
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value > lim);
#else
	if(value < lim) {
	    fprintf(stderr,"[check_int32_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
		fprintf("value=%d, lim=%d\n",value,lim);
		exit(EXIT_FAILURE);
	}
#endif
		}
	case 'E' : 
		{
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value == lim);
#else
	if(value != lim) {
	   fprintf(stderr,"[check_int32_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
	   exit(EXIT_FAILURE);
	}
#endif
		}
	case 'N' : 
		{
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value != lim);
#else
	if(value == lim) {
	   fprintf(stderr,"[check_int32_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
	   exit(EXIT_FAILURE);
	   
	}
#endif
		}
	default : 
		{
		  fprintf(stderr,"%s\n", "[check_int32_exit_failure]: Invalid option argument");
		  return;
		}
		
	}
}

/*
	Checks if first argument(value) against specific
	value(lim), which constitues a second argument.
	Argument option: L,G,E,N executes specific
	branch of switch statements:
	Valid options: L (less), G (greater), E (equal), N (not equal)
	Upon detecting an error either hard assert executes (Debug build) or
	exit(EXIT_FAILURE) (Release build) is called.
*/
void checkint64_exit_failure(  const long long value,
						     const  long long lim,
						     const char       option) {
	
     switch(option) {

	 case 'L' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value < lim);
#else
	if(value > lim) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[check_int64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_int64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
		  	       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif

	   }
	 case 'G' : {
#if CuWRF_DEBUG_ON == 1
    _ASSERTE(value > lim);
#else
	if(value < lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_int64_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE  == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_int64_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
		  	       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif

	}
	 case 'E' : {
#if CuWRF_DEBUG_ON == 1
    _ASSERTE(value == lim);
#else
	if(value != lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_int64_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_int64_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
		  	       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	}
	 case 'N' : {
#if CuWRF_DEBUG_ON == 1
    _ASSERTE(value != lim);
#else
    if(value == lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_int64_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_int64_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
		  	       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	}
	 default : {
		  fprintf(stderr,"%s\n", "[check_int64_exit_failure]: Invalid option argument to switch statement!!");
		  return;
	}

   }
}

/*
	Checks if first argument(value) against specific
	value(lim), which constitues a second argument.
	Argument option: L,G,E,N executes specific
	branch of switch statements:
	Valid options: L (less), G (greater), E (equal), N (not equal)
	Upon detecting an error either hard assert executes (Debug build) or
	exit(EXIT_FAILURE) (Release build) is called.
*/
void checkuint32_exit_failure(const unsigned int value,
							const unsigned int  lim,
							const char          option) {
	 
	 switch(option) {

	 case 'L' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value < lim);
#else
	if(value > lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_uint32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
	      fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	  }
	 case 'G' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value > lim);
#else
	if(value < lim) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[check_uint32_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
       File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
	      fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	  }
	 case 'E' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value == lim);
#else
    if(value != lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_uint32_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
       File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint32_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
	      fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	  }
	 case 'N' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value != lim);
#else
    if(value == lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_uint32_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%d, lim=%d\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint32_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
	      fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	  }
	 default : {
		 fprintf(stderr,"%s\n", "[check_uint32_exit_failure]: Invalid option argument to switch statement!!");
		 return;
	  }
   }
}

/*
	Checks if first argument(value) against specific
	value(lim), which constitues a second argument.
	Argument option: L,G,E,N executes specific
	branch of switch statements:
	Valid options: L (less), G (greater), E (equal), N (not equal)
	Upon detecting an error either hard assert executes (Debug build) or
	exit(EXIT_FAILURE) (Release build) is called.
*/
void checkuint64_exit_failure(const unsigned long long value,
							const unsigned long long lim,
							const char  option        ) {
	
	 
     switch(option) {
     
	 case 'L' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value < lim);
#else
	if(value > lim) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[check_uint64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
	   
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
		  	       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
	
#endif
		exit(EXIT_FAILURE);
	}
#endif
	   }
	 case 'G' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value > lim);
#else
	if(value < lim) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[check_uint64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
	   
#endif
#if PRINT_ERROR_TO_FILE == 1
	   File *fp;
	   fp = fopen(RunTimErrors\messages.txt,"a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint64_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
		exit(EXIT_FAILURE);
	}
#endif
	   }
	 case 'E' :  {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value == lim);
#else
	if(value != lim) {
#if PRINT_ERROR_TO_SCREEN == 1
       fprintf(stderr,"[check_uint64_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
	  
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimErrors\messages.txt","a+");
	   if(fp != NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[check_uint64_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
		exit(EXIT_FAILURE);
	}
#endif
	   }
	 case 'N' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(value != lim);
#else
	if(value == lim) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[check_uint64_exit_failure]: [value == lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%ld, lim=%ld\n",value,lim);
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimErrors\messages.txt","a+");
	   if(fp == NULL) {
          fprintf(stderr, "fopen failed with error: 0x%x", GetLastError());
	      fclose(fp);
	    }
	   else {
	      fprintf(fp,"[checkuint64_exit_failure]: [value != lim] in (line %d at address 0x%x in file %s) : \n\t" \
			       __LINE__,__FUNCTIONW__,__FILE__);
		  fprintf(fp, "value=%ld, lim=%ld\n", value,lim);
	   }
#endif
		exit(EXIT_FAILURE);
	}
#endif
	   }
	 default : {
		 fprintf(stderr,"%s\n", "[check_uint64_exit_failure]: Invalid option argument to switch statement!!");
		 return;
	   }
   }
}

/*
	Checks if first argument(value) against specific
	value(lim), which constitues a second argument.
	Argument option: L,G,E,N executes specific
	branch of switch statements:
	Valid options: L (less), G (greater), E (equal), N (not equal)
	Upon detecting an error either hard assert executes (Debug build) or
	exit(EXIT_FAILURE) is executed (called).
	Naive (trivial) floating-point comparison is used.

*/

void checkf32_exit_failure(const float value,
						 const float lim,
						 float eps,
						 const char option){
						 
	
	 if(fabsf(eps) < FLT_EPSILON) {
#if PRINT_ERROR_TO_SCREEN == 1
	    fprintf(stderr,"[checkf32_exit_failure]: **Invalid** --> fabsf(eps)=%.f9\n",fabsf(eps));

#elif PRINT_ERROR_TO_FILE == 1
	    FILE *fp;
		fp = fopen("RunTimeErrors\messages.txt","a+");
		if(fp == NULL) {
			fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
			fclose(fp);
		}
		else {
			fprintf(fp,"[checkf32_exit_failure]: **Invalid** --> fabsf(eps)=%.f9\n",fabsf(eps));
			fclose(fp);
		}
#endif
		eps = FLT_EPSILON;
	 }

	 switch(option) {
		
	 case 'L' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(fabsf(value) < fabsf(lim));
#else
	if(fabsf(value) > fabsf(lim)) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.9f, lim=%.9f\n",fabsf(value),fabsf(lim));
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf32_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.9f, lim=%.9f\n",fabsf(value),fabsf(lim)); 
		   fclose(fp);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'G' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(fabsf(value) > fabsf(lim));
#else
	if(fabsf(value) < fabsf(lim)) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf32_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.9f, lim=%.9f\n",fabsf(value),fabsf(lim));
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf32_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.9f, lim=%.9f\n",fabsf(value),fabsf(lim)); 
		   fclose(fp);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'E' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE((fabsf(value) - fabsf(lim)) <= eps);
#else
	if((fabsf(value) - fabsf(lim)) > eps) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf32_exit_failure]: [fabsf(value)-fabsf(lim)>eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.9f, lim=%.9f, eps=%.9f\n",fabsf(value),fabsf(lim),eps); 
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf32_exit_failure]: [fabsf(value)-fabsf(lim)>eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.9f, lim=%.9f, eps=%.9f\n",fabsf(value),fabsf(lim),eps); 
		   fclose(fp);
	   } 
#endif
	   exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'N' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE((fabsf(value) - fabsf(lim)) > eps);
#else
	if((fabsf(value)-fabsf(lim)) <= eps) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf32_exit_failure]: [fabsf(value)-fabsf(lim)<=eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.9f, lim=%.9f, eps=%.9f\n",fabsf(value),fabsf(lim),eps); 

#elif PRINT_ERROR_TO_FILE == 1
       FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf32_exit_failure]: [fabsf(value)-fabsf(lim)<=eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.9f, lim=%.9f, eps=%.9f\n",fabsf(value),fabsf(lim),eps); 
		   fclose(fp);
	   } 
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	    }
	 default : {
		 fprintf(stderr,"%s\n", "[checkf32_exit_failure]: Invalid option argument to switch statement!!");
		 return;
	    }
	}
	 
}



void  checkf64_exit_failure(const double value,
						  const double lim,
						  double eps,
						  const char option) {
						  
	if(fabs(eps) < DBL_EPSILON) {
#if PRINT_ERROR_TO_SCREEN == 1
		fprintf(stderr, "[checkf64_exit_failure]: Invalid eps --> fabs(eps)=%.f15\n",fabs(eps));
#elif PRINT_ERROR_TO_FILE == 1
		File * fp = NULL;
		fp = fopen("RunTimeErrors\messages.txt","a+");
		if(NULL == fp) {
			fprintf(stderr,"fopen failed with an error:0x%x", GetLastError());
			fclose(fp);
		}
		else {
			fprintf(fp,"[checkf64_exit_failure]: Invalid eps --> fabs(eps)=%.f15\n",fabs(eps));
			fclose(fp);
		}
#endif
		eps = DBL_EPSILON;

	}
	switch(option) {

	case 'L': {
#if CuWRF_DEBUG_ON == 1
		_ASSERTE(fabs(value) < fabs(lim));
#else
		if(fabs(value) > fabs(lim)) {

#if PRINT_ERROR_TO_SCREEN == 1
  fprintf(stderr,"[checkf64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.15f, lim=%.15f\n",fabs(value),fabs(lim));

#elif PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf64_exit_failure]: [value > lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.15f, lim=%.15f\n",fabsf(value),fabsf(lim)); 
		   fclose(fp);
	   }
#endif
	   exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'G' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE(fabs(value) > fabs(lim));
#else
	if(fabs(value) < fabs(lim)) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf64_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.15f, lim=%.15f\n",fabs(value),fabs(lim));
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(NULL == fp) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf64_exit_failure]: [value < lim] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.15f, lim=%.15f\n",fabs(value),fabs(lim)); 
		   fclose(fp);
	   }
#endif
	  exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'E' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE((fabs(value) - fabs(lim)) <= eps);
#else
	if((fabs(value) - fabs(lim)) > eps) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf64_exit_failure]: [fabs(value)-fabs(lim)>eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.15f, lim=%.15f, eps=%.15f\n",fabs(value),fabs(lim),eps); 
#endif
#if PRINT_ERROR_TO_FILE == 1
	   FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf64_exit_failure]: [fabs(value)-fabs(lim)>eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.15f, lim=%.15f, eps=%.15f\n",fabs(value),fabs(lim),eps); 
		   fclose(fp);
	   } 
#endif
	   exit(EXIT_FAILURE);
	}
#endif
		}
	 case 'N' : {
#if CuWRF_DEBUG_ON == 1
	_ASSERTE((fabs(value) - fabs(lim)) > eps);
#else
	if((fabs(value)-fabs(lim)) <= eps) {
#if PRINT_ERROR_TO_SCREEN == 1
	   fprintf(stderr,"[checkf64_exit_failure]: [fabs(value)-fabs(lim)<=eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	   fprintf("value=%.15f, lim=%.15f, eps=%.15f\n",fabs(value),fabs(lim),eps); 

#elif PRINT_ERROR_TO_FILE == 1
       FILE *fp;
	   fp = fopen("RunTimeErrors\messages.txt","a+");
	   if(fp == NULL) {
		  fprintf(stderr,"fopen failed with an error: 0x%x", GetLastError());
		  fclose(fp);
	   }
	   else {
		   fprintf(fp,"[checkf64_exit_failure]: [fabs(value)-fabs(lim)<=eps] in (line %d at address 0x%x in file %s) : \n\t" \
			   __LINE__,__FUNCTIONW__,__FILE__);
	       fprintf(fp,"value=%.15f, lim=%.15f, eps=%.15f\n",fabs(value),fabs(lim),eps); 
		   fclose(fp);
	   } 
#endif
	   exit(EXIT_FAILURE);
	}
#endif
	    }
	 default : {
		 fprintf(stderr,"%s\n", "[checkf364_exit_failure]: Invalid option argument to switch statement!!");
		 return;
	    }
	}
	 
}
		






