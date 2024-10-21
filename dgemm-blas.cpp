#include <cblas.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Reference dgemm.";


void square_dgemm(int n, double* A, double* B, double* C) {

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
#endif
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 1., C, n);
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
#endif

}
