#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";


void square_dgemm(int n, double* A, double* B, double* C) 
{
  #pragma omp parallel 
  {
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
#endif
    #pragma omp for
    for (int i = 0; i < n; i++)
      for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
          C[i * n + j] += A[i * n + k] * B[k * n + j];
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
#endif
  }
}
