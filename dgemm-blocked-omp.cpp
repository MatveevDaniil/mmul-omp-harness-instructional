#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"


const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


void block_dgemm(int n, double* A, double* B, double* C) 
{
  for (int k = 0; k < n; k++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        C[i * n + j] += A[i * n + k] * B[k * n + j];
}

void copy_to_block(int n, int block_size, double *M, double *block, int block_i, int block_j) {
  M += block_i * block_size * n + block_j * block_size;
  double *M_i, *B_i, *M_ij, *B_ij;
  for (M_i = M, B_i = block; M_i < M + block_size * n; M_i += n, B_i += block_size)
    for (M_ij = M_i, B_ij = B_i; M_ij < M_i + block_size; M_ij += 1, B_ij += 1)
      *B_ij = *M_ij;
}

void add_from_block(int n, int block_size, double *M, double *block, int block_i, int block_j) {
  M += block_i * block_size * n + block_j * block_size;
  double *M_i, *B_i, *M_ij, *B_ij;
  for (M_i = M, B_i = block; M_i < M + block_size * n; M_i += n, B_i += block_size)
    for (M_ij = M_i, B_ij = B_i; M_ij < M_i + block_size; M_ij += 1, B_ij += 1) {
      *M_ij += *B_ij;
      *B_ij = 0;
    }
}


void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
  std::vector<double> buf(3 * block_size * block_size);
  double* A_block = buf.data() + 0;
  double* B_block = A_block + block_size * block_size;
  double* C_block = B_block + block_size * block_size;
  #pragma omp parallel {
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(MY_MARKER_REGION_NAME); 
#endif
    #pragma omp for
    for (int i = 0; i < n / block_size; i++)
      for (int k = 0; k < n / block_size; k++)
        for (int j = 0; j < n / block_size; j++) {
          copy_to_block(n, block_size, A, A_block, i, k);
          copy_to_block(n, block_size, B, B_block, k, j);
          block_dgemm(block_size, A_block, B_block, C_block);
          add_from_block(n, block_size, C, C_block, i, j);
        }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
#endif
  }
}