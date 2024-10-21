// This code is modification of the following implementation: 
// https://github.com/lrvine/Strassen-algorithm/blob/master/strassen.c
#include <vector>

void seq_dgemm(int n, double* A, double* B, double* C) 
{
  for (int tmpi = 0; tmpi < n * n; tmpi++)
    C[tmpi] = 0;
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

void copy_to_block(int n, double *M, double *block, int block_size, int block_i, int block_j) {
  M += block_i * block_size * n + block_j * block_size;
  double *M_i, *B_i, *M_ij, *B_ij;
  for (M_i = M, B_i = block; M_i < M + block_size * n; M_i += n, B_i += block_size)
    for (M_ij = M_i, B_ij = B_i; M_ij < M_i + block_size; M_ij += 1, B_ij += 1)
      *B_ij = *M_ij;
}

void copy_from_block(int n, double *M, double *block, int block_size, int block_i, int block_j) {
  M += block_i * block_size * n + block_j * block_size;
  double *M_i, *B_i, *M_ij, *B_ij;
  for (M_i = M, B_i = block; M_i < M + block_size * n; M_i += n, B_i += block_size)
    for (M_ij = M_i, B_ij = B_i; M_ij < M_i + block_size; M_ij += 1, B_ij += 1)
      *M_ij = *B_ij;
}

void matrixAdd(int n, double *A, double *B, double *C) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      C[i * n + j] = A[i * n + j] + B[i * n + j];
}

void matrixSub(int n, double *A, double *B, double *C) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      C[i * n + j] = A[i * n + j] - B[i * n + j];
}

template <int seq_limit>
void Strassen(int n, double* X, double* Y, double* Z) 
{
  if (n <= seq_limit) {
    seq_dgemm(n, X, Y, Z);
    return;
  }
  int n_2 = n / 2;
  int n2_sq = n_2 * n_2;

  std::vector<double> buf(n2_sq * 14, 0);

  double* a = buf.data() + 0;
  double*  b = a + n2_sq;
  double*  c = b + n2_sq;
  double*  d = c + n2_sq;
  double*  A = d + n2_sq;
  double*  C = A + n2_sq;
  double*  B = C + n2_sq;
  double*  D = B + n2_sq;
  double*  t = D + n2_sq;
  double*  u = t + n2_sq;
  double*  v = u + n2_sq;
  double*  w = v + n2_sq;

  copy_to_block(n, X, a, n_2, 0, 0);
  copy_to_block(n, X, b, n_2, 0, 1);
  copy_to_block(n, X, c, n_2, 1, 0);
  copy_to_block(n, X, d, n_2, 1, 1);
  copy_to_block(n, Y, A, n_2, 0, 0);
  copy_to_block(n, Y, C, n_2, 0, 1);
  copy_to_block(n, Y, B, n_2, 1, 0);
  copy_to_block(n, Y, D, n_2, 1, 1);

  matrixSub(n_2, a, c, t);
  matrixAdd(n_2, c, d, c);
  matrixSub(n_2, C, A, w);
  matrixSub(n_2, D, C, C);
  Strassen<seq_limit>(n_2, t, C, v);
  matrixSub(n_2, c, a, u);
  Strassen<seq_limit>(n_2, a, A, t);
  matrixSub(n_2, D, w, A);
  Strassen<seq_limit>(n_2, c, w, a);
  matrixSub(n_2, A, B, w);
  Strassen<seq_limit>(n_2, d, w, c);

  matrixSub(n_2, b, u, d);
  Strassen<seq_limit>(n_2, u, A, w);
  matrixAdd(n_2, t, w, w);
  Strassen<seq_limit>(n_2, b, B, u);
  matrixAdd(n_2, t, u, t);
  matrixAdd(n_2, w, a, u);
  matrixAdd(n_2, w, v, w);
  matrixSub(n_2, w, c, v);
  matrixAdd(n_2, w, a, w);
  Strassen<seq_limit>(n_2, d, D, b);
  matrixAdd(n_2, u, b, u);

  copy_from_block(n, Z, t, n_2, 0, 0);
  copy_from_block(n, Z, u, n_2, 0, 1);
  copy_from_block(n, Z, v, n_2, 1, 0);
  copy_from_block(n, Z, w, n_2, 1, 1);
}