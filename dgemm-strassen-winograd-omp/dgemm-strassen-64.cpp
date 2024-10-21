#include "dgemm-strassen-winograd.hpp"


void square_dgemm(int n, double* A, double* B, double* C) 
{
  std::vector<double> buf(n * n);
  double *_C = buf.data() + 0;
  Strassen<64>(n, A, B, _C);
  matrixAdd(n, C, _C, C);
}