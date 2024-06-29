#pragma once

#include "DeviceFunction.cuh"
#include <cusparse.h>
#include <cusolverSp.h>

void debug(TinyCache *solvergpu);

__global__ void solve_kernel(TinyCache *solver_gpu, double *bigDual, QCost Q, RCost R, state init_state, state final_state , int* d_index);

__global__ void Second_solve_kernel(TinyCache *solver_gpu , double * d_x);

void tiny_solve_cuda(TinyCache *cache,  tinytype *bigDual, QCost Q, RCost R, state init_state, state final_state);