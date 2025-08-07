#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <c10/util/Half.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/Atomic.cuh>
#include "functions.cu"
#include <cuda_fp16.h>




#define CHECK_NAN_INF(name, val) \
    if ((isnan(val) || isinf(val))) { \
        printf("[Thread %d] %s = %.10f (isnan: %d, isinf: %d)\n", \
            threadIdx.x, name, val, isnan(val), isinf(val)); \
    }


__device__ inline void check_nan_inf(const float* arr, int len, const char* varname, const char* funcname) {
    for (int i = 0; i < len; ++i) {
        if (isnan(arr[i]) || isinf(arr[i])) {
            printf("NaN/Inf detected in %s [%s][%d] = %f\n", funcname, varname, i, arr[i]);
        }
    }
}



#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


//1 thread per edge would make the memory explode, try with a threshold for the number of threads (once reached, use a fixed number of threads, so 1 thread will process multiple edges)
#define NUM_THREADS_PER_BLOCK 256
#define MAX_BLOCKS 32

#define NUM_BLOCKS(batch_size) \
  ((batch_size + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK < MAX_BLOCKS ? \
   (batch_size + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK : \
   MAX_BLOCKS)


__device__ void
actSO3(const at::BFloat16 *q, const at::BFloat16 *X, at::BFloat16 *Y) {
  at::BFloat16 uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  //check_nan_inf(uv, 3, "uv", "actSO3");

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);

  //check_nan_inf(Y, 3, "Y", "actSO3");
}

__device__  void
actSE3(const at::BFloat16 *t, const at::BFloat16 *q, const at::BFloat16 *X, at::BFloat16 *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];

  //check_nan_inf(Y, 4, "Y", "actSE3");
}

__device__ void
adjSE3(const at::BFloat16 *t, const at::BFloat16 *q, const at::BFloat16 *X, at::BFloat16 *Y) {
  at::BFloat16 qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  //check_nan_inf(Y, 6, "Y", "adjSE3");

  at::BFloat16 u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  //check_nan_inf(u, 3, "u", "adjSE3");

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];

  //check_nan_inf(Y, 6, "Y", "adjSE3");
}

__device__ void 
relSE3(const at::BFloat16 *ti, const at::BFloat16 *qi, const at::BFloat16 *tj, const at::BFloat16 *qj, at::BFloat16 *tij, at::BFloat16 *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1];
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2];
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0];
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2];

  //check_nan_inf(qij, 4, "qij", "relSE3");
  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];

  //check_nan_inf(tij, 3, "tij", "relSE3");
}

  
__device__ void
expSO3(const at::BFloat16 *phi, at::BFloat16* q) {
  // SO3 exponential map
  at::BFloat16 theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  //check_nan_inf(&theta_sq, 1, "theta_sq", "expSO3");
  at::BFloat16 theta_p4 = theta_sq * theta_sq;
  //check_nan_inf(&theta_p4, 1, "theta_p4", "expSO3");

  at::BFloat16 theta = sqrtf(theta_sq);
  //check_nan_inf(&theta, 1, "theta", "expSO3");
  at::BFloat16 imag, real;


  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    //check_nan_inf(&imag, 1, "imag", "expSO3");
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
    //check_nan_inf(&real, 1, "real", "expSO3");
  } else {
    imag = sinf(0.5 * theta) / theta;
    //check_nan_inf(&imag, 1, "imag", "expSO3");
    real = cosf(0.5 * theta);
    //check_nan_inf(&real, 1, "real", "expSO3");
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

  //check_nan_inf(q, 4, "q", "expSO3");


}

__device__ void
crossInplace(const at::BFloat16* a, at::BFloat16 *b) {
  at::BFloat16 x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const at::BFloat16 *xi, at::BFloat16* t, at::BFloat16* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  at::BFloat16 tau[3] = {xi[0], xi[1], xi[2]};
  at::BFloat16 phi[3] = {xi[3], xi[4], xi[5]};

  //check_nan_inf(tau, 3, "tau", "expSE3");

  at::BFloat16 theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  at::BFloat16 theta = sqrtf(theta_sq);
  

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    at::BFloat16 a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    at::BFloat16 b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }

  //check_nan_inf(q, 4, "q", "expSE3");
  //check_nan_inf(t, 3, "t", "expSE3");
}


__device__ void
retrSE3(const at::BFloat16 *xi, const at::BFloat16* t, const at::BFloat16* q, at::BFloat16* t1, at::BFloat16* q1) {
  // retraction on SE3 manifold

  at::BFloat16 dt[3] = {0, 0, 0};
  at::BFloat16 dq[4] = {0, 0, 0, 1};

  expSE3(xi, dt, dq);


  //check_nan_inf(dt, 3, "dt", "retrSE3");
  //check_nan_inf(dq, 4, "dq", "retrSE3");

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  //check_nan_inf(q1, 4, "q1", "retrSE3");


  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];

  //check_nan_inf(t1, 3, "t1", "retrSE3");
}



__global__ void pose_retr_kernel(const int t0, const int t1,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(i, t1 - t0) {
    const at::BFloat16 t = t0 + i;
    at::BFloat16 t1[3], t0[3] = { poses[t][0], poses[t][1], poses[t][2] };
    at::BFloat16 q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    at::BFloat16 xi[6] = {
      update[i][0],
      update[i][1],
      update[i][2],
      update[i][3],
      update[i][4],
      update[i][5],
    };

    //here it happens the first NaN value
    //print the values of xi
    //printf("[Thread %d] xi = [%.10f, %.10f, %.10f, %.10f, %.10f, %.10f]\n", threadIdx.x,
    //       xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]);
    //printf("[Thread %d] t0 = [%.10f, %.10f, %.10f]\n", threadIdx.x,
    //       t0[0], t0[1], t0[2]);
    //printf("[Thread %d] q0 = [%.10f, %.10f, %.10f, %.10f]\n", threadIdx.x,
    //        q0[0], q0[1], q0[2], q0[3]);
    //printf("[Thread %d] t1 = [%.10f, %.10f, %.10f]\n", threadIdx.x,
    //        t1[0], t1[1], t1[2]);
    //printf("[Thread %d] q1 = [%.10f, %.10f, %.10f, %.10f]\n", threadIdx.x,
    //        q1[0], q1[1], q1[2], q1[3]);
    retrSE3(xi, t0, q0, t1, q1);

    poses[t][0] = t1[0];
    poses[t][1] = t1[1];
    poses[t][2] = t1[2];
    poses[t][3] = q1[0];
    poses[t][4] = q1[1];
    poses[t][5] = q1[2];
    poses[t][6] = q1[3];
  }
}


__global__ void patch_retr_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> index,
    torch::PackedTensorAccessor32<at::BFloat16,4,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<at::BFloat16,1,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(n, index.size(0)) {
    const int p = patches.size(2);
    const int ix = index[n];
  
    at::BFloat16 d = patches[ix][2][0][0];
    d = d + update[n];
    d = (d > at::BFloat16(20)) ? at::BFloat16(1.0) : d;
    d = max(d, at::BFloat16(1e-4));

    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
}


__global__ void reprojection_residuals_and_hessian(
    const torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<at::BFloat16,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<at::BFloat16,1,torch::RestrictPtrTraits> lmbda,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ku,
    torch::PackedTensorAccessor64<at::BFloat16,3,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor64<at::BFloat16,3,torch::RestrictPtrTraits> E,
    torch::PackedTensorAccessor64<at::BFloat16,2,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor64<at::BFloat16,2,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor64<at::BFloat16,2,torch::RestrictPtrTraits> u,
    const int t0)
{

    
    //take the thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //get the respected local tensors
    auto local_B = B[tid];
    auto local_E = E[tid];
    auto local_v = v[tid];
    auto local_C = C[tid];
    auto local_u = u[tid];



    __shared__ at::BFloat16 fx, fy, cx, cy;
    if (threadIdx.x == 0) {
        fx = intrinsics[0][0];
        fy = intrinsics[0][1];
        cx = intrinsics[0][2];
        cy = intrinsics[0][3];
    }
    __syncthreads();


    GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
      int k = ku[n];
      int ix = ii[n];
      int jx = jj[n];
      int kx = kk[n];

      //CHECK_NAN_INF("fx", fx);
      //CHECK_NAN_INF("fy", fy);
      //CHECK_NAN_INF("cx", cx);
      //CHECK_NAN_INF("cy", cy);
      //CHECK_NAN_INF("poses[ix][0]", poses[ix][0]);
      //CHECK_NAN_INF("poses[ix][1]", poses[ix][1]);
      //CHECK_NAN_INF("poses[ix][2]", poses[ix][2]);
      //CHECK_NAN_INF("poses[jx][0]", poses[jx][0]);
      //CHECK_NAN_INF("poses[jx][1]", poses[jx][1]);
      //CHECK_NAN_INF("poses[jx][2]", poses[jx][2]);
      

      at::BFloat16 ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
      at::BFloat16 tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
      at::BFloat16 qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
      at::BFloat16 qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

      //CHECK_NAN_INF("ti[0]", ti[0]);
      //CHECK_NAN_INF("ti[1]", ti[1]);
      //CHECK_NAN_INF("ti[2]", ti[2]);
      //CHECK_NAN_INF("tj[0]", tj[0]);
      //CHECK_NAN_INF("tj[1]", tj[1]);
      //CHECK_NAN_INF("tj[2]", tj[2]);
      //CHECK_NAN_INF("qi[0]", qi[0]);
      //CHECK_NAN_INF("qi[1]", qi[1]);
      //CHECK_NAN_INF("qi[2]", qi[2]);
      //CHECK_NAN_INF("qi[3]", qi[3]);

      at::BFloat16 Xi[4], Xj[4];
      Xi[0] = (patches[kx][0][1][1] - cx) / fx;
      Xi[1] = (patches[kx][1][1][1] - cy) / fy;
      Xi[2] = 1.0;
      Xi[3] = patches[kx][2][1][1];

      // if (tid == 0)
      // {
      //   printf("[Thread %d] Xi: %.10f, %.10f, %.10f, %.10f\n", threadIdx.x, Xi[0], Xi[1], Xi[2], Xi[3]);
      // }

      //CHECK_NAN_INF("Xi[0]", Xi[0]);
      //CHECK_NAN_INF("Xi[1]", Xi[1]);
      //CHECK_NAN_INF("Xi[2]", Xi[2]);
      //CHECK_NAN_INF("Xi[3]", Xi[3]);

      at::BFloat16 tij[3], qij[4];
      relSE3(ti, qi, tj, qj, tij, qij);
      actSE3(tij, qij, Xi, Xj);

      //CHECK_NAN_INF("tij[0]", tij[0]);
      //CHECK_NAN_INF("tij[1]", tij[1]);
      //CHECK_NAN_INF("tij[2]", tij[2]);
      //CHECK_NAN_INF("qij[0]", qij[0]);
      //CHECK_NAN_INF("qij[1]", qij[1]);
      //CHECK_NAN_INF("qij[2]", qij[2]);
      //CHECK_NAN_INF("qij[3]", qij[3]);

      const at::BFloat16 X = Xj[0];
      const at::BFloat16 Y = Xj[1];
      const at::BFloat16 Z = Xj[2];
      const at::BFloat16 W = Xj[3];

      //CHECK_NAN_INF("X", X);
      //CHECK_NAN_INF("Y", Y);
      //CHECK_NAN_INF("Z", Z);
      //CHECK_NAN_INF("W", W);

      const at::BFloat16 d = (Z >= 0.2) ? 1.0 / Z : 0.0; 
      const at::BFloat16 d2 = d * d;

      //CHECK_NAN_INF("d", d);
      //CHECK_NAN_INF("d2", d2);

      const at::BFloat16 x1 = fx * (X / Z) + cx;
      const at::BFloat16 y1 = fy * (Y / Z) + cy;

      const at::BFloat16 rx = target[n][0] - x1;
      const at::BFloat16 ry = target[n][1] - y1;

      //CHECK_NAN_INF("x1", x1);
      //CHECK_NAN_INF("y1", y1);
      //CHECK_NAN_INF("rx", rx);
      //CHECK_NAN_INF("ry", ry);

      const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
        (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

      const at::BFloat16 mask = in_bounds ? 1.0 : 0.0;

      //CHECK_NAN_INF("mask", mask);
      //CHECK_NAN_INF("in_bounds", in_bounds);

      ix = ix - t0;
      jx = jx - t0;

    {
      const at::BFloat16 r = target[n][0] - x1;
      const at::BFloat16 w = mask * weight[n][0];

      //CHECK_NAN_INF("r", r);
      //CHECK_NAN_INF("w", w);


      at::BFloat16 Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      at::BFloat16 Ji[6], Jj[6] = {fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      //CHECK_NAN_INF("Jz", Jz);
      //CHECK_NAN_INF("Ji[0]", Ji[0]);
      //CHECK_NAN_INF("Ji[1]", Ji[1]);
      //CHECK_NAN_INF("Ji[2]", Ji[2]);
      //CHECK_NAN_INF("Ji[3]", Ji[3]);
      //CHECK_NAN_INF("Ji[4]", Ji[4]);
      //CHECK_NAN_INF("Ji[5]", Ji[5]);

      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
          {
            
            atomicAdd(&local_B[6*ix+i][6*ix+j],  at::BFloat16(w * Ji[i] * Ji[j]));
            //check if new value added is Inf, in that case cast it to the higher half precision value
            at::BFloat16& ref = local_B[6*ix+i][6*ix+j];

          }
            
          if (jx >= 0)
          {
            
            atomicAdd(&local_B[6*jx+i][6*jx+j],  at::BFloat16(w * Jj[i] * Jj[j]));
            at::BFloat16& ref = local_B[6*jx+i][6*jx+j];


          }
          if (ix >= 0 && jx >= 0) {
            
            atomicAdd(&local_B[6*ix+i][6*jx+j], at::BFloat16(-w * Ji[i] * Jj[j]));

            //check if new value added is
            // if (isinf(local_B[6*ix+i][6*jx+j])) {
            //     local_B[6*ix+i][6*jx+j] = at::BFloat16(65504.0f);
            // }
            atomicAdd(&local_B[6*jx+i][6*ix+j], at::BFloat16(-w * Jj[i] * Ji[j]));

            //check if new value

          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
        {
          atomicAdd(&local_E[6*ix+i][k], at::BFloat16(-w * Jz * Ji[i]));

          //check if new value added is
          // if(isinf(local_E[6*ix+i][k])) {
          //   local_E[6*ix+i][k] = at::BFloat16(65504.0f);
          // }
        }
        if (jx >= 0)
        {
          atomicAdd(&local_E[6*jx+i][k],  at::BFloat16(w * Jz * Jj[i]));

          //check if new value
          // if(isinf(local_E[6*jx+i][k])) {
          //   local_E[6*jx+i][k] = at::BFloat16(65504.0f);
          // }
        }

      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
        {
          atomicAdd(&local_v[6*ix+i], at::BFloat16(-w * r * Ji[i]));

        // if (isinf(local_v[6*ix+i])) {
        //     local_v[6*ix+i] = at::BFloat16(65504.0f);
        // }
        }
        if (jx >= 0)
        {
          atomicAdd(&local_v[6*jx+i],  at::BFloat16(w * r * Jj[i]));

          //check if new value
          // if (isinf(local_v[6*jx+i])) {
          //     local_v[6*jx+i] = at::BFloat16(65504.0f);
          // }
        }

      }

      atomicAdd(&local_C[k], at::BFloat16(w * Jz * Jz));

      //check if new value
      // if (isinf(local_C[k])) {
      //     local_C[k] = at::BFloat16(65504.0f);
      // }
      atomicAdd(&local_u[k], at::BFloat16(w *  r * Jz));

      //check if new value
      // if (isinf(local_u[k])) {
      //     local_u[k] = at::BFloat16(65504.0f);
      // }
    }

    {
      const at::BFloat16 r = target[n][1] - y1;
      const at::BFloat16 w = mask * weight[n][1];
      
      at::BFloat16 Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      at::BFloat16 Ji[6], Jj[6] = {0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
      
      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
          {
            
            atomicAdd(&local_B[6*ix+i][6*ix+j],  at::BFloat16(w * Ji[i] * Ji[j]));


          }
          if (jx >= 0)
          {
            
            atomicAdd(&local_B[6*jx+i][6*jx+j],  at::BFloat16(w * Jj[i] * Jj[j]));


          }
          if (ix >= 0 && jx >= 0) {

            atomicAdd(&local_B[6*ix+i][6*jx+j], at::BFloat16(-w * Ji[i] * Jj[j]));

            
            atomicAdd(&local_B[6*jx+i][6*ix+j], at::BFloat16(-w * Jj[i] * Ji[j]));


          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
        {
          atomicAdd(&local_E[6*ix+i][k], at::BFloat16(-w * Jz * Ji[i]));

        }
        if (jx >= 0)
        {
          atomicAdd(&local_E[6*jx+i][k],  at::BFloat16(w * Jz * Jj[i]));

        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
        {
          atomicAdd(&local_v[6*ix+i], at::BFloat16(-w * r * Ji[i]));

        }
        if (jx >= 0)
        {
          atomicAdd(&local_v[6*jx+i],  at::BFloat16(w * r * Jj[i]));

          //check if new value
        }
        // if (isinf(local_v[6*jx+i])) {
        //     local_v[6*jx+i] = at::BFloat16(65504.0f);
        // }
      }

      atomicAdd(&local_C[k], at::BFloat16(w * Jz * Jz));

      //check if new value
      // if (isinf(local_C[k])) {
      //     local_C[k] = at::BFloat16(65504.0f);
      // }
      atomicAdd(&local_u[k], at::BFloat16(w *  r * Jz));

      //check if new value
      // if (isinf(local_u[k])) {
      //     local_u[k] = at::BFloat16(65504.0f);
      // }
    }
    
  } // end of GPU_1D_KERNEL_LOOP
    __syncthreads();
}





//FORWARD FUNCTION IMPLEMENTING THE BUNDLE ADJUSTMENT
std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk,
    const int t0, const int t1, const int iterations)
{


  // auto check_nan = [](const torch::Tensor& t, const std::string& name) {
  // auto nan_mask = torch::isnan(t);
  // if (nan_mask.any().item<bool>()) {
  //   std::cout << "[ERROR] Tensor '" << name << "' contains NaNs!" << std::endl;
  // }
  // };

  // // Cast to half
  // poses = poses.to(torch::kBFloat16);
  // patches = patches.to(torch::kBFloat16););
  // intrinsics = intrinsics.to(torch::kBFloat16););
  // target = target.to(torch::kBFloat16););
  // weight = weight.to(torch::kBFloat16););
  // lmbda = lmbda.to(torch::kBFloat16);

  // // Check for NaNs
  // check_nan(poses, "poses");
  // check_nan(patches, "patches");
  // check_nan(intrinsics, "intrinsics");
  // check_nan(target, "target");
  // check_nan(weight, "weight");
  // check_nan(lmbda, "lmbda");


  //std::cout << "\nStarting CUDA Bundle Adjustment..." << std::endl;

  ////std::cout << "Starting CUDA Bundle Adjustment..." << std::endl;
  ////std::cout << "N edges : " << ii.size(0) << std::endl;

  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  //print every pose in the optimization window, so from t0 to t1


  const int num = ii.size(0);
  // torch::Tensor B = torch::empty({6*N, 6*N}, opts);
  // torch::Tensor E = torch::empty({6*N, 1*M}, opts);
  // torch::Tensor C = torch::empty({M}, opts);

  // torch::Tensor v = torch::empty({6*N}, opts);
  // torch::Tensor u = torch::empty({1*M}, opts);

  int num_threads = NUM_BLOCKS(ii.size(0)) * NUM_THREADS_PER_BLOCK;  // total threads launched
  ////std::cout << "Number of threads: " << num_threads << std::endl;
  ////std::cout << "NUM BLOCKS: " << NUM_BLOCKS(ii.size(0)) << std::endl;


  for (int itr=0; itr < iterations; itr++) {
    
    //for (int i = t0; i < t1; i++) {
    //  std::cout << "Pose " << i << ": "
    //            << poses[i][0].item<float>() << ", "
    //            << poses[i][1].item<float>() << ", "
    //            << poses[i][2].item<float>() << ", "
    //            << poses[i][3].item<float>() << ", "
    //            << poses[i][4].item<float>() << ", "
    //            << poses[i][5].item<float>() << ", "
    //            << poses[i][6].item<float>() << std::endl;
    //}
    //std::cout << "Iteration " << itr + 1 << " of " << iterations << std::endl << std::endl << std::endl;  
    torch::Tensor B = torch::zeros({num_threads, 6 * N, 6 * N}, opts);
    torch::Tensor E = torch::zeros({num_threads, 6 * N, M}, opts);
    torch::Tensor C = torch::zeros({num_threads, M}, opts);
    torch::Tensor v = torch::zeros({num_threads, 6 * N}, opts);
    torch::Tensor u = torch::zeros({num_threads, M}, opts); 
    

    v = v.view({num_threads, 6*N});
    u = u.view({num_threads, 1*M});

    reprojection_residuals_and_hessian<<<NUM_BLOCKS(ii.size(0)), NUM_THREADS_PER_BLOCK>>>(
      poses.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      patches.packed_accessor32<at::BFloat16,4,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      target.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      lmbda.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      ku.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      B.packed_accessor64<at::BFloat16,3,torch::RestrictPtrTraits>(),
      E.packed_accessor64<at::BFloat16,3,torch::RestrictPtrTraits>(),
      C.packed_accessor64<at::BFloat16,2,torch::RestrictPtrTraits>(),
      v.packed_accessor64<at::BFloat16,2,torch::RestrictPtrTraits>(),
      u.packed_accessor64<at::BFloat16,2,torch::RestrictPtrTraits>(),
      t0);
    
    cudaDeviceSynchronize();
    

    // B = kahan_reduce_dim0_fp16(B);
    // E = kahan_reduce_dim0_fp16(E);
    // C = kahan_reduce_dim0_fp16(C);
    // v = kahan_reduce_dim0_fp16(v);
    // u = kahan_reduce_dim0_fp16(u);

    //copy tensor to the cpu
    auto B_cpu = B.cpu();
    auto E_cpu = E.cpu();
    auto C_cpu = C.cpu();
    auto v_cpu = v.cpu();
    auto u_cpu = u.cpu();
    //check_tensor_nan_inf(B_cpu, "B");
    //check_tensor_nan_inf(E_cpu, "E");
    //check_tensor_nan_inf(C_cpu, "C");
    //check_tensor_nan_inf(v_cpu, "v");
    //check_tensor_nan_inf(u_cpu, "u");


    B = B.sum(0);
    E = E.sum(0);
    C = C.sum(0);
    v = v.sum(0);
    u = u.sum(0);

    //clamp every value in B, E, C, v, u to the range [-65504.0, 65504.0]


    //check_tensor_nan_inf(B.cpu(), "B_reduced");
    //check_tensor_nan_inf(E.cpu(), "E_reduced");
    //check_tensor_nan_inf(C.cpu(), "C_reduced");
    //check_tensor_nan_inf(v.cpu(), "v_reduced");
    //check_tensor_nan_inf(u.cpu(), "u_reduced");

    //DIMENSION IS REDUCED AFTER (FIRST CHANNEL IS REDUCED)

    //create one more dimension for v and u
    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});
    Q = Q.to(torch::kBFloat16);
    //clamp

    //print_tensor_stats(Q, "Q");
    if (t1 - t0 == 0) {

      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS_PER_BLOCK>>>(
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<at::BFloat16,4,torch::RestrictPtrTraits>(),
        dZ.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>());

    }

    else {

      torch::Tensor EQ = E * Q;
      EQ = EQ.to(torch::kBFloat16);
      //clamp

      //check_tensor_nan_inf(EQ.cpu(), "EQ");
      //print_tensor_stats(EQ, "EQ");
      torch::Tensor Et = torch::transpose(E, 0, 1);
      Et = Et.to(torch::kBFloat16);
      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      Qt = Qt.to(torch::kBFloat16);




      torch::Tensor S = B - torch::matmul(EQ, Et);
      S = S.to(torch::kBFloat16);


      //check_tensor_nan_inf(S.cpu(), "S");
      //print_tensor_stats(S, "S");
      torch::Tensor y = v - torch::matmul(EQ,  u);
      y = y.to(torch::kBFloat16);
      //check_tensor_nan_inf(y.cpu(), "y");
      //print_tensor_stats(y, "y");

      torch::Tensor I = torch::eye(6*N, opts);
      //if (itr == 0) print_2d_tensor_to_file(S, "S_matrix_before_reg_1e-3_fp16.txt", itr);
      S += I * (1e-2 * S + 1.0);

      // Convert to FP32 for numerical stability
      S = S.to(torch::kFloat);
      y = y.to(torch::kFloat);

      // Save matrix for debug
      //remove the file if it exists
      //if (itr == 0) std::remove("S_bf16_chol.txt");
      //if (itr == 0) print_2d
      //if (itr == 0) print_2d_tensor_to_file(S, "S_bf16_chol.txt", itr);

      torch::Tensor L = torch::linalg::cholesky(S);

      torch::Tensor dX = torch::cholesky_solve(y, L);


      //if (itr == 0) std::remove("L_bf16_chol.txt");
      //if (itr == 0) print_2d
      //if (itr == 0) print_2d_tensor_to_file(L, "L_bf16_chol.txt", itr);

      dX = dX.to(torch::kBFloat16);

      //print_tensor_stats(dX, "dX");
      torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));
      //check_tensor_nan_inf(dZ.cpu(), "dZ
      dZ = dZ.to(torch::kBFloat16);

      //check_tensor_nan_inf(dZ.cpu(), "dZ");
      //check_tensor_nan_inf(dX.cpu(), "dX");
      //print_tensor_stats(torch::matmul(Et, dX), "Et * dX");
      //print_tensor_stats(dZ, "dZ");
      dX = dX.view({N, 6});
      dZ = dZ.view({M});
      //std::cout << "After solving the system: " << std::endl;


      pose_retr_kernel<<<NUM_BLOCKS(N), NUM_THREADS_PER_BLOCK>>>(t0, t1,
          poses.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
          dX.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>());
      
      //check_tensor_nan_inf(poses.cpu(), "poses after retrSE3");

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS_PER_BLOCK>>>(
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          patches.packed_accessor32<at::BFloat16,4,torch::RestrictPtrTraits>(),
          dZ.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>());
      
      //check_tensor_nan_inf(patches.cpu(), "patches after patch_retr_kernel");
      

    }
  }
  
  return {};
}
