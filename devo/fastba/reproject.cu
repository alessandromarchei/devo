#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)


__device__ void
actSO3_gpu(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3_gpu(const float *t, const float *q, const float *X, float *Y) {
  actSO3_gpu(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3_gpu(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3_gpu(qinv, &X[0], &Y[0]);
  actSO3_gpu(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3_gpu(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3_gpu(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3_gpu(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
__device__ void
expSO3_gpu(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace_gpu(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3_gpu(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3_gpu(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace_gpu(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace_gpu(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


__device__ void
retrSE3_gpu(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3_gpu(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3_gpu(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}




__global__ void reproject(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords) {

  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n];

    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    float tij[3], qij[4];
    relSE3_gpu(ti, qi, tj, qj, tij, qij);

    float Xi[4], Xj[4];
    for (int i=0; i<patches.size(2); i++) {
      for (int j=0; j<patches.size(3); j++) {
        
        Xi[0] = (patches[kx][0][i][j] - cx) / fx;
        Xi[1] = (patches[kx][1][i][j] - cy) / fy;
        Xi[2] = 1.0;
        Xi[3] = patches[kx][2][i][j];

        actSE3_gpu(tij, qij, Xi, Xj);

        coords[n][0][i][j] = fx * (Xj[0] / Xj[2]) + cx;
        coords[n][1][i][j] = fy * (Xj[1] / Xj[2]) + cy;
        // coords[n][2][i][j] = 1.0 / Xj[2];

      }
    }
  }
}



torch::Tensor cuda_reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj, 
    torch::Tensor kk)
{

  const int N = ii.size(0);
  const int P = patches.size(3); // patch size

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor coords = torch::empty({N, 2, P, P}, opts);

  reproject<<<NUM_BLOCKS(N), NUM_THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return coords.view({1, N, 2, P, P});

}