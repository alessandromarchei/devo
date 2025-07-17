
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


#define DECIMAL_PLACES 2


#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)


__global__ void round_tensor_kernel_anyshape(double* data, int64_t numel, const double factor) {
    GPU_1D_KERNEL_LOOP(i, numel) {
        double val = data[i];
        val = roundf(val * factor) / factor;
        data[i] = val;
    }
}

void round_tensor(torch::Tensor& tensor, int decimal_places) {
    double factor = powf(10.0f, (double)decimal_places);

    //std::cout << "[round_tensor] Shape: " << tensor.sizes()
              //<< ", Dtype: " << tensor.dtype()
              //<< ", Device: " << tensor.device()
              //<< ", Decimal places: " << decimal_places << std::endl;

    TORCH_CHECK(tensor.is_cuda(), "round_tensor only supports CUDA tensors.");
    TORCH_CHECK(tensor.scalar_type() == torch::kDouble, "round_tensor only supports double32 tensors.");

    torch::Tensor tensor_flat = tensor.view(-1);

    double* data_ptr = tensor_flat.data_ptr<double>();
    int64_t numel = tensor_flat.numel();

    round_tensor_kernel_anyshape<<<NUM_BLOCKS(numel), NUM_THREADS>>>(data_ptr, numel, factor);
    
    cudaDeviceSynchronize();

    //std::cout << "[output] Shape: " << tensor.sizes()
              //<< ", Dtype: " << tensor.dtype()
              //<< ", Device: " << tensor.device()
              //<< std::endl;
    //std::cout << "[round_tensor] Done." << std::endl;
}




// --------------------------- SE3 and SO3 math -----------------------------

__device__ void actSO3(const double *q, const double *X, double *Y) {
    double uv[3];
    uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
    uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
    uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

    Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
    Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
    Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__ void actSE3(const double *t, const double *q, const double *X, double *Y) {
    actSO3(q, X, Y);
    Y[3] = X[3];
    Y[0] += X[3] * t[0];
    Y[1] += X[3] * t[1];
    Y[2] += X[3] * t[2];
}

__device__ void adjSE3(const double *t, const double *q, const double *X, double *Y) {
    double qinv[4] = {-q[0], -q[1], -q[2], q[3]};
    actSO3(qinv, &X[0], &Y[0]);
    actSO3(qinv, &X[3], &Y[3]);

    double u[3], v[3];
    u[0] = t[2]*X[1] - t[1]*X[2];
    u[1] = t[0]*X[2] - t[2]*X[0];
    u[2] = t[1]*X[0] - t[0]*X[1];

    actSO3(qinv, u, v);
    Y[3] += v[0];
    Y[4] += v[1];
    Y[5] += v[2];
}

__device__ void relSE3(const double *ti, const double *qi, const double *tj, const double *qj, double *tij, double *qij) {
    qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1];
    qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2];
    qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0];
    qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2];

    actSO3(qij, ti, tij);
    tij[0] = tj[0] - tij[0];
    tij[1] = tj[1] - tij[1];
    tij[2] = tj[2] - tij[2];
}


__device__ void
crossInplace(const double* a, double *b) {
  double x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}


  
__device__ void
expSO3(const double *phi, double* q) {
  // SO3 exponential map
  double theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  double theta_p4 = theta_sq * theta_sq;

  double theta = sqrtf(theta_sq);
  double imag, real;

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
expSE3(const double *xi, double* t, double* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  double tau[3] = {xi[0], xi[1], xi[2]};
  double phi[3] = {xi[3], xi[4], xi[5]};

  double theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  double theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    double a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    double b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}




__device__ void
retrSE3(const double *xi, const double* t, const double* q, double* t1, double* q1) {
  // retraction on SE3 manifold

  double dt[3] = {0, 0, 0};
  double dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}





__global__ void pose_retr_kernel(const int t0, const int t1,
    torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(i, t1 - t0) {
    const double t = t0 + i;
    double t1[3], t0[3] = { poses[t][0], poses[t][1], poses[t][2] };
    double q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    double xi[6] = {
      update[i][0],
      update[i][1],
      update[i][2],
      update[i][3],
      update[i][4],
      update[i][5],
    };

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
    torch::PackedTensorAccessor32<double,4,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(n, index.size(0)) {
    const int p = patches.size(2);
    const int ix = index[n];
  
    double d = patches[ix][2][0][0];
    d = d + update[n];
    d = (d > 20) ? 1.0 : d;
    d = max(d, 1e-4);

    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
}




__global__ void reprojection_residuals_and_hessian(
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<double,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> lmbda,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ku,
    torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> E,
    torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<double,1,torch::RestrictPtrTraits> u, const int t0)
{

  __shared__ double fx, fy, cx, cy;
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

    double ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    double tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    double qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    double qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    double Xi[4], Xj[4];
    Xi[0] = (patches[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0;
    Xi[3] = patches[kx][2][1][1];
    
    double tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);
    actSE3(tij, qij, Xi, Xj);

    const double X = Xj[0];
    const double Y = Xj[1];
    const double Z = Xj[2];
    const double W = Xj[3];

    const double d = (Z >= 0.2) ? 1.0 / Z : 0.0; 
    const double d2 = d * d;

    const double x1 = fx * (X / Z) + cx;
    const double y1 = fy * (Y / Z) + cy;

    const double rx = target[n][0] - x1;
    const double ry = target[n][1] - y1;

    const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
      (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

    const double mask = in_bounds ? 1.0 : 0.0;

    ix = ix - t0;
    jx = jx - t0;

    {
      const double r = target[n][0] - x1;
      const double w = mask * weight[n][0];

      double Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      double Ji[6], Jj[6] = {fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
          if (jx >= 0)
            atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
          if (ix >= 0 && jx >= 0) {
            atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
            atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
        if (jx >= 0)
          atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
        if (jx >= 0)
          atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
      }

      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }

    {
      const double r = target[n][1] - y1;
      const double w = mask * weight[n][1];
      
      double Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      double Ji[6], Jj[6] = {0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
      
      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
          if (jx >= 0)
            atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
          if (ix >= 0 && jx >= 0) {
            atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
            atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
        if (jx >= 0)
          atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
        if (jx >= 0)
          atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
      }

      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }
  }
}



__global__ void reproject(
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<double,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    torch::PackedTensorAccessor32<double,4,torch::RestrictPtrTraits> coords) {

  __shared__ double fx, fy, cx, cy;
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

    double ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    double tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    double qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    double qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    double tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);

    double Xi[4], Xj[4];
    for (int i=0; i<patches.size(2); i++) {
      for (int j=0; j<patches.size(3); j++) {
        
        Xi[0] = (patches[kx][0][i][j] - cx) / fx;
        Xi[1] = (patches[kx][1][i][j] - cy) / fy;
        Xi[2] = 1.0;
        Xi[3] = patches[kx][2][i][j];

        actSE3(tij, qij, Xi, Xj);

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
    .dtype(torch::kDouble).device(torch::kCUDA);

  torch::Tensor coords = torch::empty({N, 2, P, P}, opts);

  reproject<<<NUM_BLOCKS(N), NUM_THREADS>>>(
    poses.packed_accessor32<double,2,torch::RestrictPtrTraits>(),
    patches.packed_accessor32<double,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<double,2,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<double,4,torch::RestrictPtrTraits>());

  return coords.view({1, N, 2, P, P});

}


// ---------------------- Bundle Adjustment Core -----------------------------

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
    const int t0, const int t1, const int iterations,
  const int decimal_places = DECIMAL_PLACES)
{
    auto ktuple = torch::_unique(kk, true, true);
    torch::Tensor kx = std::get<0>(ktuple);
    torch::Tensor ku = std::get<1>(ktuple);

    const int N = t1 - t0;
    const int M = kx.size(0);
    const int P = patches.size(3);

    auto opts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA);

    poses = poses.view({-1, 7});
    patches = patches.view({-1, 3, P, P});
    intrinsics = intrinsics.view({-1, 4});
    target = target.view({-1, 2});
    weight = weight.view({-1, 2});

    torch::Tensor B = torch::empty({6 * N, 6 * N}, opts);
    torch::Tensor E = torch::empty({6 * N, 1 * M}, opts);
    torch::Tensor C = torch::empty({M}, opts);
    torch::Tensor v = torch::empty({6 * N}, opts);
    torch::Tensor u = torch::empty({1 * M}, opts);


    for (int itr = 0; itr < iterations; itr++) {
        B.zero_();
        E.zero_();
        C.zero_();
        v.zero_();
        u.zero_();
        v = v.view({6*N});
        u = u.view({1*M});
        reprojection_residuals_and_hessian<<<NUM_BLOCKS(ii.size(0)), NUM_THREADS>>>(
            poses.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            patches.packed_accessor32<double, 4, torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            lmbda.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            kk.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            ku.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            B.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            E.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
            C.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
            v.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
            u.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
            t0);
        
        v = v.view({6*N, 1});
        u = u.view({1*M, 1});
        //std::cout << "truncating B" << std::endl;
        round_tensor(B, decimal_places);

        //std::cout << "truncating E" << std::endl;
        round_tensor(E, decimal_places);

        //std::cout << "truncating C" << std::endl;
        round_tensor(C, decimal_places);

        //std::cout << "truncating v" << std::endl;
        round_tensor(v, decimal_places);

        //std::cout << "truncating u" << std::endl;
        round_tensor(u, decimal_places);

        torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

        //std::cout << "truncating Q" << std::endl;
        round_tensor(Q, decimal_places);

        if (t1 - t0 == 0) {
            torch::Tensor Qt = Q.transpose(0, 1);
            torch::Tensor dZ = Qt * u;
            dZ = dZ.view({M});
            
            //std::cout << "truncating dZ" << std::endl;
            round_tensor(dZ, decimal_places);

            patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
                kx.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
                patches.packed_accessor32<double, 4, torch::RestrictPtrTraits>(),
                dZ.packed_accessor32<double, 1, torch::RestrictPtrTraits>());
        }
        else {
            torch::Tensor EQ = E * Q;
            
            torch::Tensor Et = E.transpose(0, 1);
            torch::Tensor Qt = Q.transpose(0, 1);

            torch::Tensor S = B - torch::matmul(EQ, Et);
            torch::Tensor y = v - torch::matmul(EQ, u);

            //std::cout << "truncating S" << std::endl;
            round_tensor(S, decimal_places);

            //std::cout << "truncating y" << std::endl;
            round_tensor(y, decimal_places);

            torch::Tensor I = torch::eye(6 * N, opts);
            S += I * (1e-4 * S + 1.0);

            //std::cout << "torch linalg cholesky" << std::endl;
            torch::Tensor U = torch::linalg::cholesky(S);

            //std::cout << "torch linalg cholesky_solve" << std::endl;
            torch::Tensor dX = torch::cholesky_solve(y, U);

            //std::cout << "matmul" << std::endl;  
            torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));

            dX = dX.view({N, 6});
            dZ = dZ.view({M});

            //std::cout << "truncating dX" << std::endl;
            round_tensor(dX, decimal_places);

            //std::cout << "truncating dZ" << std::endl;
            round_tensor(dZ, decimal_places);

            pose_retr_kernel<<<NUM_BLOCKS(N), NUM_THREADS>>>(
                t0, t1,
                poses.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
                dX.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

            patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
                kx.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
                patches.packed_accessor32<double, 4, torch::RestrictPtrTraits>(),
                dZ.packed_accessor32<double, 1, torch::RestrictPtrTraits>());
        }
    }

    return {};
}
