#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <algorithm>


void log_tensor_shape(const torch::Tensor& t, const std::string& name) {
    std::cout << "[LOG] Tensor " << name << " shape: ";
    for (const auto& s : t.sizes()) std::cout << s << " ";
    std::cout << std::endl;
}

template <typename T, size_t N>
void log_accessor_shape(const at::TensorAccessor<T, N>& accessor, const std::string& name) {
    std::cout << "[LOG] Accessor " << name << " shape: ";
    for (size_t i = 0; i < N; ++i) {
        std::cout << accessor.size(i) << " ";
    }
    std::cout << std::endl;
}



#define CPU_1D_KERNEL_LOOP(i, n) \
  for (int i = 0; i<n; i++)

void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
void
expSO3(const float *phi, float* q) {
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

void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}


void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
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


void pose_retr_cpu(
    int t0, int t1,
    at::TensorAccessor<float, 2> poses,
    at::TensorAccessor<float, 2> update)
{
  CPU_1D_KERNEL_LOOP(i, t1 - t0) {
    if (i % 10 == 0) std::cout << "[LOG] pose_retr_cpu loop i = " << i << std::endl;

    int t = t0 + i;
    float t1_[3], t0_[3] = { poses[t][0], poses[t][1], poses[t][2] };
    float q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    float xi[6] = {
        update[i][0],
        update[i][1],
        update[i][2],
        update[i][3],
        update[i][4],
        update[i][5],
    };

    retrSE3(xi, t0_, q0, t1_, q1);

    poses[t][0] = t1_[0];
    poses[t][1] = t1_[1];
    poses[t][2] = t1_[2];
    poses[t][3] = q1[0];
    poses[t][4] = q1[1];
    poses[t][5] = q1[2];
    poses[t][6] = q1[3];
  }
}


void patch_retr_cpu(
    at::TensorAccessor<long, 1> index,
    at::TensorAccessor<float, 4> patches,
    at::TensorAccessor<float, 1> update)
{
  CPU_1D_KERNEL_LOOP(n, index.size(0)) {
    
    if (n % 10 == 0) std::cout << "[LOG] patch_retr_cpu loop n = " << n << std::endl;
    const int p = patches.size(2);
    const int ix = index[n];

    float d = patches[ix][2][0][0];
    d = d + update[n];
    d = (d > 20) ? 1.0 : d;
    d = std::max(d, (float)1e-4);

    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
  std::cout << "[LOG] Exiting patch_retr_cpu\n";
}


void reprojection_residuals_and_hessian_cpu(
    at::TensorAccessor<float, 2> poses,
    at::TensorAccessor<float, 4> patches,
    at::TensorAccessor<float, 2> intrinsics,
    at::TensorAccessor<float, 2> target,
    at::TensorAccessor<float, 2> weight,
    at::TensorAccessor<float, 1> lmbda,
    at::TensorAccessor<long, 1> ii,
    at::TensorAccessor<long, 1> jj,
    at::TensorAccessor<long, 1> kk,
    at::TensorAccessor<long, 1> ku,
    at::TensorAccessor<float, 2> B,
    at::TensorAccessor<float, 2> E,
    at::TensorAccessor<float, 1> C,
    at::TensorAccessor<float, 1> v,
    at::TensorAccessor<float, 1> u,
    const int t0)
{
  log_accessor_shape(poses, "poses");
  log_accessor_shape(patches, "patches");
  log_accessor_shape(intrinsics, "intrinsics");
  log_accessor_shape(target, "target");
  log_accessor_shape(weight, "weight");
  log_accessor_shape(lmbda, "lmbda");
  log_accessor_shape(ii, "ii");
  log_accessor_shape(jj, "jj");
  log_accessor_shape(kk, "kk");


  float fx, fy, cx, cy;
  fx = intrinsics[0][0];
  fy = intrinsics[0][1];
  cx = intrinsics[0][2];
  cy = intrinsics[0][3];


  CPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int k = ku[n];
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n];

    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    float Xi[4], Xj[4];
    Xi[0] = (patches[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0;
    Xi[3] = patches[kx][2][1][1];
    
    float tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);
    actSE3(tij, qij, Xi, Xj);

    const float X = Xj[0];
    const float Y = Xj[1];
    const float Z = Xj[2];
    const float W = Xj[3];

    const float d = (Z >= 0.2) ? 1.0 / Z : 0.0; 
    const float d2 = d * d;

    const float x1 = fx * (X / Z) + cx;
    const float y1 = fy * (Y / Z) + cy;

    const float rx = target[n][0] - x1;
    const float ry = target[n][1] - y1;

    const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
      (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

    const float mask = in_bounds ? 1.0 : 0.0;

    ix = ix - t0;
    jx = jx - t0;

    log_accessor_shape(B, "B");
    log_accessor_shape(E, "E");
    {
      const float r = target[n][0] - x1;
      const float w = mask * weight[n][0];

      float Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      float Ji[6], Jj[6] = {fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};
      
      std::cout << "[LOG] Calculating adjSE3 for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      adjSE3(tij, qij, Jj, Ji);
      
      std::cout << "[LOG] Finished adjSE3 for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;

      std::cout << "[LOG] Updating B, E, C, v, u for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B[6*ix+i][6*jx+j] += -w * Ji[i] * Jj[j];
            B[6*jx+i][6*ix+j] += -w * Jj[i] * Ji[j];
          }
        }
      }
      

      std::cout << "[LOG] Finished updating B for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E[6*ix+i][k] += -w * Jz * Ji[i];
        if (jx >= 0)
          E[6*jx+i][k] += w * Jz * Jj[i];
      }

      std::cout << "[LOG] Finished updating E for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v[6*ix+i] += -w * r * Ji[i];
        if (jx >= 0)
          v[6*jx+i] += w * r * Jj[i];
      }
      
      C[k] += w * Jz * Jz;
      u[k] += w *  r * Jz;
    }

    {
      const float r = target[n][1] - y1;
      const float w = mask * weight[n][1];
      
      float Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      float Ji[6], Jj[6] = {0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
      
      std::cout << "[LOG] Calculating adjSE3 for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      adjSE3(tij, qij, Jj, Ji);
      std::cout << "[LOG] Finished adjSE3 for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;

      std::cout << "[LOG] Updating B, E, C, v, u for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B[6*ix+i][6*jx+j] += -w * Ji[i] * Jj[j];
            B[6*jx+i][6*ix+j] += -w * Jj[i] * Ji[j];
          }
        }
      }

      std::cout << "[LOG] Finished updating B for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E[6*ix+i][k] += -w * Jz * Ji[i];
        if (jx >= 0)
          E[6*jx+i][k] += w * Jz * Jj[i];
      }

      std::cout << "[LOG] Finished updating E for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v[6*ix+i] += -w * r * Ji[i];
        if (jx >= 0)
          v[6*jx+i] += w * r * Jj[i];
      }
      
      std::cout << "[LOG] Finished updating v for n = " << n << ", ix = " << ix << ", jx = " << jx << std::endl;
      C[k] += w * Jz * Jz;
      u[k] += w *  r * Jz;
    }
  }
}


//FORWARD FUNCTION IMPLEMENTING THE BUNDLE ADJUSTMENT
std::vector<torch::Tensor> ba_cpu(
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
  
  std::cout << "[LOG] Entering ba_cpu function\n";
  log_tensor_shape(poses, "poses");
  log_tensor_shape(patches, "patches");
  log_tensor_shape(intrinsics, "intrinsics");
  log_tensor_shape(target, "target");
  log_tensor_shape(weight, "weight");
  log_tensor_shape(lmbda, "lmbda");
  log_tensor_shape(ii, "ii");
  log_tensor_shape(jj, "jj");
  log_tensor_shape(kk, "kk");


  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCUDA);

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  torch::Tensor B = torch::empty({6*N, 6*N}, opts);
  torch::Tensor E = torch::empty({6*N, 1*M}, opts);
  torch::Tensor C = torch::empty({M}, opts);

  torch::Tensor v = torch::empty({6*N}, opts);
  torch::Tensor u = torch::empty({1*M}, opts);

  for (int itr=0; itr < iterations; itr++) {

    B.zero_();
    E.zero_();
    C.zero_();
    v.zero_();
    u.zero_();

    v = v.view({6*N});
    u = u.view({1*M});
    
    std::cout << "[LOG] reprojection_residuals_and_hessian_cpu\n";
    reprojection_residuals_and_hessian_cpu(
        poses.accessor<float, 2>(),
        patches.accessor<float, 4>(),
        intrinsics.accessor<float, 2>(),
        target.accessor<float, 2>(),
        weight.accessor<float, 2>(),
        lmbda.accessor<float, 1>(),
        ii.accessor<long, 1>(),
        jj.accessor<long, 1>(),
        kk.accessor<long, 1>(),
        ku.accessor<long, 1>(),
        B.accessor<float, 2>(),
        E.accessor<float, 2>(),
        C.accessor<float, 1>(),
        v.accessor<float, 1>(),
        u.accessor<float, 1>(),
        t0);

    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

    if (t1 - t0 == 0) {

      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      patch_retr_cpu(
          kx.accessor<long, 1>(),
          patches.accessor<float, 4>(),
          dZ.accessor<float, 1>());

    }

    else {

      torch::Tensor EQ = E * Q;
      torch::Tensor Et = torch::transpose(E, 0, 1);
      torch::Tensor Qt = torch::transpose(Q, 0, 1);

      torch::Tensor S = B - torch::matmul(EQ, Et);
      torch::Tensor y = v - torch::matmul(EQ,  u);

      torch::Tensor I = torch::eye(6*N, opts);
      S += I * (1e-4 * S + 1.0);

        std::cout << "[LOG] Cholesky decomposition\n";
      torch::Tensor U = torch::linalg::cholesky(S);
      torch::Tensor dX = torch::cholesky_solve(y, U);
      torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));

      dX = dX.view({N, 6});
      dZ = dZ.view({M});
      
      
      std::cout << "[LOG] Calling pose_retr_cpu\n";
      pose_retr_cpu(
          t0, t1,
          poses.accessor<float, 2>(),
          dX.accessor<float, 2>());    
      
      std::cout << "[LOG] Calling patch_retr_cpu\n";
      patch_retr_cpu(
        kx.accessor<long, 1>(),
        patches.accessor<float, 4>(),
        dZ.accessor<float, 1>());
      
      std::cout << "[LOG] Iteration " << itr << " completed\n";
    }
  }
      std::cout << "[LOG] Exiting ba_cpu function\n";
  return {};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ba_cpu, "BA forward operator implementation on CPU");
}