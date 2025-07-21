#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include "helper.cpp"

//specify a quad precision type, instead of using __float128 directly
typedef __float128 quad;


void actSO3(const double *q, const double *X, double *Y) {
  double uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

void actSE3(const double *t, const double *q, const double *X, double *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

void adjSE3(const double *t, const double *q, const double *X, double *Y) {
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

void relSE3(const double *ti, const double *qi, const double *tj, const double *qj, double *tij, double *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}


void expSO3(const double *phi, double* q) {
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

void crossInplace(const double* a, double *b) {
  double x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

void expSE3(const double *xi, double* t, double* q) {
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


void retrSE3(const double *xi, const double* t, const double* q, double* t1, double* q1) {
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


void actSO3(const quad *q, const quad *X, quad *Y) {
  quad uv[3];
  uv[0] = 2.0Q * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0Q * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0Q * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

void actSE3(const quad *t, const quad *q, const quad *X, quad *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

void adjSE3(const quad *t, const quad *q, const quad *X, quad *Y) {
  quad qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  quad u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

void relSE3(const quad *ti, const quad *qi, const quad *tj, const quad *qj, quad *tij, quad *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
void expSO3(const quad *phi, quad* q) {
  // SO3 exponential map
  quad theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  quad theta_p4 = theta_sq * theta_sq;

  quad theta = sqrtf(theta_sq);
  quad imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5Q - (1.0Q/48.0Q)*theta_sq + (1.0Q/3840.0Q)*theta_p4;
    real = 1.0Q - (1.0Q/ 8.0Q)*theta_sq + (1.0Q/ 384.0Q)*theta_p4;
  } else {
    imag = sinf(0.5Q * theta) / theta;
    real = cosf(0.5Q * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

void crossInplace(const quad* a, quad *b) {
  quad x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

void expSE3(const quad *xi, quad* t, quad* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  quad tau[3] = {xi[0], xi[1], xi[2]};
  quad phi[3] = {xi[3], xi[4], xi[5]};

  quad theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  quad theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4Q) {
    quad a = (1.0Q - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    quad b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}

void retrSE3(const quad *xi, const quad* t, const quad* q, quad* t1, quad* q1) {
  // retraction on SE3 manifold

  quad dt[3] = {0.0Q, 0.0Q, 0.0Q};
  quad dq[4] = {0.0Q, 0.0Q, 0.0Q, 1.0Q};
  
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



void pose_retr(const int t0, const int t1,
    torch::Tensor poses,
    torch::Tensor update)
{

  //std::cout << "Updating poses from " << t0 << " to " << t1 << std::endl;
  // move tensors to cpu
  poses = poses.to(torch::kCPU);
  update = update.to(torch::kCPU);

  //std::cout << "Pose size: " << poses.sizes() << std::endl;
  //std::cout << "Update size: " << update.sizes() << std::endl;
  //std::cout << "creating accessors..." << std::endl;
  auto poses_accessor = poses.accessor<double,2>();
  auto update_accessor = update.accessor<double,2>();

  //std::cout << "Updating poses..." << std::endl;
  for (int i=0; i < t1 - t0; i++) {
    const double t = t0 + i;
    double t0[3] = { poses_accessor[t][0], poses_accessor[t][1], poses_accessor[t][2] };
    double q0[4] = { poses_accessor[t][3], poses_accessor[t][4], poses_accessor[t][5], poses_accessor[t][6] };
    double t1[3] = {0.0, 0.0, 0.0};
    double q1[4] = {0.0, 0.0, 0.0, 1.0};

    //std::cout << "Pose " << t << ": " << t0[0] << " " << t0[1] << " " << t0[2] << " " << q0[0] << " " << q0[1] << " " << q0[2] << " " << q0[3] << std::endl;

    double xi[6] = {
      update_accessor[i][0],
      update_accessor[i][1],
      update_accessor[i][2],
      update_accessor[i][3],
      update_accessor[i][4],
      update_accessor[i][5],
    };

    //std::cout << xi[0] << " " << xi[1] << " " << xi[2] << " " << xi[3] << " " << xi[4] << " " << xi[5] << std::endl;

    //std::cout << "Retrieving pose..." << std::endl;
    retrSE3(xi, t0, q0, t1, q1);

    //std::cout << "Updated pose: " << t1[0] << " " << t1[1] << " " << t1[2] << " " << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << std::endl;
    poses_accessor[t][0] = t1[0];
    poses_accessor[t][1] = t1[1];
    poses_accessor[t][2] = t1[2];
    poses_accessor[t][3] = q1[0];
    poses_accessor[t][4] = q1[1];
    poses_accessor[t][5] = q1[2];
    poses_accessor[t][6] = q1[3];
  }
}


void patch_retr(
    torch::Tensor index,
    torch::Tensor patches,
    torch::Tensor update)
{
    //move to CPU
    //std::cout << "Updating patches..." << std::endl;
    //std::cout << "Index size: " << index.sizes() << std::endl;
    //std::cout << "Patches size: " << patches.sizes() << std::endl;
    //std::cout << "Update size: " << update.sizes() << std::endl;
    index = index.to(torch::kCPU);
    patches = patches.to(torch::kCPU);
    update = update.to(torch::kCPU);   

    //std::cout << "Creating accessors..." << std::endl;
    auto index_acc = index.accessor<int64_t, 1>();
    auto patches_acc = patches.accessor<double, 4>();
    auto update_acc = update.accessor<double, 1>();

    int p = patches.size(2);
    for (int n = 0; n < index.size(0); n++) {
        int64_t ix = index_acc[n];

        //std::cout << "Patch " << n << ": " << ix << std::endl;
        double d = patches_acc[ix][2][0][0];
        d = d + update_acc[n];
        d = (d > 20.0) ? 1.0 : d;
        d = std::max(d, 1e-4);

        //std::cout << "Updated depth: " << d << std::endl;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                patches_acc[ix][2][i][j] = d;
            }
        }
    }
}


void reprojection_residuals_and_hessian(
    std::vector<std::vector<quad>> poses,
    std::vector<std::vector<std::vector<std::vector<quad>>>> patches,
    std::vector<std::vector<quad>> intrinsics,
    std::vector<std::vector<quad>> target,
    std::vector<std::vector<quad>> weight,
    quad lmbda,
    std::vector<long> ii,
    std::vector<long> jj,
    std::vector<long> kk,
    std::vector<long> ku,
    std::vector<std::vector<quad>> B,
    std::vector<std::vector<quad>> E,
    std::vector<quad> C,
    std::vector<quad> v,
    std::vector<quad> u, const int t0)
{


  //std::cout << "Setting tensors to zero..." << std::endl;
  quad fx, fy, cx, cy;
  fx = intrinsics[0][0];
  fy = intrinsics[0][1];
  cx = intrinsics[0][2];
  cy = intrinsics[0][3];


  for (int n=0; n < ii.size(); n++) {
    //std::cout << "Processing point " << n << ": " << ii_accessor[n] << " " << jj_accessor[n] << " " << kk_accessor[n] << std::endl;
    int k = ku[n];
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n];

    //std::cout << "Point: " << n << " " << ix << " " << jx << " " << kx << std::endl;
    quad ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    quad tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    quad qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    quad qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    quad Xi[4], Xj[4];
    Xi[0] = (patches[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0Q;
    Xi[3] = patches[kx][2][1][1];
    
    quad tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);
    actSE3(tij, qij, Xi, Xj);

    const quad X = Xj[0];
    const quad Y = Xj[1];
    const quad Z = Xj[2];
    const quad W = Xj[3];

    const quad d = (Z >= 0.2Q) ? 1.0Q / Z : 0.0Q; 
    const quad d2 = d * d;

    const quad x1 = fx * (X / Z) + cx;
    const quad y1 = fy * (Y / Z) + cy;

    const quad rx = target[n][0] - x1;
    const quad ry = target[n][1] - y1;

    const bool in_bounds = (sqrtq(rx*rx + ry*ry) < 128.0Q) && (Z > 0.2Q) &&
      (x1 > -64.0Q) && (y1 > -64.0Q) && (x1 < 2.0Q*cx + 64.0Q) && (y1 < 2.0Q*cy + 64.0Q);

    const quad mask = in_bounds ? 1.0Q : 0.0Q;

    ix = ix - t0;
    jx = jx - t0;

    //std::cout << "Computing residuals for point " << n << ": " << x1 << " " << y1 << " " << Z << " " << W << std::endl;
    {
      const quad r = target[n][0] - x1;
      const quad w = mask * weight[n][0];

      quad Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      quad Ji[6], Jj[6] = {fx*W*d, 0.0Q, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E[6*ix+i][k] -= w * Jz * Ji[i];
        if (jx >= 0)
          E[6*jx+i][k] +=  w * Jz * Jj[i];
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v[6*ix+i] -= w * r * Ji[i];
        if (jx >= 0)
          v[6*jx+i] += w * r * Jj[i];
      }
      C[k] += w * Jz * Jz;
      u[k] += w * r * Jz;
    }


    {
      const quad r = target[n][1] - y1;
      const quad w = mask * weight[n][1];
      
      quad Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      quad Ji[6], Jj[6] = {0.0Q, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
      
      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E[6*ix+i][k] -= w * Jz * Ji[i];
        if (jx >= 0)
          E[6*jx+i][k] += w * Jz * Jj[i];
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v[6*ix+i] -= w * r * Ji[i];
        if (jx >= 0)
          v[6*jx+i] += w * r * Jj[i];
      }
      C[k] += w * Jz * Jz;
      u[k] += w * r * Jz;
    }
  }
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
  //move everything to CPU
  poses = poses.to(torch::kCPU);
  patches = patches.to(torch::kCPU);
  intrinsics = intrinsics.to(torch::kCPU);
  target = target.to(torch::kCPU);
  weight = weight.to(torch::kCPU);
  lmbda = lmbda.to(torch::kCPU);
  ii = ii.to(torch::kCPU);
  jj = jj.to(torch::kCPU);


  //convert the torch tensors to QUAD precision by using the helper functions
  /*INPUT SHAPES
  - poses : 1,4096,7
  - patches : 1,Npatches, 3, 3, 3 (it includes the depth channel)
  - intrinsics : 1,4096, 4
  - target : 1,Nedges, 2
  - weight : 1,Nedges, 2
  - lmbda : 1
  - ii : Nedges
  - jj : Nedges
  - kk : Nedges
  - ktuple : (kx, ku) where kx is the unique patches and ku is the indices of the patches in the original tensor
  - kx : N
  - ku : N
  - B : 6*N, 6*N
  - E : 6*N, M
  - C : M
  - v : 6*N
  - u : M
  */

  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  auto opts = torch::TensorOptions()
    .dtype(torch::kDouble).device(torch::kCPU);

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  //shape changed to

  //convert the torch tensors to QUAD precision by using the helper functions
  /*
  - poses : 4096,7 (2 dimensions)
  - patches : Npatches, 3, 3, 3 (4 dimensions)
  - intrinsics : 4096, 4 (2 dimensions)
  - target : Nedges, 2 (2 dimensions)
  - weight : Nedges, 2 (2 dimensions)
  - ii : Nedges (1 dimension)
  - jj : Nedges (1 dimension)
  - kk : Nedges (1 dimension)
  - ktuple : (kx, ku) where kx is the unique patches and ku is the indices of the patches in the original tensor
  - kx : N
  - ku : N
  - B : 6*N, 6*N
  - E : 6*N, M
  - C : M
  - v : 6*N
  - u : M
  */


  //apply conversion to quad precision here
  auto poses_q = tensor_to_quad_dim2(poses);
  auto patches_q = tensor_to_quad_dim4(patches);
  auto intrinsics_q = tensor_to_quad_dim2(intrinsics);
  auto target_q = tensor_to_quad_dim2(target);
  auto weight_q = tensor_to_quad_dim2(weight);
  auto ii_q = tensor_to_long_dim1(ii);
  auto jj_q = tensor_to_long_dim1(jj);
  auto kk_q = tensor_to_long_dim1(kk);
  auto ku_q = tensor_to_long_dim1(ku);

  double lmbda_d = lmbda.item<double>();
  quad lmbda_q = static_cast<quad>(lmbda_d);

  //debugging : print shapes on screen
  //std::cout << "Shapes after conversion to quad precision:" << std::endl;
  //std::cout << "poses_q: " << poses_q.size() << std::endl;
  //std::cout << "patches_q: " << patches_q.size() << std::endl;
  //std::cout << "intrinsics_q: " << intrinsics_q.size() << std::endl;
  //std::cout << "target_q: " << target_q.size() << std::endl;
  //std::cout << "weight_q: " << weight_q.size() << std::endl;
  //std::cout << "ii_q: " << ii_q.size() << std::endl;
  //std::cout << "jj_q: " << jj_q.size() << std::endl;
  //std::cout << "kk_q: " << kk_q.size() << std::endl;
  //std::cout << "ku_q: " << ku_q.size() << std::endl;
//

  const int num = ii.size(0);
  //torch::Tensor B = torch::empty({6*N, 6*N}, opts);
  //torch::Tensor E = torch::empty({6*N, 1*M}, opts);
  //torch::Tensor C = torch::empty({M}, opts);
//
  //torch::Tensor v = torch::empty({6*N}, opts);
  //torch::Tensor u = torch::empty({1*M}, opts);

  //creating vectors of quad precision for B, E, C, v, u
  // std::vector<std::vector<quad>> B(6*N, std::vector<quad>(6*N, 0.0Q));
  // std::vector<std::vector<quad>> E(6*N, std::vector<quad>(M, 0.0Q));
  // std::vector<quad> C(M, 0.0Q);
  // std::vector<quad> v(6*N, 0.0Q);
  // std::vector<quad> u(M, 0.0Q);

  for (int itr=0; itr < iterations; itr++) {

    //fill vectors B, E, C, v, u with zeros
    // std::fill(B.begin(), B.end(), 0.0Q);
    // std::fill(E.begin(), E.end(), 0.0Q);
    // std::fill(C.begin(), C.end(), 0.0Q);
    // std::fill(v.begin(), v.end(), 0.0Q);
    // std::fill(u.begin(), u.end(), 0.0Q);
    std::vector<std::vector<quad>> B(6*N, std::vector<quad>(6*N, 0.0Q));
    std::vector<std::vector<quad>> E(6*N, std::vector<quad>(M, 0.0Q));
    std::vector<quad> C(M, 0.0Q);
    std::vector<quad> v(6*N, 0.0Q);
    std::vector<quad> u(M, 0.0Q);


    //v = v.view({6*N});
    //u = u.view({1*M});

    //std::cout << "Iteration: " << itr << std::endl;
    //std::cout << "Computing residuals and Hessian..." << std::endl;
    reprojection_residuals_and_hessian(
      poses_q,
      patches_q,
      intrinsics_q,
      target_q,
      weight_q,
      lmbda_q,
      ii_q,
      jj_q,
      kk_q,
      ku_q,
      B,
      E,
      C,
      v,
      u, t0);

    
    //convert back the std vectors of quad precision to torch tensors
    auto B_tensor = quad_to_tensor_dim2(B);
    auto E_tensor = quad_to_tensor_dim2(E);
    auto C_tensor = quad_to_tensor_dim1(C);
    auto v_tensor = quad_to_tensor_dim1(v);
    auto u_tensor = quad_to_tensor_dim1(u);


    //std::cout << "Computing update..." << std::endl;
    v_tensor = v_tensor.view({6*N, 1});
    u_tensor = u_tensor.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C_tensor + lmbda).view({1, M});
    torch::Tensor EQ = E_tensor * Q;

    //Q shape = {1, M}
    //E matrix of shape {6*N, M}
    // EQ = E * Q (element wise multiplication) ==> output shape {6*N, M} (each row of E is multiplied by the corresponding element in Q)


    //TRANSPOSE E and Q ==> Et and Qt
    torch::Tensor Et = torch::transpose(E_tensor, 0, 1);
    torch::Tensor Qt = torch::transpose(Q, 0, 1);

    torch::Tensor S = B_tensor - torch::matmul(EQ, Et);
    torch::Tensor y = v_tensor - torch::matmul(EQ,  u_tensor);
    //std::vector<std::vector<quad>> Et(M, std::vector<quad>(6*N, 0.0Q));
    //std::vector<quad> Qt(M, 0.0Q);
    
    //transpose E
    //auto Et = transpose_quad(E);
    
    //matrix multiplication 
    //auto EqEt = matmul_quad(EQ, Et);

    //auto S = B - EqEt; // Schur complement
    ///// S is a SIMMETRIC MATRIX of shape {6*N, 6*N} (then decomposed into a lower triangular matrix U)
    
    torch::Tensor I = torch::eye(6*N, opts);
    S += I * (1e-4 * S + 1.0);

    //std::cout << "Computing Schur complement..." << std::endl;
    //torch::Tensor S = B - torch::matmul(EQ, Et);

    torch::Tensor U = torch::linalg::cholesky(S);
    torch::Tensor dX = torch::cholesky_solve(y, U);
    torch::Tensor dZ = Qt * (u_tensor - torch::matmul(Et, dX));

    //EQ is a matrix of shape {6*N, M}
    //u is a vector of shape {M, 1}
    //MATRIX VECTOR MULTIPLICATION 
    //auto EQu = matvecmul_quad(EQ, u);

    //auto y = v - EQu; // residuals
    //torch::Tensor y = v - torch::matmul(EQ,  u);


    //add scaled identity matrix to the Schur complement
    //add_scaled_identity(S, 6*N);
    //torch::Tensor I = torch::eye(6*N, opts);
    //S += I * (1e-4 * S + 1.0);

    //obtain the lower triangular matrix L from the Cholesky decomposition of S
    
    /*
    THERE IS NO EXISTING LIBRARY FOR CHOLESKY DECOMPOSITION IN QUAD PRECISION

    SO FOR SIMPLICITY, WE USE THE TORCH LIBRARY FOR CHOLESKY DECOMPOSITION, SO:

    1 CONVERT THE STD::VECTORS (QUAD PRECISION) TO TORCH TENSORS (IN DOUBLE PRECISION)
    2 USE THE TORCH LIBRARY FOR CHOLESKY DECOMPOSITION
    3 SOLVE THE LINEAR SYSTEM USING THE TORCH CHOLESKY_SOLVE FUNCTION
    4 CONVERT THE TORCH TENSORS BACK TO STD::VECTORS (QUAD PRECISION)
    5 UPDATE POSES AND PATCHES USING THE UPDATED QUAD PRECISION VECTORS
    
    */
    

    //convert S to torch tensor
    //torch::Tensor S_tensor = torch::empty({6*N, 6*N}, torch::kFloat64);
    //for (int i = 0; i < 6*N; ++i) {
    //    for (int j = 0; j < 6*N; ++j) {
    //        S_tensor[i][j] = static_cast<double>(S[i][j]);
    //    }
    //}
    ////convert y to torch tensor
    //torch::Tensor y_tensor = torch::empty({6*N, 1}, torch::kFloat64);
    //for (int i = 0; i < 6*N; ++i) {
    //    y_tensor[i][0] = static_cast<double>(y[i]);
    //}


    //torch::Tensor L_tensor = torch::linalg::cholesky(S_tensor);

    ////solve the linear system L * dX = y
    ////first argument is the right-hand side vector y, second argument is the lower triangular matrix L
    //torch::Tensor dX = torch::cholesky_solve(y_tensor, L_tensor);
    //torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));

    //std::cout << "Updating poses and patches..." << std::endl;
    dX = dX.view({N, 6});
    dZ = dZ.view({M});

    //std::cout << "Updating poses..." << std::endl;
    pose_retr(t0, t1,
        poses,
        dX);

    //std::cout << "Updating patches..." << std::endl;
    patch_retr(
        kx,
        patches,
        dZ);
  }
  
  return {};
}

