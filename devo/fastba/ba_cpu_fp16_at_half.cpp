#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

using at::Half;


void actSO3(const Half *q, const Half *X, Half *Y) {
  Half uv[3];
  uv[0] = Half(2.0f) * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = Half(2.0f) * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = Half(2.0f) * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

void actSE3(const Half *t, const Half *q, const Half *X, Half *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

void adjSE3(const Half *t, const Half *q, const Half *X, Half *Y) {
  Half qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  Half u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

void relSE3(const Half *ti, const Half *qi, const Half *tj, const Half *qj, Half *tij, Half *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
void expSO3(const Half *phi, Half* q) {
  // SO3 exponential map
  Half theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  Half theta_p4 = theta_sq * theta_sq;

  Half theta = (Half)sqrtf((float)theta_sq);
  Half imag, real;

  if (theta_sq < Half(1e-8f)) {
    imag = Half(0.5f) - (Half(1.0f)/Half(48.0f))*theta_sq + (Half(1.0f)/Half(3840.0f))*theta_p4;
    real = Half(1.0f) - (Half(1.0f)/Half(8.0f))*theta_sq + (Half(1.0f)/Half(384.0f))*theta_p4;
  } else {
    imag = sinf(Half(0.5f) * theta) / theta;
    real = cosf(Half(0.5f) * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

void crossInplace(const Half* a, Half *b) {
  Half x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

void expSE3(const Half *xi, Half* t, Half* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  Half tau[3] = {xi[0], xi[1], xi[2]};
  Half phi[3] = {xi[3], xi[4], xi[5]};

  Half theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  Half theta = (Half)sqrtf((float)theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > Half(1e-4f)) {
    Half a = (Half(1.0f) - (Half)(cosf((float)theta))) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    Half b = (theta - (Half)sinf((float)theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}

void retrSE3(const Half *xi, const Half* t, const Half* q, Half* t1, Half* q1) {
  // retraction on SE3 manifold

  Half dt[3] = {Half(0.0f), Half(0.0f), Half(0.0f)};
  Half dq[4] = {Half(0.0f), Half(0.0f), Half(0.0f), Half(1.0f)};

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
  auto poses_accessor = poses.accessor<Half,2>();
  auto update_accessor = update.accessor<Half,2>();

  //std::cout << "Updating poses..." << std::endl;
  for (int i=0; i < t1 - t0; i++) {
    const int t = t0 + i;
    Half t0[3] = { poses_accessor[t][0], poses_accessor[t][1], poses_accessor[t][2] };
    Half q0[4] = { poses_accessor[t][3], poses_accessor[t][4], poses_accessor[t][5], poses_accessor[t][6] };
    Half t1[3] = {Half(0), Half(0), Half(0)};
    Half q1[4] = {Half(0), Half(0), Half(0), Half(1)};

    //std::cout << "Pose " << t << ": " << t0[0] << " " << t0[1] << " " << t0[2] << " " << q0[0] << " " << q0[1] << " " << q0[2] << " " << q0[3] << std::endl;

    Half xi[6] = {
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
    auto patches_acc = patches.accessor<Half, 4>();
    auto update_acc = update.accessor<Half, 1>();

    int p = patches.size(2);
    for (int n = 0; n < index.size(0); n++) {
        int64_t ix = index_acc[n];

        //std::cout << "Patch " << n << ": " << ix << std::endl;
        Half d = patches_acc[ix][2][0][0];
        d = d + update_acc[n];
        d = (d > Half(20.0f)) ? Half(1.0f) : d;
        d = (Half)(std::max(d, (Half)1e-4f));

        //std::cout << "Updated depth: " << d << std::endl;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                patches_acc[ix][2][i][j] = d;
            }
        }
    }
}


void reprojection_residuals_and_hessian(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    torch::Tensor ku,
    torch::Tensor B,
    torch::Tensor E,
    torch::Tensor C,
    torch::Tensor v,
    torch::Tensor u, const int t0)
{

  // move tensors to cpu
  //std::cout << "Computing reprojection residuals and hessian..." << std::endl;
  poses = poses.to(torch::kCPU);
  patches = patches.to(torch::kCPU);
  intrinsics = intrinsics.to(torch::kCPU);
  target = target.to(torch::kCPU);
  weight = weight.to(torch::kCPU);
  lmbda = lmbda.to(torch::kCPU);
  ii = ii.to(torch::kCPU);
  jj = jj.to(torch::kCPU);
  kk = kk.to(torch::kCPU);
  ku = ku.to(torch::kCPU);


  //std::cout << "Creating accessors..." << std::endl;
  auto poses_accessor = poses.accessor<Half,2>();
  auto patches_accessor = patches.accessor<Half,4>();
  auto intrinsics_accessor = intrinsics.accessor<Half,2>();
  auto target_accessor = target.accessor<Half,2>();
  auto weight_accessor = weight.accessor<Half,2>();
  auto ii_accessor = ii.accessor<int64_t,1>();
  auto jj_accessor = jj.accessor<int64_t,1>();
  auto kk_accessor = kk.accessor<int64_t,1>();
  auto ku_accessor = ku.accessor<int64_t,1>();

  auto B_accessor = B.accessor<Half,2>();
  auto E_accessor = E.accessor<Half,2>();
  auto C_accessor = C.accessor<Half,1>();
  auto v_accessor = v.accessor<Half,1>();
  auto u_accessor = u.accessor<Half,1>();


  //std::cout << "Setting tensors to zero..." << std::endl;
  Half fx, fy, cx, cy;
  fx = (Half)intrinsics_accessor[0][0];
  fy = (Half)intrinsics_accessor[0][1];
  cx = (Half)intrinsics_accessor[0][2];
  cy = (Half)intrinsics_accessor[0][3];


  for (int n=0; n < ii.size(0); n++) {
    //std::cout << "Processing point " << n << ": " << ii_accessor[n] << " " << jj_accessor[n] << " " << kk_accessor[n] << std::endl;
    int k = ku_accessor[n];
    int ix = ii_accessor[n];
    int jx = jj_accessor[n];
    int kx = kk_accessor[n];

    //std::cout << "Point: " << n << " " << ix << " " << jx << " " << kx << std::endl;
    Half ti[3] = { poses_accessor[ix][0], poses_accessor[ix][1], poses_accessor[ix][2] };
    Half tj[3] = { poses_accessor[jx][0], poses_accessor[jx][1], poses_accessor[jx][2] };
    Half qi[4] = { poses_accessor[ix][3], poses_accessor[ix][4], poses_accessor[ix][5], poses_accessor[ix][6] };
    Half qj[4] = { poses_accessor[jx][3], poses_accessor[jx][4], poses_accessor[jx][5], poses_accessor[jx][6] };

    Half Xi[4], Xj[4];
    Xi[0] = (patches_accessor[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches_accessor[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0;
    Xi[3] = patches_accessor[kx][2][1][1];
    
    Half tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);
    actSE3(tij, qij, Xi, Xj);

    const Half X = Xj[0];
    const Half Y = Xj[1];
    const Half Z = Xj[2];
    const Half W = Xj[3];

    const Half d = (Z >= Half(0.2f)) ? Half(1.0f) / Z : Half(0.0f);
    const Half d2 = d * d;

    const Half x1 = fx * (X / Z) + cx;
    const Half y1 = fy * (Y / Z) + cy;

    const Half rx = target_accessor[n][0] - x1;
    const Half ry = target_accessor[n][1] - y1;

    const bool in_bounds = (Half)sqrtf((float)(rx*rx + ry*ry)) < Half(128.0f) && (Z > Half(0.2f)) &&
      (x1 > Half(-64.0f)) && (y1 > Half(-64.0f)) && (x1 < Half(2.0f)*cx + Half(64.0f)) && (y1 < Half(2.0f)*cy + Half(64.0f));

    const Half mask = in_bounds ? Half(1.0f) : Half(0.0f);

    ix = ix - t0;
    jx = jx - t0;

    //std::cout << "Computing residuals for point " << n << ": " << x1 << " " << y1 << " " << Z << " " << W << std::endl;
    {
      const Half r = target_accessor[n][0] - x1;
      const Half w = mask * weight_accessor[n][0];

      Half Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      Half Ji[6], Jj[6] = {fx*W*d, Half(0.0f), fx*-X*W*d2, fx*-X*Y*d2, fx*(Half(1.0f)+X*X*d2), fx*-Y*d};

      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B_accessor[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B_accessor[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B_accessor[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B_accessor[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E_accessor[6*ix+i][k] -= w * Jz * Ji[i];
        if (jx >= 0)
          E_accessor[6*jx+i][k] +=  w * Jz * Jj[i];
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v_accessor[6*ix+i] -= w * r * Ji[i];
        if (jx >= 0)
          v_accessor[6*jx+i] += w * r * Jj[i];
      }
      C_accessor[k] += w * Jz * Jz;
      u_accessor[k] += w * r * Jz;
    }


    {
      const Half r = target_accessor[n][1] - y1;
      const Half w = mask * weight_accessor[n][1];
      
      Half Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      Half Ji[6], Jj[6] = {Half(0.0f), fy*W*d, fy*-Y*W*d2, fy*(Half(-1.0f)-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
      
      adjSE3(tij, qij, Jj, Ji);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B_accessor[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          if (jx >= 0)
            B_accessor[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          if (ix >= 0 && jx >= 0) {
            B_accessor[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B_accessor[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E_accessor[6*ix+i][k] -= w * Jz * Ji[i];
        if (jx >= 0)
          E_accessor[6*jx+i][k] += w * Jz * Jj[i];
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v_accessor[6*ix+i] -= w * r * Ji[i];
        if (jx >= 0)
          v_accessor[6*jx+i] += w * r * Jj[i];
      }
      C_accessor[k] += w * Jz * Jz;
      u_accessor[k] += w * r * Jz;
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
  
  ////std::cout << "CUDA BA: " << t0 << " " << t1 << " " << iterations << std::endl;
  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  auto opts = torch::TensorOptions()
    .dtype(torch::kHalf).device(torch::kCPU);



  poses = poses.view({-1, 7});
  std::cout << "Pose type : " << poses.dtype() << std::endl;
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  const int num = ii.size(0);
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

    //std::cout << "Iteration: " << itr << std::endl;
    //std::cout << "Computing residuals and Hessian..." << std::endl;
    reprojection_residuals_and_hessian(
      poses,
      patches,
      intrinsics,
      target,
      weight,
      lmbda,
      ii,
      jj,
      kk,
      ku,
      B,
      E,
      C,
      v,
      u, t0);


    //std::cout << "Computing update..." << std::endl;
    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

    if (t1 - t0 == 0) {

      torch::Tensor Qt = torch::transpose(Q, 0, 1);
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      patch_retr(
        kx,
        patches,
        dZ);

    }


    else {


      //std::cout << "Solving linear system..." << std::endl;
      torch::Tensor EQ = E * Q;
      torch::Tensor Et = torch::transpose(E, 0, 1);
      torch::Tensor Qt = torch::transpose(Q, 0, 1);

      //std::cout << "Computing Schur complement..." << std::endl;
      torch::Tensor S = B - torch::matmul(EQ, Et);
      torch::Tensor y = v - torch::matmul(EQ,  u);

      torch::Tensor I = torch::eye(6*N, opts);
      S += I * (1e-4 * S + 1.0);

      //std::cout << "Cholesky decomposition..." << std::endl;
      torch::Tensor U = torch::linalg::cholesky(S);
      torch::Tensor dX = torch::cholesky_solve(y, U);
      torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));

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
  }
  
  return {};
}

