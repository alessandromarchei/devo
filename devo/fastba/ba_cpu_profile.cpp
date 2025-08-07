#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


uint64_t total_operations = 0;

size_t op_per_edge = 0;


#define add_ops_edge_100(x, n) \
  do { if ((n) == 100) op_per_edge += (x); } while (0)


  
int actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);
  total_operations += 9; // 6 multiplications and 3 additions

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
  total_operations += 18; // 9 multiplications and 9 additions

  int ops = 27; // total operations for actSO3
  return ops;
 
}

int actSE3(const float *t, const float *q, const float *X, float *Y) {
  int ops1 = actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
  total_operations += 6; // 3 multiplications and 3 additions

  int ops2 = 6; // total operations for actSE3
  return ops1 + ops2 + 6; // 6 additions for Y[0], Y[1], Y[2]
  
}

int adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  int ops1 = actSO3(qinv, &X[0], &Y[0]);
  int ops2 = actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];
  total_operations += 9; // 6 multiplications and 3 subtractions

  int ops3 = actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
  total_operations += 3; // 3 additions

  return ops1 + ops2 + ops3 + 12; // 9 additions and 3 multiplications

}

int relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],
  total_operations += 32; // 12 multiplications, 12 additions, 8 subtractions

  int ops1 = actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
  total_operations += 3; // 3 subtractions and 3 additions

  return ops1 + 35; // 32 from qij and tij calculations, 3 from tij subtractions
}

  
int expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;
  total_operations += 4; // 3 multiplications and 1 addition

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
    total_operations += 12; // 6 multiplications and 6 additions
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
    total_operations += 4; // 2 multiplications and 2 divisions
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;
  total_operations += 3; // 3 multiplications

  return 50; // 12 multiplications, 6 additions, 2 divisions, 2 square roots

}

int  crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  total_operations += 9; // 6 multiplications and 3 subtractions
  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];

  return 9; // 6 multiplications and 3 subtractions
}

int expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  int ops1 = expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);
  total_operations += 5; // 3 multiplications and 2 additions

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    int ops2 = crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];
    total_operations += 6; // 3 multiplications and 3 additions

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    int ops3 = crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
    total_operations += 6; // 3 multiplications and 3 additions

    return ops1 + ops2 + ops3 + 50; // 20 from expSO3, cross product and multiplications/additions
  }

  return ops1 + 50; // 20 from expSO3 and multiplications/additions
}

int retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  int ops1 = expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];
  total_operations += 28; // 12 multiplications, 12 additions, 4 subtractions

  int ops2 = actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
  total_operations += 3; // 3 additions

  return ops1 + ops2 + 31; // 28 from q1 calculations, 3 from t1 additions
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
  auto poses_accessor = poses.accessor<float,2>();
  auto update_accessor = update.accessor<float,2>();

  //std::cout << "Updating poses..." << std::endl;
  for (int i=0; i < t1 - t0; i++) {
    const float t = t0 + i;
    float t0[3] = { poses_accessor[t][0], poses_accessor[t][1], poses_accessor[t][2] };
    float q0[4] = { poses_accessor[t][3], poses_accessor[t][4], poses_accessor[t][5], poses_accessor[t][6] };
    float t1[3] = {0, 0, 0};
    float q1[4] = {0, 0, 0, 1};

    //std::cout << "Pose " << t << ": " << t0[0] << " " << t0[1] << " " << t0[2] << " " << q0[0] << " " << q0[1] << " " << q0[2] << " " << q0[3] << std::endl;

    float xi[6] = {
      update_accessor[i][0],
      update_accessor[i][1],
      update_accessor[i][2],
      update_accessor[i][3],
      update_accessor[i][4],
      update_accessor[i][5],
    };

    //std::cout << xi[0] << " " << xi[1] << " " << xi[2] << " " << xi[3] << " " << xi[4] << " " << xi[5] << std::endl;

    //std::cout << "Retrieving pose..." << std::endl;
    int ops1 = retrSE3(xi, t0, q0, t1, q1);

    //std::cout << "Updated pose: " << t1[0] << " " << t1[1] << " " << t1[2] << " " << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << std::endl;
    poses_accessor[t][0] = t1[0];
    poses_accessor[t][1] = t1[1];
    poses_accessor[t][2] = t1[2];
    poses_accessor[t][3] = q1[0];
    poses_accessor[t][4] = q1[1];
    poses_accessor[t][5] = q1[2];
    poses_accessor[t][6] = q1[3];

    add_ops_edge_100(ops1,i);
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
    auto patches_acc = patches.accessor<float, 4>();
    auto update_acc = update.accessor<float, 1>();

    int p = patches.size(2);
    for (int n = 0; n < index.size(0); n++) {
        int64_t ix = index_acc[n];

        //std::cout << "Patch " << n << ": " << ix << std::endl;
        float d = patches_acc[ix][2][0][0];
        d = d + update_acc[n];
        d = (d > 20.0f) ? 1.0f : d;
        d = std::max(d, 1e-4f);

        //std::cout << "Updated depth: " << d << std::endl;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                patches_acc[ix][2][i][j] = d;
            }
        }

        add_ops_edge_100(5, n); // 3 multiplications, 1 addition, 1 max operation
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
  auto poses_accessor = poses.accessor<float,2>();
  auto patches_accessor = patches.accessor<float,4>();
  auto intrinsics_accessor = intrinsics.accessor<float,2>();
  auto target_accessor = target.accessor<float,2>();
  auto weight_accessor = weight.accessor<float,2>();
  auto ii_accessor = ii.accessor<int64_t,1>();
  auto jj_accessor = jj.accessor<int64_t,1>();
  auto kk_accessor = kk.accessor<int64_t,1>();
  auto ku_accessor = ku.accessor<int64_t,1>();

  auto B_accessor = B.accessor<float,2>();
  auto E_accessor = E.accessor<float,2>();
  auto C_accessor = C.accessor<float,1>();
  auto v_accessor = v.accessor<float,1>();
  auto u_accessor = u.accessor<float,1>();


  //std::cout << "Setting tensors to zero..." << std::endl;
  float fx, fy, cx, cy;
  fx = intrinsics_accessor[0][0];
  fy = intrinsics_accessor[0][1];
  cx = intrinsics_accessor[0][2];
  cy = intrinsics_accessor[0][3];


  for (int n=0; n < ii.size(0); n++) {
    //std::cout << "Processing point " << n << ": " << ii_accessor[n] << " " << jj_accessor[n] << " " << kk_accessor[n] << std::endl;
    int k = ku_accessor[n];
    int ix = ii_accessor[n];
    int jx = jj_accessor[n];
    int kx = kk_accessor[n];
    //std::cout << "N : " << n << " ix: " << ix << " jx: " << jx << " kx: " << kx << std::endl;

    //std::cout << "Point: " << n << " " << ix << " " << jx << " " << kx << std::endl;
    float ti[3] = { poses_accessor[ix][0], poses_accessor[ix][1], poses_accessor[ix][2] };
    float tj[3] = { poses_accessor[jx][0], poses_accessor[jx][1], poses_accessor[jx][2] };
    float qi[4] = { poses_accessor[ix][3], poses_accessor[ix][4], poses_accessor[ix][5], poses_accessor[ix][6] };
    float qj[4] = { poses_accessor[jx][3], poses_accessor[jx][4], poses_accessor[jx][5], poses_accessor[jx][6] };

    float Xi[4], Xj[4];
    Xi[0] = (patches_accessor[kx][0][1][1] - cx) / fx;
    Xi[1] = (patches_accessor[kx][1][1][1] - cy) / fy;
    Xi[2] = 1.0;
    Xi[3] = patches_accessor[kx][2][1][1];

    add_ops_edge_100(4, n);
    total_operations += 4; // 4 divisions
    
    float tij[3], qij[4];
    add_ops_edge_100(relSE3(ti, qi, tj, qj, tij, qij), n);
    add_ops_edge_100(actSE3(tij, qij, Xi, Xj), n);

    const float X = Xj[0];
    const float Y = Xj[1];
    const float Z = Xj[2];
    const float W = Xj[3];

    const float d = (Z >= 0.2) ? 1.0 / Z : 0.0; 
    const float d2 = d * d;

    const float x1 = fx * (X / Z) + cx;
    const float y1 = fy * (Y / Z) + cy;

    const float rx = target_accessor[n][0] - x1;
    const float ry = target_accessor[n][1] - y1;

    const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
      (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

    const float mask = in_bounds ? 1.0 : 0.0;

    ix = ix - t0;
    jx = jx - t0;

    add_ops_edge_100(50, n);
    total_operations += 50; // 12 multiplications, 12 additions, 8 subtractions + sqrt counted as 20 ops

    //std::cout << "Computing residuals for point " << n << ": " << x1 << " " << y1 << " " << Z << " " << W << std::endl;
    {
      const float r = target_accessor[n][0] - x1;
      const float w = mask * weight_accessor[n][0];

      add_ops_edge_100(2, n);
      total_operations += 2; // 2 subtractions

      float Jz = fx * (tij[0] * d - tij[2] * (X * d2));
      float Ji[6], Jj[6] = {fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      add_ops_edge_100(20, n);  //probbaly more
      total_operations += 20; // 6 multiplications, 12 additions, 2 subtractions


      add_ops_edge_100(adjSE3(tij, qij, Jj, Ji), n);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B_accessor[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
            add_ops_edge_100(3, n);
            total_operations += 3; // 6 multiplications and 3 additions
          if (jx >= 0)
            B_accessor[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
            add_ops_edge_100(3, n);
            total_operations += 3; // 6 multiplications and 3 additions
          if (ix >= 0 && jx >= 0) {
            B_accessor[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B_accessor[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
            add_ops_edge_100(6, n);
            total_operations += 6; // 6 multiplications and 6 additions
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E_accessor[6*ix+i][k] -= w * Jz * Ji[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
        if (jx >= 0)
          E_accessor[6*jx+i][k] +=  w * Jz * Jj[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v_accessor[6*ix+i] -= w * r * Ji[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
        if (jx >= 0)
          v_accessor[6*jx+i] += w * r * Jj[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
      }
      C_accessor[k] += w * Jz * Jz;
      u_accessor[k] += w * r * Jz;
      add_ops_edge_100(6, n);
      total_operations += 6; // 3 multiplications and 3 additions
    }


    {
      const float r = target_accessor[n][1] - y1;
      const float w = mask * weight_accessor[n][1];

      add_ops_edge_100(2, n);
      total_operations += 2; // 2 subtractions
      
      float Jz = fy * (tij[1] * d - tij[2] * (Y * d2));
      float Ji[6], Jj[6] = {0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};

      add_ops_edge_100(20, n); //probbaly more
      total_operations += 20; // 6 multiplications, 12 additions, 2 subtractions
      
      add_ops_edge_100(adjSE3(tij, qij, Jj, Ji), n);

      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            B_accessor[6*ix+i][6*ix+j] += w * Ji[i] * Ji[j];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
          if (jx >= 0)
            B_accessor[6*jx+i][6*jx+j] += w * Jj[i] * Jj[j];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
          if (ix >= 0 && jx >= 0) {
            B_accessor[6*ix+i][6*jx+j] -= w * Ji[i] * Jj[j];
            B_accessor[6*jx+i][6*ix+j] -= w * Jj[i] * Ji[j];
            add_ops_edge_100(6, n);
            total_operations += 6; // 6 multiplications and 6 additions
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          E_accessor[6*ix+i][k] -= w * Jz * Ji[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
        if (jx >= 0)
          E_accessor[6*jx+i][k] += w * Jz * Jj[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          v_accessor[6*ix+i] -= w * r * Ji[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
        if (jx >= 0)
          v_accessor[6*jx+i] += w * r * Jj[i];
          add_ops_edge_100(3, n);
          total_operations += 3; // 6 multiplications and 3 additions
      }
      C_accessor[k] += w * Jz * Jz;
      u_accessor[k] += w * r * Jz;
      add_ops_edge_100(6, n);
      total_operations += 6; // 3 multiplications and 3 additions
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
  
  total_operations = 0;
  op_per_edge = 0;

  ////std::cout << "CUDA BA: " << t0 << " " << t1 << " " << iterations << std::endl;
  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);
  torch::Tensor ku = std::get<1>(ktuple);

  const int N = t1 - t0;    // number of poses
  const int M = kx.size(0); // number of patches
  const int P = patches.size(3); // patch size

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCPU);

  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  const int num = ii.size(0);
  std::cout << "Number of edges: " << num << std::endl;
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
    total_operations += 2* M; // 2 multiplications for each element in C and lmbda

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
      total_operations += 2 * 6 * N * M; // 2 multiplications for each element in E and Q
      torch::Tensor Et = torch::transpose(E, 0, 1);
      torch::Tensor Qt = torch::transpose(Q, 0, 1);

      //std::cout << "Computing Schur complement..." << std::endl;
      torch::Tensor S = B - torch::matmul(EQ, Et);
      total_operations += 2 * 6 * N * 6 * N; // 2 multiplications for each element in B and EQ
      torch::Tensor y = v - torch::matmul(EQ,  u);
      total_operations += 2 * 6 * N * M; // 2 multiplications for each element in v and EQ

      torch::Tensor I = torch::eye(6*N, opts);
      S += I * (1e-4 * S + 1.0);
      total_operations += 2 * 6 * N; // 2 multiplications for each element in S and I

      //std::cout << "Cholesky decomposition..." << std::endl;
      torch::Tensor U = torch::linalg::cholesky(S);
      torch::Tensor dX = torch::cholesky_solve(y, U);
      torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));
      total_operations += 2 * 6 * N * M; // 2 multiplications for each element in u and Et

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
  

  std::cout << "Total operations: " << total_operations << std::endl;
  std::cout << "Operations per edge: " << op_per_edge << std::endl;
  std::cout << "-------------------------" << std::endl << std::endl;
  return {};
}

