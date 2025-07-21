#include <torch/extension.h>
#include <quadmath.h>
#include <vector>
#include <stdexcept>

using quad = __float128;


// ---------------------- Dimension 1 ----------------------
std::vector<quad> tensor_to_quad_dim1(const torch::Tensor& t) {
    std::vector<quad> out(t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        out[i] = static_cast<quad>(t[i].item<double>());
    return out;
}

// ---------------------- Dimension 2 ----------------------
std::vector<std::vector<quad>> tensor_to_quad_dim2(const torch::Tensor& t) {
    std::vector<std::vector<quad>> out(t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        out[i] = tensor_to_quad_dim1(t[i]);
    return out;
}

// ---------------------- Dimension 3 ----------------------
std::vector<std::vector<std::vector<quad>>> tensor_to_quad_dim3(const torch::Tensor& t) {
    std::vector<std::vector<std::vector<quad>>> out(t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        out[i] = tensor_to_quad_dim2(t[i]);
    return out;
}

// ---------------------- Dimension 4 ----------------------
std::vector<std::vector<std::vector<std::vector<quad>>>> tensor_to_quad_dim4(const torch::Tensor& t) {
    std::vector<std::vector<std::vector<std::vector<quad>>>> out(t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        out[i] = tensor_to_quad_dim3(t[i]);
    return out;
}


//REVERSE : STD::VECTOR ==> TORCH TENSORS
// ---------------------- Dimension 1 ----------------------
torch::Tensor quad_to_tensor_dim1(const std::vector<quad>& t) {
    torch::Tensor out = torch::empty({(int64_t)t.size()}, torch::kDouble);
    double* out_ptr = out.data_ptr<double>();
    for (size_t i = 0; i < t.size(); ++i)
        out_ptr[i] = static_cast<double>(t[i]);
    return out;
}

// ---------------------- Dimension 2 ----------------------
torch::Tensor quad_to_tensor_dim2(const std::vector<std::vector<quad>>& t) {
    int64_t dim0 = t.size();
    int64_t dim1 = t[0].size();  // assumes non-empty and rectangular
    torch::Tensor out = torch::empty({dim0, dim1}, torch::kDouble);
    auto out_a = out.accessor<double, 2>();
    for (int64_t i = 0; i < dim0; ++i)
        for (int64_t j = 0; j < dim1; ++j)
            out_a[i][j] = static_cast<double>(t[i][j]);
    return out;
}

// ---------------------- Dimension 3 ----------------------
torch::Tensor quad_to_tensor_dim3(const std::vector<std::vector<std::vector<quad>>>& t) {
    int64_t dim0 = t.size();
    int64_t dim1 = t[0].size();
    int64_t dim2 = t[0][0].size();
    torch::Tensor out = torch::empty({dim0, dim1, dim2}, torch::kDouble);
    auto out_a = out.accessor<double, 3>();
    for (int64_t i = 0; i < dim0; ++i)
        for (int64_t j = 0; j < dim1; ++j)
            for (int64_t k = 0; k < dim2; ++k)
                out_a[i][j][k] = static_cast<double>(t[i][j][k]);
    return out;
}

// ---------------------- Dimension 4 ----------------------
torch::Tensor quad_to_tensor_dim4(const std::vector<std::vector<std::vector<std::vector<quad>>>>& t) {
    int64_t dim0 = t.size();
    int64_t dim1 = t[0].size();
    int64_t dim2 = t[0][0].size();
    int64_t dim3 = t[0][0][0].size();
    torch::Tensor out = torch::empty({dim0, dim1, dim2, dim3}, torch::kDouble);
    auto out_a = out.accessor<double, 4>();
    for (int64_t i = 0; i < dim0; ++i)
        for (int64_t j = 0; j < dim1; ++j)
            for (int64_t k = 0; k < dim2; ++k)
                for (int64_t l = 0; l < dim3; ++l)
                    out_a[i][j][k][l] = static_cast<double>(t[i][j][k][l]);
    return out;
}

// ---------------------- Dimension 1  torch tensor ==> std::long ----------------------
std::vector<long> tensor_to_long_dim1(const torch::Tensor& t) {
    std::vector<long> out(t.size(0));
    for (int64_t i = 0; i < t.size(0); ++i)
        out[i] = static_cast<long>(t[i].item<long>());
    return out;
}


std::vector<std::vector<quad>> matmul_quad(
    const std::vector<std::vector<quad>>& A,
    const std::vector<std::vector<quad>>& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    
    int rows = A.size();
    int cols = B[0].size();
    int inner_dim = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

//matrix vector multiplication
std::vector<quad> matvecmul_quad(
    const std::vector<std::vector<quad>>& A,
    const std::vector<quad>& B) {
    if (A.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
    }
    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<quad> result(rows, 0.0Q);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += A[i][j] * B[j];
        }
    }
    
    return result;
}

//element wise multiplication of a matrix and a vector
std::vector<std::vector<quad>> elementwise_mult_quad(
    const std::vector<std::vector<quad>>& A,
    const std::vector<quad>& B) {
    if (A.empty() || B.size() != A[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
    }
    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * B[j];
        }
    }
    
    return result;
}

//element wise multiplication of two matrices
std::vector<std::vector<quad>> elementwise_mult_quad(
    const std::vector<std::vector<quad>>& A,
    const std::vector<std::vector<quad>>& B) {
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
    }
    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
    
    return result;
}

//transpose a matrix
std::vector<std::vector<quad>> transpose_quad(
    const std::vector<std::vector<quad>>& A) {
    if (A.empty()) {
        throw std::invalid_argument("Matrix is empty.");
    }
    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(cols, std::vector<quad>(rows, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    
    return result;
}

//matrix subtraction
std::vector<std::vector<quad>> operator-(const std::vector<std::vector<quad>>& A, const std::vector<std::vector<quad>>& B) {
    if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }
    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    
    return result;
}


//matrix * constant multiplication
std::vector<std::vector<quad>> operator*(const std::vector<std::vector<quad>>& A, const quad& B) {
    //perform matrix multiplication with a constant
    if (A.empty()) {
        throw std::invalid_argument("Matrix is empty.");
    }

    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * B;
        }
    }
    
    return result;
}

//MATRIX + CONSTANT ADDITION
std::vector<std::vector<quad>> operator+(const std::vector<std::vector<quad>>& A, const quad& B) {
    //perform matrix summation with a constant
    if (A.empty()) {
        throw std::invalid_argument("Matrix is empty.");
    }

    
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<quad>> result(rows, std::vector<quad>(cols, 0.0Q));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B;
        }
    }
    
    return result;
}

void add_scaled_identity(std::vector<std::vector<quad>>& S, int size) {
    for (int i = 0; i < size; ++i) {
        quad scale = 1e-4Q * S[i][i] + 1.0Q;
        S[i][i] += scale;
    }
}