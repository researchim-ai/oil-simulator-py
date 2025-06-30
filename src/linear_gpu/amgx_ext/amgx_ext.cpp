#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#ifdef HAS_AMGX
#include <amgx_c.h>
#endif

// Helper: reconstruct dense from CSR (when AMGX not available)
static torch::Tensor csr_to_dense(const torch::Tensor &indptr, const torch::Tensor &indices, const torch::Tensor &data, int64_t n) {
    auto dense = torch::zeros({n, n}, data.options());
    auto indptr_cpu = indptr.cpu();
    auto indices_cpu = indices.cpu();
    auto data_cpu = data.cpu();
    for (int64_t i = 0; i < n; ++i) {
        int64_t start = indptr_cpu[i].item<int64_t>();
        int64_t end = indptr_cpu[i+1].item<int64_t>();
        for (int64_t p = start; p < end; ++p) {
            int64_t j = indices_cpu[p].item<int64_t>();
            double val = data_cpu[p].item<double>();
            dense[i][j] = val;
        }
    }
    return dense.to(data.device());
}


torch::Tensor solve(torch::Tensor indptr,
                    torch::Tensor indices,
                    torch::Tensor values,
                    torch::Tensor b,
                    double tol = 1e-8,
                    int max_iter = 1000) {
    TORCH_CHECK(indptr.dim() == 1, "indptr must be 1-D");
    int64_t n = b.size(0);
#ifndef HAS_AMGX
    // Fallback: dense solve on GPU/CPU via Torch
    auto A_dense = csr_to_dense(indptr, indices, values, n);
    return torch::linalg::solve(A_dense, b);
#else
    // TODO: полноценная интеграция с AMGX C-API
    // Для упрощения пока тот же fallback
    auto A_dense = csr_to_dense(indptr, indices, values, n);
    return torch::linalg::solve(A_dense, b);
#endif
}

PYBIND11_MODULE(amgx_ext, m) {
    m.doc() = "Torch C++ extension: solve sparse linear system with NVIDIA AMGX (placeholder)";
    m.def("solve", &solve, pybind11::arg("indptr"), pybind11::arg("indices"), pybind11::arg("values"), pybind11::arg("b"),
          pybind11::arg("tol") = 1e-8, pybind11::arg("max_iter") = 1000);
} 