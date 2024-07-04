#include <torch/extension.h>

torch::Tensor forward_kernel(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_compute", torch::wrap_pybind_function(forward_kernel), "forward_compute");
}