#include <torch/extension.h>
#include <tuple>
#include <limits>

typedef std::tuple<torch::Tensor, torch::Tensor> PairTensor;

PairTensor stacked_batch_chamfer_distance_cpu(const torch::Tensor& node_features, const torch::Tensor& indptr)
{
    // indptr[i] points to the beginning index for graph i
    // indptr[i+1] points to the ending index for graph i
    // used for slicing
    auto indptr_acc = indptr.accessor<long, 1>();
    int num_graphs = indptr.size(0) - 1;
    int num_nodes = node_features.size(0);

    float inf = std::numeric_limits<float>::infinity();
    torch::Tensor nodewise_min_dist = torch::full({ num_nodes, num_graphs }, inf, node_features.options());
    torch::Tensor nodewise_min_dist_idx = torch::arange(
        0, num_nodes, indptr.options())
        .unsqueeze(1)
        .repeat({ 1, num_graphs });

    // precompute the squared euclidean distance between all nodes within stacked node_features tensor
    torch::Tensor pairwise_dist = torch::cdist(node_features, node_features).pow(2);

    for (int graph_idx_x = 0; graph_idx_x < num_graphs; graph_idx_x++)
    {
        int start_x = indptr_acc[graph_idx_x];
        int end_x = indptr_acc[graph_idx_x + 1];
        auto x_slice = torch::indexing::Slice(start_x, end_x);

        for (int graph_idx_y = graph_idx_x + 1; graph_idx_y < num_graphs; graph_idx_y++)
        {
            int start_y = indptr_acc[graph_idx_y];
            int end_y = indptr_acc[graph_idx_y + 1];
            auto y_slice = torch::indexing::Slice(start_y, end_y);

            torch::Tensor block = pairwise_dist.index({ x_slice, y_slice });

            PairTensor x_min_data = block.min(1);
            torch::Tensor x_mins = std::get<0>(x_min_data);
            torch::Tensor x_idx = std::get<1>(x_min_data);

            PairTensor y_min_data = block.min(0);
            torch::Tensor y_mins = std::get<0>(y_min_data);
            torch::Tensor y_idx = std::get<1>(y_min_data);

            nodewise_min_dist.index_put_({ x_slice, graph_idx_y }, x_mins);
            nodewise_min_dist.index_put_({ y_slice, graph_idx_x }, y_mins);

            nodewise_min_dist_idx.index_put_({ x_slice, graph_idx_y }, x_idx + start_y);
            nodewise_min_dist_idx.index_put_({ y_slice, graph_idx_x }, y_idx + start_x);
        }
    }
    return { nodewise_min_dist, nodewise_min_dist_idx };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stacked_batch_chamfer_distance_cpu", &stacked_batch_chamfer_distance_cpu, "Stacked Batch Chamfer distance (CPU)");
}
