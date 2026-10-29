#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <cuda_runtime.h>

typedef struct {
    int *row_offsets;
    int *col_indices;
    int num_nodes;
    int num_edges;
} CSRGraph;


/*
 * Check CUDA API calls.
 */
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                            \
    } while (0)


/*
 * ------------------------------------------------------------
 * Graph generation
 * ------------------------------------------------------------
 *
 * Graph:
 *
 * Level 0:
 *              0
 *            /   \
 *
 * Level 1:
 *           1     2
 *          / \   / \
 *
 * Level 2:
 *         3   4 3   4
 *
 * Actual node IDs are unique:
 *
 * Level 0:       0
 * Level 1:       1  2
 * Level 2:       3  4
 * Level 3:       5  6
 * Level 4:       7  8
 * ...
 *
 * Every node in level L connects to every node
 * in level L+1.
 *
 * num_nodes = 1 + depth * branching_factor
 *
 * edges =
 *
 *     branching_factor
 *     +
 *     (depth - 1) * branching_factor^2
 *
 * ------------------------------------------------------------
 */

CSRGraph *create_graph(int depth, int branching_factor)
{
    CSRGraph *graph =
        (CSRGraph *) malloc(sizeof(CSRGraph));

    if (!graph) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    long long num_nodes =
        1LL +
        (long long) depth *
        branching_factor;

    long long num_edges;

    if (depth == 0) {
        num_edges = 0;
    } else {
        num_edges =
            (long long) branching_factor +
            (long long) (depth - 1) *
            branching_factor *
            branching_factor;
    }

    if (num_nodes > INT_MAX ||
        num_edges > INT_MAX) {

        fprintf(stderr,
                "Graph is too large for this int-based implementation.\n");

        free(graph);
        exit(EXIT_FAILURE);
    }

    graph->num_nodes = (int) num_nodes;
    graph->num_edges = (int) num_edges;

    graph->row_offsets =
        (int *) malloc(
            (graph->num_nodes + 1) * sizeof(int));

    graph->col_indices =
        (int *) malloc(
            graph->num_edges * sizeof(int));

    if (!graph->row_offsets ||
        (graph->num_edges > 0 &&
         !graph->col_indices)) {

        perror("malloc");

        free(graph->row_offsets);
        free(graph->col_indices);
        free(graph);

        exit(EXIT_FAILURE);
    }

    int edge_position = 0;

    for (int node = 0;
         node < graph->num_nodes;
         node++) {

        graph->row_offsets[node] =
            edge_position;

        /*
         * Root.
         */
        if (node == 0) {

            for (int j = 0;
                 j < branching_factor;
                 j++) {

                graph->col_indices[edge_position++] =
                    1 + j;
            }

        } else {

            /*
             * Determine BFS level.
             *
             * Level 1 starts at node 1.
             */
            int level =
                (node - 1) /
                branching_factor + 1;

            /*
             * Last level has no outgoing edges.
             */
            if (level < depth) {

                /*
                 * First node in next level.
                 */
                int next_level_start =
                    1 +
                    level *
                    branching_factor;

                /*
                 * Connect to every node
                 * in next level.
                 */
                for (int j = 0;
                     j < branching_factor;
                     j++) {

                    graph->col_indices[edge_position++] =
                        next_level_start + j;
                }
            }
        }
    }

    graph->row_offsets[graph->num_nodes] =
        edge_position;

    return graph;
}


void free_graph(CSRGraph *graph)
{
    if (!graph)
        return;

    free(graph->row_offsets);
    free(graph->col_indices);
    free(graph);
}


/*
 * ------------------------------------------------------------
 * CUDA BFS kernel
 * ------------------------------------------------------------
 *
 * One CUDA thread processes one node in the current frontier.
 *
 * Each thread:
 *
 *     1. Gets one node.
 *     2. Reads its CSR neighbor range.
 *     3. Checks its neighbors.
 *     4. Adds newly discovered nodes to next frontier.
 *
 * atomicCAS() is used for the visited array.
 *
 * Only the thread that changes visited[neighbor]
 * from 0 -> 1 is allowed to add that node.
 *
 * ------------------------------------------------------------
 */

__global__
void bfs_kernel(
    const int *row_offsets,
    const int *col_indices,
    int *visited,
    const int *current_frontier,
    int current_size,
    int *next_frontier,
    int *next_size)
{
    int tid =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (tid >= current_size)
        return;

    /*
     * One thread handles one frontier node.
     */
    int node =
        current_frontier[tid];

    int start =
        row_offsets[node];

    int end =
        row_offsets[node + 1];

    /*
     * Traverse this node's neighbors.
     */
    for (int i = start;
         i < end;
         i++) {

        int neighbor =
            col_indices[i];

        /*
         * Atomically mark the neighbor visited.
         *
         * Only one thread will receive
         * old_value == 0.
         */
        int old_value =
            atomicCAS(
                &visited[neighbor],
                0,
                1);

        if (old_value == 0) {

            /*
             * Atomically reserve a position
             * in the next frontier.
             */
            int position =
                atomicAdd(next_size, 1);

            next_frontier[position] =
                neighbor;
        }
    }
}


/*
 * ------------------------------------------------------------
 * GPU BFS
 * ------------------------------------------------------------
 */

void bfs_gpu(const CSRGraph *graph)
{
    int *d_row_offsets = NULL;
    int *d_col_indices = NULL;

    int *d_visited = NULL;

    int *d_current_frontier = NULL;
    int *d_next_frontier = NULL;

    int *d_next_size = NULL;

    /*
     * Copy CSR graph to GPU.
     */
    CUDA_CHECK(cudaMalloc(
        &d_row_offsets,
        (graph->num_nodes + 1) *
        sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &d_col_indices,
        graph->num_edges *
        sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_row_offsets,
        graph->row_offsets,
        (graph->num_nodes + 1) *
        sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_col_indices,
        graph->col_indices,
        graph->num_edges *
        sizeof(int),
        cudaMemcpyHostToDevice));

    /*
     * visited array.
     */
    CUDA_CHECK(cudaMalloc(
        &d_visited,
        graph->num_nodes *
        sizeof(int)));

    CUDA_CHECK(cudaMemset(
        d_visited,
        0,
        graph->num_nodes *
        sizeof(int)));

    /*
     * Frontier arrays.
     *
     * Maximum possible frontier size is
     * num_nodes.
     */
    CUDA_CHECK(cudaMalloc(
        &d_current_frontier,
        graph->num_nodes *
        sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &d_next_frontier,
        graph->num_nodes *
        sizeof(int)));

    /*
     * Counter for next frontier.
     */
    CUDA_CHECK(cudaMalloc(
        &d_next_size,
        sizeof(int)));

    /*
     * Root is the first frontier.
     */
    int root = 0;

    CUDA_CHECK(cudaMemcpy(
        d_current_frontier,
        &root,
        sizeof(int),
        cudaMemcpyHostToDevice));

    /*
     * Mark root visited.
     */
    int visited_root = 1;

    CUDA_CHECK(cudaMemcpy(
        d_visited,
        &visited_root,
        sizeof(int),
        cudaMemcpyHostToDevice));

    int current_size = 1;

    int levels = 0;
    int nodes_visited = 1;

    /*
     * Use 256 threads per block.
     */
    const int threads_per_block = 2;

    while (current_size > 0) {

        /*
         * Reset next frontier size.
         */
        int zero = 0;

        CUDA_CHECK(cudaMemcpy(
            d_next_size,
            &zero,
            sizeof(int),
            cudaMemcpyHostToDevice));

        int blocks =
            (current_size +
             threads_per_block - 1) /
            threads_per_block;

        /*
         * Process current BFS frontier.
         */
        bfs_kernel<<<blocks, threads_per_block>>>(
            d_row_offsets,
            d_col_indices,
            d_visited,
            d_current_frontier,
            current_size,
            d_next_frontier,
            d_next_size);

        CUDA_CHECK(cudaGetLastError());

        /*
         * Wait for the kernel to finish.
         */
        CUDA_CHECK(cudaDeviceSynchronize());

        /*
         * Get number of nodes discovered
         * for the next level.
         */
        int next_size = 0;

        CUDA_CHECK(cudaMemcpy(
            &next_size,
            d_next_size,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        nodes_visited += next_size;

        /*
         * Swap frontiers.
         */
        int *tmp =
            d_current_frontier;

        d_current_frontier =
            d_next_frontier;

        d_next_frontier =
            tmp;

        current_size =
            next_size;

        levels++;
    }

    printf("BFS completed\n");
    printf("Levels traversed : %d\n", levels);
    printf("Nodes visited    : %d\n", nodes_visited);

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_visited));

    CUDA_CHECK(cudaFree(d_current_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));

    CUDA_CHECK(cudaFree(d_next_size));
}


/*
 * ------------------------------------------------------------
 * Main
 * ------------------------------------------------------------
 */

int main(int argc, char **argv)
{
    int depth = 1000000;
    int branching_factor = 2;

    if (argc >= 2)
        depth = atoi(argv[1]);

    if (argc >= 3)
        branching_factor = atoi(argv[2]);

    if (depth < 0 ||
        branching_factor < 1) {

        fprintf(stderr,
                "Usage: %s <depth> <branching_factor>\n",
                argv[0]);

        return EXIT_FAILURE;
    }

    printf("Creating graph...\n");
    printf("Depth            : %d\n",
           depth);

    printf("Branching factor : %d\n",
           branching_factor);

    CSRGraph *graph =
        create_graph(
            depth,
            branching_factor);

    printf("Nodes            : %d\n",
           graph->num_nodes);

    printf("Edges            : %d\n",
           graph->num_edges);

    printf("Running GPU BFS...\n");

    /*
     * Make sure graph creation is complete
     * before timing GPU work.
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    bfs_gpu(graph);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop));

    printf("GPU BFS time     : %.3f seconds\n",
           milliseconds / 1000.0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free_graph(graph);

    return EXIT_SUCCESS;
}