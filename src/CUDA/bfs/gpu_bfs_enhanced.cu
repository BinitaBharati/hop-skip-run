
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
 * ========================================================================
 * Layered graph representation
 * ========================================================================
 *
 * The graph is NOT a traditional tree.
 *
 * It consists of multiple levels:
 *
 *   Level 0:          [0]
 *
 *   Level 1:          [1, 2, ..., B]
 *
 *   Level 2:          [B+1, ..., 2B]
 *
 *   Level 3:          [2B+1, ..., 3B]
 *
 *   ...
 *
 * where B = branching_factor.
 *
 * Every node in one level is connected to EVERY node in the next level.
 *
 * For example, with branching_factor = 2:
 *
 *   Level 0:          0
 *                    / \
 *                   /   \
 *   Level 1:       1     2
 *                  |\     /|
 *                  | \   / |
 *                  |  \ /  |
 *   Level 2:       3     4
 *                  |\     /|
 *                  | \   / |
 *                  |  \ /  |
 *   Level 3:       5     6
 *
 * More explicitly:
 *
 *   0 -> 1, 2
 *
 *   1 -> 3, 4
 *   2 -> 3, 4
 *
 *   3 -> 5, 6
 *   4 -> 5, 6
 *
 *   ...
 *
 * Therefore:
 *
 *   Number of nodes:
 *
 *       1 + depth * branching_factor
 *
 *   Edges from the root to level 1:
 *
 *       branching_factor
 *
 *   Edges between every subsequent pair of levels:
 *
 *       branching_factor * branching_factor
 *
 *   Total number of edges:
 *
 *       B + (depth - 1) * B * B
 *
 * where B = branching_factor.
 *
 * This graph is useful for demonstrating a workload with:
 *
 *   - potentially very large BFS depth
 *   - only B nodes in each non-root frontier
 *   - independent processing of nodes within the same frontier
 *   - sequential dependency between BFS levels
 *
 * For example, with B = 2, every level after the root has only
 * two frontier nodes. Therefore there are only two independent
 * node-expansion tasks available at each BFS level, regardless
 * of how large the graph depth becomes.
 */


/*
 * ========================================================================
 * CSR graph
 * ========================================================================
 *
 * row_offsets[node]     = starting position of node's neighbors
 * row_offsets[node + 1] = ending position
 *
 * Therefore the neighbors of node N are:
 *
 *     col_indices[row_offsets[N] ... row_offsets[N + 1] - 1]
 */
typedef struct
{
    int num_nodes;
    int num_edges;

    int *row_offsets;
    int *col_indices;

} Graph;


/*
 * ========================================================================
 * Create the layered graph
 * ========================================================================
 */
Graph create_graph(int depth, int branching_factor)
{
    Graph graph;

    /*
     * Level 0 contains one root node.
     *
     * Every remaining level contains branching_factor nodes.
     */
    graph.num_nodes = 1 + depth * branching_factor;

    /*
     * Root -> first level:
     *
     *     branching_factor edges
     *
     * Between every subsequent pair of levels:
     *
     *     branching_factor nodes
     *     each connected to branching_factor nodes
     *
     * Therefore:
     *
     *     branching_factor * branching_factor
     *
     * edges per level transition.
     */
    if (depth == 0)
    {
        graph.num_edges = 0;
    }
    else
    {
        graph.num_edges =
            branching_factor +
            (depth - 1) * branching_factor * branching_factor;
    }

    graph.row_offsets =
        (int *)malloc((graph.num_nodes + 1) * sizeof(int));

    graph.col_indices =
        (int *)malloc(graph.num_edges * sizeof(int));

    if (graph.row_offsets == NULL ||
        (graph.num_edges > 0 && graph.col_indices == NULL))
    {
        fprintf(stderr, "Failed to allocate graph memory\n");
        exit(EXIT_FAILURE);
    }

    int edge_index = 0;

    /*
     * ------------------------------------------------------------
     * Root node
     * ------------------------------------------------------------
     *
     * Node 0 connects to every node in level 1.
     */
    graph.row_offsets[0] = 0;

    if (depth > 0)
    {
        for (int j = 0; j < branching_factor; j++)
        {
            /*
             * First node of level 1 is node 1.
             */
            graph.col_indices[edge_index++] = 1 + j;
        }
    }

    graph.row_offsets[1] = edge_index;


    /*
     * ------------------------------------------------------------
     * Remaining levels
     * ------------------------------------------------------------
     *
     * For each level:
     *
     *   every node in the current level
     *          |
     *          +----> every node in the next level
     *
     * Example:
     *
     *   current:  1, 2
     *   next:     3, 4
     *
     * Edges:
     *
     *   1 -> 3
     *   1 -> 4
     *   2 -> 3
     *   2 -> 4
     */
    for (int level = 1; level < depth; level++)
    {
        /*
         * First node of the current level.
         *
         * Level 1 starts at node 1.
         * Level 2 starts at node 1 + B.
         * Level 3 starts at node 1 + 2B.
         */
        int current_level_start =
            1 + (level - 1) * branching_factor;

        /*
         * First node of the next level.
         */
        int next_level_start =
            1 + level * branching_factor;

        /*
         * Every node in the current level connects to
         * every node in the next level.
         */
        for (int current_node = 0;
             current_node < branching_factor;
             current_node++)
        {
            int node =
                current_level_start + current_node;

            for (int next_node = 0;
                 next_node < branching_factor;
                 next_node++)
            {
                graph.col_indices[edge_index++] =
                    next_level_start + next_node;
            }

            graph.row_offsets[node + 1] = edge_index;
        }
    }

    /*
     * If depth == 0, there is only the root node.
     *
     * Its row_offsets entry has already been initialized.
     */
    if (depth == 0)
    {
        graph.row_offsets[1] = 0;
    }

    return graph;
}


/*
 * ========================================================================
 * Free graph
 * ========================================================================
 */
void free_graph(Graph *graph)
{
    free(graph->row_offsets);
    free(graph->col_indices);

    graph->row_offsets = NULL;
    graph->col_indices = NULL;
}


/*
 * ========================================================================
 * CUDA BFS kernel
 * ========================================================================
 *
 * One CUDA thread processes one or more frontier nodes.
 *
 * The grid-stride loop is important because BLOCKS and
 * THREADS_PER_BLOCK are explicitly supplied by the user.
 *
 * Total logical CUDA threads launched:
 *
 *     blocks * threads_per_block
 *
 * If there are more frontier nodes than logical CUDA threads,
 * a thread processes multiple frontier nodes using the grid stride.
 *
 * If there are fewer frontier nodes than logical CUDA threads,
 * the extra threads simply do no work.
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
    /*
     * Global CUDA thread ID.
     *
     * Example:
     *
     *   blockDim.x = 256
     *
     *   block 0 -> thread IDs 0 ... 255
     *   block 1 -> thread IDs 256 ... 511
     *   block 2 -> thread IDs 512 ... 767
     */
    int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Number of logical CUDA threads in the entire grid.
     */
    int stride =
        blockDim.x * gridDim.x;

    /*
     * Process frontier nodes assigned to this CUDA thread.
     *
     * Normally a thread may process one frontier node.
     *
     * If the frontier is larger than the total number of
     * launched CUDA threads, the same CUDA thread will process
     * additional frontier nodes using the stride.
     */
    for (int frontier_index = tid;
         frontier_index < current_size;
         frontier_index += stride)
    {
        int node =
            current_frontier[frontier_index];

        int start =
            row_offsets[node];

        int end =
            row_offsets[node + 1];

        /*
         * Traverse this node's neighbors.
         *
         * The neighbor traversal itself is sequential within
         * this CUDA thread.
         */
        for (int i = start; i < end; i++)
        {
            int neighbor =
                col_indices[i];

            /*
             * Atomically mark the neighbor as visited.
             *
             * Only the first CUDA thread that changes the value
             * from 0 -> 1 is allowed to add the node to the
             * next frontier.
             */
            int old_value =
                atomicCAS(
                    &visited[neighbor],
                    0,
                    1);

            if (old_value == 0)
            {
                /*
                 * Atomically reserve a position in the
                 * next frontier.
                 */
                int position =
                    atomicAdd(
                        next_size,
                        1);

                next_frontier[position] =
                    neighbor;
            }
        }
    }
}


/*
 * ========================================================================
 * CUDA error checking helper
 * ========================================================================
 */
void check_cuda(
    cudaError_t error,
    const char *operation)
{
    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA error during %s: %s\n",
            operation,
            cudaGetErrorString(error));

        exit(EXIT_FAILURE);
    }
}


/*
 * ========================================================================
 * GPU BFS
 * ========================================================================
 */
void bfs_gpu(
    const Graph *graph,
    int blocks,
    int threads_per_block)
{
    int *d_row_offsets = NULL;
    int *d_col_indices = NULL;
    int *d_visited = NULL;

    int *d_current_frontier = NULL;
    int *d_next_frontier = NULL;
    int *d_next_size = NULL;


    /*
     * ------------------------------------------------------------
     * Allocate graph data on GPU
     * ------------------------------------------------------------
     */

    check_cuda(
        cudaMalloc(
            &d_row_offsets,
            (graph->num_nodes + 1) * sizeof(int)),
        "cudaMalloc(row_offsets)");

    check_cuda(
        cudaMalloc(
            &d_col_indices,
            graph->num_edges * sizeof(int)),
        "cudaMalloc(col_indices)");

    check_cuda(
        cudaMalloc(
            &d_visited,
            graph->num_nodes * sizeof(int)),
        "cudaMalloc(visited)");

    check_cuda(
        cudaMalloc(
            &d_current_frontier,
            graph->num_nodes * sizeof(int)),
        "cudaMalloc(current_frontier)");

    check_cuda(
        cudaMalloc(
            &d_next_frontier,
            graph->num_nodes * sizeof(int)),
        "cudaMalloc(next_frontier)");

    check_cuda(
        cudaMalloc(
            &d_next_size,
            sizeof(int)),
        "cudaMalloc(next_size)");


    /*
     * ------------------------------------------------------------
     * Copy graph to GPU
     * ------------------------------------------------------------
     */

    check_cuda(
        cudaMemcpy(
            d_row_offsets,
            graph->row_offsets,
            (graph->num_nodes + 1) * sizeof(int),
            cudaMemcpyHostToDevice),
        "copy row_offsets");

    if (graph->num_edges > 0)
    {
        check_cuda(
            cudaMemcpy(
                d_col_indices,
                graph->col_indices,
                graph->num_edges * sizeof(int),
                cudaMemcpyHostToDevice),
            "copy col_indices");
    }


    /*
     * ------------------------------------------------------------
     * Initialize visited array
     * ------------------------------------------------------------
     *
     * 0 = not visited
     * 1 = visited
     */
    check_cuda(
        cudaMemset(
            d_visited,
            0,
            graph->num_nodes * sizeof(int)),
        "initialize visited");


    /*
     * ------------------------------------------------------------
     * Start BFS at root node 0
     * ------------------------------------------------------------
     */

    int root = 0;

    check_cuda(
        cudaMemcpy(
            d_current_frontier,
            &root,
            sizeof(int),
            cudaMemcpyHostToDevice),
        "copy root");


    /*
     * Mark root as visited.
     */
    int visited_value = 1;

    check_cuda(
        cudaMemcpy(
            d_visited,
            &visited_value,
            sizeof(int),
            cudaMemcpyHostToDevice),
        "mark root visited");


    int current_size = 1;

    /*
     * ------------------------------------------------------------
     * BFS level by level
     * ------------------------------------------------------------
     */
    int level = 0;

    while (current_size > 0)
    {
        /*
         * Reset next frontier size.
         */
        check_cuda(
            cudaMemset(
                d_next_size,
                0,
                sizeof(int)),
            "reset next_size");


        /*
         * --------------------------------------------------------
         * Launch CUDA kernel
         * --------------------------------------------------------
         *
         * The user controls both:
         *
         *     blocks
         *     threads_per_block
         *
         * Therefore:
         *
         *     total logical CUDA threads
         *         =
         *     blocks * threads_per_block
         */
        bfs_kernel<<<blocks, threads_per_block>>>(
            d_row_offsets,
            d_col_indices,
            d_visited,
            d_current_frontier,
            current_size,
            d_next_frontier,
            d_next_size);


        check_cuda(
            cudaGetLastError(),
            "bfs_kernel launch");

        check_cuda(
            cudaDeviceSynchronize(),
            "bfs_kernel execution");


        /*
         * Get the number of nodes discovered for the next level.
         */
        int next_size = 0;

        check_cuda(
            cudaMemcpy(
                &next_size,
                d_next_size,
                sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy next_size");


        /*
         * Swap the frontier buffers.
         */
        int *tmp =
            d_current_frontier;

        d_current_frontier =
            d_next_frontier;

        d_next_frontier =
            tmp;

        current_size =
            next_size;

        level++;
    }


    /*
     * ------------------------------------------------------------
     * Cleanup
     * ------------------------------------------------------------
     */

    check_cuda(
        cudaFree(d_row_offsets),
        "cudaFree(row_offsets)");

    check_cuda(
        cudaFree(d_col_indices),
        "cudaFree(col_indices)");

    check_cuda(
        cudaFree(d_visited),
        "cudaFree(visited)");

    check_cuda(
        cudaFree(d_current_frontier),
        "cudaFree(current_frontier)");

    check_cuda(
        cudaFree(d_next_frontier),
        "cudaFree(next_frontier)");

    check_cuda(
        cudaFree(d_next_size),
        "cudaFree(next_size)");
}


/*
 * ========================================================================
 * Main
 * ========================================================================
 */
int main(int argc, char **argv)
{
    /*
     * Usage:
     *
     *   ./bfs_gpu depth branching_factor blocks threads_per_block
     *
     * Example:
     *
     *   ./bfs_gpu 1000000 2 1 2
     */
    if (argc != 5)
    {
        fprintf(
            stderr,
            "Usage:\n"
            "  %s depth branching_factor blocks threads_per_block\n\n"
            "Example:\n"
            "  %s 1000000 2 1 2\n",
            argv[0],
            argv[0]);

        return EXIT_FAILURE;
    }


    int depth =
        atoi(argv[1]);

    int branching_factor =
        atoi(argv[2]);

    int blocks =
        atoi(argv[3]);

    int threads_per_block =
        atoi(argv[4]);


    /*
     * ------------------------------------------------------------
     * Validate input
     * ------------------------------------------------------------
     */

    if (depth < 0)
    {
        fprintf(
            stderr,
            "Error: depth must be >= 0\n");

        return EXIT_FAILURE;
    }

    if (branching_factor <= 0)
    {
        fprintf(
            stderr,
            "Error: branching_factor must be > 0\n");

        return EXIT_FAILURE;
    }

    if (blocks <= 0)
    {
        fprintf(
            stderr,
            "Error: blocks must be > 0\n");

        return EXIT_FAILURE;
    }

    if (threads_per_block <= 0)
    {
        fprintf(
            stderr,
            "Error: threads_per_block must be > 0\n");

        return EXIT_FAILURE;
    }


    /*
     * ------------------------------------------------------------
     * Query CUDA device
     * ------------------------------------------------------------
     */

    int device = 0;

    cudaDeviceProp prop;

    check_cuda(
        cudaGetDeviceProperties(
            &prop,
            device),
        "cudaGetDeviceProperties");


    /*
     * CUDA imposes a hardware limit on the number of threads
     * that can exist in a single block.
     */
    if (threads_per_block >
        prop.maxThreadsPerBlock)
    {
        fprintf(
            stderr,
            "Error: threads_per_block=%d exceeds "
            "GPU maximum of %d\n",
            threads_per_block,
            prop.maxThreadsPerBlock);

        return EXIT_FAILURE;
    }


    /*
     * ------------------------------------------------------------
     * Create graph
     * ------------------------------------------------------------
     */

    Graph graph =
        create_graph(
            depth,
            branching_factor);


    /*
     * ------------------------------------------------------------
     * Print configuration
     * ------------------------------------------------------------
     */

    printf(
        "GPU BFS\n"
        "=======\n");

    printf(
        "Depth              : %d\n",
        depth);

    printf(
        "Branching factor   : %d\n",
        branching_factor);

    printf(
        "Nodes              : %d\n",
        graph.num_nodes);

    printf(
        "Edges              : %d\n",
        graph.num_edges);

    printf(
        "CUDA blocks        : %d\n",
        blocks);

    printf(
        "Threads per block  : %d\n",
        threads_per_block);

    printf(
        "Total CUDA threads : %d\n",
        blocks * threads_per_block);

    printf(
        "GPU                : %s\n",
        prop.name);

    printf("\n");


    /*
     * ------------------------------------------------------------
     * Run BFS
     * ------------------------------------------------------------
     */

    bfs_gpu(
        &graph,
        blocks,
        threads_per_block);


    printf(
        "BFS completed successfully.\n");


    /*
     * ------------------------------------------------------------
     * Cleanup
     * ------------------------------------------------------------
     */

    free_graph(&graph);

    return EXIT_SUCCESS;
}
