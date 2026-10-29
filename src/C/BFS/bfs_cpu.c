#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    int *row_offsets;
    int *col_indices;
    int num_nodes;
    int num_edges;
} CSRGraph;


/*
 * Graph structure:
   Example with branching factor = 2, every level after level 0 contains exactly 2 nodes.
 *
 *              0
 *            /   \
 *           1     2
 *            \    /
 *             3  4
 *
 *
 * Therefore:
 *
 *   num_nodes = 1 + depth * branching_factor
 *
 * This gives us large depth without exponential
 * growth in the number of nodes.
 */
CSRGraph *create_graph(int depth, int branching_factor)
{
    CSRGraph *graph = malloc(sizeof(CSRGraph));

    if (!graph) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    /*
     * num_nodes = 1 + depth * branching_factor
       1 is for the single root node
     */
    long long num_nodes =
        1LL + (long long) depth * branching_factor;

/*
 * Level 0 contains only the root, which has 'b' edges.
 *
 * Every level from 1 to depth-1 contains 'b' nodes.
 * Each of those nodes has 'b' edges to the next level.
 *
 * Therefore, each intermediate level contributes:
 *     b * b = b^2 edges
 *
 * There are (depth - 1) such levels.
 *
 * Total:
 *     edges = b + (depth - 1) * b * b
 *
 * The final level has no outgoing edges.
 */
    long long num_edges;

    if (depth == 0) {
        num_edges = 0;
    } else {
        num_edges =
            (long long) branching_factor +
            (long long)(depth - 1) *
            branching_factor *
            branching_factor;
    }

    if (num_nodes > INT32_MAX || num_edges > INT32_MAX) {
        /**
        So although you calculate num_edges using long long, you eventually need to use the edge count with int-based arrays/indexing.
        ie row_offests and col_indices is int array.
        Also, num_edges has to be a `long long`, as the calculation for num_edges can have value well beyond what 32 bit signed int can store. */
        fprintf(stderr,
                "Graph is too large for this int-based implementation.\n");
        free(graph);
        exit(EXIT_FAILURE);
    }

    graph->num_nodes = (int) num_nodes;
    graph->num_edges = (int) num_edges;

    graph->row_offsets =
        malloc((graph->num_nodes + 1) * sizeof(int));

    graph->col_indices =
        malloc(graph->num_edges * sizeof(int));

    if (!graph->row_offsets ||
        (graph->num_edges > 0 && !graph->col_indices)) {

        perror("malloc");
        free(graph->row_offsets);
        free(graph->col_indices);
        free(graph);
        exit(EXIT_FAILURE);
    }

    /*
     * Build row_offsets.
     */
    int edge_position = 0;

    for (int node = 0; node < graph->num_nodes; node++) {

        graph->row_offsets[node] = edge_position;

        /*
         * Root:
         *     connects to all nodes in level 1.
         */
        if (node == 0) {

            for (int j = 0; j < branching_factor; j++) {
                col_indices:
                graph->col_indices[edge_position++] = 1 + j;
            }

        } else {

            /*
             * Determine which level this node belongs to.
             *
             * Nodes:
             *
             * level 1 starts at node 1
             * level 2 starts at node 1+b
             * level 3 starts at node 1+2b
             */
            int level = (node - 1) / branching_factor + 1;

            /*
             * Last level has no outgoing edges.
             */
            if (level < depth) {

                /*
                 * First node of the next level.
                 */
                int next_level_start =
                    1 + level * branching_factor;

                /*
                 * Connect to every node
                 * in the next level.
                 */
                for (int j = 0; j < branching_factor; j++) {
                    graph->col_indices[edge_position++] =
                        next_level_start + j;
                }
            }
        }
    }

    graph->row_offsets[graph->num_nodes] = edge_position;

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

void print_graph_levels(const CSRGraph *graph, int branching_factor, int num_levels)
{
    printf("\nGraph structure (first %d levels):\n", num_levels);

    int max_levels = graph->num_nodes / branching_factor;

    if (num_levels > max_levels)
        num_levels = max_levels;

    for (int level = 0; level <= num_levels; level++) {

        if (level == 0) {

            printf("Level 0: 0");

            int start = graph->row_offsets[0];
            int end   = graph->row_offsets[1];

            printf(" -> ");

            for (int i = start; i < end; i++) {
                printf("%d", graph->col_indices[i]);

                if (i < end - 1)
                    printf(", ");
            }

            printf("\n");

        } else {

            int level_start =
                1 + (level - 1) * branching_factor;

            int level_end =
                level_start + branching_factor;

            printf("Level %d: ", level);

            for (int node = level_start;
                 node < level_end;
                 node++) {

                printf("%d", node);

                if (node < level_end - 1)
                    printf(", ");
            }

            printf("\n");

            /*
             * Print the edges to the next level.
             */
            if (level < num_levels && level < max_levels) {

                for (int node = level_start;
                     node < level_end;
                     node++) {

                    int start = graph->row_offsets[node];
                    int end   = graph->row_offsets[node + 1];

                    printf("          %d -> ", node);

                    for (int i = start; i < end; i++) {

                        printf("%d",
                               graph->col_indices[i]);

                        if (i < end - 1)
                            printf(", ");
                    }

                    printf("\n");
                }
            }
        }
    }

    printf("\n");
}


void bfs(const CSRGraph *graph)
{
    int *visited =
        calloc(graph->num_nodes, sizeof(int));

    int *current_frontier =
        malloc(graph->num_nodes * sizeof(int));

    int *next_frontier =
        malloc(graph->num_nodes * sizeof(int));

    if (!visited ||
        !current_frontier ||
        !next_frontier) {

        perror("malloc");
        exit(EXIT_FAILURE);
    }

    int current_size = 1;
    int next_size = 0;

    /*
     * Start BFS at node 0.
     */
    current_frontier[0] = 0;
    visited[0] = 1;

    int levels = 0;
    int nodes_visited = 1;

    while (current_size > 0) {

        next_size = 0;

        /*
         * Process all nodes in the current BFS level.
         */
        for (int i = 0; i < current_size; i++) {

            int node = current_frontier[i];

            int start = graph->row_offsets[node];
            int end   = graph->row_offsets[node + 1];

            /*
             * Traverse this node's neighbors.
             */
            for (int j = start; j < end; j++) {

                int neighbor =
                    graph->col_indices[j];

                if (!visited[neighbor]) {

                    visited[neighbor] = 1;

                    next_frontier[next_size++] =
                        neighbor;

                    nodes_visited++;
                }
            }
        }

        /*
         * Move to the next BFS level.
         */
        int *tmp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = tmp;

        current_size = next_size;
        levels++;
    }

    printf("BFS completed\n");
    printf("Levels traversed : %d\n", levels);
    printf("Nodes visited    : %d\n", nodes_visited);

    free(visited);
    free(current_frontier);
    free(next_frontier);
}


int main(int argc, char **argv)
{
    int depth = 0;
    int branching_factor = 0;

   if (argc != 3) {
    fprintf(stderr,
            "Usage: %s <depth> <branching_factor>\n",
            argv[0]);
    return EXIT_FAILURE;
    }

    depth = atoi(argv[1]);
    branching_factor = atoi(argv[2]);

    if (depth < 0 || branching_factor < 1) {
        fprintf(stderr,
                "Error: depth must be >= 0 and branching_factor must be >= 1\n");
        return EXIT_FAILURE;
    }

    printf("Creating graph...\n");
    printf("Depth            : %d\n", depth);
    printf("Branching factor : %d\n", branching_factor);

    CSRGraph *graph =
        create_graph(depth, branching_factor);

    printf("Nodes            : %d\n", graph->num_nodes);
    printf("Edges            : %d\n", graph->num_edges);

    print_graph_levels(graph, branching_factor, 5);

    printf("Running BFS...\n");

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    bfs(graph);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("BFS time         : %.3f seconds\n",
           elapsed);

    free_graph(graph);

    return EXIT_SUCCESS;
}