#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * Configuration
 * ============================================================ */

#define DEFAULT_DEPTH            1000000
#define DEFAULT_BRANCHING_FACTOR 2

/* ============================================================
 * CSR Graph
 * ============================================================ */

typedef struct
{
    int num_nodes;
    int num_edges;

    int *row_offsets;
    int *col_indices;

} CSRGraph;


/*
 * Graph structure:
 *
 *             root
 *            /    \
 *           A      B
 *          / \    / \
 *         C   D  C   D
 *
 * Every node in level L connects to every node
 * in level L+1.
 *
 * Therefore:
 *
 *   level 0 : 1 node
 *   level 1 : branching_factor nodes
 *   level 2 : branching_factor nodes
 *   ...
 *
 * Number of nodes:
 *
 *   1 + depth * branching_factor
 *
 * Number of edges:
 *
 *   branching_factor
 *   +
 *   (depth - 1) * branching_factor^2
 */
static CSRGraph *create_graph(
    int depth,
    int branching_factor)
{
    CSRGraph *graph = malloc(sizeof(CSRGraph));

    if (!graph)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    graph->num_nodes =
        1 + depth * branching_factor;

    if (depth == 0)
    {
        graph->num_edges = 0;
    }
    else
    {
        graph->num_edges =
            branching_factor +
            (depth - 1) *
            branching_factor *
            branching_factor;
    }

    graph->row_offsets =
        malloc((graph->num_nodes + 1) * sizeof(int));

    graph->col_indices =
        malloc(graph->num_edges * sizeof(int));

    if (!graph->row_offsets ||
        (graph->num_edges > 0 &&
         !graph->col_indices))
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    /*
     * Build CSR.
     */

    int edge_position = 0;

    graph->row_offsets[0] = 0;

    /*
     * Root -> first level.
     */
    for (int j = 0;
         j < branching_factor;
         j++)
    {
        graph->col_indices[edge_position++] =
            1 + j;
    }

    graph->row_offsets[1] =
        edge_position;

    /*
     * For every subsequent level:
     *
     * each node connects to every node
     * in the next level.
     */
    for (int level = 1;
         level < depth;
         level++)
    {
        int current_level_start =
            1 + (level - 1) * branching_factor;

        int next_level_start =
            1 + level * branching_factor;

        for (int i = 0;
             i < branching_factor;
             i++)
        {
            int node =
                current_level_start + i;

            (void)node;

            for (int j = 0;
                 j < branching_factor;
                 j++)
            {
                graph->col_indices[edge_position++] =
                    next_level_start + j;
            }

            graph->row_offsets[
                current_level_start + i + 1] =
                edge_position;
        }
    }

    /*
     * For depth == 0, row_offsets[1] needs
     * to be initialized.
     */
    if (depth == 0)
    {
        graph->row_offsets[1] = 0;
    }

    return graph;
}


static void free_graph(CSRGraph *graph)
{
    if (!graph)
        return;

    free(graph->row_offsets);
    free(graph->col_indices);
    free(graph);
}


/* ============================================================
 * BFS Thread Pool
 * ============================================================ */

typedef struct
{
    const CSRGraph *graph;

    /*
     * Visited is shared by all workers.
     */
    atomic_int *visited;

    /*
     * Current BFS frontier.
     */
    int *current_frontier;
    int current_size;

    /*
     * Next BFS frontier.
     */
    int *next_frontier;

    /*
     * Number of elements currently inserted
     * into next_frontier.
     */
    atomic_int next_size;

    /*
     * Number of persistent worker threads.
     */
    int num_threads;

    /*
     * Used to terminate workers.
     */
    atomic_int stop;

    /*
     * Synchronization:
     *
     * main thread + all worker threads
     * participate in both barriers.
     */
    pthread_barrier_t start_barrier;
    pthread_barrier_t done_barrier;

} BFSContext;


/* ============================================================
 * Worker
 * ============================================================ */

typedef struct
{
    BFSContext *context;
    int thread_id;

} WorkerArgument;


/*
 * Each worker repeatedly:
 *
 *   1. waits for the next BFS level
 *   2. processes its assigned frontier nodes
 *   3. waits until all workers finish
 *   4. repeats
 *
 * The threads themselves are created only once.
 */
static void *bfs_worker(void *arg)
{
    WorkerArgument *worker =
        (WorkerArgument *)arg;

    BFSContext *ctx =
        worker->context;

    int thread_id =
        worker->thread_id;

    while (1)
    {
        /*
         * Wait for the main thread to
         * release this BFS level.
         */
        pthread_barrier_wait(
            &ctx->start_barrier);

        /*
         * Main thread can tell us to stop.
         */
        if (atomic_load(&ctx->stop))
        {
            break;
        }

        /*
         * Static partitioning.
         *
         * Example:
         *
         *   current_size = 8
         *   num_threads  = 4
         *
         * Thread 0: nodes 0,4
         * Thread 1: nodes 1,5
         * Thread 2: nodes 2,6
         * Thread 3: nodes 3,7
         *
         * If current_size == 2 and
         * num_threads == 16:
         *
         * Thread 0 -> node 0
         * Thread 1 -> node 1
         * Thread 2..15 -> nothing
         *
         * This is exactly the low-parallelism
         * situation we want to demonstrate.
         */
        for (int i = thread_id;
             i < ctx->current_size;
             i += ctx->num_threads)
        {
            int node =
                ctx->current_frontier[i];

            int start =
                ctx->graph->row_offsets[node];

            int end =
                ctx->graph->row_offsets[node + 1];

            /*
             * Process all neighbors of this node.
             */
            for (int j = start;
                 j < end;
                 j++)
            {
                int neighbor =
                    ctx->graph->col_indices[j];

                /*
                 * Atomically perform:
                 *
                 *   if visited[neighbor] == 0
                 *       visited[neighbor] = 1
                 *
                 * This is the CPU equivalent of the
                 * CUDA atomicCAS used in the GPU BFS.
                 */
                int expected = 0;

                if (atomic_compare_exchange_strong(
                        &ctx->visited[neighbor],
                        &expected,
                        1))
                {
                    /*
                     * Atomically reserve a position
                     * in the next frontier.
                     *
                     * This is analogous to CUDA's
                     * atomicAdd(next_size, 1).
                     */
                    int position =
                        atomic_fetch_add(
                            &ctx->next_size,
                            1);

                    ctx->next_frontier[position] =
                        neighbor;
                }
            }
        }

        /*
         * Tell the main thread that this worker
         * has completed the current BFS level.
         */
        pthread_barrier_wait(
            &ctx->done_barrier);
    }

    return NULL;
}


/* ============================================================
 * Parallel BFS using persistent thread pool
 * ============================================================ */

static void bfs(
    const CSRGraph *graph,
    int num_threads)
{
    /*
     * One atomic visited flag per node.
     */
    atomic_int *visited =
        malloc(graph->num_nodes *
               sizeof(atomic_int));

    /*
     * Two frontier buffers.
     *
     * We reuse them throughout the BFS.
     */
    int *current_frontier =
        malloc(graph->num_nodes *
               sizeof(int));

    int *next_frontier =
        malloc(graph->num_nodes *
               sizeof(int));

    if (!visited ||
        !current_frontier ||
        !next_frontier)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    /*
     * Initialize visited.
     */
    for (int i = 0;
         i < graph->num_nodes;
         i++)
    {
        atomic_init(&visited[i], 0);
    }

    /*
     * Initial frontier contains root.
     */
    current_frontier[0] = 0;

    atomic_store(
        &visited[0],
        1);

    /*
     * Initialize BFS context.
     */
    BFSContext ctx;

    ctx.graph = graph;

    ctx.visited = visited;

    ctx.current_frontier =
        current_frontier;

    ctx.current_size = 1;

    ctx.next_frontier =
        next_frontier;

    atomic_init(
        &ctx.next_size,
        0);

    ctx.num_threads =
        num_threads;

    atomic_init(
        &ctx.stop,
        0);

    /*
     * There are:
     *
     *   num_threads workers
     *   +
     *   1 main thread
     */
    pthread_barrier_init(
        &ctx.start_barrier,
        NULL,
        num_threads + 1);

    pthread_barrier_init(
        &ctx.done_barrier,
        NULL,
        num_threads + 1);

    /*
     * Create the worker threads ONCE.
     */
    pthread_t *threads =
        malloc(num_threads *
               sizeof(pthread_t));

    WorkerArgument *arguments =
        malloc(num_threads *
               sizeof(WorkerArgument));

    if (!threads || !arguments)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0;
         i < num_threads;
         i++)
    {
        arguments[i].context = &ctx;
        arguments[i].thread_id = i;

        int rc =
            pthread_create(
                &threads[i],
                NULL,
                bfs_worker,
                &arguments[i]);

        if (rc != 0)
        {
            fprintf(
                stderr,
                "pthread_create failed: %s\n",
                strerror(rc));

            exit(EXIT_FAILURE);
        }
    }

    /*
     * BFS statistics.
     */
    int levels = 0;
    int nodes_visited = 1;

    /*
     * --------------------------------------------------------
     * BFS LEVEL LOOP
     * --------------------------------------------------------
     */
    while (ctx.current_size > 0)
    {
        /*
         * Reset next frontier size.
         */
        atomic_store(
            &ctx.next_size,
            0);

        /*
         * Release all worker threads.
         *
         * They now process the current frontier.
         */
        pthread_barrier_wait(
            &ctx.start_barrier);

        /*
         * Wait until ALL workers have finished
         * this BFS level.
         */
        pthread_barrier_wait(
            &ctx.done_barrier);

        /*
         * Every worker is now finished.
         *
         * Therefore it is safe for the main thread
         * to inspect next_size and swap frontiers.
         */
        int next_size =
            atomic_load(
                &ctx.next_size);

        nodes_visited += next_size;

        levels++;

        /*
         * Swap current and next frontier.
         */
        int *tmp =
            ctx.current_frontier;

        ctx.current_frontier =
            ctx.next_frontier;

        ctx.next_frontier =
            tmp;

        ctx.current_size =
            next_size;
    }

    /*
     * No more nodes remain.
     *
     * Tell workers to terminate.
     */
    atomic_store(
        &ctx.stop,
        1);

    /*
     * Release workers one final time.
     *
     * They wake up, see stop == 1,
     * and exit without entering done_barrier.
     */
    pthread_barrier_wait(
        &ctx.start_barrier);

    /*
     * Wait for all persistent workers.
     */
    for (int i = 0;
         i < num_threads;
         i++)
    {
        pthread_join(
            threads[i],
            NULL);
    }

    printf("\n");
    printf("CPU BFS completed\n");
    printf("-----------------\n");
    printf("CPU threads     : %d\n",
           num_threads);
    printf("Levels traversed: %d\n",
           levels);
    printf("Nodes visited   : %d\n",
           nodes_visited);

    /*
     * Cleanup.
     */
    pthread_barrier_destroy(
        &ctx.start_barrier);

    pthread_barrier_destroy(
        &ctx.done_barrier);

    free(threads);
    free(arguments);

    free(visited);
    free(current_frontier);
    free(next_frontier);
}


/* ============================================================
 * Timing
 * ============================================================ */

static double get_time_seconds(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts);

    return
        (double)ts.tv_sec +
        (double)ts.tv_nsec / 1e9;
}


/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    int depth = 0;
    int branching_factor = 0;
    int num_threads = 0;

    /*
     * All three arguments are mandatory:
     *
     *   <depth>
     *   <branching_factor>
     *   <worker_threads>
     */
    if (argc != 4)
    {
        fprintf(stderr,
                "Usage: %s <depth> <branching_factor> <worker_threads>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    depth = atoi(argv[1]);
    branching_factor = atoi(argv[2]);
    num_threads = atoi(argv[3]);

    if (depth < 0 ||
        branching_factor <= 0 ||
        num_threads <= 0)
    {
        fprintf(stderr,
                "Error: depth must be >= 0, "
                "branching_factor must be > 0, "
                "worker_threads must be > 0\n");
        return EXIT_FAILURE;
    }

    printf("========================================\n");
    printf("CPU Parallel BFS\n");
    printf("========================================\n");

    printf("Depth           : %d\n", depth);
    printf("Branching factor: %d\n", branching_factor);
    printf("CPU threads     : %d\n", num_threads);

    printf("----------------------------------------\n");

    /*
     * Create graph outside timing.
     *
     * We want to measure BFS, not graph
     * generation.
     */
    printf("Creating graph...\n");

    CSRGraph *graph =
        create_graph(
            depth,
            branching_factor);

    printf("Nodes           : %d\n",
           graph->num_nodes);

    printf("Edges           : %d\n",
           graph->num_edges);

    printf("----------------------------------------\n");

    /*
     * Time BFS only.
     */
    double start =
        get_time_seconds();

    bfs(
        graph,
        num_threads);

    double end =
        get_time_seconds();

    printf("----------------------------------------\n");

    printf("BFS time        : %.6f seconds\n",
           end - start);

    printf("========================================\n");

    free_graph(graph);

    return EXIT_SUCCESS;
}