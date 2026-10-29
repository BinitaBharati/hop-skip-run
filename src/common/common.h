#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

#include <sys/stat.h>
#include <unistd.h>
#include <sys/syscall.h>

/* ========================================================

typedef int (*ProcessVideoFunction)(void *);
        │        │                    │
        │        │                    └── function takes void *
        │        └── pointer to function
        └── function returns int

(void *) input arg can be casted to the exact type in the function implementation.
In C, * is conceptually associated with the declarator/variable, not the base type.
That is why we have (*ProcessVideoFunction)

*
* ==========================================================
*/
typedef int (*ProcessVideoFunction)(void *);



/* =========================================================
 * Video worker data
 * ========================================================= */
typedef struct
{
    char **input_files;
    int file_count;
    int next_file;
    pthread_mutex_t *mutex;
    const char *output_directory;
    int failed_count;
} VideoWorkerData;

typedef struct
{
    VideoWorkerData *data;
    ProcessVideoFunction process_video;
    void *process_arguments;
} VideoWorkerArgument;

typedef struct
{
    const char *input_path;
    const char *output_path;
    void *implementation_args;

} ProcessVideoContext;

/**
===================================================================================
VideoWorkerFunction is a pointer to a function that takes a void * argument and returns a void *.

So a compatible function looks like:

void *video_worker(void *arg)
{
    // do work

    return NULL;
}
This signature is mandated by the pthread API.
The pthread_create API expects a start routine with this signature:
void *(*start_routine)(void *)
===================================================================================
*/
typedef void *(*VideoWorkerFunction)(void *);

int create_video_workers(
    pthread_t *threads,
    int video_threads,
    void *worker_argument,
    VideoWorkerFunction worker_function
)
{
    printf(
        "Creating %d video worker threads...\n\n",
        video_threads
    );

    int created_threads = 0;

    for (int i = 0; i < video_threads; i++)
    {
        int result = pthread_create(
            &threads[i],
            NULL,
            worker_function,
            worker_argument
        );

        if (result != 0)
        {
            fprintf(
                stderr,
                "ERROR: pthread_create failed for video worker %d: %s\n",
                i,
                strerror(result)
            );
            break;
        }

        printf("Created video worker %d\n", i);

        created_threads++;
    }

    printf(
        "Created %d/%d video workers\n\n",
        created_threads,
        video_threads
    );

    return created_threads;
}


/* =========================================================
 * Video worker thread
 * This signature is required by pthread_create().
 *
 * One video worker processes one complete video at a time.
 *
 * Example with video_threads = 2:
 *
 *     Worker 1 -> 1.mp4
 *     Worker 2 -> 2.mp4
 *
 * When one finishes, it takes the next available video.
 *
 * Each worker then calls process_video(), which internally
 * creates sobel_threads pthreads for each frame.
 * ========================================================= */



static void *video_worker(void *arg)
{
    printf(
    "[VIDEO WORKER] PID=%d TID=%ld\n",
    getpid(),
    syscall(SYS_gettid)
    );

    VideoWorkerArgument *worker =
        (VideoWorkerArgument *)arg;


    printf("[VIDEO WORKER] Thread started\n");

    while (1)
    {
        int file_index;


        /*
         * Get next video.
         *
         * Only this small section needs the mutex.
         */

        pthread_mutex_lock(worker->data->mutex);


        if (worker->data->next_file >= worker->data->file_count)
        {
            //all files in input directory are already processed.
            pthread_mutex_unlock(worker->data->mutex);
            break;
        }


        file_index = worker->data->next_file;

        (worker->data->next_file)++;

        pthread_mutex_unlock(worker->data->mutex);


        /*
         * Get input filename.
         */

        const char *input_path = worker->data->input_files[file_index];


        /*
         * Extract basename.
         *
         * Example:
         *
         * resources/input/mp4/4k/1.mp4
         *
         * becomes:
         *
         * 1.mp4
         */

        const char *filename = strrchr(input_path, '/');


        if (filename != NULL)
        {
            filename++;
        }
        else
        {
            filename = input_path;
        }


        /*
         * Remove .mp4 extension.
         */

        size_t filename_length = strlen(filename);


        size_t base_length = filename_length;


        if (base_length >= 4 &&
            strcasecmp(
                filename + base_length - 4,
                ".mp4"
            ) == 0)
        {
            base_length -= 4;
        }


        /*
         * Build output filename.
         *
         * Example:
         *
         * 1.mp4
         *
         * becomes:
         *
         * 1_sobel.mp4
         */

        char output_filename[4096];


        snprintf(
            output_filename,
            sizeof(output_filename),
            "%s/%.*s_sobel.mp4",
            worker->data->output_directory,
            (int)base_length,
            filename
        );


        printf(
            "\n"
            "========================================\n"
            "[VIDEO WORKER] Processing video %d/%d\n"
            "========================================\n"
            "Input : %s\n"
            "Output: %s\n"
            "========================================\n",
            file_index + 1,
            worker->data->file_count,
            input_path,
            output_filename
        );


        /*
         * Process complete video.
         */

        /*int result =
            process_video(
                input_path,
                output_filename,
                data->sobel_threads,
                data->ffmpeg_threads
            );*/
        ProcessVideoContext process_context =
        {
            .input_path = input_path,
            .output_path = output_filename,
            .implementation_args = worker->process_arguments
        };

        int result =
        worker->process_video(
            &process_context
        );

        /*
         * Record failure.
         */

        if (result != 0)
        {
            pthread_mutex_lock(
                worker->data->mutex
            );

            (worker->data->failed_count)++;

            pthread_mutex_unlock(
                worker->data->mutex
            );


            fprintf(
                stderr,
                "\n"
                "[VIDEO WORKER] FAILED:\n"
                "  %s\n",
                input_path
            );
        }
        else
        {
            printf(
                "\n"
                "[VIDEO WORKER] DONE:\n"
                "  %s\n"
                "  -> %s\n\n",
                input_path,
                output_filename
            );

            fflush(stdout);
        }
    }


    printf(
        "[VIDEO WORKER] Thread finished\n"
    );

    fflush(stdout);


    return NULL;
}


/**
Check if directory exists.
 */

int directory_exists(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

void free_input_files(char **input_files, int file_count)
{
    for (int i = 0; i < file_count; i++)
        free(input_files[i]);

    free(input_files);
}

/**
==========================================================
Reads input files, file count and file capacity
==========================================================
*/
int read_input_files(
    const char *input_directory,
    char ***input_files,
    int *file_count
)
{
    DIR *dir = opendir(input_directory);

    if (dir == NULL)
    {
        fprintf(
            stderr,
            "ERROR: Could not open input directory '%s': %s\n",
            input_directory,
            strerror(errno)
        );
        return EXIT_FAILURE;
    }

    struct dirent *entry;
    int file_capacity = 16;

    while ((entry = readdir(dir)) != NULL)
    {
        /* Ignore "." and "..". */
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }

        /* Only process .mp4 files. */
        const char *extension = strrchr(entry->d_name, '.');

        if (extension == NULL ||
            strcasecmp(extension, ".mp4") != 0)
        {
            continue;
        }

        /* Grow array if necessary. */
        if (*file_count >= file_capacity)
        {
            file_capacity *= 2;

            char **tmp = (char **)realloc(
            *input_files,
            file_capacity * sizeof(char *)
        );

            if (tmp == NULL)
            {
                fprintf(stderr, "ERROR: realloc failed\n");
                closedir(dir);
                return EXIT_FAILURE;
            }

            *input_files = tmp;
        }

        /* Build full input path. */
        size_t path_length =
            strlen(input_directory) +
            1 +
            strlen(entry->d_name) +
            1;

        char *full_path = (char *)malloc(path_length);

        if (full_path == NULL)
        {
            fprintf(stderr, "ERROR: malloc failed\n");
            closedir(dir);
            return EXIT_FAILURE;
        }

        snprintf(
            full_path,
            path_length,
            "%s/%s",
            input_directory,
            entry->d_name
        );

        (*input_files)[*file_count] = full_path;
        (*file_count)++;
    }

    closedir(dir);
    return EXIT_SUCCESS;
}


/* ---------------------------------------------------------
 * Timing
 * --------------------------------------------------------- */

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return (double)ts.tv_sec * 1000.0
         + (double)ts.tv_nsec / 1000000.0;
}


/* ---------------------------------------------------------
 * Encode one processed frame
 *
 * frame == NULL means:
 * flush the encoder.
 *
 * The frame PTS and duration are already set by the
 * caller.
 * --------------------------------------------------------- */

static int encode_frame(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_format_ctx,
    AVStream *output_stream,
    AVFrame *frame)
{
    int ret;

    /*
     * Send frame to encoder.
     *
     * Passing NULL flushes the encoder.
     */
    ret = avcodec_send_frame(
        encoder_ctx,
        frame
    );

    if (ret < 0)
    {
        fprintf(
            stderr,
            "Error sending frame to encoder\n"
        );

        return ret;
    }


    /*
     * Receive all packets currently available
     * from the encoder.
     */
    while (1)
    {
        AVPacket *packet =
            av_packet_alloc();

        if (!packet)
        {
            fprintf(
                stderr,
                "Could not allocate output packet\n"
            );

            return AVERROR(ENOMEM);
        }


        ret = avcodec_receive_packet(
            encoder_ctx,
            packet
        );


        /*
         * Encoder needs more frames.
         */
        if (ret == AVERROR(EAGAIN) ||
            ret == AVERROR_EOF)
        {
            av_packet_free(&packet);
            break;
        }


        if (ret < 0)
        {
            fprintf(
                stderr,
                "Error receiving encoded packet\n"
            );

            av_packet_free(&packet);

            return ret;
        }


        /*
         * Convert packet timestamps from the encoder
         * time base to the output stream time base.
         */
        av_packet_rescale_ts(
            packet,
            encoder_ctx->time_base,
            output_stream->time_base
        );


        /*
         * The MP4 muxer needs the packet duration.
         *
         * For constant-frame-rate video:
         *
         *     duration = 1 / FPS
         *
         * expressed in the output stream time base.
         *
         * Example:
         *
         *     25 FPS
         *     time base = 1/12800
         *
         *     duration = 12800 / 25
         *              = 512 ticks
         */
        if (packet->duration <= 0 &&
            encoder_ctx->framerate.num > 0 &&
            encoder_ctx->framerate.den > 0)
        {
            packet->duration =
                av_rescale_q(
                    1,
                    av_inv_q(
                        encoder_ctx->framerate
                    ),
                    output_stream->time_base
                );
        }


        packet->stream_index =
            output_stream->index;


        /*
         * Write encoded packet to the output
         * container.
         */
        ret = av_interleaved_write_frame(
            output_format_ctx,
            packet
        );


        av_packet_free(&packet);


        if (ret < 0)
        {
            fprintf(
                stderr,
                "Error writing output packet\n"
            );

            return ret;
        }
    }

    return 0;
}

#endif /* COMMON_H */