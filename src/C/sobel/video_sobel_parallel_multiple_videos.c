#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <libavutil/opt.h>

#include "../../common/common.h"

#define DEFAULT_VIDEO_THREADS 2
#define DEFAULT_SOBEL_THREADS 4

/* =========================================================
 * Sobel thread data
 * ========================================================= */

typedef struct
{
    const uint8_t *input;
    uint8_t *output;

    int width;
    int height;

    int input_stride;
    int output_stride;

    int start_y;
    int end_y;

} SobelThreadData;


/* =========================================================
 * Shared Sobel job
 *
 * One decoded video frame becomes one SobelJob.
 *
 * The job is split into multiple row-band tasks.
 * Each task is processed by one worker from the shared
 * persistent Sobel thread pool.
 * ========================================================= */

typedef struct SobelJob
{
    pthread_mutex_t mutex;
    pthread_cond_t done;

    int remaining_tasks;

} SobelJob;


/* =========================================================
 * Sobel task
 *
 * Tasks from ALL videos share the same queue.
 * ========================================================= */

typedef struct SobelTask
{
    SobelJob *job;

    const uint8_t *input;
    uint8_t *output;

    int width;
    int height;

    int input_stride;
    int output_stride;

    int start_y;
    int end_y;

    struct SobelTask *next;

} SobelTask;


/* =========================================================
 * Persistent shared Sobel thread pool
 *
 * IMPORTANT:
 *
 * sobel_threads is the TOTAL number of Sobel worker
 * threads shared by ALL concurrently processed videos.
 * ========================================================= */

typedef struct
{
    pthread_t *threads;

    int num_threads;

    pthread_mutex_t mutex;
    pthread_cond_t job_available;

    SobelTask *job_head;
    SobelTask *job_tail;

    int stop;

} SobelPool;


/* =========================================================
 * Sobel worker
 * ========================================================= */

static void *sobel_worker(void *arg)
{
    SobelThreadData *data =
        (SobelThreadData *)arg;

    const uint8_t *input =
        data->input;

    uint8_t *output =
        data->output;

    int width =
        data->width;

    int input_stride =
        data->input_stride;

    int output_stride =
        data->output_stride;

    int start_y =
        data->start_y;

    int end_y =
        data->end_y;


    /*
     * Process assigned rows.
     *
     * We skip the first and last image rows because
     * Sobel requires a 3x3 neighborhood.
     */

    for (int y = start_y;
         y < end_y;
         y++)
    {
        const uint8_t *row0 =
            input + (y - 1) * input_stride;

        const uint8_t *row1 =
            input + y * input_stride;

        const uint8_t *row2 =
            input + (y + 1) * input_stride;

        uint8_t *out =
            output + y * output_stride;


        /*
         * First and last pixels don't have a complete
         * 3x3 neighborhood.
         */

        out[0] = 0;
        out[width - 1] = 0;


        for (int x = 1;
             x < width - 1;
             x++)
        {
            /*
             * Horizontal gradient.
             */

            int gx =
                -row0[x - 1]
                + row0[x + 1]
                - 2 * row1[x - 1]
                + 2 * row1[x + 1]
                - row2[x - 1]
                + row2[x + 1];


            /*
             * Vertical gradient.
             */

            int gy =
                -row0[x - 1]
                - 2 * row0[x]
                - row0[x + 1]
                + row2[x - 1]
                + 2 * row2[x]
                + row2[x + 1];


            /*
             * Approximate gradient magnitude.
             */

            int magnitude =
                abs(gx) + abs(gy);


            /*
             * Clamp to 8-bit.
             */

            if (magnitude > 255)
                magnitude = 255;


            out[x] =
                (uint8_t)magnitude;
        }
    }

    return NULL;
}


/* =========================================================
 * Persistent shared Sobel pool worker
 * ========================================================= */

static void *sobel_pool_worker(void *arg)
{
    SobelPool *pool =
        (SobelPool *)arg;


    while (1)
    {
        SobelTask *task = NULL;


        /*
         * Get one task from the shared queue.
         */

        pthread_mutex_lock(&pool->mutex);


        while (!pool->stop &&
               pool->job_head == NULL)
        {
            pthread_cond_wait(
                &pool->job_available,
                &pool->mutex
            );
        }


        /*
         * Stop only after all queued tasks have
         * been consumed.
         */

        if (pool->stop &&
            pool->job_head == NULL)
        {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }


        /*
         * Remove task from queue.
         */

        task =
            pool->job_head;

        pool->job_head =
            task->next;


        if (pool->job_head == NULL)
        {
            pool->job_tail = NULL;
        }


        pthread_mutex_unlock(&pool->mutex);


        /*
         * Process this task.
         */

        SobelThreadData data;

        data.input =
            task->input;

        data.output =
            task->output;

        data.width =
            task->width;

        data.height =
            task->height;

        data.input_stride =
            task->input_stride;

        data.output_stride =
            task->output_stride;

        data.start_y =
            task->start_y;

        data.end_y =
            task->end_y;


        sobel_worker(&data);


        /*
         * Mark this task complete.
         */

        SobelJob *job =
            task->job;


        pthread_mutex_lock(&job->mutex);

        job->remaining_tasks--;

        if (job->remaining_tasks == 0)
        {
            pthread_cond_signal(
                &job->done
            );
        }

        pthread_mutex_unlock(&job->mutex);
    }


    return NULL;
}


/* =========================================================
 * Initialize shared persistent Sobel pool
 * ========================================================= */

static int sobel_pool_init(
    SobelPool *pool,
    int num_threads)
{
    memset(
        pool,
        0,
        sizeof(*pool)
    );


    if (num_threads < 1)
        num_threads = 1;


    pool->num_threads =
        num_threads;


    pool->threads =
        calloc(
            (size_t)num_threads,
            sizeof(pthread_t)
        );


    if (!pool->threads)
    {
        return -1;
    }


    if (pthread_mutex_init(
            &pool->mutex,
            NULL) != 0)
    {
        free(pool->threads);
        pool->threads = NULL;

        return -1;
    }


    if (pthread_cond_init(
            &pool->job_available,
            NULL) != 0)
    {
        pthread_mutex_destroy(
            &pool->mutex
        );

        free(pool->threads);
        pool->threads = NULL;

        return -1;
    }


    pool->job_head = NULL;
    pool->job_tail = NULL;
    pool->stop = 0;


    /*
     * Create persistent workers.
     *
     * These threads live for the entire program run.
     */

    for (int i = 0;
         i < num_threads;
         i++)
    {
        int ret =
            pthread_create(
                &pool->threads[i],
                NULL,
                sobel_pool_worker,
                pool
            );


        if (ret != 0)
        {
            /*
               Do clean up, as pthread_create failed.
             * Stop workers that were already created.
             */

            pthread_mutex_lock(
                &pool->mutex
            );

            pool->stop = 1;

            pthread_cond_broadcast(
                &pool->job_available
            );

            pthread_mutex_unlock(
                &pool->mutex
            );


            for (int j = 0;
                 j < i;
                 j++)
            {
                pthread_join(
                    pool->threads[j],
                    NULL
                );
            }


            pthread_cond_destroy(
                &pool->job_available
            );

            pthread_mutex_destroy(
                &pool->mutex
            );

            free(pool->threads);

            memset(
                pool,
                0,
                sizeof(*pool)
            );

            return -1;
        }
    }


    return 0;
}


/* =========================================================
 * Single-threaded Sobel
 *
 * Used when:
 *
 *     sobel_threads == 1
 *
 * No additional Sobel pthread is created.
 * The video worker itself performs Sobel.
 * ========================================================= */

static void sobel_single_threaded(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height,
    int input_stride,
    int output_stride)
{
    if (width < 3 || height < 3)
    {
        memset(
            output,
            0,
            (size_t)height * output_stride
        );

        return;
    }


    /*
     * Top and bottom borders.
     */

    memset(
        output,
        0,
        output_stride
    );

    memset(
        output + (height - 1) * output_stride,
        0,
        output_stride
    );


    for (int y = 1;
         y < height - 1;
         y++)
    {
        const uint8_t *row0 =
            input + (y - 1) * input_stride;

        const uint8_t *row1 =
            input + y * input_stride;

        const uint8_t *row2 =
            input + (y + 1) * input_stride;

        uint8_t *out =
            output + y * output_stride;


        /*
         * Left and right borders.
         */

        out[0] = 0;
        out[width - 1] = 0;


        for (int x = 1;
             x < width - 1;
             x++)
        {
            int gx =
                -row0[x - 1]
                + row0[x + 1]
                - 2 * row1[x - 1]
                + 2 * row1[x + 1]
                - row2[x - 1]
                + row2[x + 1];


            int gy =
                -row0[x - 1]
                - 2 * row0[x]
                - row0[x + 1]
                + row2[x - 1]
                + 2 * row2[x]
                + row2[x + 1];


            int magnitude =
                abs(gx) + abs(gy);


            if (magnitude > 255)
                magnitude = 255;


            out[x] =
                (uint8_t)magnitude;
        }
    }
}


/* =========================================================
 * Submit one frame to the shared Sobel pool
 *
 * IMPORTANT:
 *
 * One frame is split into multiple independent row-band
 * tasks.
 *
 * Example:
 *
 *     sobel_threads = 4
 *
 *     Frame
 *       |
 *       +-- Task 1 -> Sobel worker
 *       +-- Task 2 -> Sobel worker
 *       +-- Task 3 -> Sobel worker
 *       +-- Task 4 -> Sobel worker
 *
 * Tasks from other videos can also exist in the same queue.
 * ========================================================= */

static void sobel_pool_run(
    SobelPool *pool,
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height,
    int input_stride,
    int output_stride)
{
    /*
     * Handle very small images.
     */

    if (width < 3 || height < 3)
    {
        memset(
            output,
            0,
            (size_t)height * output_stride
        );

        return;
    }


    /*
     * Rows that can actually be processed.
     *
     * Row 0 and row height-1 are borders.
     */

    int usable_rows =
        height - 2;


    /*
     * Never create more tasks than usable rows.
     */

    int task_count =
        pool->num_threads;

    if (task_count > usable_rows)
        task_count = usable_rows;


    /*
     * Allocate tasks for this frame.
     *
     * They remain valid until all workers finish because
     * sobel_pool_run() waits before returning.
     */

    SobelTask *tasks =
        calloc(
            (size_t)task_count,
            sizeof(SobelTask)
        );


    if (!tasks)
    {
        /*
         * Extremely unlikely for normal 4K frames.
         *
         * Fall back to the calling video worker.
         */
        sobel_single_threaded(
            input,
            output,
            width,
            height,
            input_stride,
            output_stride
        );

        return;
    }


    SobelJob job;


    memset(
        &job,
        0,
        sizeof(job)
    );


    pthread_mutex_init(
        &job.mutex,
        NULL
    );

    pthread_cond_init(
        &job.done,
        NULL
    );


    job.remaining_tasks =
        task_count;


    /*
     * Clear top and bottom borders.
     */

    memset(
        output,
        0,
        output_stride
    );

    memset(
        output + (height - 1) * output_stride,
        0,
        output_stride
    );


    /*
     * Divide usable rows between tasks.
     */

    int base_rows =
        usable_rows / task_count;

    int extra_rows =
        usable_rows % task_count;


    int start_y = 1;


    /*
     * Build the tasks.
     */

    for (int i = 0;
         i < task_count;
         i++)
    {
        int rows =
            base_rows +
            (i < extra_rows ? 1 : 0);


        tasks[i].job =
            &job;

        tasks[i].input =
            input;

        tasks[i].output =
            output;

        tasks[i].width =
            width;

        tasks[i].height =
            height;

        tasks[i].input_stride =
            input_stride;

        tasks[i].output_stride =
            output_stride;

        tasks[i].start_y =
            start_y;

        tasks[i].end_y =
            start_y + rows;

        tasks[i].next =
            NULL;


        start_y += rows;
    }


    /*
     * Add ALL tasks belonging to this frame to the
     * shared persistent queue.
     *
     * We use one mutex acquisition for the entire batch.
     */

    pthread_mutex_lock(
        &pool->mutex
    );


    for (int i = 0;
         i < task_count;
         i++)
    {
        if (pool->job_tail != NULL)
        {
            pool->job_tail->next =
                &tasks[i];
        }
        else
        {
            pool->job_head =
                &tasks[i];
        }

        pool->job_tail =
            &tasks[i];
    }


    /*
     * Wake all Sobel workers because this frame has
     * multiple independent tasks.
     */

    pthread_cond_broadcast(
        &pool->job_available
    );


    pthread_mutex_unlock(
        &pool->mutex
    );


    /*
     * Wait until every task belonging to this frame
     * has completed.
     */

    pthread_mutex_lock(
        &job.mutex
    );


    while (job.remaining_tasks > 0)
    {
        pthread_cond_wait(
            &job.done,
            &job.mutex
        );
    }


    pthread_mutex_unlock(
        &job.mutex
    );


    /*
     * At this point no worker references:
     *
     *     job
     *     tasks
     *
     * anymore.
     *
     * Therefore it is safe to destroy/free them.
     */

    pthread_cond_destroy(
        &job.done
    );

    pthread_mutex_destroy(
        &job.mutex
    );


    free(tasks);
}


/* =========================================================
 * Destroy shared persistent Sobel pool
 * ========================================================= */

static void sobel_pool_destroy(
    SobelPool *pool)
{
    if (!pool ||
        !pool->threads)
    {
        return;
    }


    pthread_mutex_lock(
        &pool->mutex
    );


    /*
     * Workers will continue consuming any tasks already
     * queued before exiting.
     */

    pool->stop = 1;


    pthread_cond_broadcast(
        &pool->job_available
    );


    pthread_mutex_unlock(
        &pool->mutex
    );


    /*
     * Wait for every persistent Sobel worker.
     */

    for (int i = 0;
         i < pool->num_threads;
         i++)
    {
        pthread_join(
            pool->threads[i],
            NULL
        );
    }


    pthread_cond_destroy(
        &pool->job_available
    );

    pthread_mutex_destroy(
        &pool->mutex
    );


    free(
        pool->threads
    );


    memset(
        pool,
        0,
        sizeof(*pool)
    );
}


/* =========================================================
 * Process one decoded frame
 * ========================================================= */

static int process_frame(
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_format_ctx,
    AVStream *input_stream,
    AVStream *output_stream,
    uint8_t *gray,
    uint8_t *sobel_output,
    int width,
    int height,
    int64_t frame_count,
    double *total_sobel_ms,
    int sobel_threads,
    SobelPool *sobel_pool)
{
    int ret;


    /* -----------------------------------------------------
     * Determine frame PTS
     * ----------------------------------------------------- */

    int64_t frame_pts =
        decoded_frame->pts;


    if (frame_pts == AV_NOPTS_VALUE)
    {
        frame_pts =
            frame_count *
            av_rescale_q(
                1,
                av_inv_q(
                    encoder_ctx->framerate
                ),
                encoder_ctx->time_base
            );
    }
    else
    {
        frame_pts =
            av_rescale_q(
                frame_pts,
                input_stream->time_base,
                encoder_ctx->time_base
            );
    }


    /* -----------------------------------------------------
     * Determine frame duration
     * ----------------------------------------------------- */

    int64_t frame_duration =
        decoded_frame->duration;


    if (frame_duration <= 0)
    {
        frame_duration =
            av_rescale_q(
                1,
                av_inv_q(
                    encoder_ctx->framerate
                ),
                encoder_ctx->time_base
            );
    }
    else
    {
        frame_duration =
            av_rescale_q(
                frame_duration,
                input_stream->time_base,
                encoder_ctx->time_base
            );
    }


    /* -----------------------------------------------------
     * Copy Y plane
     * ----------------------------------------------------- */

    for (int y = 0;
         y < height;
         y++)
    {
        memcpy(
            gray + y * width,

            decoded_frame->data[0]
                + y * decoded_frame->linesize[0],

            width
        );
    }


    /* -----------------------------------------------------
     * Sobel processing
     * ----------------------------------------------------- */

    double sobel_start =
        now_ms();


    if (sobel_threads == 1)
    {
        /*
         * No Sobel pthread pool.
         *
         * The video worker itself performs Sobel.
         */

        sobel_single_threaded(
            gray,
            sobel_output,
            width,
            height,
            width,
            width
        );
    }
    else
    {
        /*
         * Submit this frame to the ONE shared persistent
         * Sobel pool.
         */

        sobel_pool_run(
            sobel_pool,
            gray,
            sobel_output,
            width,
            height,
            width,
            width
        );
    }


    double sobel_end =
        now_ms();


    *total_sobel_ms +=
        sobel_end - sobel_start;


    /* -----------------------------------------------------
     * Make output frame writable
     * ----------------------------------------------------- */

    ret =
        av_frame_make_writable(
            output_frame
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Output frame is not writable\n"
        );

        return ret;
    }


    /* -----------------------------------------------------
     * Set timestamp and duration
     * ----------------------------------------------------- */

    output_frame->pts =
        frame_pts;

    output_frame->duration =
        frame_duration;


    /* -----------------------------------------------------
     * Copy Sobel result into Y plane
     * ----------------------------------------------------- */

    for (int y = 0;
         y < height;
         y++)
    {
        memcpy(
            output_frame->data[0]
                + y * output_frame->linesize[0],

            sobel_output + y * width,

            width
        );
    }


    /* -----------------------------------------------------
     * Set U/V to neutral chroma.
     *
     * U = 128
     * V = 128
     *
     * This produces grayscale output.
     * ----------------------------------------------------- */

    for (int y = 0;
         y < height / 2;
         y++)
    {
        memset(
            output_frame->data[1]
                + y * output_frame->linesize[1],

            128,

            width / 2
        );


        memset(
            output_frame->data[2]
                + y * output_frame->linesize[2],

            128,

            width / 2
        );
    }


    /* -----------------------------------------------------
     * Encode
     * ----------------------------------------------------- */

    ret =
        encode_frame(
            encoder_ctx,
            output_format_ctx,
            output_stream,
            output_frame
        );


    return ret;
}


/* =========================================================
 * CPU implementation arguments
 * ========================================================= */

typedef struct
{
    int sobel_threads;
    int ffmpeg_threads;

    /*
     * Shared persistent Sobel pool.
     *
     * NULL when sobel_threads == 1.
     */

    SobelPool *sobel_pool;

} CpuProcessArguments;


/* =========================================================
 * Process one complete video
 * ========================================================= */

static int process_video(
    const char *input_filename,
    const char *output_filename,
    int sobel_threads,
    int ffmpeg_threads,
    SobelPool *sobel_pool)
{
    int ret = 0;

    AVFormatContext *input_format_ctx = NULL;
    AVCodecContext *decoder_ctx = NULL;
    AVFormatContext *output_format_ctx = NULL;
    AVCodecContext *encoder_ctx = NULL;

    AVStream *input_stream = NULL;
    AVStream *output_stream = NULL;

    AVFrame *decoded_frame = NULL;
    AVFrame *output_frame = NULL;

    AVPacket *input_packet = NULL;

    uint8_t *gray = NULL;
    uint8_t *sobel_output = NULL;

    int video_stream_index = -1;
    int width = 0;
    int height = 0;

    int64_t frame_count = 0;

    double total_sobel_ms = 0.0;
    double total_processing_start = 0.0;
    double total_processing_end = 0.0;

    AVRational output_fps;


    /* =====================================================
     * Open input video
     *
     * Demuxer separates video/audio streams in input video.
     * ===================================================== */

    ret =
        avformat_open_input(
            &input_format_ctx,
            input_filename,
            NULL,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not open input video\n",
            input_filename
        );

        goto cleanup;
    }


    ret =
        avformat_find_stream_info(
            input_format_ctx,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not find stream information\n",
            input_filename
        );

        goto cleanup;
    }


    /* =====================================================
     * Find video stream
     * ===================================================== */

    video_stream_index =
        av_find_best_stream(
            input_format_ctx,
            AVMEDIA_TYPE_VIDEO,
            -1,
            -1,
            NULL,
            0
        );


    if (video_stream_index < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not find video stream\n",
            input_filename
        );

        ret =
            video_stream_index;

        goto cleanup;
    }


    input_stream =
        input_format_ctx->streams[
            video_stream_index
        ];


    /* =====================================================
     * Open decoder
     *
     * Decoder converts compressed video packets into raw
     * frames that Sobel can process.
     * ===================================================== */

    {
        const AVCodec *decoder =
            avcodec_find_decoder(
                input_stream->codecpar->codec_id
            );


        if (!decoder)
        {
            fprintf(
                stderr,
                "[%s] Could not find decoder\n",
                input_filename
            );

            ret =
                AVERROR_DECODER_NOT_FOUND;

            goto cleanup;
        }


        decoder_ctx =
            avcodec_alloc_context3(
                decoder
            );


        if (!decoder_ctx)
        {
            fprintf(
                stderr,
                "[%s] Could not allocate decoder context\n",
                input_filename
            );

            ret =
                AVERROR(ENOMEM);

            goto cleanup;
        }


        ret =
            avcodec_parameters_to_context(
                decoder_ctx,
                input_stream->codecpar
            );


        if (ret < 0)
        {
            fprintf(
                stderr,
                "[%s] Could not copy decoder parameters\n",
                input_filename
            );

            goto cleanup;
        }


        if (ffmpeg_threads > 0)
        {
            /*
             * ffmpeg_threads == 0:
             *
             * Do not override FFmpeg's codec threading
             * settings.
             *
             * ffmpeg_threads > 0:
             *
             * Request specified codec thread count.
             */

            decoder_ctx->thread_count =
                ffmpeg_threads;
        }


        ret =
            avcodec_open2(
                decoder_ctx,
                decoder,
                NULL
            );


        if (ret < 0)
        {
            fprintf(
                stderr,
                "[%s] Could not open decoder\n",
                input_filename
            );

            goto cleanup;
        }


        printf(
            "Decoder thread_count = %d\n",
            decoder_ctx->thread_count
        );
    }


    width =
        decoder_ctx->width;

    height =
        decoder_ctx->height;


    /* =====================================================
     * Check dimensions
     * ===================================================== */

    if ((width % 2) != 0 ||
        (height % 2) != 0)
    {
        fprintf(
            stderr,
            "[%s] Input dimensions must be even for YUV420P: %d x %d\n",
            input_filename,
            width,
            height
        );

        ret =
            AVERROR(EINVAL);

        goto cleanup;
    }


    /* =====================================================
     * Create output format
     * ===================================================== */

    ret =
        avformat_alloc_output_context2(
            &output_format_ctx,
            NULL,
            NULL,
            output_filename
        );


    if (ret < 0 ||
        !output_format_ctx)
    {
        fprintf(
            stderr,
            "[%s] Could not create output format context\n",
            input_filename
        );

        ret =
            AVERROR_UNKNOWN;

        goto cleanup;
    }


    /* =====================================================
     * Find H.264 encoder
     *
     * Prefer libx264.
     * Fall back to MPEG4.
     * ===================================================== */

    const AVCodec *encoder =
        avcodec_find_encoder_by_name(
            "libx264"
        );


    if (!encoder)
    {
        encoder =
            avcodec_find_encoder(
                AV_CODEC_ID_MPEG4
            );
    }


    if (!encoder)
    {
        fprintf(
            stderr,
            "[%s] Could not find a suitable video encoder\n",
            input_filename
        );

        ret =
            AVERROR_ENCODER_NOT_FOUND;

        goto cleanup;
    }


    /* =====================================================
     * Create output stream
     * ===================================================== */

    output_stream =
        avformat_new_stream(
            output_format_ctx,
            NULL
        );


    if (!output_stream)
    {
        fprintf(
            stderr,
            "[%s] Could not create output stream\n",
            input_filename
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* =====================================================
     * Allocate encoder
     * ===================================================== */

    encoder_ctx =
        avcodec_alloc_context3(
            encoder
        );


    if (!encoder_ctx)
    {
        fprintf(
            stderr,
            "[%s] Could not allocate encoder context\n",
            input_filename
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    encoder_ctx->codec_id =
        encoder->id;

    encoder_ctx->codec_type =
        AVMEDIA_TYPE_VIDEO;

    encoder_ctx->width =
        width;

    encoder_ctx->height =
        height;

    encoder_ctx->pix_fmt =
        AV_PIX_FMT_YUV420P;


    /* =====================================================
     * Preserve source FPS
     * ===================================================== */

    output_fps =
        input_stream->avg_frame_rate;


    if (output_fps.num <= 0 ||
        output_fps.den <= 0)
    {
        output_fps =
            input_stream->r_frame_rate;
    }


    if (output_fps.num <= 0 ||
        output_fps.den <= 0)
    {
        output_fps =
            (AVRational){25, 1};
    }


    encoder_ctx->time_base =
        input_stream->time_base;

    encoder_ctx->framerate =
        output_fps;

    output_stream->time_base =
        input_stream->time_base;

    output_stream->avg_frame_rate =
        output_fps;


    /* =====================================================
     * Encoder settings
     * ===================================================== */

    if (encoder->id ==
        AV_CODEC_ID_H264)
    {
        av_opt_set(
            encoder_ctx->priv_data,
            "preset",
            "medium",
            0
        );

        av_opt_set(
            encoder_ctx->priv_data,
            "crf",
            "23",
            0

        );

        /*
         * Leave encoder threading controlled by
         * encoder_ctx->thread_count.
         */
    }
    else
    {
        encoder_ctx->bit_rate =
            4 * 1000 * 1000;
    }


    encoder_ctx->gop_size = 50;
    encoder_ctx->max_b_frames = 0;


    if (output_format_ctx->oformat->flags &
        AVFMT_GLOBALHEADER)
    {
        encoder_ctx->flags |=
            AV_CODEC_FLAG_GLOBAL_HEADER;
    }


    /* =====================================================
     * Configure encoder threads
     * ===================================================== */

    if (ffmpeg_threads > 0)
    {
        encoder_ctx->thread_count =
            ffmpeg_threads;
    }


    /* =====================================================
     * Open encoder
     * ===================================================== */

    ret =
        avcodec_open2(
            encoder_ctx,
            encoder,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not open encoder\n",
            input_filename
        );

        goto cleanup;
    }


    printf(
        "Encoder thread_count = %d\n",
        encoder_ctx->thread_count
    );


    /* =====================================================
     * Copy encoder parameters
     * ===================================================== */

    ret =
        avcodec_parameters_from_context(
            output_stream->codecpar,
            encoder_ctx
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not copy encoder parameters\n",
            input_filename
        );

        goto cleanup;
    }


    /* =====================================================
     * Open output file
     * ===================================================== */

    if (!(output_format_ctx->oformat->flags &
          AVFMT_NOFILE))
    {
        ret =
            avio_open(
                &output_format_ctx->pb,
                output_filename,
                AVIO_FLAG_WRITE
            );


        if (ret < 0)
        {
            fprintf(
                stderr,
                "[%s] Could not open output file: %s\n",
                input_filename,
                output_filename
            );

            goto cleanup;
        }
    }


    /* =====================================================
     * Write MP4 header
     * ===================================================== */

    ret =
        avformat_write_header(
            output_format_ctx,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not write output header\n",
            input_filename
        );

        goto cleanup;
    }


    printf(
        "[%s] Final output time base: %d/%d\n",
        input_filename,
        output_stream->time_base.num,
        output_stream->time_base.den
    );


    /* =====================================================
     * Allocate frames
     * ===================================================== */

    decoded_frame =
        av_frame_alloc();

    output_frame =
        av_frame_alloc();


    if (!decoded_frame ||
        !output_frame)
    {
        fprintf(
            stderr,
            "[%s] Could not allocate frames\n",
            input_filename
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* =====================================================
     * Allocate Sobel buffers
     * ===================================================== */

    gray =
        malloc(
            (size_t)width * height
        );

    sobel_output =
        malloc(
            (size_t)width * height
        );


    if (!gray ||
        !sobel_output)
    {
        fprintf(
            stderr,
            "[%s] Could not allocate image buffers\n",
            input_filename
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* =====================================================
     * Allocate output YUV420P frame
     * ===================================================== */

    output_frame->format =
        AV_PIX_FMT_YUV420P;

    output_frame->width =
        width;

    output_frame->height =
        height;


    ret =
        av_frame_get_buffer(
            output_frame,
            32
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not allocate output frame buffer\n",
            input_filename
        );

        goto cleanup;
    }


    /* =====================================================
     * Allocate input packet
     * ===================================================== */

    input_packet =
        av_packet_alloc();


    if (!input_packet)
    {
        fprintf(
            stderr,
            "[%s] Could not allocate input packet\n",
            input_filename
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    printf("\n");
    printf(
        "[%s] Processing...\n",
        input_filename
    );

    printf(
        "[%s] Output: %s\n",
        input_filename,
        output_filename
    );

    printf("\n");

    fflush(stdout);


    total_processing_start =
        now_ms();


    /* =====================================================
     * Decode loop
     * ===================================================== */

    while (
        av_read_frame(
            input_format_ctx,
            input_packet
        ) >= 0)
    {
        if (input_packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(
                input_packet
            );

            continue;
        }


        ret =
            avcodec_send_packet(
                decoder_ctx,
                input_packet
            );


        av_packet_unref(
            input_packet
        );


        if (ret < 0)
        {
            fprintf(
                stderr,
                "[%s] Error sending packet to decoder\n",
                input_filename
            );

            goto cleanup;
        }


        while (1)
        {
            ret =
                avcodec_receive_frame(
                    decoder_ctx,
                    decoded_frame
                );


            if (ret == AVERROR(EAGAIN) ||
                ret == AVERROR_EOF)
            {
                break;
            }


            if (ret < 0)
            {
                fprintf(
                    stderr,
                    "[%s] Error decoding frame\n",
                    input_filename
                );

                goto cleanup;
            }


            ret =
                process_frame(
                    decoded_frame,
                    output_frame,
                    encoder_ctx,
                    output_format_ctx,
                    input_stream,
                    output_stream,
                    gray,
                    sobel_output,
                    width,
                    height,
                    frame_count,
                    &total_sobel_ms,
                    sobel_threads,
                    sobel_pool
                );


            if (ret < 0)
                goto cleanup;


            frame_count++;


            if (frame_count % 25 == 0)
                fflush(stdout);
        }
    }


    /* =====================================================
     * Flush decoder
     * ===================================================== */

    ret =
        avcodec_send_packet(
            decoder_ctx,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Error flushing decoder\n",
            input_filename
        );

        goto cleanup;
    }


    while (1)
    {
        ret =
            avcodec_receive_frame(
                decoder_ctx,
                decoded_frame
            );


        if (ret == AVERROR_EOF ||
            ret == AVERROR(EAGAIN))
        {
            break;
        }


        if (ret < 0)
        {
            fprintf(
                stderr,
                "[%s] Error receiving flushed frame\n",
                input_filename
            );

            goto cleanup;
        }


        ret =
            process_frame(
                decoded_frame,
                output_frame,
                encoder_ctx,
                output_format_ctx,
                input_stream,
                output_stream,
                gray,
                sobel_output,
                width,
                height,
                frame_count,
                &total_sobel_ms,
                sobel_threads,
                sobel_pool
            );


        if (ret < 0)
            goto cleanup;


        frame_count++;
    }


    /* =====================================================
     * Flush encoder
     * ===================================================== */

    ret =
        encode_frame(
            encoder_ctx,
            output_format_ctx,
            output_stream,
            NULL
        );


    if (ret < 0)
        goto cleanup;


    /* =====================================================
     * Write trailer
     * ===================================================== */

    ret =
        av_write_trailer(
            output_format_ctx
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[%s] Could not write output trailer\n",
            input_filename
        );

        goto cleanup;
    }


    total_processing_end =
        now_ms();


    /* =====================================================
     * Statistics
     * ===================================================== */

    printf("\n\n");

    printf(
        "========================================\n"
    );

    printf(
        "Processing complete\n"
    );

    printf(
        "========================================\n"
    );


    printf(
        "Input: %s\n",
        input_filename
    );

    printf(
        "Output: %s\n",
        output_filename
    );

    printf(
        "Frames processed: %lld\n",
        (long long)frame_count
    );

    printf(
        "Resolution: %d x %d\n",
        width,
        height
    );

    printf(
        "Frame rate: %d/%d\n",
        output_fps.num,
        output_fps.den
    );

    printf(
        "Time base: %d/%d\n",
        output_stream->time_base.num,
        output_stream->time_base.den
    );


    printf(
        "Total processing time: %.2f seconds\n",
        (total_processing_end -
         total_processing_start) / 1000.0
    );


    if (frame_count > 0 &&
        total_sobel_ms > 0.0)
    {
        printf(
            "Average Sobel time: %.3f ms/frame\n",
            total_sobel_ms / frame_count
        );


        printf(
            "Sobel processing FPS: %.2f\n",
            1000.0 /
            (total_sobel_ms / frame_count)
        );
    }


    printf(
        "========================================\n"
    );


    printf(
        "[DONE] %s\n",
        input_filename
    );


    printf("\n");

    fflush(stdout);


cleanup:

    /*
     * IMPORTANT:
     *
     * The Sobel pool is NOT destroyed here.
     *
     * It belongs to main() and is shared by all videos.
     */

    if (input_packet)
        av_packet_free(
            &input_packet
        );


    if (decoded_frame)
        av_frame_free(
            &decoded_frame
        );


    if (output_frame)
        av_frame_free(
            &output_frame
        );


    free(gray);
    free(sobel_output);


    if (decoder_ctx)
        avcodec_free_context(
            &decoder_ctx
        );


    if (encoder_ctx)
        avcodec_free_context(
            &encoder_ctx
        );


    if (input_format_ctx)
        avformat_close_input(
            &input_format_ctx
        );


    if (output_format_ctx)
    {
        if (output_format_ctx->pb &&
            !(output_format_ctx->oformat->flags &
              AVFMT_NOFILE))
        {
            avio_closep(
                &output_format_ctx->pb
            );
        }


        avformat_free_context(
            output_format_ctx
        );
    }


    if (ret < 0)
    {
        fprintf(
            stderr,
            "[FAILED] %s\n",
            input_filename
        );

        return 1;
    }


    return 0;
}


/* =========================================================
 * CPU video worker entry point
 * ========================================================= */

int process_video_cpu(void *arg)
{
    ProcessVideoContext *context =
        arg;


    CpuProcessArguments *cpu_args =
        context->implementation_args;


    return process_video(
        context->input_path,
        context->output_path,
        cpu_args->sobel_threads,
        cpu_args->ffmpeg_threads,
        cpu_args->sobel_pool
    );
}


/**
===============================================================================
MAIN
===============================================================================
 */

int main(int argc, char *argv[])
{
    double total_wall_start = now_ms();


    if (argc != 6)
    {
        fprintf(
            stderr,
            "Usage:\n"
            "  %s input_directory output_directory video_threads sobel_threads ffmpeg_threads\n\n"

            "Example:\n"
            "  %s resources/input/mp4/4k resources/output/mp4/4k 2 4 0\n\n"

            "Arguments:\n"
            "  input_directory   Directory containing input .mp4 files\n"
            "  output_directory  Directory where processed videos are written\n"
            "  video_threads     Number of videos processed concurrently\n"
            "  sobel_threads     TOTAL number of persistent Sobel worker threads shared by all videos\n"
            "                    1 = Sobel runs directly inside each video worker\n"
            "  ffmpeg_threads    Number of FFmpeg threads used internally to encode/decode.\n"
            "                    0 = use FFmpeg default thread count.\n"
            "                    >0 = use the specified number of threads.\n",

            argv[0],
            argv[0]
        );
        return EXIT_FAILURE;
    }


    const char *input_directory =
        argv[1];

    const char *output_directory =
        argv[2];

    int video_threads =
        atoi(argv[3]);

    int sobel_threads =
        atoi(argv[4]);

    int ffmpeg_threads =
        atoi(argv[5]);


    if (video_threads < 1 ||
        sobel_threads < 1 ||
        ffmpeg_threads < 0)
    {
        fprintf(
            stderr,
            "video and sobel thread counts must be >= 1. "
            "ffmpeg thread count must be >= 0\n"
        );

        return 1;
    }


    /* --------------------------------------------------------
     * CHECK OUTPUT DIRECTORY
     * -------------------------------------------------------- */

    if (!directory_exists(
            output_directory))
    {
        fprintf(
            stderr,
            "Output directory does not exist: %s\n",
            output_directory
        );

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Scan input directory
     * -------------------------------------------------------- */

    char **input_files =
        (char **)malloc(
            16 * sizeof(char *)
        );

    int file_count = 0;


    if (input_files == NULL)
    {
        fprintf(
            stderr,
            "Malloc failed. Failed to read input dir.\n"
        );

        return EXIT_FAILURE;
    }


    if (read_input_files(
            input_directory,
            &input_files,
            &file_count
        ) != EXIT_SUCCESS)
    {
        fprintf(
            stderr,
            "Failed to read input dir.\n"
        );

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Check whether any videos were found
     * -------------------------------------------------------- */

    if (file_count == 0)
    {
        fprintf(
            stderr,
            "No .mp4 files found in '%s'\n",
            input_directory
        );

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Don't create more video workers than videos.
     * -------------------------------------------------------- */

    if (video_threads > file_count)
        video_threads = file_count;


    /* ========================================================
     * INITIALIZE SHARED SOBEL POOL
     *
     * This happens ONCE for the entire program.
     *
     * It is NOT created once per video.
     * ======================================================== */

    SobelPool sobel_pool;

    int sobel_pool_initialized = 0;


    memset(
        &sobel_pool,
        0,
        sizeof(sobel_pool)
    );


    if (sobel_threads > 1)
    {
        if (sobel_pool_init(
                &sobel_pool,
                sobel_threads
            ) != 0)
        {
            fprintf(
                stderr,
                "ERROR: Failed to initialize shared Sobel pool\n"
            );

            return EXIT_FAILURE;
        }


        sobel_pool_initialized = 1;
    }


    /* ========================================================
     * STATUS
     * ======================================================== */

    printf("\n");

    printf(
        "=============================================\n"
    );

    printf(
        "        CPU VIDEO SOBEL PROCESSOR\n"
    );

    printf(
        "=============================================\n"
    );


    printf(
        "Input directory : %s\n",
        input_directory
    );

    printf(
        "Output directory: %s\n",
        output_directory
    );

    printf(
        "Videos found    : %d\n",
        file_count
    );

    printf(
        "Video threads   : %d\n",
        video_threads
    );

    printf(
        "Sobel threads   : %d\n",
        sobel_threads
    );


    if (sobel_threads == 1)
    {
        printf(
            "Sobel mode      : direct in video worker\n"
        );
    }
    else
    {
        printf(
            "Sobel mode      : shared persistent pool\n"
        );
    }


    printf(
        "FFmpeg threads  : %d\n",
        ffmpeg_threads
    );


    printf(
        "=============================================\n\n"
    );


    /* ========================================================
     * VIDEO WORKER MUTEX
     * ======================================================== */

    pthread_mutex_t mutex;


    if (pthread_mutex_init(
            &mutex,
            NULL
        ) != 0)
    {
        fprintf(
            stderr,
            "ERROR: pthread_mutex_init failed\n"
        );


        if (sobel_pool_initialized)
        {
            sobel_pool_destroy(
                &sobel_pool
            );
        }


        return EXIT_FAILURE;
    }


    /* ========================================================
     * Allocate video worker threads
     * ======================================================== */

    pthread_t *threads =
        (pthread_t *)malloc(
            (size_t)video_threads *
            sizeof(pthread_t)
        );


    if (threads == NULL)
    {
        fprintf(
            stderr,
            "ERROR: malloc failed\n"
        );


        pthread_mutex_destroy(
            &mutex
        );


        if (sobel_pool_initialized)
        {
            sobel_pool_destroy(
                &sobel_pool
            );
        }


        return EXIT_FAILURE;
    }


    /* ========================================================
     * Video worker data
     * ======================================================== */

    VideoWorkerData worker_data;


    worker_data.input_files =
        input_files;

    worker_data.file_count =
        file_count;

    worker_data.next_file =
        0;

    worker_data.mutex =
        &mutex;

    worker_data.output_directory =
        output_directory;

    worker_data.failed_count =
        0;


    /* ========================================================
     * Worker argument
     * ======================================================== */

    VideoWorkerArgument worker_argument;


    worker_argument.data =
        &worker_data;


    worker_argument.process_video =
        process_video_cpu;


    /* ========================================================
     * CPU implementation arguments
     * ======================================================== */

    CpuProcessArguments cpu_arguments;


    cpu_arguments.sobel_threads =
        sobel_threads;

    cpu_arguments.ffmpeg_threads =
        ffmpeg_threads;


    /*
     * All video workers receive the SAME SobelPool pointer.
     */

    cpu_arguments.sobel_pool =
        sobel_pool_initialized
            ? &sobel_pool
            : NULL;


    worker_argument.process_arguments =
        &cpu_arguments;


    /* ========================================================
     * Create video worker threads
     * ======================================================== */

    int created_threads =
        create_video_workers(
            threads,
            video_threads,
            &worker_argument,
            video_worker
        );


    printf(
        "Waiting for video workers...\n"
    );


    /* ========================================================
     * Join video workers
     * ======================================================== */

    for (int i = 0;
         i < created_threads;
         i++)
    {
        printf(
            "[MAIN] Joining video worker %d...\n",
            i
        );

        fflush(stdout);


        int result =
            pthread_join(
                threads[i],
                NULL
            );


        if (result != 0)
        {
            fprintf(
                stderr,
                "ERROR: pthread_join failed for worker %d: %s\n",
                i,
                strerror(result)
            );
        }


        printf(
            "[MAIN] Joined video worker %d\n",
            i
        );

        fflush(stdout);


        printf(
            "[MAIN] Worker %d/%d joined\n",
            i + 1,
            created_threads
        );

        fflush(stdout);
    }


    printf(
        "[MAIN] ALL VIDEO WORKERS JOINED\n"
    );

    fflush(stdout);


    /* ========================================================
     * Destroy shared Sobel pool
     *
     * IMPORTANT:
     *
     * This happens ONLY after all video workers have finished.
     * ======================================================== */

    if (sobel_pool_initialized)
    {
        sobel_pool_destroy(
            &sobel_pool
        );

        sobel_pool_initialized = 0;
    }


    /* ========================================================
     * Final timing
     * ======================================================== */

    double total_wall_end =
        now_ms();


    double total_wall_elapsed =
        total_wall_end -
        total_wall_start;


    printf("\n");

    printf(
        "=============================================\n"
    );

    printf(
        "              FINAL RESULTS\n"
    );

    printf(
        "=============================================\n"
    );


    printf(
        "Total wall time    : %.3f sec (%.3f ms)\n",
        total_wall_elapsed / 1000.0,
        total_wall_elapsed
    );


    printf(
        "Videos processed   : %d\n",
        file_count
    );


    printf(
        "Video threads      : %d\n",
        video_threads
    );


    printf(
        "Sobel threads      : %d\n",
        sobel_threads
    );


    printf(
        "Failed videos      : %d\n",
        worker_data.failed_count
    );


    printf(
        "=============================================\n"
    );


    free(threads);

    pthread_mutex_destroy(
        &mutex
    );


    /*
     * Existing code intentionally leaves input_files
     * cleanup commented out.
     */

    //free_input_files(input_files, file_count);


    printf("\n");

    printf(
        "=============================================\n"
    );


    if (worker_data.failed_count == 0)
    {
        printf(
            "All videos processed successfully.\n"
        );
    }
    else
    {
        printf(
            "Processing completed with %d failed video(s).\n",
            worker_data.failed_count
        );
    }


    printf(
        "=============================================\n"
    );


    return
        (worker_data.failed_count == 0)
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
}