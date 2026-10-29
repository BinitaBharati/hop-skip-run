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

/* ---------------------------------------------------------
 * Persistent Sobel thread pool
 * --------------------------------------------------------- */

typedef struct
{
    pthread_t *threads;
    SobelThreadData *thread_data;

    int num_threads;

    pthread_mutex_t mutex;

    pthread_cond_t job_available;
    pthread_cond_t job_done;

    int job_generation;
    int completed_threads;

    int stop;

    /* Current frame job */
    const uint8_t *input;
    uint8_t *output;

    int width;
    int height;

    int input_stride;
    int output_stride;

} SobelPool;

/* =========================================================
 * Video worker data
 *
 * These workers process COMPLETE VIDEOS.
 *
 * Each video worker calls:
 *
 *     process_video()
 *
 * and process_video() internally creates Sobel threads.
 * ========================================================= */



/* =========================================================
 * Sobel worker thread
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

        out[0] =
            0;

        out[width - 1] =
            0;


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

static void *sobel_pool_worker(void *arg)
{
    SobelPool *pool = (SobelPool *)arg;

    int my_generation = 0;

    while (1)
    {
        pthread_mutex_lock(&pool->mutex);

        while (!pool->stop &&
               pool->job_generation == my_generation)
        {
            pthread_cond_wait(
                &pool->job_available,
                &pool->mutex
            );
        }

        if (pool->stop)
        {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        my_generation = pool->job_generation;

        /*
         * Copy the job parameters locally.
         *
         * The main thread will not modify them until
         * all workers have completed.
         */
        const uint8_t *input = pool->input;
        uint8_t *output = pool->output;

        int width = pool->width;
        int height = pool->height;

        int input_stride = pool->input_stride;
        int output_stride = pool->output_stride;

        int thread_index = -1;

        pthread_t self = pthread_self();

        for (int i = 0; i < pool->num_threads; i++)
        {
            if (pthread_equal(
                    pool->threads[i],
                    self))
            {
                thread_index = i;
                break;
            }
        }

        pthread_mutex_unlock(&pool->mutex);

        /*
         * Divide the image into horizontal bands.
         *
         * Only rows 1 .. height-2 are processed because
         * Sobel needs the rows above and below.
         */
        int usable_rows = height - 2;

        int base_rows =
            usable_rows / pool->num_threads;

        int extra_rows =
            usable_rows % pool->num_threads;

        int start_y = 1;

        for (int i = 0; i < thread_index; i++)
        {
            start_y +=
                base_rows +
                (i < extra_rows ? 1 : 0);
        }

        int rows =
            base_rows +
            (thread_index < extra_rows ? 1 : 0);

        int end_y =
            start_y + rows;

        SobelThreadData data;

        data.input = input;
        data.output = output;

        data.width = width;
        data.height = height;

        data.input_stride = input_stride;
        data.output_stride = output_stride;

        data.start_y = start_y;
        data.end_y = end_y;

        sobel_worker(&data);

        pthread_mutex_lock(&pool->mutex);

        pool->completed_threads++;

        if (pool->completed_threads ==
            pool->num_threads)
        {
            pthread_cond_signal(
                &pool->job_done
            );
        }

        pthread_mutex_unlock(&pool->mutex);
    }

    return NULL;
}

static int sobel_pool_init(
    SobelPool *pool,
    int num_threads)
{
    memset(pool, 0, sizeof(*pool));

    if (num_threads < 1)
        num_threads = 1;

    pool->num_threads = num_threads;

    pool->threads =
        calloc(
            num_threads,
            sizeof(pthread_t)
        );

    pool->thread_data =
        calloc(
            num_threads,
            sizeof(SobelThreadData)
        );

    if (!pool->threads ||
        !pool->thread_data)
    {
        free(pool->threads);
        free(pool->thread_data);

        return -1;
    }

    pthread_mutex_init(
        &pool->mutex,
        NULL
    );

    pthread_cond_init(
        &pool->job_available,
        NULL
    );

    pthread_cond_init(
        &pool->job_done,
        NULL
    );

    pool->job_generation = 0;
    pool->completed_threads = 0;
    pool->stop = 0;

    for (int i = 0; i < num_threads; i++)
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
             * Stop threads that were already created.
             */
            pthread_mutex_lock(&pool->mutex);

            pool->stop = 1;

            pthread_cond_broadcast(
                &pool->job_available
            );

            pthread_mutex_unlock(&pool->mutex);

            for (int j = 0; j < i; j++)
            {
                pthread_join(
                    pool->threads[j],
                    NULL
                );
            }

            pthread_cond_destroy(
                &pool->job_available
            );

            pthread_cond_destroy(
                &pool->job_done
            );

            pthread_mutex_destroy(
                &pool->mutex
            );

            free(pool->threads);
            free(pool->thread_data);

            memset(pool, 0, sizeof(*pool));

            return -1;
        }
    }

    return 0;
}

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
    if (height < 3 || width < 3)
    {
        memset(
            output,
            0,
            (size_t)height * output_stride
        );

        return;
    }

    pthread_mutex_lock(&pool->mutex);

    pool->input = input;
    pool->output = output;

    pool->width = width;
    pool->height = height;

    pool->input_stride = input_stride;
    pool->output_stride = output_stride;

    pool->completed_threads = 0;

    /*
     * Clear border rows.
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
     * New frame/job.
     */
    pool->job_generation++;

    pthread_cond_broadcast(
        &pool->job_available
    );

    /*
     * Wait until every persistent Sobel thread
     * finishes this frame.
     */
    while (pool->completed_threads <
           pool->num_threads)
    {
        pthread_cond_wait(
            &pool->job_done,
            &pool->mutex
        );
    }

    pthread_mutex_unlock(&pool->mutex);
}

static void sobel_pool_destroy(
    SobelPool *pool)
{
    if (!pool->threads)
        return;

    pthread_mutex_lock(&pool->mutex);

    pool->stop = 1;

    pthread_cond_broadcast(
        &pool->job_available
    );

    pthread_mutex_unlock(&pool->mutex);

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

    pthread_cond_destroy(
        &pool->job_done
    );

    pthread_mutex_destroy(
        &pool->mutex
    );

    free(pool->threads);
    free(pool->thread_data);

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


   sobel_pool_run(
    sobel_pool,
    gray,
    sobel_output,
    width,
    height,
    width,
    width
);

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

typedef struct
{
    int sobel_threads;
    int ffmpeg_threads;
} CpuProcessArguments;

/* =========================================================
 * Process one complete video
 * ========================================================= */

static int process_video(
    const char *input_filename,
    const char *output_filename,
    int sobel_threads,
    int ffmpeg_threads
)
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

    SobelPool sobel_pool;
    int sobel_pool_initialized = 0;

    memset(&sobel_pool, 0, sizeof(sobel_pool));

    /* =====================================================
     * Open input video
       Demuxer (seperates video, audio streams in input video)
     * ===================================================== */

    ret = avformat_open_input(&input_format_ctx, input_filename, NULL, NULL);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not open input video\n", input_filename);
        goto cleanup;
    }

    ret = avformat_find_stream_info(input_format_ctx, NULL);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not find stream information\n", input_filename);
        goto cleanup;
    }

    /* =====================================================
     * Find video stream
     * ===================================================== */

    video_stream_index = av_find_best_stream(
        input_format_ctx,
        AVMEDIA_TYPE_VIDEO,
        -1,
        -1,
        NULL,
        0
    );

    if (video_stream_index < 0)
    {
        fprintf(stderr, "[%s] Could not find video stream\n", input_filename);
        ret = video_stream_index;
        goto cleanup;
    }

    input_stream = input_format_ctx->streams[video_stream_index];

    /* =====================================================
     * Open decoder
       The decoder takes the compressed video packets produced by the demuxer and converts them into raw video frames that your Sobel code can process.
     * ===================================================== */

    {
        const AVCodec *decoder = avcodec_find_decoder(
            input_stream->codecpar->codec_id
        );

        if (!decoder)
        {
            fprintf(stderr, "[%s] Could not find decoder\n", input_filename);
            ret = AVERROR_DECODER_NOT_FOUND;
            goto cleanup;
        }

        decoder_ctx = avcodec_alloc_context3(decoder);

        if (!decoder_ctx)
        {
            fprintf(stderr, "[%s] Could not allocate decoder context\n", input_filename);
            ret = AVERROR(ENOMEM);
            goto cleanup;
        }

        ret = avcodec_parameters_to_context(
            decoder_ctx,
            input_stream->codecpar
        );

        if (ret < 0)
        {
            fprintf(stderr, "[%s] Could not copy decoder parameters\n", input_filename);
            goto cleanup;
        }


        if (ffmpeg_threads > 0) {
           /*
            * ffmpeg_threads == 0:
            *     Do not override FFmpeg's codec threading settings.
            *     FFmpeg/the codec chooses the default.
            *
            * ffmpeg_threads > 0:
            *     Request the specified codec thread count.
            */
            decoder_ctx->thread_count = ffmpeg_threads;
        }
                  
        ret = avcodec_open2(decoder_ctx, decoder, NULL);

        if (ret < 0)
        {
            fprintf(stderr, "[%s] Could not open decoder\n", input_filename);
            goto cleanup;
        }

        printf("Decoder thread_count = %d\n", decoder_ctx->thread_count);
    }

    width = decoder_ctx->width;
    height = decoder_ctx->height;

    /* =====================================================
     * Check dimensions
     * ===================================================== */

    if ((width % 2) != 0 || (height % 2) != 0)
    {
        fprintf(
            stderr,
            "[%s] Input dimensions must be even for YUV420P: %d x %d\n",
            input_filename,
            width,
            height
        );

        ret = AVERROR(EINVAL);
        goto cleanup;
    }

    /* =====================================================
     * Create output format
     * ===================================================== */

    ret = avformat_alloc_output_context2(
        &output_format_ctx,
        NULL,
        NULL,
        output_filename
    );

    if (ret < 0 || !output_format_ctx)
    {
        fprintf(stderr, "[%s] Could not create output format context\n", input_filename);
        ret = AVERROR_UNKNOWN;
        goto cleanup;
    }

    /* =====================================================
     * Find H.264 encoder
       The encoder converts raw video frames into compressed video data.
       Use the libx264 encoder to encode my processed raw frames into H.264."
       Why libx264 is good for your project
        Mature and stable
        Excellent H.264 compression quality
        Widely supported by browsers, macOS, Ubuntu, TVs, phones, etc.
        Good CPU performance
        Lots of configurable options such as CRF, preset, threading, bitrate, GOP, etc.
        Available in FFmpeg when FFmpeg was built with x264 support.

     * ===================================================== */

    const AVCodec *encoder = avcodec_find_encoder_by_name("libx264");

    if (!encoder)
        encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG4);

    if (!encoder)
    {
        fprintf(stderr, "[%s] Could not find a suitable video encoder\n", input_filename);
        ret = AVERROR_ENCODER_NOT_FOUND;
        goto cleanup;
    }

    /* =====================================================
     * Create output stream
     * ===================================================== */

    output_stream = avformat_new_stream(output_format_ctx, NULL);

    if (!output_stream)
    {
        fprintf(stderr, "[%s] Could not create output stream\n", input_filename);
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* =====================================================
     * Allocate encoder
     * ===================================================== */

    encoder_ctx = avcodec_alloc_context3(encoder);

    if (!encoder_ctx)
    {
        fprintf(stderr, "[%s] Could not allocate encoder context\n", input_filename);
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    encoder_ctx->codec_id = encoder->id;
    encoder_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    encoder_ctx->width = width;
    encoder_ctx->height = height;
    encoder_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    /* =====================================================
     * Preserve source FPS
     * ===================================================== */

    output_fps = input_stream->avg_frame_rate;

    if (output_fps.num <= 0 || output_fps.den <= 0)
        output_fps = input_stream->r_frame_rate;

    if (output_fps.num <= 0 || output_fps.den <= 0)
        output_fps = (AVRational){25, 1};

    encoder_ctx->time_base = input_stream->time_base;
    encoder_ctx->framerate = output_fps;

    output_stream->time_base = input_stream->time_base;
    output_stream->avg_frame_rate = output_fps;

    /* =====================================================
     * Encoder settings
     * ===================================================== */

    if (encoder->id == AV_CODEC_ID_H264)
    {
        av_opt_set(encoder_ctx->priv_data, "preset", "medium", 0);
        av_opt_set(encoder_ctx->priv_data, "crf", "23", 0);
        //av_opt_set(encoder_ctx->priv_data, "threads", "1", 0);
    }
    else
    {
        encoder_ctx->bit_rate = 4 * 1000 * 1000;
    }

    encoder_ctx->gop_size = 50;
    encoder_ctx->max_b_frames = 0;

    if (output_format_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        encoder_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    /* =====================================================
     * Open encoder
     * ===================================================== */
     if (ffmpeg_threads > 0) {
            encoder_ctx->thread_count = ffmpeg_threads;
     }


    ret = avcodec_open2(encoder_ctx, encoder, NULL);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not open encoder\n", input_filename);
        goto cleanup;
    }
    printf("Encoder thread_count = %d\n", encoder_ctx->thread_count);

    /* =====================================================
     * Copy encoder parameters
     * ===================================================== */

    ret = avcodec_parameters_from_context(
        output_stream->codecpar,
        encoder_ctx
    );

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not copy encoder parameters\n", input_filename);
        goto cleanup;
    }

    /* =====================================================
     * Open output file
     * ===================================================== */

    if (!(output_format_ctx->oformat->flags & AVFMT_NOFILE))
    {
        ret = avio_open(
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

    ret = avformat_write_header(output_format_ctx, NULL);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not write output header\n", input_filename);
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

    decoded_frame = av_frame_alloc();
    output_frame = av_frame_alloc();

    if (!decoded_frame || !output_frame)
    {
        fprintf(stderr, "[%s] Could not allocate frames\n", input_filename);
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* =====================================================
     * Initialize persistent Sobel thread pool
     * ===================================================== */

    ret = sobel_pool_init(&sobel_pool, sobel_threads);

    if (ret != 0)
    {
        fprintf(
            stderr,
            "[%s] Failed to initialize Sobel thread pool\n",
            input_filename
        );

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    sobel_pool_initialized = 1;

    /* =====================================================
     * Allocate Sobel buffers
     * ===================================================== */

    gray = malloc((size_t)width * height);
    sobel_output = malloc((size_t)width * height);

    if (!gray || !sobel_output)
    {
        fprintf(stderr, "[%s] Could not allocate image buffers\n", input_filename);
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* =====================================================
     * Allocate output YUV420P frame
     * ===================================================== */

    output_frame->format = AV_PIX_FMT_YUV420P;
    output_frame->width = width;
    output_frame->height = height;

    ret = av_frame_get_buffer(output_frame, 32);

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

    input_packet = av_packet_alloc();

    if (!input_packet)
    {
        fprintf(stderr, "[%s] Could not allocate input packet\n", input_filename);
        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    printf("\n");
    printf("[%s] Processing...\n", input_filename);
    printf("[%s] Output: %s\n", input_filename, output_filename);
    printf("\n");

    fflush(stdout);

    total_processing_start = now_ms();

    /* =====================================================
     * Decode loop
     * ===================================================== */

    while (av_read_frame(input_format_ctx, input_packet) >= 0)
    {
        if (input_packet->stream_index != video_stream_index)
        {
            av_packet_unref(input_packet);
            continue;
        }

        ret = avcodec_send_packet(decoder_ctx, input_packet);
        av_packet_unref(input_packet);

        if (ret < 0)
        {
            fprintf(stderr, "[%s] Error sending packet to decoder\n", input_filename);
            goto cleanup;
        }

        while (1)
        {
            ret = avcodec_receive_frame(decoder_ctx, decoded_frame);

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;

            if (ret < 0)
            {
                fprintf(stderr, "[%s] Error decoding frame\n", input_filename);
                goto cleanup;
            }

            ret = process_frame(
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
                &sobel_pool
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

    ret = avcodec_send_packet(decoder_ctx, NULL);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Error flushing decoder\n", input_filename);
        goto cleanup;
    }

    while (1)
    {
        ret = avcodec_receive_frame(decoder_ctx, decoded_frame);

        if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
            break;

        if (ret < 0)
        {
            fprintf(stderr, "[%s] Error receiving flushed frame\n", input_filename);
            goto cleanup;
        }

        ret = process_frame(
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
            &sobel_pool
        );

        if (ret < 0)
            goto cleanup;

        frame_count++;
    }

    /* =====================================================
     * Flush encoder
     * ===================================================== */

    ret = encode_frame(
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

    ret = av_write_trailer(output_format_ctx);

    if (ret < 0)
    {
        fprintf(stderr, "[%s] Could not write output trailer\n", input_filename);
        goto cleanup;
    }

    total_processing_end = now_ms();

    /* =====================================================
     * Statistics
     * ===================================================== */

    printf("\n\n");
    printf("========================================\n");
    printf("Processing complete\n");
    printf("========================================\n");

    printf("Input: %s\n", input_filename);
    printf("Output: %s\n", output_filename);
    printf("Frames processed: %lld\n", (long long)frame_count);
    printf("Resolution: %d x %d\n", width, height);
    printf("Frame rate: %d/%d\n", output_fps.num, output_fps.den);
    printf(
        "Time base: %d/%d\n",
        output_stream->time_base.num,
        output_stream->time_base.den
    );

    printf(
        "Total processing time: %.2f seconds\n",
        (total_processing_end - total_processing_start) / 1000.0
    );

    if (frame_count > 0 && total_sobel_ms > 0.0)
    {
        printf(
            "Average Sobel time: %.3f ms/frame\n",
            total_sobel_ms / frame_count
        );

        printf(
            "Sobel processing FPS: %.2f\n",
            1000.0 / (total_sobel_ms / frame_count)
        );
    }

    printf("========================================\n");
    printf("[DONE] %s\n", input_filename);
    printf("\n");

    fflush(stdout);

cleanup:

    /* =====================================================
     * Destroy Sobel pool first
     * ===================================================== */

    if (sobel_pool_initialized)
    {
        sobel_pool_destroy(&sobel_pool);
        sobel_pool_initialized = 0;
    }

    /* =====================================================
     * Free packet and frames
     * ===================================================== */

    if (input_packet)
        av_packet_free(&input_packet);

    if (decoded_frame)
        av_frame_free(&decoded_frame);

    if (output_frame)
        av_frame_free(&output_frame);

    /* =====================================================
     * Free Sobel buffers
     * ===================================================== */

    free(gray);
    free(sobel_output);

    /* =====================================================
     * Free codec contexts
     * ===================================================== */

    if (decoder_ctx)
        avcodec_free_context(&decoder_ctx);

    if (encoder_ctx)
        avcodec_free_context(&encoder_ctx);

    /* =====================================================
     * Close input
     * ===================================================== */

    if (input_format_ctx)
        avformat_close_input(&input_format_ctx);

    /* =====================================================
     * Close output
     * ===================================================== */

    if (output_format_ctx)
    {
        if (output_format_ctx->pb &&
            !(output_format_ctx->oformat->flags & AVFMT_NOFILE))
        {
            avio_closep(&output_format_ctx->pb);
        }

        avformat_free_context(output_format_ctx);
    }

    /* =====================================================
     * Final status
     * ===================================================== */

    if (ret < 0)
    {
        fprintf(stderr, "[FAILED] %s\n", input_filename);
        return 1;
    }

    return 0;
}

int process_video_cpu(void *arg)
{
    ProcessVideoContext *context = arg;

    CpuProcessArguments *cpu_args =
        context->implementation_args;

    return process_video(
        context->input_path,
        context->output_path,
        cpu_args->sobel_threads,
        cpu_args->ffmpeg_threads
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
            "  sobel_threads     Number of pthreads used by Sobel per video\n"
            "  ffmpeg_threads    Number of FFmpeg threads used internally to encode/decode.\n"
            "                    0 = use FFmpeg default thread count.\n"
            "                    >0 = use the specified number of threads.\n",
            argv[0],
            argv[0]
        );

        return EXIT_FAILURE;
    }

    const char *input_directory = argv[1];
    const char *output_directory = argv[2];
    int video_threads = atoi(argv[3]);
    int sobel_threads = atoi(argv[4]);
    int ffmpeg_threads = atoi(argv[5]);

    if (video_threads < 1 || sobel_threads < 1 || ffmpeg_threads < 0)
    {
        fprintf(stderr, "video and sobel thread counts must be >= 1.ffmpeg thread count must be >= 0\n");
        return 1;
    }

     /* --------------------------------------------------------
     * CHECK OUTPUT DIRECTORY
     * -------------------------------------------------------- */
    if (!directory_exists(output_directory))
    {
        fprintf(
            stderr,
            "Output directory does not exist: %s\n",
            output_directory);

        return EXIT_FAILURE;
    }

    /* Scan input directory. */
    char **input_files =(char **)malloc(16 * sizeof(char *));
    int file_count = 0;

    if (input_files == NULL)
    {
        fprintf(stderr,"Malloc failed. Failed to read input dir.\n ");
        return EXIT_FAILURE;
    }

    if (read_input_files(
            input_directory,
            &input_files,
            &file_count
        ) != EXIT_SUCCESS)
    {
        fprintf(stderr,"Malloc failed. Failed to read input dir.\n ");
        return EXIT_FAILURE;
    }

    /* Check whether any videos were found. */
    if (file_count == 0)
    {
        fprintf(
            stderr,
            "No .mp4 files found in '%s'\n",
            input_directory
        );

        return EXIT_FAILURE;
    }

    /* Don't create more video workers than videos. */
    if (video_threads > file_count)
        video_threads = file_count;

    printf("\n");
    printf("=============================================\n");
    printf("        CPU VIDEO SOBEL PROCESSOR\n");
    printf("=============================================\n");
    printf("Input directory : %s\n", input_directory);
    printf("Output directory: %s\n", output_directory);
    printf("Videos found    : %d\n", file_count);
    printf("Video threads   : %d\n", video_threads);
    printf("Sobel threads   : %d\n", sobel_threads);
    printf("ffmpreg threads   : %d\n", ffmpeg_threads);

    printf("=============================================\n\n");

    pthread_mutex_t mutex;

    if (pthread_mutex_init(&mutex, NULL) != 0)
    {
        fprintf(stderr, "ERROR: pthread_mutex_init failed\n");
        return EXIT_FAILURE;
    }

    pthread_t *threads =
    (pthread_t *)malloc(
        video_threads * sizeof(pthread_t));

    if (threads == NULL)
    {
        fprintf(stderr, "ERROR: malloc failed\n");
        return EXIT_FAILURE;
    }

    VideoWorkerData worker_data;

    worker_data.input_files = input_files;
    worker_data.file_count = file_count;
    worker_data.next_file = 0;
    worker_data.mutex = &mutex;
    worker_data.output_directory = output_directory;
    worker_data.failed_count = 0;

    VideoWorkerArgument worker_argument;
    worker_argument.data = &worker_data;

    worker_argument.process_video = process_video_cpu;

    CpuProcessArguments cpu_arguments;
    cpu_arguments.sobel_threads = sobel_threads;
    cpu_arguments.ffmpeg_threads = ffmpeg_threads;
    worker_argument.process_arguments = &cpu_arguments;

    //Create video worker threads.
    int created_threads = create_video_workers(threads, video_threads, &worker_argument, video_worker);

    printf("Waiting for video workers...\n");

    for (int i = 0; i < created_threads; i++)
    {
        printf("[MAIN] Joining video worker %d...\n", i);
        fflush(stdout);

        int result = pthread_join(threads[i], NULL);

        if (result != 0)
        {
            fprintf(
                stderr,
                "ERROR: pthread_join failed for worker %d: %s\n",
                i,
                strerror(result)
            );
        }

        printf("[MAIN] Joined video worker %d\n", i);
        fflush(stdout);

        printf(
            "[MAIN] Worker %d/%d joined\n",
            i + 1,
            created_threads
        );
        fflush(stdout);
    }

    printf("[MAIN] ALL VIDEO WORKERS JOINED\n");
    fflush(stdout);

    double total_wall_end = now_ms();
    double total_wall_elapsed =
        total_wall_end - total_wall_start;

    printf("\n");
    printf("=============================================\n");
    printf("              FINAL RESULTS\n");
    printf("=============================================\n");
    printf(
        "Total wall time    : %.3f sec (%.3f ms)\n",
        total_wall_elapsed / 1000.0,
        total_wall_elapsed
    );
    printf("Videos processed   : %d\n", file_count);
    printf("Video threads      : %d\n", video_threads);
    printf("Sobel threads      : %d\n", sobel_threads);
    printf("Failed videos      : %d\n", worker_data.failed_count);
    printf("=============================================\n");

    free(threads);
    pthread_mutex_destroy(&mutex);

    //free_input_files(input_files, file_count);

    printf("\n");
    printf("=============================================\n");

    if (worker_data.failed_count == 0)
    {
        printf("All videos processed successfully.\n");
    }
    else
    {
        printf("Processing completed with %d failed video(s).\n",worker_data.failed_count);
    }

    printf("=============================================\n");

    return (worker_data.failed_count == 0)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}