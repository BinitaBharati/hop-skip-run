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

#include "./common.h"

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

typedef struct
{
    char **input_files;
    int file_count;

    int *next_file;

    pthread_mutex_t *mutex;

    const char *output_directory;

    int sobel_threads;

    int *failed_count;

} VideoWorkerData;


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


/* =========================================================
 * Multithreaded Sobel filter
 * ========================================================= */

static void sobel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height,
    int input_stride,
    int output_stride,
    int num_threads)
{
    /*
     * Need at least 3 rows for Sobel.
     */

    if (height < 3)
    {
        memset(
            output,
            0,
            (size_t)height * output_stride
        );

        return;
    }


    /*
     * Don't create more threads than useful rows.
     */

    if (num_threads < 1)
        num_threads = 1;

    if (num_threads > height - 2)
        num_threads = height - 2;


    pthread_t threads[num_threads];

    SobelThreadData thread_data[num_threads];


    /*
     * Clear first row.
     */

    memset(
        output,
        0,
        output_stride
    );


    /*
     * Clear last row.
     */

    memset(
        output + (height - 1) * output_stride,
        0,
        output_stride
    );


    /*
     * Rows 1 through height-2 can be processed.
     */

    int total_rows =
        height - 2;


    /*
     * Divide rows between threads.
     */

    int rows_per_thread =
        total_rows / num_threads;


    int remaining_rows =
        total_rows % num_threads;


    int current_y =
        1;


    /*
     * Create worker threads.
     */

    for (int i = 0;
         i < num_threads;
         i++)
    {
        int rows =
            rows_per_thread;


        /*
         * Distribute leftover rows.
         */

        if (i < remaining_rows)
            rows++;


        thread_data[i].input =
            input;

        thread_data[i].output =
            output;

        thread_data[i].width =
            width;

        thread_data[i].height =
            height;

        thread_data[i].input_stride =
            input_stride;

        thread_data[i].output_stride =
            output_stride;

        thread_data[i].start_y =
            current_y;

        thread_data[i].end_y =
            current_y + rows;


        current_y += rows;


        int result =
            pthread_create(
                &threads[i],
                NULL,
                sobel_worker,
                &thread_data[i]
            );


        if (result != 0)
        {
            /*
             * This should normally never happen.
             *
             * For simplicity, abort this Sobel invocation.
             */

            fprintf(
                stderr,
                "ERROR: pthread_create for Sobel worker failed: %s\n",
                strerror(result)
            );

            /*
             * Join workers already created.
             */

            for (int j = 0;
                 j < i;
                 j++)
            {
                pthread_join(
                    threads[j],
                    NULL
                );
            }

            /*
             * Fall back to single-threaded Sobel.
             */

            SobelThreadData fallback;

            fallback.input =
                input;

            fallback.output =
                output;

            fallback.width =
                width;

            fallback.height =
                height;

            fallback.input_stride =
                input_stride;

            fallback.output_stride =
                output_stride;

            fallback.start_y =
                1;

            fallback.end_y =
                height - 1;

            sobel_worker(
                &fallback
            );

            return;
        }
    }


    /*
     * Wait for all threads.
     */

    for (int i = 0;
         i < num_threads;
         i++)
    {
        pthread_join(
            threads[i],
            NULL
        );
    }
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
    int sobel_threads)
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


    sobel(
        gray,
        sobel_output,
        width,
        height,
        width,
        width,
        sobel_threads
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


/* =========================================================
 * Process one complete video
 * ========================================================= */

static int process_video(
    const char *input_filename,
    const char *output_filename,
    int sobel_threads)
{
    int ret = 0;


    AVFormatContext *input_format_ctx =
        NULL;

    AVCodecContext *decoder_ctx =
        NULL;

    AVFormatContext *output_format_ctx =
        NULL;

    AVCodecContext *encoder_ctx =
        NULL;


    AVStream *input_stream =
        NULL;

    AVStream *output_stream =
        NULL;


    AVFrame *decoded_frame =
        NULL;

    AVFrame *output_frame =
        NULL;


    AVPacket *input_packet =
        NULL;


    uint8_t *gray =
        NULL;

    uint8_t *sobel_output =
        NULL;


    int video_stream_index =
        -1;


    int width;
    int height;


    int64_t frame_count =
        0;


    double total_sobel_ms =
        0.0;


    double total_processing_start;
    double total_processing_end;


    AVRational output_fps;


    /* =====================================================
     * Open input video
     * ===================================================== */

    /*printf(
        "\n[%s]\n"
        "Opening input: %s\n",
        input_filename,
        input_filename
    );*/

    fflush(stdout);


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
        input_format_ctx
            ->streams[video_stream_index];


    /* =====================================================
     * Open decoder
     * ===================================================== */

    {
        const AVCodec *decoder =
            avcodec_find_decoder(
                input_stream
                    ->codecpar
                    ->codec_id
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
    }


    width =
        decoder_ctx->width;

    height =
        decoder_ctx->height;


    /*printf(
        "[%s] Resolution: %d x %d\n",
        input_filename,
        width,
        height
    );*/


    if (input_stream->avg_frame_rate.den != 0)
    {
        /*printf(
            "[%s] Source FPS: %.2f\n",
            input_filename,
            av_q2d(
                input_stream->avg_frame_rate
            )
        );*/
    }


    /* printf(
        "[%s] Codec: %s\n",
        input_filename,
        avcodec_get_name(
            decoder_ctx->codec_id
        )
    );


    printf(
        "[%s] Input time base: %d/%d\n",
        input_filename,
        input_stream->time_base.num,
        input_stream->time_base.den
    );


    printf(
        "[%s] Sobel threads: %d\n",
        input_filename,
        sobel_threads
    );*/


    /* =====================================================
     * Check dimensions
     * ===================================================== */

    if ((width % 2) != 0 ||
        (height % 2) != 0)
    {
        fprintf(
            stderr,

            "[%s] Input dimensions must be even "
            "for YUV420P: %d x %d\n",

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
     * ===================================================== */

    const AVCodec *encoder =
        avcodec_find_encoder_by_name(
            "libx264"
        );


    if (!encoder)
    {
        /*printf(
            "[%s] libx264 not available. "
            "Using MPEG-4 encoder instead.\n",
            input_filename
        );*/


        encoder =
            avcodec_find_encoder(
                AV_CODEC_ID_MPEG4
            );
    }
    else
    {
        /*printf(
            "[%s] Using encoder: libx264\n",
            input_filename
        );*/
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


    /*printf(
        "[%s] Output FPS: %d/%d\n",
        input_filename,
        output_fps.num,
        output_fps.den
    );


    printf(
        "[%s] Output time base: %d/%d\n",
        input_filename,
        output_stream->time_base.num,
        output_stream->time_base.den
    );*/


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
    }
    else
    {
        encoder_ctx->bit_rate =
            4 * 1000 * 1000;
    }


    encoder_ctx->gop_size =
        50;


    encoder_ctx->max_b_frames =
        0;


    if (output_format_ctx
            ->oformat
            ->flags &
        AVFMT_GLOBALHEADER)
    {
        encoder_ctx->flags |=
            AV_CODEC_FLAG_GLOBAL_HEADER;
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

    if (!(output_format_ctx
              ->oformat
              ->flags &
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
            (size_t)width *
            height
        );


    sobel_output =
        malloc(
            (size_t)width *
            height
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
        /*
         * Ignore audio and other streams.
         */

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
                    sobel_threads
                );


            if (ret < 0)
                goto cleanup;


            frame_count++;


            if (frame_count % 25 == 0)
            {
                /*printf(
                    "\r[%s] Frames processed: %lld",
                    input_filename,
                    (long long)frame_count
                );*/

                fflush(stdout);
            }
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
                sobel_threads
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

        (
            total_processing_end -
            total_processing_start
        ) / 1000.0
    );


    if (frame_count > 0 &&
        total_sobel_ms > 0.0)
    {
        printf(
            "Average Sobel time: %.3f ms/frame\n",

            total_sobel_ms /
            frame_count
        );


        printf(
            "Sobel processing FPS: %.2f\n",

            1000.0 /
            (
                total_sobel_ms /
                frame_count
            )
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

    if (input_packet)
    {
        av_packet_free(
            &input_packet
        );
    }


    if (decoded_frame)
    {
        av_frame_free(
            &decoded_frame
        );
    }


    if (output_frame)
    {
        av_frame_free(
            &output_frame
        );
    }


    free(gray);

    free(sobel_output);


    if (decoder_ctx)
    {
        avcodec_free_context(
            &decoder_ctx
        );
    }


    if (encoder_ctx)
    {
        avcodec_free_context(
            &encoder_ctx
        );
    }


    if (input_format_ctx)
    {
        avformat_close_input(
            &input_format_ctx
        );
    }


    if (output_format_ctx)
    {
        if (output_format_ctx->pb &&
            !(output_format_ctx
                  ->oformat
                  ->flags &
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
 * Video worker thread
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
    VideoWorkerData *data =
        (VideoWorkerData *)arg;


    printf(
        "[VIDEO WORKER] Thread started\n"
    );

    fflush(stdout);


    while (1)
    {
        int file_index;


        /*
         * Get next video.
         *
         * Only this small section needs the mutex.
         */

        pthread_mutex_lock(
            data->mutex
        );


        if (*(data->next_file) >=
            data->file_count)
        {
            pthread_mutex_unlock(
                data->mutex
            );

            break;
        }


        file_index =
            *(data->next_file);


        (*(data->next_file))++;


        pthread_mutex_unlock(
            data->mutex
        );


        /*
         * Get input filename.
         */

        const char *input_path =
            data->input_files[file_index];


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

        const char *filename =
            strrchr(
                input_path,
                '/'
            );


        if (filename != NULL)
        {
            filename++;
        }
        else
        {
            filename =
                input_path;
        }


        /*
         * Remove .mp4 extension.
         */

        size_t filename_length =
            strlen(filename);


        size_t base_length =
            filename_length;


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
            data->output_directory,
            (int)base_length,
            filename
        );


        printf(
            "\n"
            "========================================\n"
            "[VIDEO WORKER] Starting video %d/%d\n"
            "========================================\n"
            "Input : %s\n"
            "Output: %s\n"
            "Sobel threads: %d\n"
            "========================================\n",
            file_index + 1,
            data->file_count,
            input_path,
            output_filename,
            data->sobel_threads
        );

        fflush(stdout);


        /*
         * Process complete video.
         */

        int result =
            process_video(
                input_path,
                output_filename,
                data->sobel_threads
            );


        /*
         * Record failure.
         */

        if (result != 0)
        {
            pthread_mutex_lock(
                data->mutex
            );

            (*(data->failed_count))++;

            pthread_mutex_unlock(
                data->mutex
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


/* =========================================================
 * Main
 * ========================================================= */

int main(
    int argc,
    char *argv[])
{

      double total_wall_start =
        now_ms();

    if (argc != 5)
    {
        fprintf(
            stderr,

            "Usage:\n"
            "  %s input_directory output_directory "
            "video_threads sobel_threads\n\n"

            "Example:\n"
            "  %s "
            "resources/input/mp4/4k "
            "resources/output/mp4/4k "
            "2 "
            "4\n\n"

            "Arguments:\n"
            "  input_directory   Directory containing input .mp4 files\n"
            "  output_directory  Directory where processed videos are written\n"
            "  video_threads     Number of videos processed concurrently\n"
            "  sobel_threads     Number of pthreads used by Sobel per video\n",

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


    if (video_threads <= 0)
    {
        fprintf(
            stderr,
            "ERROR: video_threads must be > 0\n"
        );

        return EXIT_FAILURE;
    }


    if (sobel_threads <= 0)
    {
        fprintf(
            stderr,
            "ERROR: sobel_threads must be > 0\n"
        );

        return EXIT_FAILURE;
    }

    /* =====================================================
     * Create output directory
     * ===================================================== */

    if (mkdir(
            output_directory,
            0755
        ) != 0)
    {
        if (errno != EEXIST)
        {
            fprintf(
                stderr,
                "ERROR: Could not create output directory "
                "'%s': %s\n",
                output_directory,
                strerror(errno)
            );

            return EXIT_FAILURE;
        }
    }


    /* =====================================================
     * Scan input directory
     * ===================================================== */

    DIR *dir =
        opendir(
            input_directory
        );


    if (dir == NULL)
    {
        fprintf(
            stderr,
            "ERROR: Could not open input directory "
            "'%s': %s\n",
            input_directory,
            strerror(errno)
        );

        return EXIT_FAILURE;
    }


    char **input_files =
        NULL;


    int file_count =
        0;


    int file_capacity =
        16;


    input_files =
        malloc(
            file_capacity *
            sizeof(char *)
        );


    if (input_files == NULL)
    {
        fprintf(
            stderr,
            "ERROR: malloc failed\n"
        );

        closedir(dir);

        return EXIT_FAILURE;
    }


    struct dirent *entry;


    while (
        (entry = readdir(dir)) != NULL)
    {
        /*
         * Ignore "." and "..".
         */

        if (strcmp(
                entry->d_name,
                "."
            ) == 0 ||
            strcmp(
                entry->d_name,
                ".."
            ) == 0)
        {
            continue;
        }


        /*
         * Only process .mp4 files.
         */

        const char *extension =
            strrchr(
                entry->d_name,
                '.'
            );


        if (extension == NULL)
        {
            continue;
        }


        if (strcasecmp(
                extension,
                ".mp4"
            ) != 0)
        {
            continue;
        }


        /*
         * Grow array if necessary.
         */

        if (file_count >=
            file_capacity)
        {
            file_capacity *= 2;


            char **tmp =
                realloc(
                    input_files,
                    file_capacity *
                    sizeof(char *)
                );


            if (tmp == NULL)
            {
                fprintf(
                    stderr,
                    "ERROR: realloc failed\n"
                );


                for (int i = 0;
                     i < file_count;
                     i++)
                {
                    free(
                        input_files[i]
                    );
                }


                free(input_files);

                closedir(dir);

                return EXIT_FAILURE;
            }


            input_files =
                tmp;
        }


        /*
         * Build full input path.
         */

        size_t path_length =
            strlen(input_directory) +
            1 +
            strlen(entry->d_name) +
            1;


        char *full_path =
            malloc(
                path_length
            );


        if (full_path == NULL)
        {
            fprintf(
                stderr,
                "ERROR: malloc failed\n"
            );


            for (int i = 0;
                 i < file_count;
                 i++)
            {
                free(
                    input_files[i]
                );
            }


            free(input_files);

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


        input_files[
            file_count++
        ] =
            full_path;
    }


    closedir(dir);


    /* =====================================================
     * Check whether any videos were found
     * ===================================================== */

    if (file_count == 0)
    {
        fprintf(
            stderr,
            "No .mp4 files found in '%s'\n",
            input_directory
        );


        free(input_files);

        return EXIT_FAILURE;
    }


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

    printf(
        "=============================================\n"
    );

    printf("\n");


    /* =====================================================
     * Limit video threads to number of videos
     * ===================================================== */

    if (video_threads >
        file_count)
    {
        video_threads =
            file_count;
    }


    /*
     * Shared work index.
     */

    int next_file =
        0;


    /*
     * Number of failed videos.
     */

    int failed_count =
        0;


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


        for (int i = 0;
             i < file_count;
             i++)
        {
            free(
                input_files[i]
            );
        }


        free(input_files);

        return EXIT_FAILURE;
    }


    /* =====================================================
     * Allocate video worker threads
     * ===================================================== */

    pthread_t *threads =
        malloc(
            video_threads *
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


        for (int i = 0;
             i < file_count;
             i++)
        {
            free(
                input_files[i]
            );
        }


        free(input_files);

        return EXIT_FAILURE;
    }


    /* =====================================================
     * Worker data
     * ===================================================== */

    VideoWorkerData worker_data;


    worker_data.input_files =
        input_files;


    worker_data.file_count =
        file_count;


    worker_data.next_file =
        &next_file;


    worker_data.mutex =
        &mutex;


    worker_data.output_directory =
        output_directory;


    worker_data.sobel_threads =
        sobel_threads;


    worker_data.failed_count =
        &failed_count;


    /* =====================================================
     * Create video worker threads
     * ===================================================== */

    printf(
        "Creating %d video worker threads...\n\n",
        video_threads
    );


    int created_threads =
        0;


    for (int i = 0;
         i < video_threads;
         i++)
    {
        int result =
            pthread_create(
                &threads[i],
                NULL,

                /*
                 * IMPORTANT:
                 *
                 * This MUST be video_worker.
                 *
                 * NOT sobel_worker.
                 */

                video_worker,

                &worker_data
            );


        if (result != 0)
        {
            fprintf(
                stderr,

                "ERROR: pthread_create failed "
                "for video worker %d: %s\n",

                i,

                strerror(result)
            );

            break;
        }


        printf(
            "Created video worker %d\n",
            i
        );


        created_threads++;
    }


    printf(
        "\nCreated %d/%d video workers\n\n",
        created_threads,
        video_threads
    );


    /* =====================================================
     * Wait for all video workers
     * ===================================================== */
     printf("\nWaiting for video workers...\n");
     fflush(stdout);

    for (int i = 0;
         i < created_threads;
         i++)
    {
        printf("[MAIN] Joining video worker %d...\n", i);
        fflush(stdout);
        int result =
            pthread_join(
                threads[i],
                NULL
            );

        printf("[MAIN] Joined video worker %d\n", i);
        fflush(stdout);

        if (result != 0)
        {
            fprintf(
                stderr,
                "ERROR: pthread_join failed "
                "for worker %d: %s\n",
                i,
                strerror(result)
            );
        }

        printf(
        "[MAIN] Worker %d/%d joined\n",
        i + 1,
        created_threads
    );
    fflush(stdout);
    }

    printf("[MAIN] ALL VIDEO WORKERS JOINED\n");
    fflush(stdout);

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
    failed_count
);

printf(
    "=============================================\n"
);   
    



    /* =====================================================
     * Cleanup
     * ===================================================== */

    free(threads);


    pthread_mutex_destroy(
        &mutex
    );


    for (int i = 0;
         i < file_count;
         i++)
    {
        free(
            input_files[i]
        );
    }


    free(input_files);


    /* =====================================================
     * Final status
     * ===================================================== */

    printf("\n");

    printf(
        "=============================================\n"
    );


    if (failed_count == 0 &&
        created_threads == video_threads)
    {
        printf(
            "All videos processed successfully.\n"
        );
    }
    else
    {
        printf(
            "Processing completed with %d failed video(s).\n",
            failed_count
        );
    }


    printf(
        "=============================================\n"
    );


    return (
        failed_count == 0 &&
        created_threads == video_threads
    )
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}