#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "./common.h"


/* ---------------------------------------------------------
 * Multithreaded Sobel filter
 *
 * The image is divided into horizontal bands.
 * Each pthread processes a range of rows.
 * --------------------------------------------------------- */

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
 * Worker thread
 * --------------------------------------------------------- */

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


/* ---------------------------------------------------------
 * Multithreaded Sobel filter
 * --------------------------------------------------------- */

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
     * Number of CPU threads.
     *
     * Change this value to test different levels
     * of CPU parallelism.
     */
    //const int num_threads = 10;


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


        pthread_create(
            &threads[i],
            NULL,
            sobel_worker,
            &thread_data[i]
        );
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


/* ---------------------------------------------------------
 * Process one decoded frame
 *
 * This function:
 *
 * 1. Preserves the input frame timestamp.
 * 2. Copies the Y plane.
 * 3. Runs multithreaded Sobel.
 * 4. Builds YUV420P output frame.
 * 5. Preserves frame duration.
 * 6. Sends frame to encoder.
 * --------------------------------------------------------- */

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
    int num_threads)
{
    int ret;


    /* -----------------------------------------------------
     * Determine frame PTS
     * ----------------------------------------------------- */

    int64_t frame_pts =
        decoded_frame->pts;


    if (frame_pts == AV_NOPTS_VALUE)
    {
        /*
         * Fallback for streams without a usable PTS.
         */
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
        /*
         * Convert PTS from input stream time base
         * into encoder time base.
         */
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

    /*int64_t frame_duration =
        decoded_frame->duration;*/
    int64_t frame_duration =
    GET_FRAME_DURATION(decoded_frame);


    if (frame_duration <= 0)
    {
        /*
         * Fall back to 1/FPS.
         */
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
        /*
         * Convert duration from input time base
         * into encoder time base.
         */
        frame_duration =
            av_rescale_q(
                frame_duration,
                input_stream->time_base,
                encoder_ctx->time_base
            );
    }


    /* -----------------------------------------------------
     * Copy Y plane
     *
     * For YUV video:
     *
     *     data[0] = Y
     *     data[1] = U
     *     data[2] = V
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
        num_threads
    );


    double sobel_end =
        now_ms();


    *total_sobel_ms +=
        sobel_end - sobel_start;


    /* -----------------------------------------------------
     * Make output frame writable
     * ----------------------------------------------------- */

    ret = av_frame_make_writable(
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

    /*output_frame->duration =
        frame_duration;*/
    SET_FRAME_DURATION(
    output_frame,
    frame_duration
    );     


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

    ret = encode_frame(
        encoder_ctx,
        output_format_ctx,
        output_stream,
        output_frame
    );

    return ret;
}


/* ---------------------------------------------------------
 * Main
 * --------------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(
        stderr,

        "Usage:\n"
        "  %s input.mp4 output.mp4 thread_count_for_sobel\n\n"

        "Example:\n"
        "  %s "
        "resources/input/mp4/"
        "4k/1.mp4 "
        "output_sobel.mp4 "
        "10\n",

        argv[0],
        argv[0]
    );

        return 1;
    }

    int num_threads = atoi(argv[3]);


    const char *input_filename =
        argv[1];

    const char *output_filename =
        argv[2];


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


    /* -----------------------------------------------------
     * Open input video
     * ----------------------------------------------------- */

    printf(
        "Opening input: %s\n",
        input_filename
    );


    ret = avformat_open_input(
        &input_format_ctx,
        input_filename,
        NULL,
        NULL
    );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Could not open input video\n"
        );

        goto cleanup;
    }


    ret = avformat_find_stream_info(
        input_format_ctx,
        NULL
    );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Could not find stream information\n"
        );

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Find video stream
     * ----------------------------------------------------- */

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
            "Could not find video stream\n"
        );

        ret =
            video_stream_index;

        goto cleanup;
    }


    input_stream =
        input_format_ctx
            ->streams[video_stream_index];


    /* -----------------------------------------------------
     * Open decoder
     * ----------------------------------------------------- */

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
                "Could not find decoder\n"
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
                "Could not allocate decoder context\n"
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
                "Could not copy decoder parameters\n"
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
                "Could not open decoder\n"
            );

            goto cleanup;
        }
    }


    width =
        decoder_ctx->width;

    height =
        decoder_ctx->height;


    printf(
        "Resolution: %d x %d\n",
        width,
        height
    );


    if (input_stream->avg_frame_rate.den != 0)
    {
        printf(
            "Source FPS: %.2f\n",
            av_q2d(
                input_stream->avg_frame_rate
            )
        );
    }


    printf(
        "Codec: %s\n",
        avcodec_get_name(
            decoder_ctx->codec_id
        )
    );


    printf(
        "Input time base: %d/%d\n",
        input_stream->time_base.num,
        input_stream->time_base.den
    );


    /* -----------------------------------------------------
     * Check dimensions
     * ----------------------------------------------------- */

    if ((width % 2) != 0 ||
        (height % 2) != 0)
    {
        fprintf(
            stderr,

            "Input dimensions must be even "
            "for YUV420P: %d x %d\n",

            width,
            height
        );

        ret =
            AVERROR(EINVAL);

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Create output format
     * ----------------------------------------------------- */

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
            "Could not create output format context\n"
        );

        ret =
            AVERROR_UNKNOWN;

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Find H.264 encoder
     *
     * Prefer libx264.
     * Fall back to MPEG-4.
     * ----------------------------------------------------- */

    const AVCodec *encoder =
        avcodec_find_encoder_by_name(
            "libx264"
        );


    if (!encoder)
    {
        printf(
            "libx264 not available. "
            "Using MPEG-4 encoder instead.\n"
        );


        encoder =
            avcodec_find_encoder(
                AV_CODEC_ID_MPEG4
            );
    }
    else
    {
        printf(
            "Using encoder: libx264\n"
        );
    }


    if (!encoder)
    {
        fprintf(
            stderr,
            "Could not find a suitable video encoder\n"
        );

        ret =
            AVERROR_ENCODER_NOT_FOUND;

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Create output stream
     * ----------------------------------------------------- */

    output_stream =
        avformat_new_stream(
            output_format_ctx,
            NULL
        );


    if (!output_stream)
    {
        fprintf(
            stderr,
            "Could not create output stream\n"
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate encoder
     * ----------------------------------------------------- */

    encoder_ctx =
        avcodec_alloc_context3(
            encoder
        );


    if (!encoder_ctx)
    {
        fprintf(
            stderr,
            "Could not allocate encoder context\n"
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


    /*
     * Sobel result goes into Y.
     *
     * U/V are 128.
     *
     * Therefore output is grayscale
     * YUV420P video.
     */
    encoder_ctx->pix_fmt =
        AV_PIX_FMT_YUV420P;


    /* -----------------------------------------------------
     * Preserve source FPS
     * ----------------------------------------------------- */

    AVRational output_fps =
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


    /*
     * Preserve the input stream time base.
     */
    encoder_ctx->time_base =
        input_stream->time_base;


    /*
     * Tell encoder the source frame rate.
     */
    encoder_ctx->framerate =
        output_fps;


    /*
     * Initially use the same time base for
     * the output stream.
     *
     * avformat_write_header() may modify the
     * final stream time base, so all packets
     * are always rescaled using the final
     * output_stream->time_base.
     */
    output_stream->time_base =
        input_stream->time_base;


    output_stream->avg_frame_rate =
        output_fps;


    printf(
        "Output FPS: %d/%d\n",
        output_fps.num,
        output_fps.den
    );


    printf(
        "Output time base: %d/%d\n",
        output_stream->time_base.num,
        output_stream->time_base.den
    );


    /* -----------------------------------------------------
     * Encoder settings
     * ----------------------------------------------------- */

    if (encoder->id ==
        AV_CODEC_ID_H264)
    {
        /*
         * CRF controls quality for libx264.
         */
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


    /*
     * Use a fixed GOP.
     */
    encoder_ctx->gop_size =
        50;


    /*
     * Disable B-frames.
     *
     * This makes timestamp handling simpler
     * and is useful for comparing the CPU
     * implementation with the CUDA/NVENC
     * implementation.
     */
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


    /* -----------------------------------------------------
     * Open encoder
     * ----------------------------------------------------- */

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
            "Could not open encoder\n"
        );

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Copy encoder parameters to output stream
     * ----------------------------------------------------- */

    ret =
        avcodec_parameters_from_context(
            output_stream->codecpar,
            encoder_ctx
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Could not copy encoder parameters\n"
        );

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Open output file
     * ----------------------------------------------------- */

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
                "Could not open output file: %s\n",
                output_filename
            );

            goto cleanup;
        }
    }


    /* -----------------------------------------------------
     * Write MP4 header
     * ----------------------------------------------------- */

    ret =
        avformat_write_header(
            output_format_ctx,
            NULL
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Could not write output header\n"
        );

        goto cleanup;
    }


    /*
     * The muxer may change the output stream time base
     * when writing the header.
     *
     * Print the actual final value.
     */
    printf(
        "Final output time base: %d/%d\n",
        output_stream->time_base.num,
        output_stream->time_base.den
    );


    /* -----------------------------------------------------
     * Allocate frames
     * ----------------------------------------------------- */

    decoded_frame =
        av_frame_alloc();


    output_frame =
        av_frame_alloc();


    if (!decoded_frame ||
        !output_frame)
    {
        fprintf(
            stderr,
            "Could not allocate frames\n"
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate Sobel buffers
     * ----------------------------------------------------- */

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
            "Could not allocate image buffers\n"
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate output YUV420P frame
     * ----------------------------------------------------- */

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
            "Could not allocate output frame buffer\n"
        );

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate input packet
     * ----------------------------------------------------- */

    input_packet =
        av_packet_alloc();


    if (!input_packet)
    {
        fprintf(
            stderr,
            "Could not allocate input packet\n"
        );

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    printf("\n");
    printf("Processing...\n");
    printf(
        "Output: %s\n",
        output_filename
    );
    printf("\n");


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
                "Error sending packet to decoder\n"
            );

            goto cleanup;
        }


        /* -------------------------------------------------
         * Receive all decoded frames.
         * ------------------------------------------------- */

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
                    "Error decoding frame\n"
                );

                goto cleanup;
            }


            /* -------------------------------------------------
             * Process frame.
             * ------------------------------------------------- */

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
                    num_threads
                );


            if (ret < 0)
                goto cleanup;


            frame_count++;


            /* -------------------------------------------------
             * Progress.
             * ------------------------------------------------- */

            if (frame_count % 25 == 0)
            {
                printf(
                    "\rFrames processed: %lld",
                    (long long)frame_count
                );

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
            "Error flushing decoder\n"
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
                "Error receiving flushed frame\n"
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
                num_threads
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
     * Write MP4 trailer
     * ===================================================== */

    ret =
        av_write_trailer(
            output_format_ctx
        );


    if (ret < 0)
    {
        fprintf(
            stderr,
            "Could not write output trailer\n"
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


    printf("\n");

    printf(
        "Output video:\n"
    );

    printf(
        "%s\n",
        output_filename
    );


    printf(
        "========================================\n"
    );


cleanup:

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


    return ret < 0 ? 1 : 0;
}