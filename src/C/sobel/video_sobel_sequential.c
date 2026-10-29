
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "./common.h"

/* ---------------------------------------------------------
 * Sobel filter
 *
 * Input:
 *   grayscale image
 *
 * Output:
 *   grayscale edge image
 * --------------------------------------------------------- */

static void sobel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height,
    int input_stride,
    int output_stride)
{
    static const int gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    static const int gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    /*
     * Clear first/last rows.
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

    for (int y = 1; y < height - 1; y++)
    {
        const uint8_t *row0 =
            input + (y - 1) * input_stride;

        const uint8_t *row1 =
            input + y * input_stride;

        const uint8_t *row2 =
            input + (y + 1) * input_stride;

        uint8_t *out =
            output + y * output_stride;

        out[0] = 0;
        out[width - 1] = 0;

        for (int x = 1; x < width - 1; x++)
        {
            int gx =
                -row0[x - 1] + row0[x + 1]
                -2 * row1[x - 1] + 2 * row1[x + 1]
                -row2[x - 1] + row2[x + 1];

            int gy =
                -row0[x - 1] - 2 * row0[x] - row0[x + 1]
                +row2[x - 1] + 2 * row2[x] + row2[x + 1];

            int magnitude =
                abs(gx) + abs(gy);

            if (magnitude > 255)
                magnitude = 255;

            out[x] =
                (uint8_t)magnitude;
        }
    }
}

/* ---------------------------------------------------------
 * Main
 * --------------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr,
            "Usage:\n"
            "  %s input.mp4 output.mp4\n\n"
            "Example:\n"
            "  %s resources/input/input.mp4 "
            "output.mp4\n",
            argv[0],
            argv[0]);

        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

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

    int width;
    int height;

    int64_t frame_count = 0;

    double total_sobel_ms = 0.0;
    double total_processing_start;
    double total_processing_end;


    /* -----------------------------------------------------
     * Open input video
     * ----------------------------------------------------- */

    printf("Opening input: %s\n", input_filename);

    ret = avformat_open_input(
        &input_format_ctx,
        input_filename,
        NULL,
        NULL
    );

    if (ret < 0)
    {
        fprintf(stderr, "Could not open input video\n");
        goto cleanup;
    }

    ret = avformat_find_stream_info(
        input_format_ctx,
        NULL
    );

    if (ret < 0)
    {
        fprintf(stderr, "Could not find stream information\n");
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Find video stream
     * ----------------------------------------------------- */

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
        fprintf(stderr, "Could not find video stream\n");
        ret = video_stream_index;
        goto cleanup;
    }

    input_stream =
        input_format_ctx->streams[video_stream_index];


    /* -----------------------------------------------------
     * Open decoder
     * ----------------------------------------------------- */

    {
        const AVCodec *decoder =
            avcodec_find_decoder(
                input_stream->codecpar->codec_id
            );

        if (!decoder)
        {
            fprintf(stderr, "Could not find decoder\n");
            ret = AVERROR_DECODER_NOT_FOUND;
            goto cleanup;
        }

        decoder_ctx = avcodec_alloc_context3(decoder);

        if (!decoder_ctx)
        {
            fprintf(stderr,
                    "Could not allocate decoder context\n");
            ret = AVERROR(ENOMEM);
            goto cleanup;
        }

        ret = avcodec_parameters_to_context(
            decoder_ctx,
            input_stream->codecpar
        );

        if (ret < 0)
        {
            fprintf(stderr,
                    "Could not copy decoder parameters\n");
            goto cleanup;
        }

        ret = avcodec_open2(
            decoder_ctx,
            decoder,
            NULL
        );

        if (ret < 0)
        {
            fprintf(stderr,
                    "Could not open decoder\n");
            goto cleanup;
        }
    }


    width = decoder_ctx->width;
    height = decoder_ctx->height;

    printf("Resolution: %d x %d\n", width, height);

    if (input_stream->avg_frame_rate.den != 0)
    {
        double fps =
            av_q2d(input_stream->avg_frame_rate);

        printf("Source FPS: %.2f\n", fps);
    }

    printf("Codec: %s\n",
           avcodec_get_name(decoder_ctx->codec_id));


    /* -----------------------------------------------------
     * Check dimensions
     *
     * YUV420P requires even width/height.
     * ----------------------------------------------------- */

    if ((width % 2) != 0 || (height % 2) != 0)
    {
        fprintf(stderr,
                "Input dimensions must be even for YUV420P: "
                "%d x %d\n",
                width,
                height);

        ret = AVERROR(EINVAL);
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Create output format
     * ----------------------------------------------------- */

    ret = avformat_alloc_output_context2(
        &output_format_ctx,
        NULL,
        NULL,
        output_filename
    );

    if (ret < 0 || !output_format_ctx)
    {
        fprintf(stderr,
                "Could not create output format context\n");
        ret = AVERROR_UNKNOWN;
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Find H.264 encoder
     *
     * Prefer libx264 because it produces a normal H.264 MP4.
     * Fall back to MPEG-4 if libx264 isn't available.
     * ----------------------------------------------------- */

    const AVCodec *encoder =
        avcodec_find_encoder_by_name("libx264");

    if (!encoder)
    {
        printf("libx264 not available. "
               "Using MPEG-4 encoder instead.\n");

        encoder =
            avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    }
    else
    {
        printf("Using encoder: libx264\n");
    }

    if (!encoder)
    {
        fprintf(stderr,
                "Could not find a suitable video encoder\n");

        ret = AVERROR_ENCODER_NOT_FOUND;
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
        fprintf(stderr,
                "Could not create output stream\n");

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate encoder
     * ----------------------------------------------------- */

    encoder_ctx =
        avcodec_alloc_context3(encoder);

    if (!encoder_ctx)
    {
        fprintf(stderr,
                "Could not allocate encoder context\n");

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }

    encoder_ctx->codec_id = encoder->id;
    encoder_ctx->codec_type = AVMEDIA_TYPE_VIDEO;

    encoder_ctx->width = width;
    encoder_ctx->height = height;

    /*
     * Output frame format.
     *
     * The Sobel output becomes the Y plane.
     * U/V are set to 128, producing grayscale video.
     */
    encoder_ctx->pix_fmt = AV_PIX_FMT_YUV420P;


    /* -----------------------------------------------------
     * Output FPS
     * ----------------------------------------------------- */

    AVRational output_fps =
        input_stream->avg_frame_rate;

    if (output_fps.num <= 0 || output_fps.den <= 0)
    {
        output_fps = (AVRational){25, 1};
    }

    encoder_ctx->time_base =
        av_inv_q(output_fps);

    output_stream->time_base =
        encoder_ctx->time_base;


    /* -----------------------------------------------------
     * Encoder settings
     * ----------------------------------------------------- */

    if (encoder->id == AV_CODEC_ID_H264)
    {
        /*
         * CRF controls quality for libx264.
         *
         * 23 is a reasonable normal-quality setting.
         *
         * We are NOT intentionally degrading quality yet.
         * We'll change this in the next step.
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
        encoder_ctx->bit_rate = 4 * 1000 * 1000;
    }

    encoder_ctx->gop_size = 50;
    encoder_ctx->max_b_frames = 2;


    if (output_format_ctx->oformat->flags &
        AVFMT_GLOBALHEADER)
    {
        encoder_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }


    /* -----------------------------------------------------
     * Open encoder
     * ----------------------------------------------------- */

    ret = avcodec_open2(
        encoder_ctx,
        encoder,
        NULL
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Could not open encoder\n");

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Copy encoder parameters to output stream
     * ----------------------------------------------------- */

    ret = avcodec_parameters_from_context(
        output_stream->codecpar,
        encoder_ctx
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Could not copy encoder parameters\n");

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Open output file
     * ----------------------------------------------------- */

    if (!(output_format_ctx->oformat->flags &
          AVFMT_NOFILE))
    {
        ret = avio_open(
            &output_format_ctx->pb,
            output_filename,
            AVIO_FLAG_WRITE
        );

        if (ret < 0)
        {
            fprintf(stderr,
                    "Could not open output file: %s\n",
                    output_filename);

            goto cleanup;
        }
    }


    /* -----------------------------------------------------
     * Write MP4 header
     * ----------------------------------------------------- */

    ret = avformat_write_header(
        output_format_ctx,
        NULL
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Could not write output header\n");

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate frames
     * ----------------------------------------------------- */

    decoded_frame = av_frame_alloc();

    output_frame = av_frame_alloc();

    if (!decoded_frame || !output_frame)
    {
        fprintf(stderr,
                "Could not allocate frames\n");

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate Sobel buffers
     * ----------------------------------------------------- */

    gray =
        malloc((size_t)width * height);

    sobel_output =
        malloc((size_t)width * height);

    if (!gray || !sobel_output)
    {
        fprintf(stderr,
                "Could not allocate image buffers\n");

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate output YUV420P frame
     * ----------------------------------------------------- */

    output_frame->format = AV_PIX_FMT_YUV420P;
    output_frame->width = width;
    output_frame->height = height;

    ret = av_frame_get_buffer(
        output_frame,
        32
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Could not allocate output frame buffer\n");

        goto cleanup;
    }


    /* -----------------------------------------------------
     * Allocate packet
     * ----------------------------------------------------- */

    input_packet = av_packet_alloc();

    if (!input_packet)
    {
        fprintf(stderr,
                "Could not allocate input packet\n");

        ret = AVERROR(ENOMEM);
        goto cleanup;
    }


    printf("\n");
    printf("Processing...\n");
    printf("Output: %s\n", output_filename);
    printf("\n");

    total_processing_start = now_ms();


    /* -----------------------------------------------------
     * Decode loop
     * ----------------------------------------------------- */

    while (av_read_frame(
               input_format_ctx,
               input_packet) >= 0)
    {
        if (input_packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(input_packet);
            continue;
        }

        ret = avcodec_send_packet(
            decoder_ctx,
            input_packet
        );

        av_packet_unref(input_packet);

        if (ret < 0)
        {
            fprintf(stderr,
                    "Error sending packet to decoder\n");

            goto cleanup;
        }


        while (1)
        {
            ret = avcodec_receive_frame(
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
                fprintf(stderr,
                        "Error decoding frame\n");

                goto cleanup;
            }


            /* -------------------------------------------------
             * Get grayscale/luminance plane
             *
             * For YUV video, data[0] is the Y plane.
             * ------------------------------------------------- */

            for (int y = 0; y < height; y++)
            {
                memcpy(
                    gray + y * width,
                    decoded_frame->data[0]
                        + y * decoded_frame->linesize[0],
                    width
                );
            }


            /* -------------------------------------------------
             * Sobel processing
             * ------------------------------------------------- */

            double sobel_start = now_ms();

            sobel(
                gray,
                sobel_output,
                width,
                height,
                width,
                width
            );

            double sobel_end = now_ms();

            total_sobel_ms +=
                sobel_end - sobel_start;


            /* -------------------------------------------------
             * Prepare output frame
             *
             * Y = Sobel result
             * U = 128
             * V = 128
             *
             * This creates grayscale edge video.
             * ------------------------------------------------- */

            ret = av_frame_make_writable(
                output_frame
            );

            if (ret < 0)
            {
                fprintf(stderr,
                        "Output frame is not writable\n");

                goto cleanup;
            }

            for (int y = 0; y < height; y++)
            {
                memcpy(
                    output_frame->data[0]
                        + y * output_frame->linesize[0],
                    sobel_output + y * width,
                    width
                );
            }


            /*
             * Neutral chroma.
             *
             * YUV420P U/V are half resolution.
             */
            for (int y = 0; y < height / 2; y++)
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


            /* -------------------------------------------------
             * Encode
             * ------------------------------------------------- */

            ret = encode_frame(
                encoder_ctx,
                output_format_ctx,
                output_stream,
                output_frame,
                frame_count
            );

            if (ret < 0)
            {
                goto cleanup;
            }

            frame_count++;


            /*
             * Progress indicator.
             */
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


    /* -----------------------------------------------------
     * Flush decoder
     *
     * Some codecs may have frames buffered internally.
     * ----------------------------------------------------- */

    ret = avcodec_send_packet(
        decoder_ctx,
        NULL
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Error flushing decoder\n");

        goto cleanup;
    }

    while (1)
    {
        ret = avcodec_receive_frame(
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
            fprintf(stderr,
                    "Error receiving flushed frame\n");

            goto cleanup;
        }


        /* Copy Y plane */
        for (int y = 0; y < height; y++)
        {
            memcpy(
                gray + y * width,
                decoded_frame->data[0]
                    + y * decoded_frame->linesize[0],
                width
            );
        }


        /* Sobel */
        double sobel_start = now_ms();

        sobel(
            gray,
            sobel_output,
            width,
            height,
            width,
            width
        );

        double sobel_end = now_ms();

        total_sobel_ms +=
            sobel_end - sobel_start;


        /* Prepare output frame */

        ret = av_frame_make_writable(
            output_frame
        );

        if (ret < 0)
        {
            goto cleanup;
        }

        for (int y = 0; y < height; y++)
        {
            memcpy(
                output_frame->data[0]
                    + y * output_frame->linesize[0],
                sobel_output + y * width,
                width
            );
        }

        for (int y = 0; y < height / 2; y++)
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


        ret = encode_frame(
            encoder_ctx,
            output_format_ctx,
            output_stream,
            output_frame,
            frame_count
        );

        if (ret < 0)
        {
            goto cleanup;
        }

        frame_count++;
    }


    /* -----------------------------------------------------
     * Flush encoder
     * ----------------------------------------------------- */

    ret = encode_frame(
        encoder_ctx,
        output_format_ctx,
        output_stream,
        NULL,
        frame_count
    );

    if (ret < 0)
    {
        goto cleanup;
    }


    /* -----------------------------------------------------
     * Write MP4 trailer
     * ----------------------------------------------------- */

    ret = av_write_trailer(
        output_format_ctx
    );

    if (ret < 0)
    {
        fprintf(stderr,
                "Could not write output trailer\n");

        goto cleanup;
    }


    total_processing_end = now_ms();


    /* -----------------------------------------------------
     * Statistics
     * ----------------------------------------------------- */

    printf("\n\n");
    printf("========================================\n");
    printf("Processing complete\n");
    printf("========================================\n");

    printf("Frames processed: %lld\n",
           (long long)frame_count);

    printf("Resolution: %d x %d\n",
           width,
           height);

    printf("Total processing time: %.2f seconds\n",
           (total_processing_end -
            total_processing_start) / 1000.0);

    if (frame_count > 0)
    {
        printf("Average Sobel time: %.3f ms/frame\n",
               total_sobel_ms / frame_count);

        printf("Sobel processing FPS: %.2f\n",
               1000.0 /
               (total_sobel_ms / frame_count));
    }

    printf("\nOutput video:\n");
    printf("%s\n", output_filename);

    printf("========================================\n");


cleanup:

    if (input_packet)
        av_packet_free(&input_packet);

    if (decoded_frame)
        av_frame_free(&decoded_frame);

    if (output_frame)
        av_frame_free(&output_frame);

    free(gray);
    free(sobel_output);

    if (decoder_ctx)
        avcodec_free_context(&decoder_ctx);

    if (encoder_ctx)
        avcodec_free_context(&encoder_ctx);

    if (input_format_ctx)
        avformat_close_input(&input_format_ctx);

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

    return ret < 0 ? 1 : 0;
}
