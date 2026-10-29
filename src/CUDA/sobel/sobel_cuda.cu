#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}


/* =========================================================
 * CUDA error checking
 * ========================================================= */

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess)                               \
        {                                                     \
            fprintf(stderr,                                  \
                    "CUDA error at %s:%d: %s\n",              \
                    __FILE__,                                 \
                    __LINE__,                                 \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)


/* =========================================================
 * Timing
 * ========================================================= */

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1000000000.0;
}


/* =========================================================
 * CUDA Sobel kernel
 *
 * One CUDA thread processes one pixel.
 * ========================================================= */

__global__ void sobel_kernel(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height,
    int input_stride,
    int output_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /*
     * Ignore threads outside the image.
     */
    if (x >= width || y >= height)
        return;

    /*
     * Border pixels have no complete 3x3 neighborhood.
     */
    if (x == 0 ||
        x == width - 1 ||
        y == 0 ||
        y == height - 1)
    {
        output[y * output_stride + x] = 0;
        return;
    }

    const uint8_t *row0 =
        input + (y - 1) * input_stride;

    const uint8_t *row1 =
        input + y * input_stride;

    const uint8_t *row2 =
        input + (y + 1) * input_stride;

    int gx =
        -row0[x - 1] + row0[x + 1]
        -2 * row1[x - 1] + 2 * row1[x + 1]
        -row2[x - 1] + row2[x + 1];

    int gy =
        -row0[x - 1] - 2 * row0[x] - row0[x + 1]
        +row2[x - 1] + 2 * row2[x] + row2[x + 1];

    /*
     * Same magnitude calculation as the CPU version.
     */
    int magnitude = abs(gx) + abs(gy);

    if (magnitude > 255)
        magnitude = 255;

    output[y * output_stride + x] =
        (uint8_t)magnitude;
}


/* =========================================================
 * CUDA Sobel context
 *
 * GPU memory is allocated once and reused for all frames.
 * ========================================================= */

typedef struct
{
    uint8_t *d_input;
    uint8_t *d_output;

    size_t buffer_size;

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;

    double total_kernel_ms;

} SobelCudaContext;


/* =========================================================
 * Find CUDA GPU by name
 * ========================================================= */

static int find_gpu_by_name(const char *name)
{
    int device_count = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&device_count));

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;

        CUDA_CHECK(
            cudaGetDeviceProperties(&prop, i));

        printf("CUDA device %d: %s\n",
               i,
               prop.name);

        if (strstr(prop.name, name) != NULL)
        {
            return i;
        }
    }

    return -1;
}


/* =========================================================
 * Initialize CUDA Sobel
 * ========================================================= */

static void sobel_cuda_init(
    SobelCudaContext *ctx,
    int width,
    int height)
{
    memset(ctx, 0, sizeof(*ctx));


    /* -----------------------------------------------------
     * Enumerate CUDA devices
     * ----------------------------------------------------- */

    int device_count = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&device_count));

    printf("CUDA device count: %d\n",
           device_count);


    if (device_count == 0)
    {
        fprintf(stderr,
                "No CUDA devices found\n");

        exit(EXIT_FAILURE);
    }


    /* -----------------------------------------------------
     * Find RTX 4060 Ti
     * ----------------------------------------------------- */

    int selected_device =
        find_gpu_by_name("RTX 4060 Ti");

    if (selected_device < 0)
    {
        fprintf(stderr,
                "RTX 4060 Ti not found\n");

        exit(EXIT_FAILURE);
    }


    /* -----------------------------------------------------
     * Select GPU
     * ----------------------------------------------------- */

    printf("Selecting CUDA device %d...\n",
           selected_device);

    CUDA_CHECK(
        cudaSetDevice(selected_device));


    /* -----------------------------------------------------
     * Verify selected device
     * ----------------------------------------------------- */

    int current_device = -1;

    CUDA_CHECK(
        cudaGetDevice(&current_device));

    printf("Current CUDA device: %d\n",
           current_device);


    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            current_device));


    printf("\nCUDA GPU:\n");
    printf("  Device: %s\n",
           prop.name);

    printf("  Compute capability: %d.%d\n",
           prop.major,
           prop.minor);

    printf("  Global memory: %.2f GB\n",
           (double)prop.totalGlobalMem /
           (1024.0 * 1024.0 * 1024.0));

    printf("\n");


    /* -----------------------------------------------------
     * Allocate GPU buffers
     *
     * These are allocated AFTER selecting the GPU.
     * ----------------------------------------------------- */

    ctx->buffer_size =
        (size_t)width * height;

    CUDA_CHECK(
        cudaMalloc(
            (void **)&ctx->d_input,
            ctx->buffer_size));

    CUDA_CHECK(
        cudaMalloc(
            (void **)&ctx->d_output,
            ctx->buffer_size));


    /* -----------------------------------------------------
     * Create CUDA timing events
     * ----------------------------------------------------- */

    CUDA_CHECK(
        cudaEventCreate(
            &ctx->kernel_start));

    CUDA_CHECK(
        cudaEventCreate(
            &ctx->kernel_stop));

    ctx->total_kernel_ms = 0.0;
}


/* =========================================================
 * Run CUDA Sobel for one frame
 * ========================================================= */

static void sobel_cuda_process(
    SobelCudaContext *ctx,
    const uint8_t *input,
    uint8_t *output,
    int width,
    int height)
{
    size_t size =
        (size_t)width * height;


    /* -----------------------------------------------------
     * CPU -> GPU
     * ----------------------------------------------------- */

    CUDA_CHECK(
        cudaMemcpy(
            ctx->d_input,
            input,
            size,
            cudaMemcpyHostToDevice));


    /* -----------------------------------------------------
     * CUDA launch configuration
     *
     * 16 x 16 = 256 threads per block.
     * ----------------------------------------------------- */

    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);


    /* -----------------------------------------------------
     * Start GPU kernel timing
     * ----------------------------------------------------- */

    CUDA_CHECK(
        cudaEventRecord(
            ctx->kernel_start));


    sobel_kernel<<<grid, block>>>(
        ctx->d_input,
        ctx->d_output,
        width,
        height,
        width,
        width);


    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaEventRecord(
            ctx->kernel_stop));


    CUDA_CHECK(
        cudaEventSynchronize(
            ctx->kernel_stop));


    float kernel_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &kernel_ms,
            ctx->kernel_start,
            ctx->kernel_stop));


    ctx->total_kernel_ms +=
        kernel_ms;


    /* -----------------------------------------------------
     * GPU -> CPU
     * ----------------------------------------------------- */

    CUDA_CHECK(
        cudaMemcpy(
            output,
            ctx->d_output,
            size,
            cudaMemcpyDeviceToHost));
}


/* =========================================================
 * Free CUDA resources
 * ========================================================= */

static void sobel_cuda_destroy(
    SobelCudaContext *ctx)
{
    if (ctx->kernel_start)
        cudaEventDestroy(
            ctx->kernel_start);

    if (ctx->kernel_stop)
        cudaEventDestroy(
            ctx->kernel_stop);

    if (ctx->d_input)
        cudaFree(
            ctx->d_input);

    if (ctx->d_output)
        cudaFree(
            ctx->d_output);

    memset(ctx, 0, sizeof(*ctx));
}


/* =========================================================
 * Encode one frame
 * ========================================================= */

static int encode_frame(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_format_ctx,
    AVStream *output_stream,
    AVFrame *frame,
    int64_t pts)
{
    int ret;

    if (frame != NULL)
        frame->pts = pts;

    ret = avcodec_send_frame(
        encoder_ctx,
        frame);

    if (ret < 0)
    {
        fprintf(stderr,
                "Error sending frame to encoder\n");

        return ret;
    }

    while (1)
    {
        AVPacket *packet =
            av_packet_alloc();

        if (!packet)
        {
            fprintf(stderr,
                    "Could not allocate output packet\n");

            return AVERROR(ENOMEM);
        }

        ret = avcodec_receive_packet(
            encoder_ctx,
            packet);

        if (ret == AVERROR(EAGAIN) ||
            ret == AVERROR_EOF)
        {
            av_packet_free(&packet);
            break;
        }

        if (ret < 0)
        {
            fprintf(stderr,
                    "Error receiving encoded packet\n");

            av_packet_free(&packet);

            return ret;
        }

        av_packet_rescale_ts(
            packet,
            encoder_ctx->time_base,
            output_stream->time_base);

        packet->stream_index =
            output_stream->index;

        ret = av_interleaved_write_frame(
            output_format_ctx,
            packet);

        av_packet_free(&packet);

        if (ret < 0)
        {
            fprintf(stderr,
                    "Error writing output packet\n");

            return ret;
        }
    }

    return 0;
}


/* =========================================================
 * Main
 * ========================================================= */

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr,
                "Usage: %s input.mp4 output.mp4\n",
                argv[0]);

        return EXIT_FAILURE;
    }

    const char *input_filename =
        argv[1];

    const char *output_filename =
        argv[2];


    /* -----------------------------------------------------
     * Input
     * ----------------------------------------------------- */

    AVFormatContext *input_format_ctx = NULL;

    if (avformat_open_input(
            &input_format_ctx,
            input_filename,
            NULL,
            NULL) < 0)
    {
        fprintf(stderr,
                "Could not open input file\n");

        return EXIT_FAILURE;
    }

    if (avformat_find_stream_info(
            input_format_ctx,
            NULL) < 0)
    {
        fprintf(stderr,
                "Could not find stream information\n");

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Find video stream
     * ----------------------------------------------------- */

    int video_stream_index = -1;

    for (unsigned int i = 0;
         i < input_format_ctx->nb_streams;
         i++)
    {
        if (input_format_ctx->streams[i]
                ->codecpar->codec_type ==
            AVMEDIA_TYPE_VIDEO)
        {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index < 0)
    {
        fprintf(stderr,
                "No video stream found\n");

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }

    AVStream *input_stream =
        input_format_ctx
            ->streams[video_stream_index];


    /* -----------------------------------------------------
     * Decoder
     * ----------------------------------------------------- */

    const AVCodec *decoder =
        avcodec_find_decoder(
            input_stream->codecpar->codec_id);

    if (!decoder)
    {
        fprintf(stderr,
                "Decoder not found\n");

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }

    AVCodecContext *decoder_ctx =
        avcodec_alloc_context3(decoder);

    if (!decoder_ctx)
    {
        fprintf(stderr,
                "Could not allocate decoder context\n");

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }

    if (avcodec_parameters_to_context(
            decoder_ctx,
            input_stream->codecpar) < 0)
    {
        fprintf(stderr,
                "Could not copy decoder parameters\n");

        avcodec_free_context(
            &decoder_ctx);

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }

    if (avcodec_open2(
            decoder_ctx,
            decoder,
            NULL) < 0)
    {
        fprintf(stderr,
                "Could not open decoder\n");

        avcodec_free_context(
            &decoder_ctx);

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }


    int width =
        decoder_ctx->width;

    int height =
        decoder_ctx->height;


    printf("Input video:\n");

    printf("  Resolution: %dx%d\n",
           width,
           height);

    printf("  Pixel format: %s\n",
           av_get_pix_fmt_name(
               (AVPixelFormat)decoder_ctx->pix_fmt));

    printf("\n");


    /* -----------------------------------------------------
     * Allocate input/output image buffers
     * ----------------------------------------------------- */

    uint8_t *gray =
        (uint8_t *)malloc(
            (size_t)width * height);

    uint8_t *edges =
        (uint8_t *)malloc(
            (size_t)width * height);

    if (!gray || !edges)
    {
        fprintf(stderr,
                "Could not allocate image buffers\n");

        free(gray);
        free(edges);

        avcodec_free_context(
            &decoder_ctx);

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * CUDA initialization
     *
     * GPU buffers are allocated ONCE here.
     * ----------------------------------------------------- */

    SobelCudaContext cuda_ctx;

    sobel_cuda_init(
        &cuda_ctx,
        width,
        height);


    /* -----------------------------------------------------
     * Output format
     * ----------------------------------------------------- */

    AVFormatContext *output_format_ctx =
        NULL;

    if (avformat_alloc_output_context2(
            &output_format_ctx,
            NULL,
            NULL,
            output_filename) < 0)
    {
        fprintf(stderr,
                "Could not create output context\n");

        sobel_cuda_destroy(
            &cuda_ctx);

        free(gray);
        free(edges);

        avcodec_free_context(
            &decoder_ctx);

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Encoder
     *
     * Prefer libx264.
     * ----------------------------------------------------- */

    const AVCodec *encoder =
        avcodec_find_encoder_by_name(
            "libx264");

    if (!encoder)
    {
        fprintf(stderr,
                "libx264 not available, "
                "using MPEG-4 encoder\n");

        encoder =
            avcodec_find_encoder(
                AV_CODEC_ID_MPEG4);
    }

    if (!encoder)
    {
        fprintf(stderr,
                "Could not find encoder\n");

        avformat_free_context(
            output_format_ctx);

        sobel_cuda_destroy(
            &cuda_ctx);

        free(gray);
        free(edges);

        avcodec_free_context(
            &decoder_ctx);

        avformat_close_input(
            &input_format_ctx);

        return EXIT_FAILURE;
    }


    AVStream *output_stream =
        avformat_new_stream(
            output_format_ctx,
            NULL);

    if (!output_stream)
    {
        fprintf(stderr,
                "Could not create output stream\n");

        return EXIT_FAILURE;
    }


    AVCodecContext *encoder_ctx =
        avcodec_alloc_context3(encoder);

    if (!encoder_ctx)
    {
        fprintf(stderr,
                "Could not allocate encoder context\n");

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Frame rate
     * ----------------------------------------------------- */

    AVRational frame_rate =
        av_guess_frame_rate(
            input_format_ctx,
            input_stream,
            NULL);

    if (frame_rate.num <= 0 ||
        frame_rate.den <= 0)
    {
        frame_rate.num = 25;
        frame_rate.den = 1;
    }


    encoder_ctx->width = width;
    encoder_ctx->height = height;

    encoder_ctx->pix_fmt =
        AV_PIX_FMT_YUV420P;

    encoder_ctx->time_base =
        av_inv_q(frame_rate);

    encoder_ctx->framerate =
        frame_rate;


    /* -----------------------------------------------------
     * H.264 settings
     * ----------------------------------------------------- */

    if (encoder->id == AV_CODEC_ID_H264)
    {
        av_opt_set(
            encoder_ctx->priv_data,
            "preset",
            "medium",
            0);

        av_opt_set(
            encoder_ctx->priv_data,
            "crf",
            "23",
            0);
    }


    if (output_format_ctx->oformat->flags &
        AVFMT_GLOBALHEADER)
    {
        encoder_ctx->flags |=
            AV_CODEC_FLAG_GLOBAL_HEADER;
    }


    if (avcodec_open2(
            encoder_ctx,
            encoder,
            NULL) < 0)
    {
        fprintf(stderr,
                "Could not open encoder\n");

        return EXIT_FAILURE;
    }


    if (avcodec_parameters_from_context(
            output_stream->codecpar,
            encoder_ctx) < 0)
    {
        fprintf(stderr,
                "Could not copy encoder parameters\n");

        return EXIT_FAILURE;
    }

    output_stream->time_base =
        encoder_ctx->time_base;


    /* -----------------------------------------------------
     * Open output file
     * ----------------------------------------------------- */

    if (!(output_format_ctx->oformat->flags &
          AVFMT_NOFILE))
    {
        if (avio_open(
                &output_format_ctx->pb,
                output_filename,
                AVIO_FLAG_WRITE) < 0)
        {
            fprintf(stderr,
                    "Could not open output file\n");

            return EXIT_FAILURE;
        }
    }


    if (avformat_write_header(
            output_format_ctx,
            NULL) < 0)
    {
        fprintf(stderr,
                "Could not write output header\n");

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Allocate FFmpeg frames
     * ----------------------------------------------------- */

    AVFrame *decoded_frame =
        av_frame_alloc();

    AVFrame *output_frame =
        av_frame_alloc();

    if (!decoded_frame ||
        !output_frame)
    {
        fprintf(stderr,
                "Could not allocate frames\n");

        return EXIT_FAILURE;
    }


    output_frame->format =
        AV_PIX_FMT_YUV420P;

    output_frame->width =
        width;

    output_frame->height =
        height;

    if (av_frame_get_buffer(
            output_frame,
            32) < 0)
    {
        fprintf(stderr,
                "Could not allocate output frame buffer\n");

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Packet
     * ----------------------------------------------------- */

    AVPacket *packet =
        av_packet_alloc();

    if (!packet)
    {
        fprintf(stderr,
                "Could not allocate packet\n");

        return EXIT_FAILURE;
    }


    /* -----------------------------------------------------
     * Processing statistics
     * ----------------------------------------------------- */

    int frame_count = 0;

    double total_processing_start =
        now_seconds();

    double total_gpu_stage_ms = 0.0;

    double total_processing_end = 0.0;
    double total_processing_seconds = 0.0;


    /* -----------------------------------------------------
     * Decode/process loop
     * ----------------------------------------------------- */

    while (av_read_frame(
               input_format_ctx,
               packet) >= 0)
    {
        if (packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(packet);
            continue;
        }


        int ret =
            avcodec_send_packet(
                decoder_ctx,
                packet);

        av_packet_unref(packet);

        if (ret < 0)
        {
            fprintf(stderr,
                    "Error sending packet to decoder\n");

            break;
        }


        while (1)
        {
            ret =
                avcodec_receive_frame(
                    decoder_ctx,
                    decoded_frame);

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
             * We expect YUV420P.
             *
             * The Sobel input is the Y/luma plane.
             * ------------------------------------------------- */

            if (decoded_frame->format !=
                AV_PIX_FMT_YUV420P)
            {
                fprintf(stderr,
                        "Unexpected pixel format: %s\n",
                        av_get_pix_fmt_name(
                            (AVPixelFormat)decoded_frame->format));

                goto cleanup;
            }


            /* -------------------------------------------------
             * Copy Y plane into contiguous CPU buffer.
             * ------------------------------------------------- */

            for (int y = 0;
                 y < height;
                 y++)
            {
                memcpy(
                    gray + y * width,
                    decoded_frame->data[0]
                        + y * decoded_frame->linesize[0],
                    width);
            }


            /* -------------------------------------------------
             * CUDA Sobel
             *
             * Measure complete CUDA stage:
             *
             * CPU -> GPU
             * kernel
             * GPU -> CPU
             * ------------------------------------------------- */

            double gpu_start =
                now_seconds();

            sobel_cuda_process(
                &cuda_ctx,
                gray,
                edges,
                width,
                height);

            double gpu_end =
                now_seconds();

            total_gpu_stage_ms +=
                (gpu_end - gpu_start) * 1000.0;


            /* -------------------------------------------------
             * Prepare output YUV420P frame
             *
             * Y = Sobel result
             * U = 128
             * V = 128
             * ------------------------------------------------- */

            if (av_frame_make_writable(
                    output_frame) < 0)
            {
                fprintf(stderr,
                        "Output frame is not writable\n");

                goto cleanup;
            }


            /* -------------------------------------------------
             * Copy Sobel result into Y plane.
             * ------------------------------------------------- */

            for (int y = 0;
                 y < height;
                 y++)
            {
                memcpy(
                    output_frame->data[0]
                        + y * output_frame->linesize[0],
                    edges + y * width,
                    width);
            }


            /* -------------------------------------------------
             * U and V = 128 produces grayscale output.
             * ------------------------------------------------- */

            for (int y = 0;
                 y < height / 2;
                 y++)
            {
                memset(
                    output_frame->data[1]
                        + y * output_frame->linesize[1],
                    128,
                    width / 2);

                memset(
                    output_frame->data[2]
                        + y * output_frame->linesize[2],
                    128,
                    width / 2);
            }


            /* -------------------------------------------------
             * Encode
             * ------------------------------------------------- */

            if (encode_frame(
                    encoder_ctx,
                    output_format_ctx,
                    output_stream,
                    output_frame,
                    frame_count) < 0)
            {
                goto cleanup;
            }


            frame_count++;


            if (frame_count % 25 == 0)
            {
                printf(
                    "Processed %d frames\n",
                    frame_count);
            }
        }
    }


    /* -----------------------------------------------------
     * Flush decoder
     * ----------------------------------------------------- */

    if (avcodec_send_packet(
            decoder_ctx,
            NULL) >= 0)
    {
        while (avcodec_receive_frame(
                   decoder_ctx,
                   decoded_frame) >= 0)
        {
            /*
             * Normally not reached for many MP4 inputs.
             */
        }
    }


    /* -----------------------------------------------------
     * Flush encoder
     * ----------------------------------------------------- */

    if (encode_frame(
            encoder_ctx,
            output_format_ctx,
            output_stream,
            NULL,
            frame_count) < 0)
    {
        goto cleanup;
    }


    av_write_trailer(
        output_format_ctx);


    /* -----------------------------------------------------
     * Statistics
     * ----------------------------------------------------- */

    total_processing_end =
        now_seconds();

    total_processing_seconds =
        total_processing_end -
        total_processing_start;


    printf("\n========================================\n");
    printf("CUDA Sobel results\n");
    printf("========================================\n");

    printf("Frames processed: %d\n",
           frame_count);

    printf("Resolution: %dx%d\n",
           width,
           height);

    printf("Total processing time: %.3f seconds\n",
           total_processing_seconds);

    if (frame_count > 0)
    {
        printf("Average CUDA stage: %.3f ms/frame\n",
               total_gpu_stage_ms /
               frame_count);

        printf("CUDA stage FPS: %.2f\n",
               1000.0 /
               (total_gpu_stage_ms /
                frame_count));

        printf("Average CUDA kernel: %.3f ms/frame\n",
               cuda_ctx.total_kernel_ms /
               frame_count);

        printf("CUDA kernel FPS: %.2f\n",
               1000.0 /
               (cuda_ctx.total_kernel_ms /
                frame_count));
    }

    printf("========================================\n");


cleanup:

    av_packet_free(
        &packet);

    av_frame_free(
        &decoded_frame);

    av_frame_free(
        &output_frame);

    avcodec_free_context(
        &encoder_ctx);

    avcodec_free_context(
        &decoder_ctx);

    if (output_format_ctx)
    {
        if (!(output_format_ctx->oformat->flags &
              AVFMT_NOFILE))
        {
            avio_closep(
                &output_format_ctx->pb);
        }

        avformat_free_context(
            output_format_ctx);
    }

    avformat_close_input(
        &input_format_ctx);

    sobel_cuda_destroy(
        &cuda_ctx);

    free(gray);
    free(edges);

    return EXIT_SUCCESS;
}
