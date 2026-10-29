/*
 * ============================================================
 * GPU VIDEO PROCESSOR
 *
 * Pipeline:
 *
 *     Input MP4
 *        |
 *        v
 *     FFmpeg demuxer
 *        |
 *        v
 *     NVIDIA CUVID / NVDEC
 *        |
 *        v
 *     AV_PIX_FMT_CUDA / NV12
 *        |
 *        v
 *     CUDA Sobel
 *        |
 *        v
 *     AV_PIX_FMT_CUDA / NV12
 *        |
 *        v
 *     NVIDIA NVENC H.264
 *        |
 *        v
 *     MP4 muxer
 *
 * Usage:
 *
 *     ./gpu_video_processor input.mp4 output.mp4
 *
 * Build:
 *
 * nvcc -O3 \
 *     -std=c++17 \
 *     -arch=sm_89 \
 *     -I/usr/include/x86_64-linux-gnu \
 *     sobel_cuda_enhanced.cu \
 *     -o gpu_video_processor \
 *     -lavformat -lavcodec -lavutil \
 *     -lcudart
 *
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

extern "C" {

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>

}


/* ============================================================
 * CUDA error checking
 * ============================================================ */

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
            return AVERROR_EXTERNAL;                          \
        }                                                     \
    } while (0)


/* ============================================================
 * FFmpeg error helper
 * ============================================================ */

static void print_ffmpeg_error(
    const char *message,
    int err)
{
    char error_string[AV_ERROR_MAX_STRING_SIZE];

    av_strerror(
        err,
        error_string,
        sizeof(error_string));

    fprintf(
        stderr,
        "%s: %s\n",
        message,
        error_string);
}


/* ============================================================
 * CUDA Sobel kernel
 *
 * Input:
 *
 *     NV12 Y plane
 *
 * Output:
 *
 *     Sobel Y plane
 *
 * UV:
 *
 *     128 / 128
 *
 * This produces grayscale Sobel output.
 * ============================================================ */

__global__ void sobel_nv12_kernel(
    const uint8_t *input_y,
    int input_y_stride,

    uint8_t *output_y,
    int output_y_stride,

    uint8_t *output_uv,
    int output_uv_stride,

    int width,
    int height)
{
    int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;


    /* --------------------------------------------------------
     * Y plane
     * -------------------------------------------------------- */

    if (x < width &&
        y < height)
    {
        if (x == 0 ||
            x == width - 1 ||
            y == 0 ||
            y == height - 1)
        {
            output_y[
                y * output_y_stride + x
            ] = 0;
        }
        else
        {
            const uint8_t *row0 =
                input_y +
                (y - 1) * input_y_stride;

            const uint8_t *row1 =
                input_y +
                y * input_y_stride;

            const uint8_t *row2 =
                input_y +
                (y + 1) * input_y_stride;


            int gx =
                -row0[x - 1] +
                 row0[x + 1]

                -2 * row1[x - 1] +
                 2 * row1[x + 1]

                -row2[x - 1] +
                 row2[x + 1];


            int gy =
                -row0[x - 1] -
                 2 * row0[x] -
                 row0[x + 1]

                +row2[x - 1] +
                 2 * row2[x] +
                 row2[x + 1];


            int magnitude =
                abs(gx) +
                abs(gy);


            if (magnitude > 255)
                magnitude = 255;


            output_y[
                y * output_y_stride + x
            ] = (uint8_t)magnitude;
        }
    }


    /* --------------------------------------------------------
     * UV plane
     *
     * One thread writes one UV pair.
     * -------------------------------------------------------- */

    if (x < width / 2 &&
        y < height / 2)
    {
        uint8_t *uv_row =
            output_uv +
            y * output_uv_stride;

        uv_row[
            x * 2 + 0
        ] = 128;

        uv_row[
            x * 2 + 1
        ] = 128;
    }
}


/* ============================================================
 * Sobel context
 * ============================================================ */

typedef struct
{
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;

    double total_kernel_ms;

    int64_t frames_processed;

} SobelCudaContext;


/* ============================================================
 * Initialize Sobel
 * ============================================================ */

static int sobel_cuda_init(
    SobelCudaContext *ctx)
{
    memset(
        ctx,
        0,
        sizeof(*ctx));


    cudaError_t err;


    err =
        cudaEventCreate(
            &ctx->kernel_start);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventCreate failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    err =
        cudaEventCreate(
            &ctx->kernel_stop);

    if (err != cudaSuccess)
    {
        cudaEventDestroy(
            ctx->kernel_start);

        fprintf(
            stderr,
            "cudaEventCreate failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    return 0;
}


/* ============================================================
 * Destroy Sobel
 * ============================================================ */

static void sobel_cuda_destroy(
    SobelCudaContext *ctx)
{
    if (!ctx)
        return;


    if (ctx->kernel_start)
    {
        cudaEventDestroy(
            ctx->kernel_start);

        ctx->kernel_start = nullptr;
    }


    if (ctx->kernel_stop)
    {
        cudaEventDestroy(
            ctx->kernel_stop);

        ctx->kernel_stop = nullptr;
    }
}


/* ============================================================
 * Process CUDA frame
 * ============================================================ */

static int sobel_cuda_process_frame(
    SobelCudaContext *ctx,
    const AVFrame *input_frame,
    AVFrame *output_frame,
    int width,
    int height)
{
    if (!input_frame ||
        !output_frame)
    {
        return AVERROR(EINVAL);
    }


    if (input_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Input frame is not AV_PIX_FMT_CUDA\n");

        return AVERROR(EINVAL);
    }


    if (output_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Output frame is not AV_PIX_FMT_CUDA\n");

        return AVERROR(EINVAL);
    }


    const uint8_t *d_input_y =
        input_frame->data[0];

    uint8_t *d_output_y =
        output_frame->data[0];

    uint8_t *d_output_uv =
        output_frame->data[1];


    if (!d_input_y ||
        !d_output_y ||
        !d_output_uv)
    {
        fprintf(
            stderr,
            "Invalid CUDA frame plane pointers\n");

        return AVERROR(EINVAL);
    }


    int input_y_stride =
        input_frame->linesize[0];

    int output_y_stride =
        output_frame->linesize[0];

    int output_uv_stride =
        output_frame->linesize[1];


    dim3 block(
        16,
        16);


    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);


    cudaError_t err;


    err =
        cudaEventRecord(
            ctx->kernel_start,
            0);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventRecord failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    sobel_nv12_kernel<<<
        grid,
        block
    >>>(
        d_input_y,
        input_y_stride,

        d_output_y,
        output_y_stride,

        d_output_uv,
        output_uv_stride,

        width,
        height);


    err =
        cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "Sobel kernel launch failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    err =
        cudaEventRecord(
            ctx->kernel_stop,
            0);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventRecord failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    /*
     * Synchronize before NVENC receives the frame.
     */

    err =
        cudaEventSynchronize(
            ctx->kernel_stop);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventSynchronize failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    float elapsed_ms = 0.0f;


    err =
        cudaEventElapsedTime(
            &elapsed_ms,
            ctx->kernel_start,
            ctx->kernel_stop);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventElapsedTime failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


    ctx->total_kernel_ms +=
        elapsed_ms;

    ctx->frames_processed++;


    return 0;
}


/* ============================================================
 * Create CUDA device
 * ============================================================ */

static int create_cuda_device(
    int cuda_device_index,
    AVBufferRef **hw_device_ctx)
{
    char device_string[32];


    snprintf(
        device_string,
        sizeof(device_string),
        "%d",
        cuda_device_index);


    int ret =
        av_hwdevice_ctx_create(
            hw_device_ctx,
            AV_HWDEVICE_TYPE_CUDA,
            device_string,
            NULL,
            0);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not create CUDA hardware device",
            ret);

        return ret;
    }


    return 0;
}


/* ============================================================
 * FFmpeg CUDA pixel format callback
 * ============================================================ */

static enum AVPixelFormat get_hw_format(
    AVCodecContext *ctx,
    const enum AVPixelFormat *pix_fmts)
{
    (void)ctx;


    for (const enum AVPixelFormat *p = pix_fmts;
         *p != AV_PIX_FMT_NONE;
         ++p)
    {
        if (*p == AV_PIX_FMT_CUDA)
        {
            return AV_PIX_FMT_CUDA;
        }
    }


    fprintf(
        stderr,
        "Decoder did not provide AV_PIX_FMT_CUDA\n");


    return AV_PIX_FMT_NONE;
}


/* ============================================================
 * Create CUDA frame context
 * ============================================================ */

static int create_cuda_frames_context(
    AVBufferRef *hw_device_ctx,
    AVBufferRef **frames_ref,
    int width,
    int height)
{
    AVBufferRef *frames_ctx_ref =
        av_hwframe_ctx_alloc(
            hw_device_ctx);


    if (!frames_ctx_ref)
    {
        fprintf(
            stderr,
            "Could not allocate CUDA frame context\n");

        return AVERROR(ENOMEM);
    }


    AVHWFramesContext *frames_ctx =
        (AVHWFramesContext *)
        frames_ctx_ref->data;


    frames_ctx->format =
        AV_PIX_FMT_CUDA;

    frames_ctx->sw_format =
        AV_PIX_FMT_NV12;

    frames_ctx->width =
        width;

    frames_ctx->height =
        height;

    frames_ctx->initial_pool_size =
        16;


    int ret =
        av_hwframe_ctx_init(
            frames_ctx_ref);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not initialize CUDA frame context",
            ret);

        av_buffer_unref(
            &frames_ctx_ref);

        return ret;
    }


    *frames_ref =
        frames_ctx_ref;


    return 0;
}


/* ============================================================
 * Open decoder
 * ============================================================ */

static int open_decoder(
    AVFormatContext *input_ctx,
    int video_stream_index,
    AVBufferRef *hw_device_ctx,
    AVCodecContext **decoder_ctx_out)
{
    if (!input_ctx ||
        video_stream_index < 0)
    {
        return AVERROR(EINVAL);
    }


    AVStream *stream =
        input_ctx->streams[
            video_stream_index];


    const AVCodec *decoder =
        avcodec_find_decoder_by_name(
            "h264_cuvid");


    if (!decoder)
    {
        fprintf(
            stderr,
            "h264_cuvid decoder not found\n");

        return AVERROR_DECODER_NOT_FOUND;
    }


    AVCodecContext *decoder_ctx =
        avcodec_alloc_context3(
            decoder);


    if (!decoder_ctx)
    {
        return AVERROR(ENOMEM);
    }


    int ret =
        avcodec_parameters_to_context(
            decoder_ctx,
            stream->codecpar);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not copy decoder parameters",
            ret);

        avcodec_free_context(
            &decoder_ctx);

        return ret;
    }


    decoder_ctx->get_format =
        get_hw_format;


    decoder_ctx->hw_device_ctx =
        av_buffer_ref(
            hw_device_ctx);


    if (!decoder_ctx->hw_device_ctx)
    {
        avcodec_free_context(
            &decoder_ctx);

        return AVERROR(ENOMEM);
    }


    decoder_ctx->thread_count =
        1;


    ret =
        avcodec_open2(
            decoder_ctx,
            decoder,
            NULL);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not open h264_cuvid",
            ret);

        avcodec_free_context(
            &decoder_ctx);

        return ret;
    }


    *decoder_ctx_out =
        decoder_ctx;


    return 0;
}


/* ============================================================
 * Open NVENC encoder
 * ============================================================ */

static int open_encoder(
    AVBufferRef *hw_device_ctx,
    AVBufferRef *encoder_frames_ref,
    int width,
    int height,
    AVRational input_time_base,
    AVRational input_framerate,
    AVCodecContext **encoder_ctx_out)
{
    const AVCodec *encoder =
        avcodec_find_encoder_by_name(
            "h264_nvenc");


    if (!encoder)
    {
        fprintf(
            stderr,
            "h264_nvenc encoder not found\n");

        return AVERROR_ENCODER_NOT_FOUND;
    }


    AVCodecContext *encoder_ctx =
        avcodec_alloc_context3(
            encoder);


    if (!encoder_ctx)
    {
        return AVERROR(ENOMEM);
    }


    encoder_ctx->width =
        width;

    encoder_ctx->height =
        height;


    /*
     * NVENC receives CUDA frames.
     */

    encoder_ctx->pix_fmt =
        AV_PIX_FMT_CUDA;


    /*
     * IMPORTANT:
     *
     * Keep the same timestamp clock as the
     * original video stream.
     */

    encoder_ctx->time_base =
        input_time_base;


    encoder_ctx->framerate =
        input_framerate;


    encoder_ctx->bit_rate =
        5 * 1000 * 1000;


    encoder_ctx->gop_size =
        60;


    encoder_ctx->max_b_frames =
        0;


    /*
     * Hardware device.
     */

    encoder_ctx->hw_device_ctx =
        av_buffer_ref(
            hw_device_ctx);


    if (!encoder_ctx->hw_device_ctx)
    {
        avcodec_free_context(
            &encoder_ctx);

        return AVERROR(ENOMEM);
    }


    /*
     * CUDA output frame pool.
     */

    encoder_ctx->hw_frames_ctx =
        av_buffer_ref(
            encoder_frames_ref);


    if (!encoder_ctx->hw_frames_ctx)
    {
        avcodec_free_context(
            &encoder_ctx);

        return AVERROR(ENOMEM);
    }


    /*
     * NVENC settings.
     */

    av_opt_set(
        encoder_ctx->priv_data,
        "preset",
        "p4",
        0);


    av_opt_set(
        encoder_ctx->priv_data,
        "tune",
        "hq",
        0);


    av_opt_set(
        encoder_ctx->priv_data,
        "profile",
        "high",
        0);


    /*
     * Avoid NVENC introducing B-frame
     * reordering into our PTS timeline.
     */

    av_opt_set(
        encoder_ctx->priv_data,
        "bf",
        "0",
        0);


    int ret =
        avcodec_open2(
            encoder_ctx,
            encoder,
            NULL);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not open h264_nvenc",
            ret);

        avcodec_free_context(
            &encoder_ctx);

        return ret;
    }


    *encoder_ctx_out =
        encoder_ctx;


    return 0;
}


/* ============================================================
 * Encode frame and write packets
 * ============================================================ */

static int encode_and_write(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    AVFrame *frame)
{
    int ret =
        avcodec_send_frame(
            encoder_ctx,
            frame);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Error sending frame to encoder",
            ret);

        return ret;
    }


    AVPacket *packet =
        av_packet_alloc();


    if (!packet)
    {
        return AVERROR(ENOMEM);
    }


    while (1)
    {
        ret =
            avcodec_receive_packet(
                encoder_ctx,
                packet);


        if (ret == AVERROR(EAGAIN) ||
            ret == AVERROR_EOF)
        {
            ret = 0;
            break;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error receiving encoded packet",
                ret);

            break;
        }


        /*
         * Encoder PTS/DTS are in encoder
         * time base.
         *
         * Convert to MP4 stream time base.
         */

        av_packet_rescale_ts(
            packet,
            encoder_ctx->time_base,
            output_stream->time_base);


        packet->stream_index =
            output_stream->index;


        ret =
            av_interleaved_write_frame(
                output_ctx,
                packet);


        av_packet_unref(
            packet);


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error writing encoded packet",
                ret);

            break;
        }
    }


    av_packet_free(
        &packet);


    return ret;
}


/* ============================================================
 * Process decoded frame
 * ============================================================ */

static int process_decoded_frame(
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    AVBufferRef *output_frames_ref,
    SobelCudaContext *sobel_ctx,
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    int width,
    int height)
{
    if (!decoded_frame)
        return AVERROR(EINVAL);


    /*
     * Make sure we have a CUDA frame.
     */

    if (decoded_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Decoder returned format %s instead of CUDA\n",
            av_get_pix_fmt_name(
                (AVPixelFormat)decoded_frame->format));

        return AVERROR(EINVAL);
    }


    /*
     * Release previous output frame.
     */

    av_frame_unref(
        output_frame);


    output_frame->format =
        AV_PIX_FMT_CUDA;

    output_frame->width =
        width;

    output_frame->height =
        height;


    /*
     * Allocate GPU frame.
     */

    int ret =
        av_hwframe_get_buffer(
            output_frames_ref,
            output_frame,
            0);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not allocate CUDA output frame",
            ret);

        return ret;
    }


    /*
     * CRITICAL:
     *
     * Preserve the decoder PTS exactly.
     *
     * Decoder PTS and encoder time_base are both
     * based on the original input stream clock.
     */

    output_frame->pts =
        decoded_frame->pts;


    /*
     * Preserve duration if the decoder provides it.
     */

    output_frame->duration =
        decoded_frame->duration;


    ret =
        sobel_cuda_process_frame(
            sobel_ctx,
            decoded_frame,
            output_frame,
            width,
            height);


    if (ret < 0)
    {
        return ret;
    }


    ret =
        encode_and_write(
            encoder_ctx,
            output_ctx,
            output_stream,
            output_frame);


    return ret;
}


/* ============================================================
 * Find GPU by name
 * ============================================================ */

static int find_gpu_by_name(
    const char *name)
{
    int device_count = 0;


    cudaError_t err =
        cudaGetDeviceCount(
            &device_count);


    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaGetDeviceCount failed: %s\n",
            cudaGetErrorString(err));

        return -1;
    }


    for (int i = 0;
         i < device_count;
         ++i)
    {
        cudaDeviceProp prop;


        err =
            cudaGetDeviceProperties(
                &prop,
                i);


        if (err != cudaSuccess)
        {
            continue;
        }


        printf(
            "CUDA device %d: %s\n",
            i,
            prop.name);


        if (strstr(
                prop.name,
                name) != NULL)
        {
            return i;
        }
    }


    return -1;
}


/* ============================================================
 * Main
 * ============================================================ */

int main(
    int argc,
    char **argv)
{
    /*
     * --------------------------------------------------------
     * Arguments
     * --------------------------------------------------------
     */

    if (argc != 3)
    {
        fprintf(
            stderr,
            "\nUsage:\n"
            "  %s input.mp4 output.mp4\n\n",
            argv[0]);

        return EXIT_FAILURE;
    }


    const char *input_filename =
        argv[1];

    const char *output_filename =
        argv[2];


    /*
     * --------------------------------------------------------
     * All resources declared FIRST.
     *
     * This deliberately avoids C++ goto initialization
     * problems.
     * --------------------------------------------------------
     */

    int ret = 0;

    int exit_code =
        EXIT_FAILURE;


    AVFormatContext *input_ctx =
        NULL;

    AVFormatContext *output_ctx =
        NULL;


    AVBufferRef *hw_device_ctx =
        NULL;

    AVBufferRef *output_frames_ref =
        NULL;


    AVCodecContext *decoder_ctx =
        NULL;

    AVCodecContext *encoder_ctx =
        NULL;


    AVStream *input_stream =
        NULL;

    AVStream *output_stream =
        NULL;


    AVPacket *packet =
        NULL;

    AVFrame *decoded_frame =
        NULL;

    AVFrame *output_frame =
        NULL;


    SobelCudaContext sobel_ctx;

    memset(
        &sobel_ctx,
        0,
        sizeof(sobel_ctx));


    int video_stream_index =
        -1;


    int width =
        0;

    int height =
        0;


    AVRational input_time_base =
        {0, 1};

    AVRational input_framerate =
        {0, 1};


    int64_t frame_count =
        0;


    int64_t first_input_pts =
        AV_NOPTS_VALUE;

    int64_t last_input_pts =
        AV_NOPTS_VALUE;


    int64_t first_output_pts =
        AV_NOPTS_VALUE;

    int64_t last_output_pts =
        AV_NOPTS_VALUE;


    double start_time =
        0.0;

    double end_time =
        0.0;

    double elapsed =
        0.0;


    /*
     * --------------------------------------------------------
     * CUDA GPU selection
     * --------------------------------------------------------
     */

    int selected_device =
        find_gpu_by_name(
            "RTX 4060 Ti");


    if (selected_device < 0)
    {
        fprintf(
            stderr,
            "RTX 4060 Ti not found\n");

        goto cleanup;
    }


    printf(
        "\nSelecting CUDA device %d...\n",
        selected_device);


    {
        cudaError_t cuda_ret =
            cudaSetDevice(
                selected_device);


        if (cuda_ret != cudaSuccess)
        {
            fprintf(
                stderr,
                "cudaSetDevice failed: %s\n",
                cudaGetErrorString(cuda_ret));

            goto cleanup;
        }
    }


    {
        cudaDeviceProp device_prop;


        cudaError_t cuda_ret =
            cudaGetDeviceProperties(
                &device_prop,
                selected_device);


        if (cuda_ret != cudaSuccess)
        {
            fprintf(
                stderr,
                "cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(cuda_ret));

            goto cleanup;
        }


        printf(
            "Using CUDA device %d: %s\n",
            selected_device,
            device_prop.name);
    }


    /*
     * --------------------------------------------------------
     * Open input
     * --------------------------------------------------------
     */

    printf(
        "\nOpening input: %s\n",
        input_filename);


    ret =
        avformat_open_input(
            &input_ctx,
            input_filename,
            NULL,
            NULL);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not open input",
            ret);

        goto cleanup;
    }


    ret =
        avformat_find_stream_info(
            input_ctx,
            NULL);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not find stream information",
            ret);

        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Find video stream
     * --------------------------------------------------------
     */

    video_stream_index =
        av_find_best_stream(
            input_ctx,
            AVMEDIA_TYPE_VIDEO,
            -1,
            -1,
            NULL,
            0);


    if (video_stream_index < 0)
    {
        print_ffmpeg_error(
            "Could not find video stream",
            video_stream_index);

        goto cleanup;
    }


    input_stream =
        input_ctx->streams[
            video_stream_index];


    if (!input_stream)
    {
        fprintf(
            stderr,
            "Invalid input stream\n");

        goto cleanup;
    }


    width =
        input_stream->codecpar->width;

    height =
        input_stream->codecpar->height;


    input_time_base =
        input_stream->time_base;


    /*
     * Prefer average frame rate.
     */

    input_framerate =
        input_stream->avg_frame_rate;


    if (input_framerate.num <= 0 ||
        input_framerate.den <= 0)
    {
        input_framerate =
            input_stream->r_frame_rate;
    }


    printf(
        "\nInput video\n");

    printf(
        "  Stream index : %d\n",
        video_stream_index);

    printf(
        "  Codec        : %s\n",
        avcodec_get_name(
            input_stream->codecpar->codec_id));

    printf(
        "  Resolution   : %dx%d\n",
        width,
        height);

    printf(
        "  Time base    : %d/%d\n",
        input_time_base.num,
        input_time_base.den);

    printf(
        "  Frame rate   : %d/%d (%.6f fps)\n",
        input_framerate.num,
        input_framerate.den,
        av_q2d(input_framerate));

    printf(
        "  Duration     : %" PRId64 " ticks\n",
        input_stream->duration);

    printf(
        "  Duration     : %.6f seconds\n",
        input_stream->duration == AV_NOPTS_VALUE
            ? 0.0
            : input_stream->duration *
              av_q2d(input_stream->time_base));


    /*
     * --------------------------------------------------------
     * CUDA hardware device
     * --------------------------------------------------------
     */

    ret =
        create_cuda_device(
            selected_device,
            &hw_device_ctx);


    if (ret < 0)
    {
        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Decoder
     * --------------------------------------------------------
     */

    ret =
        open_decoder(
            input_ctx,
            video_stream_index,
            hw_device_ctx,
            &decoder_ctx);


    if (ret < 0)
    {
        goto cleanup;
    }


    width =
        decoder_ctx->width;

    height =
        decoder_ctx->height;


    /*
     * --------------------------------------------------------
     * CUDA output frames
     * --------------------------------------------------------
     */

    ret =
        create_cuda_frames_context(
            hw_device_ctx,
            &output_frames_ref,
            width,
            height);


    if (ret < 0)
    {
        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Create MP4 output
     * --------------------------------------------------------
     */

    ret =
        avformat_alloc_output_context2(
            &output_ctx,
            NULL,
            "mp4",
            output_filename);


    if (ret < 0 ||
        !output_ctx)
    {
        fprintf(
            stderr,
            "Could not create output format context\n");

        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Encoder
     * --------------------------------------------------------
     *
     * IMPORTANT:
     *
     * Encoder time_base == input stream time_base.
     *
     * This allows input PTS to pass through without
     * an accidental timestamp conversion.
     * --------------------------------------------------------
     */

    ret =
        open_encoder(
            hw_device_ctx,
            output_frames_ref,
            width,
            height,
            input_time_base,
            input_framerate,
            &encoder_ctx);


    if (ret < 0)
    {
        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Output stream
     * --------------------------------------------------------
     */

    output_stream =
        avformat_new_stream(
            output_ctx,
            NULL);


    if (!output_stream)
    {
        fprintf(
            stderr,
            "Could not create output stream\n");

        goto cleanup;
    }


    /*
     * Use exactly the encoder time base.
     */

    output_stream->time_base =
        encoder_ctx->time_base;


    ret =
        avcodec_parameters_from_context(
            output_stream->codecpar,
            encoder_ctx);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not copy encoder parameters",
            ret);

        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Output file
     * --------------------------------------------------------
     */

    if (!(output_ctx->oformat->flags &
          AVFMT_NOFILE))
    {
        ret =
            avio_open(
                &output_ctx->pb,
                output_filename,
                AVIO_FLAG_WRITE);


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Could not open output file",
                ret);

            goto cleanup;
        }
    }


    /*
     * --------------------------------------------------------
     * Write MP4 header
     * --------------------------------------------------------
     */

    ret =
        avformat_write_header(
            output_ctx,
            NULL);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not write output header",
            ret);

        goto cleanup;
    }



    /*
     * --------------------------------------------------------
     * Sobel
     * --------------------------------------------------------
     */

    ret =
        sobel_cuda_init(
            &sobel_ctx);


    if (ret < 0)
    {
        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Allocate packet/frame objects
     * --------------------------------------------------------
     */

    packet =
        av_packet_alloc();

    decoded_frame =
        av_frame_alloc();

    output_frame =
        av_frame_alloc();


    if (!packet ||
        !decoded_frame ||
        !output_frame)
    {
        fprintf(
            stderr,
            "Could not allocate packet/frame\n");

        ret =
            AVERROR(ENOMEM);

        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Start timing
     * --------------------------------------------------------
     */

    start_time =
        (double)clock() /
        CLOCKS_PER_SEC;


    printf(
        "\nStarting GPU processing...\n\n");


    /*
     * --------------------------------------------------------
     * MAIN DEMUX LOOP
     * --------------------------------------------------------
     */

    while (1)
    {
        ret =
            av_read_frame(
                input_ctx,
                packet);


        if (ret == AVERROR_EOF)
        {
            /*
             * Normal end of file.
             */

            ret = 0;

            break;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error reading input packet",
                ret);

            goto cleanup;
        }


        /*
         * Ignore audio and other streams.
         */

        if (packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(
                packet);

            continue;
        }


        /*
         * Send compressed H.264 packet
         * to NVDEC.
         */

        ret =
            avcodec_send_packet(
                decoder_ctx,
                packet);


        av_packet_unref(
            packet);


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error sending packet to decoder",
                ret);

            goto cleanup;
        }


        /*
         * Receive ALL frames generated
         * by this packet.
         */

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
                print_ffmpeg_error(
                    "Error receiving decoded frame",
                    ret);

                goto cleanup;
            }


            /*
             * Track original PTS.
             */

            if (decoded_frame->pts !=
                AV_NOPTS_VALUE)
            {
                if (first_input_pts ==
                    AV_NOPTS_VALUE)
                {
                    first_input_pts =
                        decoded_frame->pts;
                }

                last_input_pts =
                    decoded_frame->pts;
            }


            /*
             * Process frame.
             */

            ret =
                process_decoded_frame(
                    decoded_frame,
                    output_frame,
                    output_frames_ref,
                    &sobel_ctx,
                    encoder_ctx,
                    output_ctx,
                    output_stream,
                    width,
                    height);


            if (ret < 0)
            {
                goto cleanup;
            }


            /*
             * Track output PTS.
             *
             * The PTS is the same value we assigned
             * to output_frame.
             */

            if (output_frame->pts !=
                AV_NOPTS_VALUE)
            {
                if (first_output_pts ==
                    AV_NOPTS_VALUE)
                {
                    first_output_pts =
                        output_frame->pts;
                }

                last_output_pts =
                    output_frame->pts;
            }


            frame_count++;


            if ((frame_count % 30) == 0)
            {
                printf(
                    "\rProcessed frames: %" PRId64,
                    frame_count);

                fflush(stdout);
            }
        }
    }


    printf(
        "\n");


    /*
     * --------------------------------------------------------
     * FLUSH DECODER
     * --------------------------------------------------------
     *
     * This is important for codecs with delayed frames.
     *
     * Without this, the final decoded frames can be lost.
     * --------------------------------------------------------
     */

    ret =
        avcodec_send_packet(
            decoder_ctx,
            NULL);


    if (ret < 0 &&
        ret != AVERROR_EOF)
    {
        print_ffmpeg_error(
            "Could not send decoder flush",
            ret);

        goto cleanup;
    }


    while (1)
    {
        ret =
            avcodec_receive_frame(
                decoder_ctx,
                decoded_frame);


        if (ret == AVERROR_EOF ||
            ret == AVERROR(EAGAIN))
        {
            break;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error receiving flushed decoder frame",
                ret);

            goto cleanup;
        }


        if (decoded_frame->pts !=
            AV_NOPTS_VALUE)
        {
            if (first_input_pts ==
                AV_NOPTS_VALUE)
            {
                first_input_pts =
                    decoded_frame->pts;
            }

            last_input_pts =
                decoded_frame->pts;
        }


        ret =
            process_decoded_frame(
                decoded_frame,
                output_frame,
                output_frames_ref,
                &sobel_ctx,
                encoder_ctx,
                output_ctx,
                output_stream,
                width,
                height);


        if (ret < 0)
        {
            goto cleanup;
        }


        if (output_frame->pts !=
            AV_NOPTS_VALUE)
        {
            if (first_output_pts ==
                AV_NOPTS_VALUE)
            {
                first_output_pts =
                    output_frame->pts;
            }

            last_output_pts =
                output_frame->pts;
        }


        frame_count++;
    }


    /*
     * --------------------------------------------------------
     * FLUSH NVENC
     * --------------------------------------------------------
     */

    ret =
        encode_and_write(
            encoder_ctx,
            output_ctx,
            output_stream,
            NULL);


    if (ret < 0)
    {
        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Calculate expected duration
     * --------------------------------------------------------
     *
     * For CFR video:
     *
     *     duration ~= number_of_frames / fps
     *
     * But PTS is the authoritative timeline.
     *
     * We use the original stream duration when available.
     * --------------------------------------------------------
     */

    if (input_stream->duration !=
        AV_NOPTS_VALUE)
    {
        output_stream->duration =
            av_rescale_q(
                input_stream->duration,
                input_stream->time_base,
                output_stream->time_base);
    }
    else if (frame_count > 0 &&
             input_framerate.num > 0 &&
             input_framerate.den > 0)
    {
        int64_t duration_pts =
            av_rescale_q(
                frame_count,
                av_inv_q(input_framerate),
                output_stream->time_base);


        output_stream->duration =
            duration_pts;
    }


    /*
     * --------------------------------------------------------
     * Write trailer
     * --------------------------------------------------------
     */

    ret =
        av_write_trailer(
            output_ctx);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not write output trailer",
            ret);

        goto cleanup;
    }


    /*
     * --------------------------------------------------------
     * Timing
     * --------------------------------------------------------
     */

    end_time =
        (double)clock() /
        CLOCKS_PER_SEC;


    elapsed =
        end_time -
        start_time;


    /*
     * --------------------------------------------------------
     * Statistics
     * --------------------------------------------------------
     */

    printf(
        "\n========================================\n");

    printf(
        "GPU VIDEO PROCESSING COMPLETE\n");

    printf(
        "========================================\n");


    printf(
        "Frames processed : %" PRId64 "\n",
        frame_count);


    printf(
        "CUDA Sobel time  : %.3f ms\n",
        sobel_ctx.total_kernel_ms);


    printf(
        "Average Sobel    : %.3f ms/frame\n",
        frame_count > 0
            ? sobel_ctx.total_kernel_ms /
              frame_count
            : 0.0);


    printf(
        "CPU wall time    : %.3f seconds\n",
        elapsed);


    /*
     * Input timeline.
     */

    if (first_input_pts != AV_NOPTS_VALUE &&
        last_input_pts != AV_NOPTS_VALUE)
    {
        double input_pts_span =
            (last_input_pts -
             first_input_pts) *
            av_q2d(input_time_base);


        printf(
            "Input first PTS  : %" PRId64 "\n",
            first_input_pts);


        printf(
            "Input last PTS   : %" PRId64 "\n",
            last_input_pts);


        printf(
            "Input PTS span   : %.6f seconds\n",
            input_pts_span);
    }


    /*
     * Output timeline.
     */

    if (first_output_pts != AV_NOPTS_VALUE &&
        last_output_pts != AV_NOPTS_VALUE)
    {
        double output_pts_span =
            (last_output_pts -
             first_output_pts) *
            av_q2d(output_stream->time_base);


        printf(
            "Output first PTS : %" PRId64 "\n",
            first_output_pts);


        printf(
            "Output last PTS  : %" PRId64 "\n",
            last_output_pts);


        printf(
            "Output PTS span  : %.6f seconds\n",
            output_pts_span);
    }


    if (output_stream->duration !=
        AV_NOPTS_VALUE)
    {
        double output_duration =
            output_stream->duration *
            av_q2d(
                output_stream->time_base);


        printf(
            "Output duration  : %.6f seconds\n",
            output_duration);
    }


    printf(
        "Output           : %s\n",
        output_filename);


    printf(
        "========================================\n");


    exit_code =
        EXIT_SUCCESS;


cleanup:

    /*
     * --------------------------------------------------------
     * Cleanup
     * --------------------------------------------------------
     *
     * No goto jumps over variable initialization because all
     * variables were declared at the beginning of main().
     * --------------------------------------------------------
     */

    if (packet)
    {
        av_packet_free(
            &packet);
    }


    if (decoded_frame)
    {
        av_frame_free(
            &decoded_frame);
    }


    if (output_frame)
    {
        av_frame_free(
            &output_frame);
    }


    sobel_cuda_destroy(
        &sobel_ctx);


    if (encoder_ctx)
    {
        avcodec_free_context(
            &encoder_ctx);
    }


    if (decoder_ctx)
    {
        avcodec_free_context(
            &decoder_ctx);
    }


    if (output_frames_ref)
    {
        av_buffer_unref(
            &output_frames_ref);
    }


    if (hw_device_ctx)
    {
        av_buffer_unref(
            &hw_device_ctx);
    }


    if (output_ctx)
    {
        if (!(output_ctx->oformat->flags &
              AVFMT_NOFILE))
        {
            if (output_ctx->pb)
            {
                avio_closep(
                    &output_ctx->pb);
            }
        }


        avformat_free_context(
            output_ctx);

        output_ctx =
            NULL;
    }


    if (input_ctx)
    {
        avformat_close_input(
            &input_ctx);
    }


    if (exit_code != EXIT_SUCCESS)
    {
        fprintf(
            stderr,
            "\nProcessing failed.\n");
    }
    else
    {
        printf(
            "\nSuccessfully created: %s\n",
            output_filename);
    }


    return exit_code;
}
