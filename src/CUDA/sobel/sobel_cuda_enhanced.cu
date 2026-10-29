/*
 * ============================================================
 * GENERIC GPU VIDEO PROCESSOR
 *
 * Pipeline:
 *
 *     H.264 MP4
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
 *
 * Usage:
 *
 *     ./gpu_video_processor input.mp4 output.mp4
 *
 * Optional:
 *
 *     CUDA_DEVICE_INDEX=1 ./gpu_video_processor input.mp4 output.mp4
 *
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

extern "C"
{

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>

}

/* ============================================================
 * Wall-clock timing
 * ============================================================ */

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts);

    return (double)ts.tv_sec * 1000.0
         + (double)ts.tv_nsec / 1000000.0;
}


/* ============================================================
 * CUDA error helper
 * ============================================================ */

#define CUDA_CHECK_RET(call)                                      \
    do                                                            \
    {                                                             \
        cudaError_t err__ = (call);                               \
        if (err__ != cudaSuccess)                                 \
        {                                                         \
            fprintf(stderr,                                       \
                    "CUDA error at %s:%d: %s\n",                  \
                    __FILE__,                                     \
                    __LINE__,                                     \
                    cudaGetErrorString(err__));                   \
            return AVERROR_EXTERNAL;                              \
        }                                                         \
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
 * Rational printer
 * ============================================================ */

static void print_rational(
    const char *name,
    AVRational value)
{
    printf(
        "%s: %d/%d",
        name,
        value.num,
        value.den);

    if (value.den != 0)
    {
        printf(
            " (%.6f)",
            av_q2d(value));
    }

    printf("\n");
}


/* ============================================================
 * PTS printer
 * ============================================================ */

static void print_pts_value(
    const char *name,
    int64_t pts,
    AVRational time_base)
{
    if (pts == AV_NOPTS_VALUE)
    {
        printf(
            "%s: NOPTS\n",
            name);

        return;
    }

    printf(
        "%s: %" PRId64 " -> %.6f sec\n",
        name,
        pts,
        pts * av_q2d(time_base));
}


/* ============================================================
 * Statistics
 * ============================================================ */

typedef struct
{
    int64_t packets_read;

    int64_t video_packets_read;

    int64_t video_packets_sent;

    int64_t decoder_frames_received;

    int64_t decoder_frames_drained;

    int64_t cuda_frames_processed;

    int64_t encoder_frames_submitted;

    int64_t encoder_packets_received;

    int64_t encoder_packets_written;

} PipelineStats;


/* ============================================================
 * CUDA Sobel kernel
 *
 * Input:
 *     NV12 Y plane
 *
 * Output:
 *     Sobel Y plane
 *
 * UV:
 *     U = 128
 *     V = 128
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
    const int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;


    /*
     * --------------------------------------------------------
     * Luma
     * --------------------------------------------------------
     */

    if (x < width && y < height)
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


            const int gx =
                -row0[x - 1]
                + row0[x + 1]

                - 2 * row1[x - 1]
                + 2 * row1[x + 1]

                - row2[x - 1]
                + row2[x + 1];


            const int gy =
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


            output_y[
                y * output_y_stride + x
            ] = (uint8_t)magnitude;
        }
    }


    /*
     * --------------------------------------------------------
     * Chroma
     * --------------------------------------------------------
     */

    if ((x % 2) == 0 &&
        (y % 2) == 0)
    {
        const int uv_x =
            x / 2;

        const int uv_y =
            y / 2;

        const int uv_width =
            (width + 1) / 2;

        const int uv_height =
            (height + 1) / 2;


        if (uv_x < uv_width &&
            uv_y < uv_height)
        {
            uint8_t *uv_row =
                output_uv +
                uv_y * output_uv_stride;


            uv_row[
                uv_x * 2 + 0
            ] = 128;


            uv_row[
                uv_x * 2 + 1
            ] = 128;
        }
    }
}


/* ============================================================
 * Sobel CUDA context
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


    cudaError_t err =
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
        fprintf(
            stderr,
            "cudaEventCreate failed: %s\n",
            cudaGetErrorString(err));


        cudaEventDestroy(
            ctx->kernel_start);


        ctx->kernel_start =
            NULL;


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

        ctx->kernel_start =
            NULL;
    }


    if (ctx->kernel_stop)
    {
        cudaEventDestroy(
            ctx->kernel_stop);

        ctx->kernel_stop =
            NULL;
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
    if (!ctx ||
        !input_frame ||
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
            "Invalid CUDA plane pointer\n");

        return AVERROR(EINVAL);
    }


    const int input_y_stride =
        input_frame->linesize[0];


    const int output_y_stride =
        output_frame->linesize[0];


    const int output_uv_stride =
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
            "cudaEventRecord(start) failed: %s\n",
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
            "cudaEventRecord(stop) failed: %s\n",
            cudaGetErrorString(err));

        return AVERROR_EXTERNAL;
    }


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


    float elapsed_ms =
        0.0f;


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
 * Find CUDA device
 * ============================================================ */

static int find_gpu(void)
{
    int device_count =
        0;


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


    if (device_count <= 0)
    {
        fprintf(
            stderr,
            "No CUDA devices found\n");

        return -1;
    }


    printf(
        "\nAvailable CUDA devices:\n");


    for (int i = 0;
         i < device_count;
         ++i)
    {
        cudaDeviceProp prop;


        if (cudaGetDeviceProperties(
                &prop,
                i) == cudaSuccess)
        {
            printf(
                "  [%d] %s\n",
                i,
                prop.name);
        }
    }


    const char *env =
        getenv("CUDA_DEVICE_INDEX");


    if (env &&
        env[0] != '\0')
    {
        char *end =
            NULL;


        long requested =
            strtol(
                env,
                &end,
                10);


        if (end != env &&
            *end == '\0' &&
            requested >= 0 &&
            requested < device_count)
        {
            return (int)requested;
        }


        fprintf(
            stderr,
            "Invalid CUDA_DEVICE_INDEX=%s\n",
            env);


        return -1;
    }


    /*
     * Convenience preference only.
     */

    for (int i = 0;
         i < device_count;
         ++i)
    {
        cudaDeviceProp prop;


        if (cudaGetDeviceProperties(
                &prop,
                i) != cudaSuccess)
        {
            continue;
        }


        if (strstr(
                prop.name,
                "RTX 4060 Ti") != NULL)
        {
            return i;
        }
    }


    return 0;
}


/* ============================================================
 * Create CUDA device
 * ============================================================ */

static int create_cuda_device(
    int device_index,
    AVBufferRef **hw_device_ctx)
{
    char device_string[32];


    snprintf(
        device_string,
        sizeof(device_string),
        "%d",
        device_index);


    return av_hwdevice_ctx_create(
        hw_device_ctx,
        AV_HWDEVICE_TYPE_CUDA,
        device_string,
        NULL,
        0);
}


/* ============================================================
 * CUDA format callback
 * ============================================================ */

static enum AVPixelFormat get_hw_format(
    AVCodecContext *ctx,
    const enum AVPixelFormat *pix_fmts)
{
    (void)ctx;


    for (const enum AVPixelFormat *p =
             pix_fmts;
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
        "Decoder did not offer AV_PIX_FMT_CUDA\n");


    return AV_PIX_FMT_NONE;
}


/* ============================================================
 * Create CUDA frames context
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
        32;


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
 * Open CUVID decoder
 *
 * IMPORTANT FIX:
 *
 * decoder_ctx->pkt_timebase MUST match the time base of
 * AVPacket PTS/DTS supplied by the demuxer.
 * ============================================================ */

static int open_decoder(
    AVFormatContext *input_ctx,
    int video_stream_index,
    AVBufferRef *hw_device_ctx,
    AVCodecContext **decoder_ctx_out)
{
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
            "\nh264_cuvid decoder not found.\n"
            "Your FFmpeg installation must contain "
            "the NVIDIA CUVID decoder.\n\n");

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


    /*
     * ========================================================
     * CRITICAL TIMESTAMP FIX
     * ========================================================
     *
     * AVPacket PTS/DTS from the demuxer are expressed in
     * stream->time_base.
     *
     * CUVID uses AVCodecContext::pkt_timebase when handling
     * packet timestamps.
     */

    decoder_ctx->pkt_timebase =
        stream->time_base;


    /*
     * Give decoder an explicit frame rate.
     */

    decoder_ctx->framerate =
        av_guess_frame_rate(
            input_ctx,
            stream,
            NULL);


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


    /*
     * CUVID handles its own decode scheduling.
     */

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
 * Determine frame rate
 * ============================================================ */

static AVRational get_input_frame_rate(
    AVFormatContext *input_ctx,
    AVStream *stream)
{
    AVRational rate =
        av_guess_frame_rate(
            input_ctx,
            stream,
            NULL);


    if (rate.num > 0 &&
        rate.den > 0)
    {
        return rate;
    }


    if (stream->avg_frame_rate.num > 0 &&
        stream->avg_frame_rate.den > 0)
    {
        return stream->avg_frame_rate;
    }


    if (stream->r_frame_rate.num > 0 &&
        stream->r_frame_rate.den > 0)
    {
        return stream->r_frame_rate;
    }


    return AVRational{25, 1};
}


/* ============================================================
 * Open NVENC
 * ============================================================ */

static int open_encoder(
    AVBufferRef *hw_device_ctx,
    AVBufferRef *encoder_frames_ref,
    int width,
    int height,
    AVRational encoder_time_base,
    AVRational frame_rate,
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


    encoder_ctx->pix_fmt =
        AV_PIX_FMT_CUDA;


    /*
     * Preserve the original stream time base.
     */

    encoder_ctx->time_base =
        encoder_time_base;


    encoder_ctx->framerate =
        frame_rate;


    encoder_ctx->bit_rate =
        5 * 1000 * 1000;


    encoder_ctx->gop_size =
        60;


    /*
     * No B frames.
     *
     * This simplifies timestamp ordering.
     */

    encoder_ctx->max_b_frames =
        0;


    encoder_ctx->hw_device_ctx =
        av_buffer_ref(
            hw_device_ctx);


    if (!encoder_ctx->hw_device_ctx)
    {
        avcodec_free_context(
            &encoder_ctx);

        return AVERROR(ENOMEM);
    }


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
     * NVENC configuration.
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
 * Encode frame and drain available packets
 * ============================================================ */

static int encode_and_write(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    AVFrame *frame,
    PipelineStats *stats)
{
    int ret;


    ret =
        avcodec_send_frame(
            encoder_ctx,
            frame);


    if (ret < 0)
    {
        if (frame == NULL &&
            ret == AVERROR_EOF)
        {
            return 0;
        }


        print_ffmpeg_error(
            "Error sending frame to NVENC",
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


        if (ret == AVERROR(EAGAIN))
        {
            ret = 0;
            break;
        }


        if (ret == AVERROR_EOF)
        {
            ret = 0;
            break;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error receiving NVENC packet",
                ret);

            break;
        }


        stats->encoder_packets_received++;


        /*
         * Convert encoder time base -> muxer time base.
         */
av_packet_rescale_ts(
    packet,
    encoder_ctx->time_base,
    output_stream->time_base);


/*
 * Make sure every encoded packet has the duration
 * of one video frame.
 *
 * 25 FPS + 1/12800 time base = 512 ticks/frame.
 */
packet->duration =
    av_rescale_q(
        1,
        av_inv_q(encoder_ctx->framerate),
        output_stream->time_base);

        packet->stream_index =
            output_stream->index;


        ret =
            av_interleaved_write_frame(
                output_ctx,
                packet);


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error writing encoded packet",
                ret);

            av_packet_unref(
                packet);

            break;
        }


        stats->encoder_packets_written++;


        av_packet_unref(
            packet);
    }


    av_packet_free(
        &packet);


    return ret;
}


/* ============================================================
 * Process one decoded frame
 *
 * This function is shared by normal decode and decoder drain.
 * ============================================================ */

static int process_decoded_frame(
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    AVBufferRef *output_frames_ref,
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    SobelCudaContext *sobel_ctx,
    PipelineStats *stats,
    int width,
    int height,
    int64_t *fallback_pts,
    int64_t pts_step)
{
    if (decoded_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Decoder returned non-CUDA frame\n");

        return AVERROR(EINVAL);
    }


    /*
     * --------------------------------------------------------
     * Determine output PTS.
     *
     * Normally preserve decoder PTS.
     *
     * If decoder does not provide PTS, use a sequential
     * fallback based on the input frame rate.
     * --------------------------------------------------------
     */

    int64_t frame_pts =
        decoded_frame->pts;


    if (frame_pts == AV_NOPTS_VALUE)
    {
        frame_pts =
            *fallback_pts;
    }


    /*
     * Advance fallback clock.
     */

    if (frame_pts != AV_NOPTS_VALUE)
    {
        *fallback_pts =
            frame_pts + pts_step;
    }


    /*
     * --------------------------------------------------------
     * Allocate CUDA output frame.
     * --------------------------------------------------------
     */

    av_frame_unref(
        output_frame);


    output_frame->format =
        AV_PIX_FMT_CUDA;


    output_frame->width =
        width;


    output_frame->height =
        height;


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
     * Exact PTS.
     */

    output_frame->pts =
        frame_pts;


    /*
     * Preserve duration when available.
     */

    output_frame->duration =
        //decoded_frame->duration;
        pts_step;



    /*
     * --------------------------------------------------------
     * CUDA Sobel.
     * --------------------------------------------------------
     */

    ret =
        sobel_cuda_process_frame(
            sobel_ctx,
            decoded_frame,
            output_frame,
            width,
            height);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "CUDA Sobel processing failed",
            ret);

        return ret;
    }


    stats->cuda_frames_processed++;


    /*
     * --------------------------------------------------------
     * Submit to NVENC.
     * --------------------------------------------------------
     */

    ret =
        encode_and_write(
            encoder_ctx,
            output_ctx,
            output_stream,
            output_frame,
            stats);


    if (ret < 0)
    {
        return ret;
    }


    stats->encoder_frames_submitted++;


    return 0;
}


/* ============================================================
 * Drain decoder
 * ============================================================ */

static int drain_decoder(
    AVCodecContext *decoder_ctx,
    AVBufferRef *output_frames_ref,
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    SobelCudaContext *sobel_ctx,
    PipelineStats *stats,
    int width,
    int height,
    int64_t *fallback_pts,
    int64_t pts_step)
{
    while (1)
    {
        int ret =
            avcodec_receive_frame(
                decoder_ctx,
                decoded_frame);


        if (ret == AVERROR(EAGAIN))
        {
            return 0;
        }


        if (ret == AVERROR_EOF)
        {
            return 0;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error receiving flushed CUVID frame",
                ret);

            return ret;
        }


        stats->decoder_frames_drained++;


        ret =
            process_decoded_frame(
                decoded_frame,
                output_frame,
                output_frames_ref,
                encoder_ctx,
                output_ctx,
                output_stream,
                sobel_ctx,
                stats,
                width,
                height,
                fallback_pts,
                pts_step);


        if (ret < 0)
        {
            return ret;
        }


        av_frame_unref(
            decoded_frame);
    }
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
     * Validate command line.
     * --------------------------------------------------------
     */

    if (argc != 3)
    {
        fprintf(
            stderr,
            "\nUsage:\n"
            "  %s input.mp4 output.mp4\n\n"
            "Optional:\n"
            "  CUDA_DEVICE_INDEX=1 %s input.mp4 output.mp4\n\n",
            argv[0],
            argv[0]);


        return EXIT_FAILURE;
    }


    const char *input_filename =
        argv[1];


    const char *output_filename =
        argv[2];


    /*
     * --------------------------------------------------------
     * ALL variables are declared here.
     *
     * This avoids NVCC goto initialization errors.
     * --------------------------------------------------------
     */

    int ret =
        0;


    int selected_device =
        -1;


    int video_stream_index =
        -1;


    int width =
        0;


    int height =
        0;


    int64_t first_input_pts =
        AV_NOPTS_VALUE;


    int64_t last_input_pts =
        AV_NOPTS_VALUE;


    int64_t first_output_pts =
        AV_NOPTS_VALUE;


    int64_t last_output_pts =
        AV_NOPTS_VALUE;


    /*
     * Fallback PTS clock.
     */

    int64_t fallback_pts =
        0;


  int64_t pts_step =
    0;

int64_t decoded_total =
    0;

AVRational input_time_base =
    AVRational{0, 1};


    AVRational frame_rate =
        AVRational{0, 1};


    AVRational encoder_time_base =
        AVRational{0, 1};


    double processing_start_ms =
        0.0;


    double processing_end_ms =
        0.0;


    double processing_elapsed_ms =
        0.0;



    AVFormatContext *input_ctx =
        NULL;


    AVFormatContext *output_ctx =
        NULL;


    AVCodecContext *decoder_ctx =
        NULL;


    AVCodecContext *encoder_ctx =
        NULL;


    AVStream *input_stream =
        NULL;


    AVStream *output_stream =
        NULL;


    AVBufferRef *hw_device_ctx =
        NULL;


    AVBufferRef *output_frames_ref =
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


    PipelineStats stats;


    memset(
        &stats,
        0,
        sizeof(stats));


    /*
     * --------------------------------------------------------
     * Select GPU.
     * --------------------------------------------------------
     */

    selected_device =
        find_gpu();


    if (selected_device < 0)
    {
        fprintf(
            stderr,
            "Could not select CUDA device\n");

        return EXIT_FAILURE;
    }


    printf(
        "\nSelected CUDA device: %d\n",
        selected_device);


    cudaError_t cuda_ret =
        cudaSetDevice(
            selected_device);


    if (cuda_ret != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaSetDevice failed: %s\n",
            cudaGetErrorString(cuda_ret));

        return EXIT_FAILURE;
    }


    cudaDeviceProp device_prop;


    cuda_ret =
        cudaGetDeviceProperties(
            &device_prop,
            selected_device);


    if (cuda_ret != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaGetDeviceProperties failed: %s\n",
            cudaGetErrorString(cuda_ret));

        return EXIT_FAILURE;
    }


    printf(
        "Using GPU %d: %s\n",
        selected_device,
        device_prop.name);


    /*
     * --------------------------------------------------------
     * Open input.
     * --------------------------------------------------------
     */

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

        goto final_cleanup;
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

        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Find video stream.
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


        ret =
            video_stream_index;


        goto final_cleanup;
    }


    input_stream =
        input_ctx->streams[
            video_stream_index];


    /*
     * --------------------------------------------------------
     * Verify H.264.
     * --------------------------------------------------------
     */

    if (input_stream->codecpar->codec_id !=
        AV_CODEC_ID_H264)
    {
        fprintf(
            stderr,
            "\nERROR: This implementation expects H.264 input.\n"
            "Detected codec: %s\n\n",
            avcodec_get_name(
                input_stream->codecpar->codec_id));


        ret =
            AVERROR(EINVAL);


        goto final_cleanup;
    }


    width =
        input_stream->codecpar->width;


    height =
        input_stream->codecpar->height;


    input_time_base =
        input_stream->time_base;


    frame_rate =
        get_input_frame_rate(
            input_ctx,
            input_stream);


    /*
     * --------------------------------------------------------
     * Calculate frame duration in input time-base units.
     *
     * For:
     *
     *     25 FPS
     *     time_base = 1/12800
     *
     * this becomes:
     *
     *     12800 / 25 = 512 ticks/frame
     * --------------------------------------------------------
     */

    if (frame_rate.num > 0 &&
        frame_rate.den > 0 &&
        input_time_base.num > 0 &&
        input_time_base.den > 0)
    {
        AVRational frame_duration =
            av_div_q(
                input_time_base,
                av_inv_q(frame_rate));


        pts_step =
            av_rescale_q(
                1,
                av_inv_q(frame_rate),
                input_time_base);


        if (pts_step <= 0)
        {
            pts_step =
                av_rescale_q(
                    1,
                    frame_duration,
                    input_time_base);
        }
    }


    if (pts_step <= 0)
    {
        pts_step =
            1;
    }


    printf(
        "\n================ INPUT =================\n");


    printf(
        "Input file       : %s\n",
        input_filename);


    printf(
        "Codec            : %s\n",
        avcodec_get_name(
            input_stream->codecpar->codec_id));


    printf(
        "Resolution       : %dx%d\n",
        width,
        height);


    print_rational(
        "Input time base  ",
        input_time_base);


    print_rational(
        "Frame rate       ",
        frame_rate);


    printf(
        "PTS step         : %" PRId64 " ticks/frame\n",
        pts_step);


    if (input_stream->start_time !=
        AV_NOPTS_VALUE)
    {
        print_pts_value(
            "Input start PTS ",
            input_stream->start_time,
            input_time_base);
    }


    if (input_stream->duration !=
        AV_NOPTS_VALUE)
    {
        printf(
            "Input duration   : %.6f sec\n",
            input_stream->duration *
            av_q2d(input_time_base));
    }


    if (input_stream->nb_frames > 0)
    {
        printf(
            "Input nb_frames  : %" PRId64 "\n",
            input_stream->nb_frames);
    }


    printf(
        "==========================================\n");


    /*
     * --------------------------------------------------------
     * Create CUDA device.
     * --------------------------------------------------------
     */

    ret =
        create_cuda_device(
            selected_device,
            &hw_device_ctx);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not create CUDA hardware device",
            ret);

        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Open CUVID/NVDEC.
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
        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * IMPORTANT diagnostic.
     * --------------------------------------------------------
     */

    printf(
        "\nCUVID decoder configuration:\n");


    print_rational(
        "  pkt_timebase   ",
        decoder_ctx->pkt_timebase);


    print_rational(
        "  decoder fps    ",
        decoder_ctx->framerate);


    width =
        decoder_ctx->width;


    height =
        decoder_ctx->height;


    /*
     * --------------------------------------------------------
     * CUDA output frame pool.
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
        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Encoder uses same time base.
     * --------------------------------------------------------
     */

    encoder_time_base =
        input_time_base;


    /*
     * --------------------------------------------------------
     * Create output context.
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


        if (ret >= 0)
            ret = AVERROR(EINVAL);


        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Open NVENC.
     * --------------------------------------------------------
     */

    ret =
        open_encoder(
            hw_device_ctx,
            output_frames_ref,
            width,
            height,
            encoder_time_base,
            frame_rate,
            &encoder_ctx);


    if (ret < 0)
    {
        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Create output stream.
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


        ret =
            AVERROR(ENOMEM);


        goto final_cleanup;
    }


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

        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Open output file.
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

            goto final_cleanup;
        }
    }


    /*
     * --------------------------------------------------------
     * Write header.
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

        goto final_cleanup;
    }



    /*
     * --------------------------------------------------------
     * Initialize Sobel.
     * --------------------------------------------------------
     */

    ret =
        sobel_cuda_init(
            &sobel_ctx);


    if (ret < 0)
    {
        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Allocate packet/frame objects.
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


        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Initial fallback PTS.
     * --------------------------------------------------------
     */

    if (input_stream->start_time !=
        AV_NOPTS_VALUE)
    {
        fallback_pts =
            input_stream->start_time;
    }
    else
    {
        fallback_pts =
            0;
    }


    /*
     * --------------------------------------------------------
     * Start timing.
     * --------------------------------------------------------
     */

    processing_start_ms =
      now_ms();


    printf(
        "\nStarting GPU processing...\n\n");


    /*
     * ========================================================
     * MAIN DEMUX LOOP
     * ========================================================
     */

    while (1)
    {
        ret =
            av_read_frame(
                input_ctx,
                packet);


        if (ret == AVERROR_EOF)
        {
            ret =
                0;

            break;
        }


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error reading input packet",
                ret);

            goto final_cleanup;
        }


        stats.packets_read++;


        /*
         * Ignore non-video streams.
         */

        if (packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(
                packet);

            continue;
        }


        stats.video_packets_read++;


        /*
         * ----------------------------------------------------
         * Diagnostic packet timestamps.
         * ----------------------------------------------------
         */

        if (stats.video_packets_read <= 5 ||
            stats.video_packets_read % 50 == 0)
        {
            printf(
                "[PACKET] #%" PRId64
                " pts=%" PRId64
                " dts=%" PRId64
                " duration=%" PRId64
                "\n",
                stats.video_packets_read,
                packet->pts,
                packet->dts,
                packet->duration);
        }


        /*
         * ----------------------------------------------------
         * Send compressed packet to CUVID.
         * ----------------------------------------------------
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
                "Error sending packet to CUVID",
                ret);

            goto final_cleanup;
        }


        stats.video_packets_sent++;


        /*
         * ----------------------------------------------------
         * Receive ALL frames produced by this packet.
         * ----------------------------------------------------
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
                    "Error receiving CUVID frame",
                    ret);

                goto final_cleanup;
            }


            stats.decoder_frames_received++;


            /*
             * Record first/last decoder PTS.
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
             * ------------------------------------------------
             * Diagnostic decoder output.
             * ------------------------------------------------
             */

            if (stats.decoder_frames_received <= 5 ||
                stats.decoder_frames_received % 50 == 0)
            {
                printf(
                    "[DECODE] frame=%" PRId64
                    " pts=%" PRId64
                    " duration=%" PRId64
                    " format=%s\n",
                    stats.decoder_frames_received,
                    decoded_frame->pts,
                    decoded_frame->duration,
                    av_get_pix_fmt_name(
                        (AVPixelFormat)
                        decoded_frame->format));
            }


            /*
             * ------------------------------------------------
             * Process decoded CUDA frame.
             * ------------------------------------------------
             */

            ret =
                process_decoded_frame(
                    decoded_frame,
                    output_frame,
                    output_frames_ref,
                    encoder_ctx,
                    output_ctx,
                    output_stream,
                    &sobel_ctx,
                    &stats,
                    width,
                    height,
                    &fallback_pts,
                    pts_step);


            if (ret < 0)
            {
                goto final_cleanup;
            }


            /*
             * Record output PTS.
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


            av_frame_unref(
                decoded_frame);


            /*
             * Progress.
             */

            if ((stats.cuda_frames_processed % 30) == 0)
            {
                printf(
                    "\rProcessed frames: %" PRId64,
                    stats.cuda_frames_processed);

                fflush(stdout);
            }
        }


        av_packet_unref(
            packet);
    }


    printf(
        "\n\nInput demux reached EOF.\n");


    /*
     * ========================================================
     * DRAIN CUVID
     * ========================================================
     */

    printf(
        "\nSending NULL packet to CUVID...\n");


    ret =
        avcodec_send_packet(
            decoder_ctx,
            NULL);


    if (ret < 0 &&
        ret != AVERROR_EOF)
    {
        print_ffmpeg_error(
            "Could not flush CUVID",
            ret);

        goto final_cleanup;
    }


    ret =
        drain_decoder(
            decoder_ctx,
            output_frames_ref,
            encoder_ctx,
            output_ctx,
            output_stream,
            decoded_frame,
            output_frame,
            &sobel_ctx,
            &stats,
            width,
            height,
            &fallback_pts,
            pts_step);


    if (ret < 0)
    {
        goto final_cleanup;
    }


    printf(
        "CUVID drain complete.\n");


    /*
     * ========================================================
     * DRAIN NVENC
     * ========================================================
     */

    printf(
        "Sending NULL frame to NVENC...\n");


    ret =
        encode_and_write(
            encoder_ctx,
            output_ctx,
            output_stream,
            NULL,
            &stats);


    if (ret < 0)
    {
        goto final_cleanup;
    }


    printf(
        "NVENC drain complete.\n");


    /*
     * ========================================================
     * WRITE TRAILER
     * ========================================================
     */

    ret =
        av_write_trailer(
            output_ctx);


    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not write MP4 trailer",
            ret);

        goto final_cleanup;
    }


    /*
     * --------------------------------------------------------
     * Timing.
     * --------------------------------------------------------
     */

    processing_end_ms =
       now_ms();


    processing_elapsed_ms =
        processing_end_ms -
        processing_start_ms;


    /*
     * ========================================================
     * FINAL DIAGNOSTICS
     * ========================================================
     */

    printf(
        "\n\n"
        "============================================================\n"
        "PIPELINE DIAGNOSTICS\n"
        "============================================================\n");


    printf(
        "Input packets read             : %" PRId64 "\n",
        stats.packets_read);


    printf(
        "Video packets read             : %" PRId64 "\n",
        stats.video_packets_read);


    printf(
        "Video packets sent to CUVID    : %" PRId64 "\n",
        stats.video_packets_sent);


    printf(
        "CUVID frames received          : %" PRId64 "\n",
        stats.decoder_frames_received);


    printf(
        "CUVID frames received on drain : %" PRId64 "\n",
        stats.decoder_frames_drained);


    printf(
        "CUDA Sobel frames processed    : %" PRId64 "\n",
        stats.cuda_frames_processed);


    printf(
        "Frames submitted to NVENC      : %" PRId64 "\n",
        stats.encoder_frames_submitted);


    printf(
        "Encoded packets received       : %" PRId64 "\n",
        stats.encoder_packets_received);


    printf(
        "Encoded packets written        : %" PRId64 "\n",
        stats.encoder_packets_written);


    printf(
        "============================================================\n");


    /*
     * --------------------------------------------------------
     * Frame-loss diagnosis.
     * --------------------------------------------------------
     */

decoded_total =
    stats.decoder_frames_received +
    stats.decoder_frames_drained;

    printf(
        "\nFRAME ACCOUNTING\n");


    printf(
        "  CUVID total frames : %" PRId64 "\n",
        decoded_total);


    printf(
        "  CUDA total frames  : %" PRId64 "\n",
        stats.cuda_frames_processed);


    printf(
        "  NVENC submissions  : %" PRId64 "\n",
        stats.encoder_frames_submitted);


    if (stats.video_packets_read > 0)
    {
        printf(
            "  Video packets      : %" PRId64 "\n",
            stats.video_packets_read);
    }


    /*
     * If the input container reports nb_frames, compare it.
     */

    if (input_stream->nb_frames > 0)
    {
        printf(
            "  Container nb_frames: %" PRId64 "\n",
            input_stream->nb_frames);


        if (decoded_total !=
            input_stream->nb_frames)
        {
            printf(
                "\n"
                "WARNING: CUVID produced %" PRId64
                " frames but the container reports %" PRId64
                " frames.\n"
                "This indicates the missing frames are before "
                "the CUDA/NVENC stage.\n",
                decoded_total,
                input_stream->nb_frames);
        }
    }


    if (stats.encoder_frames_submitted !=
        stats.cuda_frames_processed)
    {
        printf(
            "\n"
            "WARNING: CUDA processed %" PRId64
            " frames but only %" PRId64
            " were submitted to NVENC.\n",
            stats.cuda_frames_processed,
            stats.encoder_frames_submitted);
    }


    /*
     * --------------------------------------------------------
     * PTS diagnostics.
     * --------------------------------------------------------
     */

    printf(
        "\nPTS DIAGNOSTICS\n");


    if (first_input_pts !=
        AV_NOPTS_VALUE)
    {
        printf(
            "  First decoder PTS : %" PRId64
            " (%.6f sec)\n",
            first_input_pts,
            first_input_pts *
            av_q2d(input_time_base));


        printf(
            "  Last decoder PTS  : %" PRId64
            " (%.6f sec)\n",
            last_input_pts,
            last_input_pts *
            av_q2d(input_time_base));
    }
    else
    {
        printf(
            "  Decoder PTS       : unavailable\n");
    }


    if (first_output_pts !=
        AV_NOPTS_VALUE)
    {
        printf(
            "  First output PTS  : %" PRId64
            " (%.6f sec)\n",
            first_output_pts,
            first_output_pts *
            av_q2d(output_stream->time_base));


        printf(
            "  Last output PTS   : %" PRId64
            " (%.6f sec)\n",
            last_output_pts,
            last_output_pts *
            av_q2d(output_stream->time_base));
    }


    if (first_output_pts !=
            AV_NOPTS_VALUE &&
        last_output_pts !=
            AV_NOPTS_VALUE)
    {
        double pts_span =
            (last_output_pts -
             first_output_pts) *
            av_q2d(output_stream->time_base);


        printf(
            "  Output PTS span   : %.6f sec\n",
            pts_span);


        printf(
            "  Expected duration : %.6f sec\n",
            (stats.encoder_frames_submitted) *
            av_q2d(
                av_inv_q(frame_rate)));


        printf(
            "  Last frame + 1    : %.6f sec\n",
            ((last_output_pts -
              first_output_pts) +
             pts_step) *
            av_q2d(
                output_stream->time_base));
    }


    /*
     * --------------------------------------------------------
     * Sobel timing.
     * --------------------------------------------------------
     */

   
printf(
    "\nPERFORMANCE\n");


printf(
    "  CUDA Sobel time       : %.3f ms\n",
    sobel_ctx.total_kernel_ms);


printf(
    "  Average Sobel        : %.3f ms/frame\n",
    sobel_ctx.frames_processed > 0
        ? sobel_ctx.total_kernel_ms /
          (double)sobel_ctx.frames_processed
        : 0.0);


printf(
    "  Total processing time : %.3f sec (%.3f ms)\n",
    processing_elapsed_ms / 1000.0,
    processing_elapsed_ms);

    printf(
        "\n============================================================\n"
        "GPU VIDEO PROCESSING COMPLETE\n"
        "============================================================\n");


    ret =
        0;


/* ============================================================
 * FINAL CLEANUP
 * ============================================================ */

final_cleanup:

    /*
     * We intentionally do not attempt to write another trailer
     * here. If processing failed, the file may be incomplete.
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
    }


    if (input_ctx)
    {
        avformat_close_input(
            &input_ctx);
    }


    if (ret < 0)
    {
        fprintf(
            stderr,
            "\n"
            "============================================================\n"
            "PROCESSING FAILED\n"
            "============================================================\n");


        return EXIT_FAILURE;
    }


    printf(
        "\nSuccessfully created:\n"
        "  %s\n\n",
        output_filename);


    return EXIT_SUCCESS;
}
