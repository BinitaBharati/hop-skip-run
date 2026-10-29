#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include <cuda_runtime.h>

extern "C" {

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

#include "../../common/common.h"


/* ============================================================
 * CONSTANTS
 * ============================================================ */

#define MAX_PATH_LEN 4096
#define MAX_INPUT_FILES 4096

#define SOBEL_BLOCK_X 16
#define SOBEL_BLOCK_Y 16

#define OUTPUT_BITRATE 5000000
#define OUTPUT_GOP 60

#define CUDA_FRAME_POOL_SIZE 32


/* ============================================================
 * GLOBAL LOGGING
 * ============================================================ */

static pthread_mutex_t performance_log_mutex =
    PTHREAD_MUTEX_INITIALIZER;

/* ============================================================
 * ERROR HELPERS
 * ============================================================ */

static void print_av_error(
    const char *prefix,
    int error)
{
    char error_string[
        AV_ERROR_MAX_STRING_SIZE];

    av_strerror(
        error,
        error_string,
        sizeof(error_string));

    fprintf(
        stderr,
        "%s: %s\n",
        prefix,
        error_string);
}


/* ============================================================
 * PIPELINE STATISTICS
 * ============================================================ */

typedef struct
{
    int64_t packets_read;

    int64_t packets_sent_to_decoder;

    int64_t decoded_frames;

    int64_t cuda_frames_processed;

    int64_t encoder_frames_submitted;

    int64_t encoder_packets_received;

    int64_t encoder_packets_written;

    int64_t fallback_pts_frames;

    int64_t first_input_pts;
    int64_t last_input_pts;

    int64_t first_output_pts;
    int64_t last_output_pts;

    int64_t output_duration;

    int64_t frames_from_decoder;

} PipelineStats;


/* ============================================================
 * CUDA SOBEL CONTEXT
 *
 * Each video gets its own CUDA stream and CUDA timing events.
 *
 * This allows multiple video pipelines to overlap on the GPU.
 * ============================================================ */

 typedef struct
{
    int cuda_device;
} CudaProcessArguments;

typedef struct
{
    cudaStream_t stream;

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;

    double total_kernel_ms;

    int64_t frames_processed;

} SobelCudaContext;


/* ============================================================
 * CUDA SOBEL KERNEL
 *
 * Input:
 *   NV12 frame
 *
 * Y plane:
 *   Sobel edge detection
 *
 * UV plane:
 *   Neutral chroma = 128
 * ============================================================ */

__global__ void sobel_nv12_kernel(
    const uint8_t *input_y,
    uint8_t *output_y,
    uint8_t *output_uv,
    int input_stride,
    int output_stride,
    int width,
    int height)
{
    int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (x >= width ||
        y >= height)
    {
        return;
    }

    int output_index =
        y * output_stride + x;

    /*
     * Borders are black.
     */
    if (x == 0 ||
        y == 0 ||
        x >= width - 1 ||
        y >= height - 1)
    {
        output_y[output_index] = 0;
    }
    else
    {
        int p00 =
            input_y[
                (y - 1) * input_stride +
                (x - 1)];

        int p01 =
            input_y[
                (y - 1) * input_stride +
                x];

        int p02 =
            input_y[
                (y - 1) * input_stride +
                (x + 1)];

        int p10 =
            input_y[
                y * input_stride +
                (x - 1)];

        int p12 =
            input_y[
                y * input_stride +
                (x + 1)];

        int p20 =
            input_y[
                (y + 1) * input_stride +
                (x - 1)];

        int p21 =
            input_y[
                (y + 1) * input_stride +
                x];

        int p22 =
            input_y[
                (y + 1) * input_stride +
                (x + 1)];

        int gx =
            -p00 + p02
            -2 * p10 + 2 * p12
            -p20 + p22;

        int gy =
            -p00 - 2 * p01 - p02
            +p20 + 2 * p21 + p22;

        int magnitude =
            abs(gx) + abs(gy);

        if (magnitude > 255)
            magnitude = 255;

        output_y[output_index] =
            (uint8_t)magnitude;
    }

    /*
     * Write neutral chroma.
     *
     * Each UV sample represents two horizontal pixels.
     */
    if ((x & 1) == 0)
    {
        int uv_y =
            y / 2;

        int uv_index =
            uv_y * output_stride + x;

        output_uv[uv_index] = 128;
        output_uv[uv_index + 1] = 128;
    }
}


/* ============================================================
 * CUDA SOBEL INIT
 * ============================================================ */

static int sobel_cuda_init(
    SobelCudaContext *context)
{
    memset(
        context,
        0,
        sizeof(*context));

    cudaError_t error;

    error =
        cudaStreamCreateWithFlags(
            &context->stream,
            cudaStreamNonBlocking);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaStreamCreateWithFlags failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    error =
        cudaEventCreate(
            &context->kernel_start);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventCreate failed: %s\n",
            cudaGetErrorString(error));

        cudaStreamDestroy(
            context->stream);

        return -1;
    }

    error =
        cudaEventCreate(
            &context->kernel_stop);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventCreate failed: %s\n",
            cudaGetErrorString(error));

        cudaEventDestroy(
            context->kernel_start);

        cudaStreamDestroy(
            context->stream);

        return -1;
    }

    return 0;
}


/* ============================================================
 * CUDA SOBEL DESTROY
 * ============================================================ */

static void sobel_cuda_destroy(
    SobelCudaContext *context)
{
    if (!context)
        return;

    if (context->kernel_start)
    {
        cudaEventDestroy(
            context->kernel_start);
    }

    if (context->kernel_stop)
    {
        cudaEventDestroy(
            context->kernel_stop);
    }

    if (context->stream)
    {
        cudaStreamDestroy(
            context->stream);
    }

    memset(
        context,
        0,
        sizeof(*context));
}


/* ============================================================
 * CUDA SOBEL PROCESS
 * ============================================================ */

static int sobel_cuda_process(
    SobelCudaContext *context,
    AVFrame *input_frame,
    AVFrame *output_frame)
{
    if (!input_frame ||
        !output_frame)
    {
        fprintf(
            stderr,
            "Invalid frame passed to Sobel\n");

        return -1;
    }

    if (input_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Input frame is not AV_PIX_FMT_CUDA\n");

        return -1;
    }

    if (output_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Output frame is not AV_PIX_FMT_CUDA\n");

        return -1;
    }

    int width =
        input_frame->width;

    int height =
        input_frame->height;

    uint8_t *input_y =
        input_frame->data[0];

    uint8_t *output_y =
        output_frame->data[0];

    uint8_t *output_uv =
        output_frame->data[1];

    int input_stride =
        input_frame->linesize[0];

    int output_stride =
        output_frame->linesize[0];

    dim3 block(
        SOBEL_BLOCK_X,
        SOBEL_BLOCK_Y);

    dim3 grid(
        (width + block.x - 1) /
            block.x,
        (height + block.y - 1) /
            block.y);

    cudaError_t error;

    error =
        cudaEventRecord(
            context->kernel_start,
            context->stream);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventRecord(start) failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    sobel_nv12_kernel<<<
        grid,
        block,
        0,
        context->stream>>>(
            input_y,
            output_y,
            output_uv,
            input_stride,
            output_stride,
            width,
            height);

    error =
        cudaGetLastError();

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "Sobel kernel launch failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    error =
        cudaEventRecord(
            context->kernel_stop,
            context->stream);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventRecord(stop) failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    /*
     * Synchronize this video stream.
     *
     * This makes the measured kernel time complete before
     * the frame is submitted to NVENC.
     */
    error =
        cudaEventSynchronize(
            context->kernel_stop);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventSynchronize failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    float elapsed_ms = 0.0f;

    error =
        cudaEventElapsedTime(
            &elapsed_ms,
            context->kernel_start,
            context->kernel_stop);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaEventElapsedTime failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    context->total_kernel_ms +=
        (double)elapsed_ms;

    context->frames_processed++;

    return 0;
}


/* ============================================================
 * FIND CUDA GPU
 * ============================================================ */

static int find_cuda_gpu(void)
{
    int device_count = 0;

    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaGetDeviceCount failed: %s\n",
            cudaGetErrorString(error));

        return -1;
    }

    if (device_count <= 0)
    {
        fprintf(stderr,"No CUDA devices found\n");
        return EXIT_FAILURE;
    }

    printf("CUDA DEVICES\n");

    for (int i = 0;i < device_count;i++)
    {
        cudaDeviceProp properties;

        memset(&properties,0,sizeof(properties));

        error = cudaGetDeviceProperties(&properties,i);

        if (error != cudaSuccess)
        {
            fprintf(
                stderr,
                "cudaGetDeviceProperties(%d) failed: %s\n",
                i,
                cudaGetErrorString(error));

            continue;
        }

        printf(
            "  [%d] %s\n",
            i,
            properties.name);
    }

    /*
     * Explicit environment variable wins.
     *
     * Example:
     *
     * CUDA_DEVICE_INDEX=1 ./gpu_video_processor ...
     */
    const char *environment_device = getenv("CUDA_DEVICE_INDEX");

    if (environment_device)
    {
        int requested =
            atoi(environment_device);

        if (requested >= 0 &&
            requested < device_count)
        {
            printf(
                "Using CUDA_DEVICE_INDEX=%d\n",
                requested);

            return requested;
        }

        fprintf(
            stderr,
            "Invalid CUDA_DEVICE_INDEX=%s\n",
            environment_device);
    }

    /*
     * Prefer RTX 4060 Ti.
     */
    for (int i = 0; i < device_count;i++)
    {
        cudaDeviceProp properties;

        error =
            cudaGetDeviceProperties(
                &properties,
                i);

        if (error != cudaSuccess)
            continue;

        if (strstr(
                properties.name,
                "RTX 4060 Ti"))
        {
            printf(
                "Selected RTX 4060 Ti: device %d\n",
                i);

            return i;
        }
    }

    printf(
        "Using CUDA device 0\n");

    return 0;
}


/* ============================================================
 * CREATE CUDA HW DEVICE
 * ============================================================ */

static int create_cuda_device(
    int selected_device,
    AVBufferRef **hw_device_ctx)
{
    char device_string[32];

    int written =
        snprintf(
            device_string,
            sizeof(device_string),
            "%d",
            selected_device);

    if (written < 0 ||
        (size_t)written >= sizeof(device_string))
    {
        fprintf(
            stderr,
            "Invalid CUDA device number: %d\n",
            selected_device);

        return AVERROR(EINVAL);
    }

    int result =
        av_hwdevice_ctx_create(
            hw_device_ctx,
            AV_HWDEVICE_TYPE_CUDA,
            device_string,
            NULL,
            0);

    if (result < 0)
    {
        print_av_error(
            "av_hwdevice_ctx_create",
            result);

        return result;
    }

    return 0;
}


/* ============================================================
 * GET HW FORMAT
 * ============================================================ */

static enum AVPixelFormat get_hw_format(
    AVCodecContext *decoder_context,
    const enum AVPixelFormat *formats)
{
    /*
     * The callback receives the decoder context, but we only
     * need the format list.
     */
    (void)decoder_context;

    const enum AVPixelFormat *format;

    for (format = formats;
         *format != AV_PIX_FMT_NONE;
         format++)
    {
        if (*format == AV_PIX_FMT_CUDA)
            return *format;
    }

    fprintf(
        stderr,
        "Decoder does not support AV_PIX_FMT_CUDA\n");

    return AV_PIX_FMT_NONE;
}


/* ============================================================
 * CREATE CUDA FRAME CONTEXT
 * ============================================================ */

static int create_cuda_frames_context(
    AVBufferRef *hw_device_ctx,
    int width,
    int height,
    AVBufferRef **frames_ref)
{
    *frames_ref =
        av_hwframe_ctx_alloc(
            hw_device_ctx);

    if (!*frames_ref)
    {
        fprintf(
            stderr,
            "av_hwframe_ctx_alloc failed\n");

        return AVERROR(ENOMEM);
    }

    AVHWFramesContext *frames_ctx =
        (AVHWFramesContext *)
        (*frames_ref)->data;

    frames_ctx->format =
        AV_PIX_FMT_CUDA;

    frames_ctx->sw_format =
        AV_PIX_FMT_NV12;

    frames_ctx->width =
        width;

    frames_ctx->height =
        height;

    frames_ctx->initial_pool_size =
        CUDA_FRAME_POOL_SIZE;

    int result =
        av_hwframe_ctx_init(
            *frames_ref);

    if (result < 0)
    {
        print_av_error(
            "av_hwframe_ctx_init",
            result);

        av_buffer_unref(
            frames_ref);

        return result;
    }

    return 0;
}


/* ============================================================
 * OPEN DECODER
 * ============================================================ */

static int open_decoder(
    AVFormatContext *input_format,
    int video_stream_index,
    AVBufferRef *hw_device_ctx,
    AVCodecContext **decoder_context)
{
    AVStream *stream =
        input_format->streams[
            video_stream_index];

    /**
    h264_cuvid is FFmpeg's interface to NVIDIA's hardware video decoding through CUVID/NVDEC.
    */
    const AVCodec *decoder =
        avcodec_find_decoder_by_name(
            "h264_cuvid");

    if (!decoder)
    {
        /*fprintf(
            stderr,
            "h264_cuvid decoder not found\n");*/

        return AVERROR_DECODER_NOT_FOUND;
    }

    AVCodecContext *context =
        avcodec_alloc_context3(
            decoder);

    if (!context)
        return AVERROR(ENOMEM);

    int result =
        avcodec_parameters_to_context(
            context,
            stream->codecpar);

    if (result < 0)
    {
        print_av_error(
            "avcodec_parameters_to_context",
            result);

        avcodec_free_context(
            &context);

        return result;
    }

    context->pkt_timebase =
        stream->time_base;

    context->framerate =
        av_guess_frame_rate(
            input_format,
            stream,
            NULL);

    context->get_format =
        get_hw_format;

    context->hw_device_ctx =
        av_buffer_ref(
            hw_device_ctx);

    if (!context->hw_device_ctx)
    {
        avcodec_free_context(
            &context);

        return AVERROR(ENOMEM);
    }

    result =
        avcodec_open2(
            context,
            decoder,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "avcodec_open2 decoder",
            result);

        avcodec_free_context(
            &context);

        return result;
    }

    *decoder_context =
        context;

    return 0;
}


/* ============================================================
 * INPUT FRAME RATE
 * ============================================================ */

static AVRational get_input_frame_rate(
    AVFormatContext *input_format,
    AVStream *stream)
{
    AVRational frame_rate;

    frame_rate =
        av_guess_frame_rate(
            input_format,
            stream,
            NULL);

    if (frame_rate.num > 0 &&
        frame_rate.den > 0)
    {
        return frame_rate;
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

    return (AVRational){25, 1};
}


/* ============================================================
 * OPEN ENCODER
 * ============================================================ */

static int open_encoder(
    AVFormatContext *output_format,
    AVStream *input_stream,
    AVBufferRef *hw_device_ctx,
    AVBufferRef *output_frames_ref,
    AVCodecContext **encoder_context)
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

    AVCodecContext *context =
        avcodec_alloc_context3(
            encoder);

    if (!context)
        return AVERROR(ENOMEM);

    AVRational frame_rate =
        get_input_frame_rate(
            output_format,
            input_stream);

    context->width =
        input_stream->codecpar->width;

    context->height =
        input_stream->codecpar->height;

    context->pix_fmt =
        AV_PIX_FMT_CUDA;

    context->time_base =
        input_stream->time_base;

    context->framerate =
        frame_rate;

    context->bit_rate =
        OUTPUT_BITRATE;

    context->gop_size =
        OUTPUT_GOP;

    context->max_b_frames =
        0;

    context->hw_device_ctx =
        av_buffer_ref(
            hw_device_ctx);

    if (!context->hw_device_ctx)
    {
        avcodec_free_context(
            &context);

        return AVERROR(ENOMEM);
    }

    context->hw_frames_ctx =
        av_buffer_ref(
            output_frames_ref);

    if (!context->hw_frames_ctx)
    {
        avcodec_free_context(
            &context);

        return AVERROR(ENOMEM);
    }

    av_opt_set(
        context->priv_data,
        "preset",
        "p4",
        0);

    av_opt_set(
        context->priv_data,
        "tune",
        "hq",
        0);

    av_opt_set(
        context->priv_data,
        "profile",
        "high",
        0);

    int result =
        avcodec_open2(
            context,
            encoder,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "avcodec_open2 encoder",
            result);

        avcodec_free_context(
            &context);

        return result;
    }

    *encoder_context =
        context;

    return 0;
}


/* ============================================================
 * ENCODE + WRITE
 * ============================================================ */

static int encode_and_write(
    AVCodecContext *encoder_context,
    AVFormatContext *output_format,
    AVStream *output_stream,
    AVFrame *frame,
    PipelineStats *stats)
{
    int result;

    result =
        avcodec_send_frame(
            encoder_context,
            frame);

    if (result < 0)
    {
        print_av_error(
            "avcodec_send_frame",
            result);

        return result;
    }

    AVPacket *packet =
        av_packet_alloc();

    if (!packet)
        return AVERROR(ENOMEM);

    while (1)
    {
        result =
            avcodec_receive_packet(
                encoder_context,
                packet);

        if (result == AVERROR(EAGAIN) ||
            result == AVERROR_EOF)
        {
            break;
        }

        if (result < 0)
        {
            print_av_error(
                "avcodec_receive_packet",
                result);

            av_packet_free(
                &packet);

            return result;
        }

        stats->encoder_packets_received++;

        av_packet_rescale_ts(
            packet,
            encoder_context->time_base,
            output_stream->time_base);

        /*
         * Explicit one-frame duration.
         */
        packet->duration =
            av_rescale_q(
                1,
                av_inv_q(
                    encoder_context->framerate),
                output_stream->time_base);

        packet->stream_index =
            output_stream->index;

        result =
            av_interleaved_write_frame(
                output_format,
                packet);

        if (result < 0)
        {
            print_av_error(
                "av_interleaved_write_frame",
                result);

            av_packet_free(
                &packet);

            return result;
        }

        stats->encoder_packets_written++;

        av_packet_unref(
            packet);
    }

    av_packet_free(
        &packet);

    return 0;
}


/* ============================================================
 * PROCESS DECODED FRAME
 * ============================================================ */

static int process_decoded_frame(
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    AVBufferRef *output_frames_ref,
    SobelCudaContext *sobel_context,
    AVCodecContext *encoder_context,
    AVFormatContext *output_format,
    AVStream *output_stream,
    AVRational input_time_base,
    AVRational frame_rate,
    AVRational pts_step,
    int64_t *fallback_pts,
    PipelineStats *stats)
{
    if (!decoded_frame)
        return 0;

    if (decoded_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Decoded frame is not CUDA\n");

        return -1;
    }

    int64_t pts =
        decoded_frame->pts;

    if (pts == AV_NOPTS_VALUE)
    {
        pts =
            *fallback_pts;

        stats->fallback_pts_frames++;
    }

    /*
     * Advance fallback PTS by one frame in the
     * decoder/input stream time base.
     *
     * The old code used:
     *
     *     pts_step.num / pts_step.den
     *
     * which performs integer division and becomes zero
     * for common frame rates such as 1/30.
     */
    int64_t pts_increment =
        av_rescale_q(
            1,
            av_inv_q(frame_rate),
            input_time_base);

    if (pts_increment <= 0)
        pts_increment = 1;

    *fallback_pts =
        pts + pts_increment;

    av_frame_unref(
        output_frame);

    output_frame->format =
        AV_PIX_FMT_CUDA;

    output_frame->width =
        decoded_frame->width;

    output_frame->height =
        decoded_frame->height;

    int result =
        av_hwframe_get_buffer(
            output_frames_ref,
            output_frame,
            0);

    if (result < 0)
    {
        print_av_error(
            "av_hwframe_get_buffer",
            result);

        return result;
    }

    output_frame->pts =
        pts;

    output_frame->duration =
        av_rescale_q(
            1,
            pts_step,
            output_stream->time_base);

    result =
        sobel_cuda_process(
            sobel_context,
            decoded_frame,
            output_frame);

    if (result < 0)
        return result;

    stats->cuda_frames_processed++;

    stats->encoder_frames_submitted++;

    result =
        encode_and_write(
            encoder_context,
            output_format,
            output_stream,
            output_frame,
            stats);

    if (result < 0)
        return result;

    if (stats->cuda_frames_processed == 1)
    {
        stats->first_output_pts =
            output_frame->pts;
    }

    stats->last_output_pts =
        output_frame->pts;

    return 0;
}


/* ============================================================
 * DRAIN DECODER
 * ============================================================ */

static int drain_decoder(
    AVCodecContext *decoder_context,
    AVFrame *decoded_frame,
    AVFrame *output_frame,
    AVBufferRef *output_frames_ref,
    SobelCudaContext *sobel_context,
    AVCodecContext *encoder_context,
    AVFormatContext *output_format,
    AVStream *output_stream,
    AVRational input_time_base,
    AVRational frame_rate,
    AVRational pts_step,
    int64_t *fallback_pts,
    PipelineStats *stats)
{
    int result;

    result =
        avcodec_send_packet(
            decoder_context,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "decoder drain send",
            result);

        return result;
    }

    while (1)
    {
        result =
            avcodec_receive_frame(
                decoder_context,
                decoded_frame);

        if (result == AVERROR_EOF ||
            result == AVERROR(EAGAIN))
        {
            break;
        }

        if (result < 0)
        {
            print_av_error(
                "decoder drain receive",
                result);

            return result;
        }

        stats->decoded_frames++;

        result =
            process_decoded_frame(
                decoded_frame,
                output_frame,
                output_frames_ref,
                sobel_context,
                encoder_context,
                output_format,
                output_stream,
                input_time_base,
                frame_rate,
                pts_step,
                fallback_pts,
                stats);

        if (result < 0)
            return result;
    }

    return 0;
}


/* ============================================================
 * DRAIN ENCODER
 * ============================================================ */

static int drain_encoder(
    AVCodecContext *encoder_context,
    AVFormatContext *output_format,
    AVStream *output_stream,
    PipelineStats *stats)
{
    return encode_and_write(
        encoder_context,
        output_format,
        output_stream,
        NULL,
        stats);
}


/* ============================================================
 * FIND VIDEO STREAM
 * ============================================================ */

static int find_video_stream(
    AVFormatContext *format)
{
    for (unsigned int i = 0;
         i < format->nb_streams;
         i++)
    {
        if (format->streams[i]->codecpar->codec_type ==
            AVMEDIA_TYPE_VIDEO)
        {
            return (int)i;
        }
    }

    return -1;
}

/* ============================================================
 * PROCESS ONE VIDEO
 * ============================================================ */

static int process_video(
    const char *input_filename,
    const char *output_filename,
    int selected_device)
{
    /*
     * Keep declarations at the top.
     *
     * This is important when compiling the file with NVCC,
     * because cleanup uses goto and C++ does not allow a goto
     * to jump over initialized variables.
     */

    int result = 0;

    AVFormatContext *input_format = NULL;
    AVFormatContext *output_format = NULL;

    AVCodecContext *decoder_context = NULL;
    AVCodecContext *encoder_context = NULL;

    AVBufferRef *hw_device_ctx = NULL;
    AVBufferRef *output_frames_ref = NULL;

    AVFrame *decoded_frame = NULL;
    AVFrame *output_frame = NULL;

    AVPacket *packet = NULL;

    AVStream *input_stream = NULL;
    AVStream *output_stream = NULL;

    int video_stream_index = -1;

    AVRational frame_rate =
        (AVRational){0, 1};

    AVRational input_time_base =
        (AVRational){0, 1};

    AVRational pts_step =
        (AVRational){1, 25};

    int width = 0;
    int height = 0;

    int64_t fallback_pts = 0;

    double processing_start_ms = 0.0;
    double processing_end_ms = 0.0;
    double processing_elapsed_ms = 0.0;

    PipelineStats stats;

    SobelCudaContext sobel_context;

    memset(
        &stats,
        0,
        sizeof(stats));

    memset(
        &sobel_context,
        0,
        sizeof(sobel_context));

    /*
     * These are important for PTS diagnostics.
     */
    stats.first_input_pts =
        AV_NOPTS_VALUE;

    stats.last_input_pts =
        AV_NOPTS_VALUE;

    stats.first_output_pts =
        AV_NOPTS_VALUE;

    stats.last_output_pts =
        AV_NOPTS_VALUE;


    /* --------------------------------------------------------
     * HEADER
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    /*printf(
        "\n============================================================\n");

    printf(
        "VIDEO PROCESSING\n");

    printf(
        "Input file  : %s\n",
        input_filename);

    printf(
        "Output file : %s\n",
        output_filename);

    printf(
        "CUDA device : %d\n",
        selected_device);*/

    printf(
        "============================================================\n");

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * CUDA DEVICE
     * -------------------------------------------------------- */

    cudaError_t cuda_result =
        cudaSetDevice(
            selected_device);

    if (cuda_result != cudaSuccess)
    {
        fprintf(
            stderr,
            "[%s] cudaSetDevice failed: %s\n",
            input_filename,
            cudaGetErrorString(cuda_result));

        result = -1;

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * CUDA SOBEL CONTEXT
     * -------------------------------------------------------- */

    result =
        sobel_cuda_init(
            &sobel_context);

    if (result < 0)
        goto final_cleanup;


    /* --------------------------------------------------------
     * OPEN INPUT
     * -------------------------------------------------------- */

    result =
        avformat_open_input(
            &input_format,
            input_filename,
            NULL,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "avformat_open_input",
            result);

        goto final_cleanup;
    }

    result =
        avformat_find_stream_info(
            input_format,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "avformat_find_stream_info",
            result);

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * VIDEO STREAM
     * -------------------------------------------------------- */

    video_stream_index =
        find_video_stream(
            input_format);

    if (video_stream_index < 0)
    {
        fprintf(
            stderr,
            "No video stream found\n");

        result = -1;

        goto final_cleanup;
    }

    input_stream =
        input_format->streams[
            video_stream_index];

    input_time_base =
        input_stream->time_base;

    width =
        input_stream->codecpar->width;

    height =
        input_stream->codecpar->height;

    frame_rate =
        get_input_frame_rate(
            input_format,
            input_stream);

    pts_step =
        av_inv_q(frame_rate);


    /* --------------------------------------------------------
     * INPUT INFORMATION
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    /*printf(
        "\nINPUT\n");

    printf(
        "  Codec             : %s\n",
        avcodec_get_name(
            input_stream->codecpar->codec_id));

    printf(
        "  Resolution        : %dx%d\n",
        width,
        height);

    printf(
        "  Input time base   : %d/%d\n",
        input_time_base.num,
        input_time_base.den);

    printf(
        "  Frame rate        : %d/%d (%.3f FPS)\n",
        frame_rate.num,
        frame_rate.den,
        av_q2d(frame_rate));

    printf(
        "  PTS step          : %d/%d\n",
        pts_step.num,
        pts_step.den);

    printf(
        "  Input start PTS   : %" PRId64 "\n",
        input_stream->start_time);

    printf(
        "  Input duration    : %" PRId64 "\n",
        input_stream->duration);

    printf(
        "  Input nb_frames   : %" PRId64 "\n",
        input_stream->nb_frames); */

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * CUDA DEVICE CONTEXT
     * -------------------------------------------------------- */

    result =
        create_cuda_device(
            selected_device,
            &hw_device_ctx);

    if (result < 0)
        goto final_cleanup;


    /* --------------------------------------------------------
     * DECODER
     * -------------------------------------------------------- */

    result =
        open_decoder(
            input_format,
            video_stream_index,
            hw_device_ctx,
            &decoder_context);

    if (result < 0)
        goto final_cleanup;


    pthread_mutex_lock(
        &performance_log_mutex);

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * OUTPUT FORMAT
     * -------------------------------------------------------- */

    result =
        avformat_alloc_output_context2(
            &output_format,
            NULL,
            "mp4",
            output_filename);

    if (result < 0 ||
        !output_format)
    {
        print_av_error(
            "avformat_alloc_output_context2",
            result);

        if (result >= 0)
            result = AVERROR(EINVAL);

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * OUTPUT STREAM
     * -------------------------------------------------------- */

    output_stream =
        avformat_new_stream(
            output_format,
            NULL);

    if (!output_stream)
    {
        fprintf(
            stderr,
            "avformat_new_stream failed\n");

        result = AVERROR(ENOMEM);

        goto final_cleanup;
    }

    output_stream->time_base =
        input_stream->time_base;


    /* --------------------------------------------------------
     * CUDA FRAME POOL
     * -------------------------------------------------------- */

    result =
        create_cuda_frames_context(
            hw_device_ctx,
            width,
            height,
            &output_frames_ref);

    if (result < 0)
        goto final_cleanup;


    /* --------------------------------------------------------
     * ENCODER
     * -------------------------------------------------------- */

    result =
        open_encoder(
            output_format,
            input_stream,
            hw_device_ctx,
            output_frames_ref,
            &encoder_context);

    if (result < 0)
        goto final_cleanup;


    result =
        avcodec_parameters_from_context(
            output_stream->codecpar,
            encoder_context);

    if (result < 0)
    {
        print_av_error(
            "avcodec_parameters_from_context",
            result);

        goto final_cleanup;
    }

    output_stream->time_base =
        encoder_context->time_base;


    /* --------------------------------------------------------
     * OPEN OUTPUT
     * -------------------------------------------------------- */

    if (!(output_format->oformat->flags &
          AVFMT_NOFILE))
    {
        result =
            avio_open(
                &output_format->pb,
                output_filename,
                AVIO_FLAG_WRITE);

        if (result < 0)
        {
            print_av_error(
                "avio_open",
                result);

            goto final_cleanup;
        }
    }

    result =
        avformat_write_header(
            output_format,
            NULL);

    if (result < 0)
    {
        print_av_error(
            "avformat_write_header",
            result);

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * FRAMES / PACKET
     * -------------------------------------------------------- */

    decoded_frame =
        av_frame_alloc();

    output_frame =
        av_frame_alloc();

    packet =
        av_packet_alloc();

    if (!decoded_frame ||
        !output_frame ||
        !packet)
    {
        fprintf(
            stderr,
            "Frame/packet allocation failed\n");

        result = AVERROR(ENOMEM);

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * START PERFORMANCE TIMER
     * -------------------------------------------------------- */

    processing_start_ms =
        now_ms();

    pthread_mutex_lock(
        &performance_log_mutex);

    printf(
        "\nStarting GPU processing...\n\n");

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * READ / DECODE / SOBEL / ENCODE
     * -------------------------------------------------------- */

    while (1)
    {
        result =
            av_read_frame(
                input_format,
                packet);

        if (result == AVERROR_EOF)
        {
            pthread_mutex_lock(
                &performance_log_mutex);

            printf(
                "\n[%s]\n",
                input_filename);

            /*printf(
                "Input EOF reached. Draining decoder...\n");*/

            pthread_mutex_unlock(
                &performance_log_mutex);

            break;
        }

        if (result < 0)
        {
            print_av_error(
                "av_read_frame",
                result);

            goto final_cleanup;
        }

        stats.packets_read++;

        if (stats.packets_read <= 5 ||
            stats.packets_read % 50 == 0)
        {
            pthread_mutex_lock(
                &performance_log_mutex);

            /*printf(
                "Packet %" PRId64
                "  stream=%d size=%d pts=%" PRId64 "\n",
                stats.packets_read,
                packet->stream_index,
                packet->size,
                packet->pts);*/

            pthread_mutex_unlock(
                &performance_log_mutex);
        }

        if (packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(
                packet);

            continue;
        }

        stats.packets_sent_to_decoder++;

        result =
            avcodec_send_packet(
                decoder_context,
                packet);

        av_packet_unref(
            packet);

        if (result < 0)
        {
            print_av_error(
                "avcodec_send_packet",
                result);

            goto final_cleanup;
        }

        while (1)
        {
            result =
                avcodec_receive_frame(
                    decoder_context,
                    decoded_frame);

            if (result == AVERROR(EAGAIN) ||
                result == AVERROR_EOF)
            {
                break;
            }

            if (result < 0)
            {
                print_av_error(
                    "avcodec_receive_frame",
                    result);

                goto final_cleanup;
            }

            stats.decoded_frames++;

            if (stats.decoded_frames == 1)
            {
                stats.first_input_pts =
                    decoded_frame->pts;

                fallback_pts =
                    decoded_frame->pts != AV_NOPTS_VALUE
                        ? decoded_frame->pts
                        : 0;
            }

            stats.last_input_pts =
                decoded_frame->pts;

            if (stats.decoded_frames <= 5 ||
                stats.decoded_frames % 50 == 0)
            {
                pthread_mutex_lock(
                    &performance_log_mutex);

                /*printf(
                    "Decoded frame %" PRId64
                    "  pts=%" PRId64
                    "  format=%s\n",
                    stats.decoded_frames,
                    decoded_frame->pts,
                    av_get_pix_fmt_name(
                        (AVPixelFormat)
                        decoded_frame->format));*/

                pthread_mutex_unlock(
                    &performance_log_mutex);
            }

            result =
                process_decoded_frame(
                    decoded_frame,
                    output_frame,
                    output_frames_ref,
                    &sobel_context,
                    encoder_context,
                    output_format,
                    output_stream,
                    input_time_base,
                    frame_rate,
                    pts_step,
                    &fallback_pts,
                    &stats);

            if (result < 0)
                goto final_cleanup;

            if (stats.decoded_frames % 30 == 0)
            {
                pthread_mutex_lock(
                    &performance_log_mutex);

                /*printf(
                    "Progress: %" PRId64
                    " frames processed\n",
                    stats.decoded_frames);*/

                pthread_mutex_unlock(
                    &performance_log_mutex);
            }
        }
    }


    /* --------------------------------------------------------
     * DRAIN DECODER
     * -------------------------------------------------------- */

    result =
        drain_decoder(
            decoder_context,
            decoded_frame,
            output_frame,
            output_frames_ref,
            &sobel_context,
            encoder_context,
            output_format,
            output_stream,
            input_time_base,
            frame_rate,
            pts_step,
            &fallback_pts,
            &stats);

    if (result < 0)
        goto final_cleanup;


    pthread_mutex_lock(
        &performance_log_mutex);

    //printf(
    //    "Decoder drain complete.\n");

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * DRAIN ENCODER
     * -------------------------------------------------------- */

    result =
        drain_encoder(
            encoder_context,
            output_format,
            output_stream,
            &stats);

    if (result < 0)
        goto final_cleanup;


    pthread_mutex_lock(
        &performance_log_mutex);

    //printf(
    //    "Encoder drain complete.\n");

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * WRITE TRAILER
     * -------------------------------------------------------- */

    result =
        av_write_trailer(
            output_format);

    if (result < 0)
    {
        print_av_error(
            "av_write_trailer",
            result);

        goto final_cleanup;
    }


    /* --------------------------------------------------------
     * END PERFORMANCE TIMER
     * -------------------------------------------------------- */

    processing_end_ms =
        now_ms();

    processing_elapsed_ms =
        processing_end_ms -
        processing_start_ms;


    /* --------------------------------------------------------
     * PIPELINE DIAGNOSTICS
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    /* printf(
        "\nPIPELINE DIAGNOSTICS\n");

    printf(
        "  Packets read             : %" PRId64 "\n",
        stats.packets_read);

    printf(
        "  Packets sent to decoder : %" PRId64 "\n",
        stats.packets_sent_to_decoder);

    printf(
        "  Decoded frames          : %" PRId64 "\n",
        stats.decoded_frames);

    printf(
        "  CUDA frames processed   : %" PRId64 "\n",
        stats.cuda_frames_processed);

    printf(
        "  Encoder frames submitted: %" PRId64 "\n",
        stats.encoder_frames_submitted);

    printf(
        "  Encoder packets received: %" PRId64 "\n",
        stats.encoder_packets_received);

    printf(
        "  Encoder packets written : %" PRId64 "\n",
        stats.encoder_packets_written);

    printf(
        "  Fallback PTS frames     : %" PRId64 "\n",
        stats.fallback_pts_frames); */

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * FRAME ACCOUNTING
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    /* printf(
        "\nFRAME ACCOUNTING\n");

    printf(
        "  Decoded frames          : %" PRId64 "\n",
        stats.decoded_frames);

    printf(
        "  CUDA processed          : %" PRId64 "\n",
        stats.cuda_frames_processed);

    printf(
        "  Encoder submitted       : %" PRId64 "\n",
        stats.encoder_frames_submitted);

    printf(
        "  Input nb_frames         : %" PRId64 "\n",
        input_stream->nb_frames);

    printf(
        "  Output nb_frames        : %" PRId64 "\n",
        output_stream->nb_frames); */

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * PTS DIAGNOSTICS
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    /* printf(
        "\nPTS DIAGNOSTICS\n");

    printf(
        "  First input PTS         : %" PRId64 "\n",
        stats.first_input_pts);

    printf(
        "  Last input PTS          : %" PRId64 "\n",
        stats.last_input_pts);

    printf(
        "  First output PTS        : %" PRId64 "\n",
        stats.first_output_pts);

    printf(
        "  Last output PTS         : %" PRId64 "\n",
        stats.last_output_pts);*/

    if (stats.first_output_pts !=
            AV_NOPTS_VALUE &&
        stats.last_output_pts !=
            AV_NOPTS_VALUE)
    {
        int64_t output_pts_span =
            stats.last_output_pts -
            stats.first_output_pts;

        double output_seconds =
            output_pts_span *
            av_q2d(
                output_stream->time_base);

        double expected_duration =
            stats.cuda_frames_processed *
            av_q2d(
                av_inv_q(
                    encoder_context->framerate));

        int64_t last_frame_plus_one =
            stats.last_output_pts +
            av_rescale_q(
                1,
                av_inv_q(
                    encoder_context->framerate),
                output_stream->time_base);

        double last_frame_plus_one_seconds =
            last_frame_plus_one *
            av_q2d(
                output_stream->time_base);

        /* printf(
            "  Output PTS span         : %" PRId64 "\n",
            output_pts_span);

        printf(
            "  Output PTS duration     : %.6f sec\n",
            output_seconds);

        printf(
            "  Expected duration       : %.6f sec\n",
            expected_duration);

        printf(
            "  Last frame + 1 duration : %.6f sec\n",
            last_frame_plus_one_seconds);*/
    }

    pthread_mutex_unlock(
        &performance_log_mutex);


    /* --------------------------------------------------------
     * PERFORMANCE
     *
     * Keep these exactly aligned with the single-video
     * benchmark output.
     * -------------------------------------------------------- */

    pthread_mutex_lock(
        &performance_log_mutex);

    printf(
        "\nPERFORMANCE\n");

    printf(
        "  CUDA Sobel time       : %.3f ms\n",
        sobel_context.total_kernel_ms);

    printf(
        "  Average Sobel        : %.3f ms/frame\n",
        sobel_context.frames_processed > 0
            ? sobel_context.total_kernel_ms /
              (double)sobel_context.frames_processed
            : 0.0);

    printf(
        "  Total processing time : %.3f sec (%.3f ms)\n",
        processing_elapsed_ms / 1000.0,
        processing_elapsed_ms);

    printf(
        "\nCompleted: %s\n",
        input_filename);

    printf(
        "Output:    %s\n",
        output_filename);

    printf(
        "============================================================\n");

    pthread_mutex_unlock(
        &performance_log_mutex);


    result = 0;


final_cleanup:

    if (result < 0)
    {
        pthread_mutex_lock(
            &performance_log_mutex);

        fprintf(
            stderr,
            "\nFAILED: %s\n",
            input_filename);

        pthread_mutex_unlock(
            &performance_log_mutex);
    }

    if (packet)
        av_packet_free(
            &packet);

    if (decoded_frame)
        av_frame_free(
            &decoded_frame);

    if (output_frame)
        av_frame_free(
            &output_frame);

    if (encoder_context)
        avcodec_free_context(
            &encoder_context);

    if (decoder_context)
        avcodec_free_context(
            &decoder_context);

    if (output_format)
    {
        if (output_format->pb)
        {
            avio_closep(
                &output_format->pb);
        }

        avformat_free_context(
            output_format);
    }

    if (input_format)
    {
        avformat_close_input(
            &input_format);
    }

    /*
     * output_frames_ref owns the CUDA frame pool.
     */
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

    sobel_cuda_destroy(
        &sobel_context);

    return result;
}


int process_video_cuda(void *arg)
{
    ProcessVideoContext *context =
        (ProcessVideoContext *)arg;

    CudaProcessArguments *cuda_args =
        (CudaProcessArguments *)context->implementation_args;

    return process_video(
        context->input_path,
        context->output_path,
        cuda_args->cuda_device);
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(int argc,char **argv)
{

    double total_start_ms = now_ms();

    if (argc != 4)
    {
      
        fprintf(
        stderr,
        "Usage:\n"
        "  %s input_directory output_directory video_threads\n\n"
        "Example:\n"
        "  %s resources/input/mp4/4k resources/output/mp4/4k 2\n\n"
        "Arguments:\n"
        "  input_directory   Directory containing input .mp4 files\n"
        "  output_directory  Directory where processed videos are written\n"
        "  video_threads     Number of videos processed concurrently\n",
        argv[0],
        argv[0]
        );
        return EXIT_FAILURE;
    }

    const char *input_directory = argv[1];
    const char *output_directory = argv[2];
    int video_threads = atoi(argv[3]);



    if (video_threads < 1)
    {
        fprintf(stderr, "video thread counts must be >= 1.\n");
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



    /* --------------------------------------------------------
     * FIND GPU
     * -------------------------------------------------------- */

    int selected_device = find_cuda_gpu();

    if (selected_device < 0)
    {
        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * SCAN INPUT DIRECTORY
     * -------------------------------------------------------- */

    int file_count = 0;
    char **input_files = (char **)malloc(16 * sizeof(char *));
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


    /*
     * There is no point creating more workers than files.
     */
    if (video_threads > file_count)
    {
        video_threads = file_count;
    }


    /* --------------------------------------------------------
     * STARTUP INFORMATION
     * -------------------------------------------------------- */

    printf(
        "\n============================================================\n");

    printf(
        "MULTI-VIDEO CUDA PROCESSOR\n");

    printf(
        "============================================================\n");

    printf(
        "Input directory  : %s\n",
        input_directory);

    printf(
        "Output directory : %s\n",
        output_directory);

    printf(
        "Videos found     : %d\n",
        file_count);

    printf(
        "Video threads    : %d\n",
        video_threads);

    printf(
        "CUDA device      : %d\n",
        selected_device);

    printf(
        "\nEach video thread runs a complete pipeline:\n");

    printf(
        "  H.264 -> NVDEC -> CUDA Sobel -> NVENC -> MP4\n");

    printf(
        "\nThe Sobel operation itself uses CUDA GPU threads.\n");

    printf(
        "============================================================\n");

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


    /* --------------------------------------------------------
     * WORKER DATA
     * -------------------------------------------------------- */

    VideoWorkerData worker_data;

    worker_data.input_files = input_files;
    worker_data.file_count = file_count;
    worker_data.next_file = 0;
    worker_data.mutex = &mutex;
    worker_data.output_directory = output_directory;
    worker_data.failed_count = 0;

    VideoWorkerArgument worker_argument;
    worker_argument.data = &worker_data;

    worker_argument.process_video = process_video_cuda;

    CudaProcessArguments cuda_args;
    cuda_args.cuda_device = selected_device;
    worker_argument.process_arguments = &cuda_args;

    //Create video worker threads.
    int created_threads = create_video_workers(threads, video_threads, &worker_argument, video_worker);

    /* --------------------------------------------------------
     * JOIN WORKERS
     * -------------------------------------------------------- */
    printf("Waiting for video workers...\n");

    for (int i = 0; i < created_threads;i++)
    {
        pthread_join(threads[i],NULL);
    }


    double total_end_ms =
        now_ms();

    double total_elapsed_ms =
        total_end_ms -
        total_start_ms;


    /* --------------------------------------------------------
     * FINAL SUMMARY
     * -------------------------------------------------------- */

    printf(
        "\n============================================================\n");

    printf(
        "MULTI-VIDEO SUMMARY\n");

    printf(
        "============================================================\n");

    printf(
        "Videos found       : %d\n",
        file_count);

    printf(
        "Video threads      : %d\n",
        video_threads);

    printf(
        "Videos failed      : %d\n",
        worker_data.failed_count);

    printf(
        "Total processing time    : %.3f sec (%.3f ms)\n",
        total_elapsed_ms / 1000.0,
        total_elapsed_ms);

    printf(
        "============================================================\n");


    /* --------------------------------------------------------
     * CLEANUP
     * -------------------------------------------------------- */

    free(threads);

    pthread_mutex_destroy(
        &mutex);

    return worker_data.failed_count == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}