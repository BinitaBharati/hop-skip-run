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
 *     CUDA Sobel kernel
 *        |
 *        v
 *     AV_PIX_FMT_CUDA / NV12
 *        |
 *        v
 *     NVIDIA NVENC H.264
 *        |
 *        v
 *     FFmpeg MP4 muxer
 *
 * No CPU frame copy is performed between:
 *
 *     NVDEC -> CUDA Sobel -> NVENC
 *
 * ============================================================
 *
 * Usage:
 *
 *     ./gpu_video_processor input.mp4 output.mp4
 *
 * ============================================================
 */

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
            exit(EXIT_FAILURE);                               \
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
 * UV is set to 128/128, producing grayscale output.
 *
 * One CUDA thread processes one Y pixel.
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
     * Sobel Y plane
     * -------------------------------------------------------- */

    if (x < width && y < height)
    {
        /*
         * Border pixels.
         */

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


            /*
             * Sobel X.
             */

            int gx =
                -row0[x - 1] +
                 row0[x + 1]

                -2 * row1[x - 1] +
                 2 * row1[x + 1]

                -row2[x - 1] +
                 row2[x + 1];


            /*
             * Sobel Y.
             */

            int gy =
                -row0[x - 1] -
                 2 * row0[x] -
                 row0[x + 1]

                +row2[x - 1] +
                 2 * row2[x] +
                 row2[x + 1];


            /*
             * L1 magnitude approximation:
             *
             *     |gx| + |gy|
             *
             * instead of:
             *
             *     sqrt(gx*gx + gy*gy)
             */

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
     * NV12 chroma plane
     *
     * Each chroma sample contains:
     *
     *     U V
     *
     * U=128, V=128 means neutral chroma.
     * -------------------------------------------------------- */

    int uv_x = x;
    int uv_y = y;

    int uv_width =
        width / 2;

    int uv_height =
        height / 2;


    if (uv_x < uv_width &&
        uv_y < uv_height)
    {
        uint8_t *uv_row =
            output_uv +
            uv_y * output_uv_stride;

        uv_row[uv_x * 2 + 0] = 128;
        uv_row[uv_x * 2 + 1] = 128;
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
 * Initialize Sobel context
 * ============================================================ */

static void sobel_cuda_init(
    SobelCudaContext *ctx)
{
    memset(
        ctx,
        0,
        sizeof(*ctx));

    CUDA_CHECK(
        cudaEventCreate(
            &ctx->kernel_start));

    CUDA_CHECK(
        cudaEventCreate(
            &ctx->kernel_stop));
}


/* ============================================================
 * Destroy Sobel context
 * ============================================================ */

static void sobel_cuda_destroy(
    SobelCudaContext *ctx)
{
    cudaEventDestroy(
        ctx->kernel_start);

    cudaEventDestroy(
        ctx->kernel_stop);
}


/* ============================================================
 * Process one CUDA frame
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
            "Input frame is not CUDA format\n");

        return AVERROR(EINVAL);
    }


    if (output_frame->format !=
        AV_PIX_FMT_CUDA)
    {
        fprintf(
            stderr,
            "Output frame is not CUDA format\n");

        return AVERROR(EINVAL);
    }


    /*
     * FFmpeg CUDA frame data[] contains
     * pointers to CUDA device memory.
     */

    const uint8_t *d_input_y =
        input_frame->data[0];

    uint8_t *d_output_y =
        output_frame->data[0];

    uint8_t *d_output_uv =
        output_frame->data[1];


    int input_y_stride =
        input_frame->linesize[0];

    int output_y_stride =
        output_frame->linesize[0];

    int output_uv_stride =
        output_frame->linesize[1];


    if (!d_input_y ||
        !d_output_y ||
        !d_output_uv)
    {
        fprintf(
            stderr,
            "Invalid CUDA plane pointers\n");

        return AVERROR(EINVAL);
    }


    /*
     * 16x16 CUDA block.
     */

    dim3 block(
        16,
        16);


    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);


    /*
     * Start GPU timer.
     */

    CUDA_CHECK(
        cudaEventRecord(
            ctx->kernel_start,
            0));


    /*
     * Launch kernel.
     */

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


    /*
     * Check launch errors.
     */

    CUDA_CHECK(
        cudaGetLastError());


    /*
     * Stop timer.
     */

    CUDA_CHECK(
        cudaEventRecord(
            ctx->kernel_stop,
            0));


    /*
     * Wait until kernel has completed.
     *
     * This is required before handing the
     * output frame to the encoder.
     */

    CUDA_CHECK(
        cudaEventSynchronize(
            ctx->kernel_stop));


    float elapsed_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            ctx->kernel_start,
            ctx->kernel_stop));


    ctx->total_kernel_ms +=
        elapsed_ms;

    ctx->frames_processed++;


    return 0;
}


/* ============================================================
 * Create CUDA hardware device context
 * ============================================================ */

static int create_cuda_device(
    AVBufferRef **hw_device_ctx)
{
    int ret;

    /*
     * "0" selects NVIDIA GPU 0.
     *
     * We will later explicitly select the RTX 4060 Ti
     * if necessary using CUDA device selection.
     */

    ret =
        av_hwdevice_ctx_create(
            hw_device_ctx,
            AV_HWDEVICE_TYPE_CUDA,
            "0",
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
 * Select CUDA pixel format for decoder
 * ============================================================ */

static enum AVPixelFormat get_hw_format(
    AVCodecContext *ctx,
    const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++)
    {
        if (*p == AV_PIX_FMT_CUDA)
            return AV_PIX_FMT_CUDA;
    }

    fprintf(
        stderr,
        "Decoder did not offer AV_PIX_FMT_CUDA\n");

    return AV_PIX_FMT_NONE;
}


/* ============================================================
 * Create CUDA output frame pool
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

    /*
     * Number of frames in the CUDA pool.
     */

    frames_ctx->initial_pool_size =
        8;


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
    int ret;

    AVStream *stream =
        input_ctx->streams[
            video_stream_index];


    /*
     * Use the NVIDIA CUVID H.264 decoder.
     *
     * This example specifically targets H.264 input.
     */

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
        return AVERROR(ENOMEM);


    ret =
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
     * Ask FFmpeg for CUDA frames.
     */

    decoder_ctx->get_format =
        get_hw_format;


    /*
     * Give decoder access to CUDA device.
     */

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
     * Threading is generally handled internally
     * by the hardware decoder.
     */

    decoder_ctx->thread_count = 1;


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
    AVCodecContext **encoder_ctx_out)
{
    int ret;


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
        return AVERROR(ENOMEM);


    encoder_ctx->width =
        width;

    encoder_ctx->height =
        height;


    /*
     * NVENC accepts CUDA frames.
     */

    encoder_ctx->pix_fmt =
        AV_PIX_FMT_CUDA;


    encoder_ctx->time_base =
        input_time_base;


    encoder_ctx->framerate =
        av_inv_q(input_time_base);


    /*
     * H.264 stream parameters.
     */

    encoder_ctx->bit_rate =
        5 * 1000 * 1000;


    encoder_ctx->gop_size =
        60;


    encoder_ctx->max_b_frames =
        0;


    /*
     * CUDA hardware context.
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
     * CUDA frames that we created for
     * the Sobel output.
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
     * NVENC options.
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


    ret =
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
 * Encode one frame and mux resulting packets
 * ============================================================ */

static int encode_and_write(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream,
    AVFrame *frame)
{
    int ret;

    ret =
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
        return AVERROR(ENOMEM);


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
         * Convert encoder time base to
         * output stream time base.
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

        av_packet_unref(packet);


        if (ret < 0)
        {
            print_ffmpeg_error(
                "Error writing output packet",
                ret);

            break;
        }
    }


    av_packet_free(
        &packet);

    return ret;
}


/* ============================================================
 * Flush encoder
 * ============================================================ */

static int flush_encoder(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_ctx,
    AVStream *output_stream)
{
    return encode_and_write(
        encoder_ctx,
        output_ctx,
        output_stream,
        NULL);
}

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

/* ============================================================
 * Main
 * ============================================================ */

int main(
    int argc,
    char **argv)
{
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


    int ret;

double start_time = 0.0;
double end_time = 0.0;
double elapsed = 0.0;



    /* --------------------------------------------------------
     * CUDA device
     * -------------------------------------------------------- */

    /*
     * Your RTX 4060 Ti is CUDA device 1 according to
     * nvidia-smi.
     *
     * Therefore select CUDA device 1.
     */
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


cudaDeviceProp device_prop;

CUDA_CHECK(
    cudaGetDeviceProperties(
        &device_prop,
        selected_device));

printf(
    "Using CUDA device %d: %s\n",
    selected_device,
    device_prop.name);



    /* --------------------------------------------------------
     * Input
     * -------------------------------------------------------- */

    AVFormatContext *input_ctx =
        NULL;


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

        return EXIT_FAILURE;
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

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Find video stream
     * -------------------------------------------------------- */

    int video_stream_index =
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

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    AVStream *input_stream =
        input_ctx->streams[
            video_stream_index];


    printf(
        "Input video: %dx%d\n",
        input_stream->codecpar->width,
        input_stream->codecpar->height);


    /* --------------------------------------------------------
     * CUDA hardware device
     * -------------------------------------------------------- */

    AVBufferRef *hw_device_ctx =
        NULL;


    ret =
        create_cuda_device(
            &hw_device_ctx);

    if (ret < 0)
    {
        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Decoder
     * -------------------------------------------------------- */

    AVCodecContext *decoder_ctx =
        NULL;


    ret =
        open_decoder(
            input_ctx,
            video_stream_index,
            hw_device_ctx,
            &decoder_ctx);

    if (ret < 0)
    {
        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    int width =
        decoder_ctx->width;

    int height =
        decoder_ctx->height;


    AVRational time_base =
        input_stream->time_base;


    printf(
        "Time base: %d/%d\n",
        time_base.num,
        time_base.den);


    /* --------------------------------------------------------
     * Output CUDA frame pool
     * -------------------------------------------------------- */

    AVBufferRef *output_frames_ref =
        NULL;


    ret =
        create_cuda_frames_context(
            hw_device_ctx,
            &output_frames_ref,
            width,
            height);

    if (ret < 0)
    {
        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Output MP4
     * -------------------------------------------------------- */

    AVFormatContext *output_ctx =
        NULL;


    ret =
        avformat_alloc_output_context2(
            &output_ctx,
            NULL,
            "mp4",
            output_filename);

    if (ret < 0 || !output_ctx)
    {
        fprintf(
            stderr,
            "Could not create output context\n");

        av_buffer_unref(
            &output_frames_ref);

        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Encoder
     * -------------------------------------------------------- */

    AVCodecContext *encoder_ctx =
        NULL;


    ret =
        open_encoder(
            hw_device_ctx,
            output_frames_ref,
            width,
            height,
            time_base,
            &encoder_ctx);

    if (ret < 0)
    {
        avformat_free_context(
            output_ctx);

        av_buffer_unref(
            &output_frames_ref);

        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Output stream
     * -------------------------------------------------------- */

    AVStream *output_stream =
        avformat_new_stream(
            output_ctx,
            NULL);

    if (!output_stream)
    {
        fprintf(
            stderr,
            "Could not create output stream\n");

        avcodec_free_context(
            &encoder_ctx);

        avformat_free_context(
            output_ctx);

        av_buffer_unref(
            &output_frames_ref);

        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
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

        avcodec_free_context(
            &encoder_ctx);

        avformat_free_context(
            output_ctx);

        av_buffer_unref(
            &output_frames_ref);

        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Open output file
     * -------------------------------------------------------- */

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

            avcodec_free_context(
                &encoder_ctx);

            avformat_free_context(
                output_ctx);

            av_buffer_unref(
                &output_frames_ref);

            avcodec_free_context(
                &decoder_ctx);

            av_buffer_unref(
                &hw_device_ctx);

            avformat_close_input(
                &input_ctx);

            return EXIT_FAILURE;
        }
    }


    /* --------------------------------------------------------
     * Write MP4 header
     * -------------------------------------------------------- */

    ret =
        avformat_write_header(
            output_ctx,
            NULL);

    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not write output header",
            ret);

        if (!(output_ctx->oformat->flags &
              AVFMT_NOFILE))
        {
            avio_closep(
                &output_ctx->pb);
        }

        avcodec_free_context(
            &encoder_ctx);

        avformat_free_context(
            output_ctx);

        av_buffer_unref(
            &output_frames_ref);

        avcodec_free_context(
            &decoder_ctx);

        av_buffer_unref(
            &hw_device_ctx);

        avformat_close_input(
            &input_ctx);

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Sobel initialization
     * -------------------------------------------------------- */

    SobelCudaContext sobel_ctx;

    sobel_cuda_init(
        &sobel_ctx);


    /* --------------------------------------------------------
     * Allocate packets and frames
     * -------------------------------------------------------- */

    AVPacket *packet =
        av_packet_alloc();

    AVFrame *decoded_frame =
        av_frame_alloc();

    AVFrame *output_frame =
        av_frame_alloc();


    if (!packet ||
        !decoded_frame ||
        !output_frame)
    {
        fprintf(
            stderr,
            "Could not allocate packet/frame\n");

        return EXIT_FAILURE;
    }


    /* --------------------------------------------------------
     * Processing loop
     * -------------------------------------------------------- */

    int64_t frame_count = 0;

 start_time =
    (double)clock() /
    CLOCKS_PER_SEC;


    printf(
        "Starting GPU processing...\n");


    while (1)
    {
        ret =
            av_read_frame(
                input_ctx,
                packet);

        if (ret < 0)
        {
            /*
             * End of input.
             */

            break;
        }


        if (packet->stream_index !=
            video_stream_index)
        {
            av_packet_unref(
                packet);

            continue;
        }


        /*
         * Send compressed H.264 packet
         * to NVDEC/CUVID.
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

            break;
        }


        /*
         * Receive all decoded frames
         * produced by this packet.
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


            if (decoded_frame->format !=
                AV_PIX_FMT_CUDA)
            {
                fprintf(
                    stderr,
                    "Decoder returned non-CUDA frame\n");

                goto cleanup;
            }


            /*
             * Allocate a fresh CUDA output frame.
             */

            av_frame_unref(
                output_frame);


            output_frame->format =
                AV_PIX_FMT_CUDA;

            output_frame->width =
                width;

            output_frame->height =
                height;


            ret =
                av_hwframe_get_buffer(
                    output_frames_ref,
                    output_frame,
                    0);

            if (ret < 0)
            {
                print_ffmpeg_error(
                    "Could not allocate CUDA output frame",
                    ret);

                goto cleanup;
            }


            /*
             * Preserve presentation timestamp.
             */

            output_frame->pts =
                decoded_frame->pts;


            /*
             * ------------------------------------------------
             * CUDA SOBEL
             * ------------------------------------------------
             *
             * GPU:
             *
             * decoded_frame
             *      ↓
             * CUDA kernel
             *      ↓
             * output_frame
             *
             * No CPU image copy.
             */

            ret =
                sobel_cuda_process_frame(
                    &sobel_ctx,
                    decoded_frame,
                    output_frame,
                    width,
                    height);

            if (ret < 0)
            {
                print_ffmpeg_error(
                    "CUDA Sobel failed",
                    ret);

                goto cleanup;
            }


            /*
             * ------------------------------------------------
             * NVENC
             * ------------------------------------------------
             *
             * output_frame is still:
             *
             *     AV_PIX_FMT_CUDA
             *
             * Therefore NVENC receives the GPU frame.
             */

            ret =
                encode_and_write(
                    encoder_ctx,
                    output_ctx,
                    output_stream,
                    output_frame);

            if (ret < 0)
            {
                goto cleanup;
            }


            frame_count++;


            if ((frame_count % 30) == 0)
            {
                printf(
                    "\rProcessed frames: %ld",
                    (long)frame_count);

                fflush(stdout);
            }
        }
    }


    printf("\n");


    /* --------------------------------------------------------
     * Flush decoder
     * -------------------------------------------------------- */

    ret =
        avcodec_send_packet(
            decoder_ctx,
            NULL);

    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not flush decoder",
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
                "Error flushing decoder",
                ret);

            goto cleanup;
        }


        av_frame_unref(
            output_frame);


        output_frame->format =
            AV_PIX_FMT_CUDA;

        output_frame->width =
            width;

        output_frame->height =
            height;

        output_frame->pts =
            decoded_frame->pts;


        ret =
            av_hwframe_get_buffer(
                output_frames_ref,
                output_frame,
                0);

        if (ret < 0)
        {
            print_ffmpeg_error(
                "Could not allocate flush frame",
                ret);

            goto cleanup;
        }


        ret =
            sobel_cuda_process_frame(
                &sobel_ctx,
                decoded_frame,
                output_frame,
                width,
                height);

        if (ret < 0)
            goto cleanup;


        ret =
            encode_and_write(
                encoder_ctx,
                output_ctx,
                output_stream,
                output_frame);

        if (ret < 0)
            goto cleanup;


        frame_count++;
    }


    /* --------------------------------------------------------
     * Flush NVENC
     * -------------------------------------------------------- */

    ret =
        flush_encoder(
            encoder_ctx,
            output_ctx,
            output_stream);

    if (ret < 0)
        goto cleanup;


    /* --------------------------------------------------------
     * Write MP4 trailer
     * -------------------------------------------------------- */

    ret =
        av_write_trailer(
            output_ctx);

    if (ret < 0)
    {
        print_ffmpeg_error(
            "Could not write trailer",
            ret);

        goto cleanup;
    }


    /* --------------------------------------------------------
     * Statistics
     * -------------------------------------------------------- */

    end_time =
    (double)clock() /
    CLOCKS_PER_SEC;

elapsed =
    end_time -
    start_time;



    printf(
        "\n========================================\n");

    printf(
        "GPU VIDEO PROCESSING COMPLETE\n");

    printf(
        "========================================\n");

    printf(
        "Frames processed : %ld\n",
        (long)frame_count);

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

    printf(
        "Output           : %s\n",
        output_filename);

    printf(
        "========================================\n");


cleanup:

    av_frame_free(
        &decoded_frame);

    av_frame_free(
        &output_frame);

    av_packet_free(
        &packet);


    sobel_cuda_destroy(
        &sobel_ctx);


    avcodec_free_context(
        &encoder_ctx);

    avcodec_free_context(
        &decoder_ctx);


    av_buffer_unref(
        &output_frames_ref);

    av_buffer_unref(
        &hw_device_ctx);


    if (output_ctx)
    {
        if (!(output_ctx->oformat->flags &
              AVFMT_NOFILE))
        {
            avio_closep(
                &output_ctx->pb);
        }

        avformat_free_context(
            output_ctx);
    }


    avformat_close_input(
        &input_ctx);


    if (ret < 0)
    {
        fprintf(
            stderr,
            "\nProcessing failed.\n");

        return EXIT_FAILURE;
    }


    printf(
        "\nSuccessfully created: %s\n",
        output_filename);


    return EXIT_SUCCESS;
}
