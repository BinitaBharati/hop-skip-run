#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <libavutil/version.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

#if LIBAVUTIL_VERSION_MAJOR >= 60
#define GET_FRAME_DURATION(frame) ((frame)->duration)
#define SET_FRAME_DURATION(frame, value) ((frame)->duration = (value))
#else
#define GET_FRAME_DURATION(frame) ((frame)->pkt_duration)
#define SET_FRAME_DURATION(frame, value) ((frame)->pkt_duration = (value))
#endif


/* ---------------------------------------------------------
 * Timing
 * --------------------------------------------------------- */

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return (double)ts.tv_sec * 1000.0
         + (double)ts.tv_nsec / 1000000.0;
}


/* ---------------------------------------------------------
 * Encode one processed frame
 *
 * frame == NULL means:
 * flush the encoder.
 *
 * The frame PTS and duration are already set by the
 * caller.
 * --------------------------------------------------------- */

static int encode_frame(
    AVCodecContext *encoder_ctx,
    AVFormatContext *output_format_ctx,
    AVStream *output_stream,
    AVFrame *frame)
{
    int ret;

    /*
     * Send frame to encoder.
     *
     * Passing NULL flushes the encoder.
     */
    ret = avcodec_send_frame(
        encoder_ctx,
        frame
    );

    if (ret < 0)
    {
        fprintf(
            stderr,
            "Error sending frame to encoder\n"
        );

        return ret;
    }


    /*
     * Receive all packets currently available
     * from the encoder.
     */
    while (1)
    {
        AVPacket *packet =
            av_packet_alloc();

        if (!packet)
        {
            fprintf(
                stderr,
                "Could not allocate output packet\n"
            );

            return AVERROR(ENOMEM);
        }


        ret = avcodec_receive_packet(
            encoder_ctx,
            packet
        );


        /*
         * Encoder needs more frames.
         */
        if (ret == AVERROR(EAGAIN) ||
            ret == AVERROR_EOF)
        {
            av_packet_free(&packet);
            break;
        }


        if (ret < 0)
        {
            fprintf(
                stderr,
                "Error receiving encoded packet\n"
            );

            av_packet_free(&packet);

            return ret;
        }


        /*
         * Convert packet timestamps from the encoder
         * time base to the output stream time base.
         */
        av_packet_rescale_ts(
            packet,
            encoder_ctx->time_base,
            output_stream->time_base
        );


        /*
         * The MP4 muxer needs the packet duration.
         *
         * For constant-frame-rate video:
         *
         *     duration = 1 / FPS
         *
         * expressed in the output stream time base.
         *
         * Example:
         *
         *     25 FPS
         *     time base = 1/12800
         *
         *     duration = 12800 / 25
         *              = 512 ticks
         */
        if (packet->duration <= 0 &&
            encoder_ctx->framerate.num > 0 &&
            encoder_ctx->framerate.den > 0)
        {
            packet->duration =
                av_rescale_q(
                    1,
                    av_inv_q(
                        encoder_ctx->framerate
                    ),
                    output_stream->time_base
                );
        }


        packet->stream_index =
            output_stream->index;


        /*
         * Write encoded packet to the output
         * container.
         */
        ret = av_interleaved_write_frame(
            output_format_ctx,
            packet
        );


        av_packet_free(&packet);


        if (ret < 0)
        {
            fprintf(
                stderr,
                "Error writing output packet\n"
            );

            return ret;
        }
    }

    return 0;
}

#endif /* COMMON_H */