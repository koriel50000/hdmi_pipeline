#include "overlay.hpp"

constexpr int PARAM_SIZES[] = {
    // YuNetBackbone stage0
    // Conv_head
    16 * 9 + 16 * 4,
    // Conv_head ConvDPUnit
    16 * 1 + 16 * 4,
    16 * 1 + 16 * 4,
    // YuNetBackbone stage1
    // YuNetBackbone Conv4layerBlock 1
    16 * 1 + 16 * 4,
    16 * 1 + 16 * 4,
    // YuNetBackbone Conv4layerBlock 2
    64 * 1 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone stage2
    // YuNetBackbone Conv4layerBlock 1
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone Conv4layerBlock 2
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone stage3
    // YuNetBackbone Conv4layerBlock 1
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone Conv4layerBlock 2
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone stage4
    // YuNetBackbone Conv4layerBlock 1
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone Conv4layerBlock 2
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone stage5
    // YuNetBackbone Conv4layerBlock 1
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNetBackbone Conv4layerBlock 2
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,

    // TFPN stride32
    // TFPN ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // TFPN stride16
    // TFPN ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // TFPN stride8
    // TFPN ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNet_Head stride8
    // YuNet_Head shared ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNet_Head stride16
    // YuNet_Head shared ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,
    // YuNet_Head stride32
    // YuNet_Head shared ConvDPUnit
    64 * 1 * 4 + 64 * 4,
    64 * 1 + 64 * 4,

    // YuNet_Head cls ConvDPUnit
    // YuNet_Head stride8
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,
    // YuNet_Head stride16
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,
    // YuNet_Head stride32
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,

    // YuNet_Head bbox ConvDPUnit
    // YuNet_Head stride8
    4 * 1 * 4 + 4 * 4,
    4 * 1 + 4 * 4,
    // YuNet_Head stride16
    4 * 1 * 4 + 4 * 4,
    4 * 1 + 4 * 4,
    // YuNet_Head stride32
    4 * 1 * 4 + 4 * 4,
    4 * 1 + 4 * 4,

    // YuNet_Head obj ConvDPUnit
    // YuNet_Head stride8
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,
    // YuNet_Head stride16
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,
    // YuNet_Head stride32
    1 * 1 * 4 + 1 * 4,
    1 * 1 + 1 * 4,

    // YuNet_Head kps ConvDPUnit
    // YuNet_Head stride8
    10 * 1 * 4 + 10 * 4,
    10 * 1 + 10 * 4,
    // YuNet_Head stride16
    10 * 1 * 4 + 10 * 4,
    10 * 1 + 10 * 4,
    // YuNet_Head stride32
    10 * 1 * 4 + 10 * 4,
    10 * 1 + 10 * 4
};

constexpr int PARAM_BLOCK_COUNT = sizeof(PARAM_SIZES) / sizeof(PARAM_SIZES[0]);

void select_line_sprites(const Detect detects[MAX_DETECTIONS], const ap_uint<8> detect_count,
    const uint16_t y, LineSprite line_sprites[MAX_LINE_SPRITES])
{
#pragma HLS inline

    int count = 0;
    for (int i = 0; i < MAX_DETECTIONS; i++) {
#pragma HLS pipeline
        if (i < detect_count && count < MAX_LINE_SPRITES) {
            if (detects[i].y1 <= y && y <= detects[i].y2) {
                line_sprites[count].x1 = detects[i].x1;
                line_sprites[count].x2 = detects[i].x2;
                line_sprites[count].enable = true;
                count++;
            }
        }
    }

    for (int i = count; i < MAX_LINE_SPRITES; i++) {
#pragma HLS unroll
        line_sprites[i].enable = false;
    }
}

void set_sprite_pixel(const LineSprite line_sprites[MAX_LINE_SPRITES], const uint16_t x, pixel_t& pix) {
#pragma HLS inline

    for (int i = 0; i < MAX_LINE_SPRITES; i++) {
#pragma HLS unroll
        if (line_sprites[i].enable && line_sprites[i].x1 <= x && x <= line_sprites[i].x2) {
            pix.data = 0x0000ff;
        }
    }
}

// void write_params(const ap_uint<64> params[PARAM_COUNT]) {
void write_params(const ap_uint<64> params[PARAM_COUNT], fifo<axis_data64>& yunet_ins) {
    int ptr = 0;
    for (int j = 0; j < PARAM_BLOCK_COUNT; j++) {
        for (int i = 0; i < PARAM_SIZES[j]; i++) {
#pragma HLS pipeline 
            axis_data64 pkt;
            pkt.data = params[ptr++];
            pkt.last = (i == PARAM_SIZES[j] - 1);
            yunet_ins.write(pkt);
        }
    }
}

// void read_detects(Detect detects[MAX_DETECTIONS], ap_uint<8>& count) {
void read_detects(fifo<axis_data8>& outs, Detect detects[MAX_DETECTIONS], ap_uint<8>& detect_count) {
    detect_count = outs.read().data;

    // detects[0] = Detect{ 48, 36, 84, 84, 49153, { 56, 53, 67, 51, 61, 59, 57, 66, 70, 66 } };
    // detects[1] = Detect{ 110, 65, 146, 113, 45942, { 126, 83, 138, 83, 134, 89, 129, 98, 138, 98 } };
    // detects[2] = Detect{ 11, 121, 47, 157, 45942, { 24, 136, 35, 136, 30, 139, 24, 147, 35, 145 } };
    // detects[3] = Detect{ 13, 33, 35, 63, 42237, { 19, 43, 25, 43, 22, 49, 19, 54, 27, 54 } };
    // detects[4] = Detect{ 131, 38, 157, 68, 38874, { 144, 49, 150, 49, 150, 53, 145, 59, 154, 59 } };
    // detects[5] = Detect{ 70, 97, 106, 145, 35739, { 88, 115, 96, 115, 93, 121, 88, 130, 96, 130 } };

    for (int i = 0; i < MAX_DETECTIONS; i++) {
#pragma HLS pipeline 
        if (i < detect_count) {
            ap_uint<8> x1 = outs.read().data;
            ap_uint<8> y1 = outs.read().data;
            ap_uint<8> x2 = outs.read().data;
            ap_uint<8> y2 = outs.read().data;
            detects[i].x1 = x1 * 8;
            detects[i].y1 = y1 * 8;
            detects[i].x2 = x2 * 8;
            detects[i].y2 = y2 * 8;
            ap_int<8> hi = outs.read().data;
            ap_int<8> lo = outs.read().data;
            // detects[i].score = (hi, lo);
            for (int k = 0; k < 10; k++) {
                ap_uint<8> kps = outs.read().data;
                // detects[i].kps[k] = kps * 8;
            }
        }
    }
}

// void yunet(fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
//     Detect detects[MAX_DETECTIONS], ap_uint<8>& detect_count,
//     const ap_uint<64> params[PARAM_COUNT])
// {
// #pragma HLS dataflow

//     write_params(params, yunet_ins);
//     read_detects(yunet_outs, detects, detect_count);    
// }

void pattern_overlay(fifo<pixel_t>& pin, fifo<pixel_t>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    const ap_uint<64> params[PARAM_COUNT])
{
#pragma HLS interface axis port=pin
#pragma HLS interface axis port=pout
#pragma HLS interface axis port=yunet_ins
#pragma HLS interface axis port=yunet_outs
#pragma HLS interface m_axi port=params offset=slave bundle=gmem
// #pragma HLS interface m_axi port=result offset=slave bundle=gmem
#pragma HLS interface s_axilite port=params bundle=ctrl
// #pragma HLS interface s_axilite port=result bundle=ctrl
#pragma HLS interface s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTIONS];
    static ap_uint<8> detect_count = 0;

    LineSprite line_sprites[MAX_LINE_SPRITES];
    LineBuffer line_buffer[INPUT_SIZE];

    axis_data64 pkt;

    bool line_boundary = true;
    for (uint16_t y = 0; y < HEIGHT; y++) {
        select_line_sprites(detects, detect_count, y, line_sprites);
        uint16_t r_sum = 0;
        uint16_t g_sum = 0;
        uint16_t b_sum = 0;
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            pixel_t pix = pin.read();
            set_sprite_pixel(line_sprites, x, pix);
            pout.write(pix);

            ap_uint<24> rbg = pix.data;
            r_sum += rbg.range(23, 16);
            g_sum += rbg.range(7, 0);
            b_sum += rbg.range(15, 8);
            if ((x & 0x7) == 7) {
                ap_uint<8> r = r_sum >> 3;
                ap_uint<8> g = g_sum >> 3;
                ap_uint<8> b = b_sum >> 3;
                uint8_t cx = x >> 3;
                if (line_boundary) {
                    line_buffer[cx].r = r;
                    line_buffer[cx].g = g;
                    line_buffer[cx].b = b;
                } else {
                    line_buffer[cx].r = (line_buffer[cx].r + r) / 2;
                    line_buffer[cx].g = (line_buffer[cx].g + g) / 2;
                    line_buffer[cx].b = (line_buffer[cx].b + b) / 2;
                }
                r_sum = 0;
                g_sum = 0;
                b_sum = 0;
            }
        }
        line_boundary = ((y & 0x07) == 7);
        if (line_boundary) {
            for (uint8_t cx = 0; cx < INPUT_SIZE; cx++) {
#pragma HLS pipeline
                LineBuffer& buf = line_buffer[cx];
                ap_uint<1> b0 = 0;
                ap_uint<12> rgb = (b0, buf.r.range(7, 5), b0, buf.g.range(7, 5), b0, buf.b.range(7, 5));
                pkt.data = rgb.to_uint64();
                pkt.last = (cx == INPUT_SIZE - 1);
                yunet_ins.write(pkt);                
            }
        }
    }

    for (uint8_t cy = 0; cy < 70; cy++) {
        for (uint8_t cx = 0; cx < INPUT_SIZE; cx++) {
#pragma HLS pipeline
            pkt.data = 0;
            pkt.last = (cx == INPUT_SIZE - 1);
            yunet_ins.write(pkt);            
        }
    }

    // yunet(yunet_ins, yunet_outs, detects, detect_count, params);
#pragma HLS dataflow

    write_params(params, yunet_ins);
    read_detects(yunet_outs, detects, detect_count);    

//     int ptr = 0;
//     result[ptr++] = detect_count;
//     for (int i = 0; i < MAX_DETECTIONS; i++) {
// #pragma HLS pipeline
//         if (i < detect_count) {
//             result[ptr++] = detects[i].x1;
//             result[ptr++] = detects[i].y1;
//             result[ptr++] = detects[i].x2;
//             result[ptr++] = detects[i].y2;
//             result[ptr++] = 0;
//             result[ptr++] = 0;
//             for (int k = 0; k < 10; k++) {
//                 result[ptr++] = 0;
//             }
//         }
//     }
}
