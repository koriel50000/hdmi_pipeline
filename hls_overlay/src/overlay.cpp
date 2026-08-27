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

// void write_params(const ap_uint<64> params[PARAM_COUNT]) {
void write_params(const ap_uint<64> params[PARAM_COUNT], fifo<axis_data64>& ins) {
    int ptr = 0;
    axis_data64 pkt;
    for (int j = 0; j < PARAM_BLOCK_COUNT; j++) {
        for (int i = 0; i < PARAM_SIZES[j]; i++) {
#pragma HLS pipeline 
            pkt.data = params[ptr++];
            pkt.last = (i == PARAM_SIZES[j] - 1);
            ins.write(pkt);
        }
    }
}

// void read_detects(Detect detects[MAX_DETECTIONS], ap_uint<8>& count) {
void read_detects(fifo<axis_data8>& outs, Detect detects[MAX_DETECTIONS], ap_uint<8>& count) {
    count = outs.read().data;

    // detects[0] = Detect{ 48, 36, 84, 84, 49153, { 56, 53, 67, 51, 61, 59, 57, 66, 70, 66 } };
    // detects[1] = Detect{ 110, 65, 146, 113, 45942, { 126, 83, 138, 83, 134, 89, 129, 98, 138, 98 } };
    // detects[2] = Detect{ 11, 121, 47, 157, 45942, { 24, 136, 35, 136, 30, 139, 24, 147, 35, 145 } };
    // detects[3] = Detect{ 13, 33, 35, 63, 42237, { 19, 43, 25, 43, 22, 49, 19, 54, 27, 54 } };
    // detects[4] = Detect{ 131, 38, 157, 68, 38874, { 144, 49, 150, 49, 150, 53, 145, 59, 154, 59 } };
    // detects[5] = Detect{ 70, 97, 106, 145, 35739, { 88, 115, 96, 115, 93, 121, 88, 130, 96, 130 } };

    for (int i = 0; i < MAX_DETECTIONS; i++) {
#pragma HLS pipeline 
        if (i < count) {
            // detects[i].x1 = detects[i].x1 * 8;
            // detects[i].y1 = detects[i].y1 * 9 / 2;
            // detects[i].x2 = detects[i].x2 * 8;
            // detects[i].y2 = detects[i].y2 * 9 / 2;
            detects[i].x1 = outs.read().data * 8;
            detects[i].y1 = outs.read().data * 9 / 2;
            detects[i].x2 = outs.read().data * 8;
            detects[i].y2 = outs.read().data * 9 / 2;
            ap_int<8> hi = outs.read().data;
            ap_int<8> lo = outs.read().data;
            detects[i].score = (hi, lo);
            for (int k = 0; k < 10; k++) {
                detects[i].kps[k] = outs.read().data;
            }
        }
    }
}

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
#pragma HLS pipeline
        line_sprites[i].enable = false;
    }
}

void set_sprite_pixel(const LineSprite line_sprites[MAX_LINE_SPRITES], const uint16_t x, ap_uint<24>& pix) {
#pragma HLS inline

    for (int i = 0; i < MAX_LINE_SPRITES; i++) {
#pragma HLS unroll
        if (line_sprites[i].enable && line_sprites[i].x1 <= x && x <= line_sprites[i].x2) {
            pix = 0x0000ff;
        }
    }
}

void pattern_overlay(fifo<pixel_t>& pin,fifo<pixel_t>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    const ap_uint<64> params[PARAM_COUNT])
{
#pragma HLS interface axis port=pin
#pragma HLS interface axis port=pout
#pragma HLS interface axis port=yunet_ins
#pragma HLS interface axis port=yunet_outs
#pragma HLS interface m_axi port=params offset=slave bundle=gmem
#pragma HLS interface s_axilite port=params bundle=ctrl
#pragma HLS interface s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTIONS];
    static ap_uint<8> detect_count = 0;

    LineSprite line_sprites[MAX_LINE_SPRITES];

    pixel_t p;
    p.data = 0;
    p.keep = 0x7;
    p.strb = 0x7;
    p.user = 0;
    p.last = 0;
    p.id = 0;
    p.dest = 0;

    axis_data64 pkt;

    uint16_t dy = HEIGHT / 2;
    for (uint16_t y = 0; y < HEIGHT; y++) {
        select_line_sprites(detects, detect_count, y, line_sprites);
        bool hactive = false;
        dy -= INPUT_SIZE;
        if (dy < 0) {
            dy += HEIGHT;
            hactive = true;
        }
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            ap_uint<24> pix = pin.read().data;
            set_sprite_pixel(line_sprites, x, pix);
            p.data = pix;
            p.user[0] = (x == 0 && y == 0);
            p.last    = (x == WIDTH - 1);
            pout.write(p);

            if ((x & 0x7) == 4 && dy < 0) {
                pkt.data = (pix.range(23, 20), pix.range(7, 4), pix.range(15, 12)); //images[ptr++];
                pkt.last = (x == WIDTH - 4);
                yunet_ins.write(pkt);
            }            
        }
    }

    // int ptr = 0;
    // axis_data64 pkt;
    // for (int j = 0; j < 160; j += 20) {
    //     for (int i = 0; i < 160 * 20; i++) {
    //         pkt.data = 0; //images[ptr++];
    //         pkt.last = (i == 160 * 20 - 1);
    //         yunet_ins.write(pkt);
    //     }
    // }

#pragma HLS dataflow

    write_params(params, yunet_ins);
    read_detects(yunet_outs, detects, detect_count);
}
