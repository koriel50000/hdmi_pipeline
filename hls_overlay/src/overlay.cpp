#include "overlay.hpp"
#include "image.hpp"

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
        
// void write_params(ap_uint<64> params[PARAM_COUNT], fifo<axis_data64>& ins) {
void write_params(ap_uint<64> params[PARAM_COUNT]) {
    int ptr = 0;
	axis_data64 pkt;
    for (int j = 0; j < PARAM_BLOCK_COUNT; j++) {
#pragma HLS pipeline 
        for (int i = 0; i < PARAM_SIZES[j]; i++) {
            pkt.data = params[ptr++];
            pkt.last = (i == PARAM_SIZES[j] - 1);
            // ins.write(pkt);
        }
    }
}

// void read_detects(fifo<axis_data8>& outs, Detect detects[MAX_DETECTS], ap_uint<8>& count) {
void read_detects(Detect detects[MAX_DETECTS], ap_uint<8>& count) {
    axis_data8 data;
    // data = outs.read();
    count = 0; //data.data;

    for (int i = 0; i < MAX_DETECTS; i++) {
        if (i < count) {
            // detects[i].x1 = outs.read().data;
            // detects[i].y1 = outs.read().data;
            // detects[i].x2 = outs.read().data;
            // detects[i].y2 = outs.read().data;
            // ap_int<8> hi = outs.read().data;
            // ap_int<8> lo = outs.read().data;
            // detects[i].score = (hi, lo);
            // detects[i].kps_x[0] = outs.read().data;
            // detects[i].kps_y[0] = outs.read().data;
            // detects[i].kps_x[1] = outs.read().data;
            // detects[i].kps_y[1] = outs.read().data;
            // detects[i].kps_x[2] = outs.read().data;
            // detects[i].kps_y[2] = outs.read().data;
            // detects[i].kps_x[3] = outs.read().data;
            // detects[i].kps_y[3] = outs.read().data;
            // detects[i].kps_x[4] = outs.read().data;
            // detects[i].kps_y[4] = outs.read().data;
        }
    }
}

void pattern_overlay(
    // fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    ap_uint<64> params[PARAM_COUNT])
{
// #pragma HLS INTERFACE axis port=yunet_ins
// #pragma HLS INTERFACE axis port=yunet_outs
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=params bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTS];
    static ap_uint<8> detect_count = 0;

    int ptr = 0;
    axis_data64 pkt;
    for (int j = 0; j < 160; j += 20) {
        for (int i = 0; i < 160 * 20; i++) {
            pkt.data = images[ptr++];
            pkt.last = (i == 160 * 20 - 1);
            // yunet_ins.write(pkt);
        }
    }

// #pragma HLS dataflow

    // write_params(params, yunet_ins);
    // read_detects(yunet_outs, detects, detect_count);
    write_params(params);
    read_detects(detects, detect_count);
}
