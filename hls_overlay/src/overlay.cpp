#include "overlay.hpp"
#include "image.hpp"

void write_params(ap_uint<64>* params, fifo<axis_data64>& ins) {
    static constexpr int param_counts[] = {
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
    
    int ptr = 0;
	axis_data64 pkt;
    for (int j = 0; j < sizeof(param_counts) / sizeof(param_counts[0]); j++) {
        printf("len=%d\n", param_counts[j]);
        for (int i = 0; i < param_counts[j]; i++) {
            pkt.data = params[ptr++];
            pkt.last = (i == param_counts[j] - 1);
            ins.write(pkt);
        }
    }
}

void pattern_overlay(fifo<axis_data8>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    ap_uint<64>* params, ap_uint<32> params_size)
{
#pragma HLS INTERFACE axis port=pout
#pragma HLS INTERFACE axis port=yunet_ins
#pragma HLS INTERFACE axis port=yunet_outs
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=params bundle=ctrl
#pragma HLS INTERFACE s_axilite port=params_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTS];
    static ap_uint<8> detect_count = 0;

    int ptr = 0;
    axis_data64 ival;
    for (int j = 0; j < 160; j += 20) {
        for (int i = 0; i < 160 * 20; i++) {
            ival.data = images[ptr++];
            ival.last = (i == 160 * 20 - 1);
            yunet_ins.write(ival);
        }
    }
    write_params(params, yunet_ins);

    // read_detects(yunet_outs, detects, detect_count);
    axis_data8 oval;
    oval = yunet_outs.read();
    int count = 1;
    // oval.data = params_size / 256;
    oval.last = 1;
    pout.write(oval);
    for (int i = 0; i < MAX_DETECTS; i++) {
        if (i < count) {
            ap_uint<64> v = params[0];
            for (int j = 0; j < 16; j++) {
                oval.data = (v >> (8 * j)) & 0xff; //yunet_outs.read();
                oval.last = (j == 16 - 1);
                pout.write(oval);
            }
        }
    }
}
