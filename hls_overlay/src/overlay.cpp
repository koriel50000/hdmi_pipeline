#include "overlay.hpp"

void write_params(fifo<axis_data64>& ins) {
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
    
	axis_data64 pkt;
    int ptr = 0;
    for (int j = 0; j < sizeof(param_counts) / sizeof(param_counts[0]); j++) {
        printf("len=%d\n", param_counts[j]);
        for (int i = 0; i < param_counts[j]; i++) {
            pkt.data = 0LL;
            pkt.last = (i == param_counts[j] - 1);
            ins.write(pkt);
        }
    }
}

void pattern_overlay(fifo<axis_data8>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs)
{
#pragma HLS INTERFACE axis port=pout
#pragma HLS INTERFACE axis port=yunet_ins
#pragma HLS INTERFACE axis port=yunet_outs
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTS];
    static ap_uint<8> detect_count = 0;

    axis_data64 ival;
    for (int j = 0; j < 160; j += 20) {
        for (int i = 0; i < 160 * 20; i++) {
            ival.data = 0LL;
            ival.last = (i == 160 * 20 - 1);
            yunet_ins.write(ival);
        }
    }
    write_params(yunet_ins);

    // read_detects(yunet_outs, detects, detect_count);
    axis_data8 oval;
    // oval = yunet_outs.read();
    yunet_outs.read_nb(oval);
    int count = 0; //oval.data;
    oval.data = 0;
    oval.last = 1;
    pout.write(oval);
    for (int i = 0; i < MAX_DETECTS; i++) {
        if (i < count) {
            for (int j = 0; j < 16; j++) {
                oval.data = 0; //yunet_outs.read();
                oval.last = (j == 16 - 1);
                pout.write(oval);
            }
        }
    }
}
