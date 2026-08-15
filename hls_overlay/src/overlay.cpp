#include "overlay.hpp"

void pattern_overlay(fifo<axis_data64>& pin, fifo<axis_data8>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs)
{
#pragma HLS INTERFACE axis port=pin
#pragma HLS INTERFACE axis port=pout
#pragma HLS INTERFACE axis port=yunet_ins
#pragma HLS INTERFACE axis port=yunet_outs
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTS];
    static ap_uint<8> detect_count = 0;

    // write_params(pin, yunet_ins);
    axis_data64 ival;
    for (int i = 0; i < 160 * 160; i++) {
        ival = pin.read();
        yunet_ins.write(ival);
    }
    for (int i = 0; i < 13440; i++) {
        ival = pin.read();
        yunet_ins.write(ival);
    }

    // read_detects(yunet_outs, detects, detect_count);
    axis_data8 oval;
    oval = yunet_outs.read();
    int count = oval.data;
    pout.write(oval);
    for (int i = 0; i < MAX_DETECTS; i++) {
        if (i < count) {
            for (int j = 0; j < 16; j++) {
                oval = yunet_outs.read();
                pout.write(oval);
            }
        }
    }
}
