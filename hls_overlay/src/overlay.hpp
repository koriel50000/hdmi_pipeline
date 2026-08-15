#pragma once
#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include <ap_int.h>

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;
constexpr int MAX_DETECTS = 64;

struct Detect {
    ap_uint<8> x1;
    ap_uint<8> y1;
    ap_uint<8> x2;
    ap_uint<8> y2;

    ap_uint<16> score;

    ap_uint<8> kps_x[5];
    ap_uint<8> kps_y[5];
};

// @see ug1399, HLS Programmers Guide > Customizing-AXI4-Stream-Interfaces
using axis_data64 = ap_axis<64, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;
using axis_data8 = ap_axis<8, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;

// using pixel_t = ap_axiu<24,1,1,1>;

template <typename T>
using fifo = hls::stream<T>;

extern "C" {
void pattern_overlay(fifo<axis_data8>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs);
}
