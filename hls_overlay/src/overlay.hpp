#pragma once
#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include <ap_int.h>

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

constexpr int PARAM_COUNT = 13440;
constexpr int MAX_DETECTS = 32;

// @see ug1399, HLS Programmers Guide > Customizing-AXI4-Stream-Interfaces
using axis_data64 = ap_axis<64, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;
using axis_data8 = ap_axis<8, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;

using pixel_t = ap_axiu<24,1,1,1>;

template <typename T>
using fifo = hls::stream<T>;

struct Detect {
    ap_uint<8> x1;
    ap_uint<8> y1;
    ap_uint<8> x2;
    ap_uint<8> y2;
    uint16_t score;
    ap_uint<8> kps[10];
    bool started;
    bool ended;
};

extern "C" {
void pattern_overlay(fifo<pixel_t>& pin, fifo<pixel_t>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    const ap_uint<64> params[PARAM_COUNT]);
}
