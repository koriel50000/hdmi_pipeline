#include <cstdint>
#include <stdio.h>
#include "overlay.hpp"

int main ()
{
    fifo<pixel_t> pin;
    fifo<pixel_t> pout;
    fifo<axis_data64> yunet_ins;
    fifo<axis_data8> yunet_outs;
    const ap_uint<64> params[PARAM_COUNT] = {};

    for (uint16_t y = 0; y < HEIGHT; y++) {
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            pixel_t p;
            pin.write(p);
        }
    }

    axis_data8 pkt;
    pkt.data = 0;
    pkt.last = 1;
    yunet_outs.write(pkt);

    pattern_overlay(pin, pout, yunet_ins, yunet_outs, params);

    return 0;
}