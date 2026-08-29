#include <cstdint>
#include <stdio.h>
#include "overlay.hpp"
#include "image.hpp"

int main ()
{
    fifo<pixel_t> pin;
    fifo<pixel_t> pout;
    fifo<axis_data64> yunet_ins;
    fifo<axis_data8> yunet_outs;
    const ap_uint<64> params[PARAM_COUNT] = {};

    // int ptr = 0;
    // for (uint16_t y = 0; y < INPUT_SIZE; y++) {
    //     for (uint16_t x = 0; x < INPUT_SIZE; x++) {
    //         ap_uint<12> p = images[ptr++];
    //         ap_uint<9> rgb = (p.range(10, 8), p.range(6, 4), p.range(2, 0));
    //         printf("0x%03x, ", rgb.to_uint());
    //     }
    //     printf("\n");
    // }

//     for (uint16_t y = 0; y < HEIGHT; y++) {
//         for (uint16_t x = 0; x < WIDTH; x++) {
// #pragma HLS pipeline
//             pixel_t p;
//             pin.write(p);
//         }
//     }

    // axis_data8 pkt;
    // pkt.data = 0;
    // pkt.last = 1;
    // yunet_outs.write(pkt);

    printf("pattern_overlay-in\n");
    pattern_overlay(pin, pout, yunet_ins, yunet_outs, params);
    printf("pattern_overlay-out\n");

    return 0;
}