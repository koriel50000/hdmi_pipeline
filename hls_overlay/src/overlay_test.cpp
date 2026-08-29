#include <stdio.h>
#include "overlay.hpp"

int main ()
{
    fifo<pixel_t> pin;
    fifo<pixel_t> pout;

    pixel_t p;
    p.data = 0;
    p.keep = 0x7;
    p.strb = 0x7;
    p.user = 0;
    p.last = 0;
    p.id = 0;
    p.dest = 0;

    for (uint16_t y = 0; y < HEIGHT; y++) {
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            p.data = 0;
            p.user[0] = (x == 0 && y == 0);
            p.last    = (x == WIDTH - 1);
            pin.write(p);
        }
    }

    pattern_overlay(pin, pout);
    return 0;
}