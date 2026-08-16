#include <stdio.h>
#include "overlay.hpp"

int main ()
{
    fifo<axis_data64> ins;
    fifo<axis_data8> outs;
    pattern_overlay(ins, outs);
    return 0;
}