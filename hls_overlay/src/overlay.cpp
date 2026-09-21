#include "overlay.hpp"
#include <cstdint>

void write_params(const ap_uint<64> params[PARAM_COUNT], fifo<axis_data64>& yunet_ins) {
    int ptr = 0;
    for (int j = 0; j < PARAM_BLOCK_COUNT; j++) {
        int size = params[ptr++];
        assert(size <= 512);
        for (int i = 0; i < size; i++) {
#pragma HLS pipeline 
            axis_data64 pkt;
            pkt.data = params[ptr++];
            pkt.last = (i == size - 1);
            yunet_ins.write(pkt);
        }
    }
}

void read_detects(fifo<axis_data8>& outs, Detect detects[MAX_DETECTIONS], uint8_t& detect_count) {
    detect_count = outs.read().data;

    for (int i = 0; i < MAX_DETECTIONS; i++) {
#pragma HLS pipeline 
        if (i < detect_count) {
            ap_uint<8> x1 = outs.read().data;
            ap_uint<8> y1 = outs.read().data;
            ap_uint<8> x2 = outs.read().data;
            ap_uint<8> y2 = outs.read().data;
            detects[i].x1 = x1 * 8;
            detects[i].y1 = y1 * 8;
            detects[i].x2 = x2 * 8;
            detects[i].y2 = y2 * 8;
            ap_int<8> hi = outs.read().data;
            ap_int<8> lo = outs.read().data;
            // detects[i].score = (hi, lo);
            for (int k = 0; k < 10; k++) {
                ap_uint<8> kps = outs.read().data;
                // detects[i].kps[k] = kps * 8;
            }
        }
    }
}

void update_line_buffer(LineBuffer line_buffer[INPUT_SIZE],
    const uint16_t x, const bool line_boundary, const ap_uint<24>& rbg)
{
#pragma HLS inline

    static uint16_t r_sum = 0;
    static uint16_t g_sum = 0;
    static uint16_t b_sum = 0;

    r_sum += rbg.range(23, 16);
    g_sum += rbg.range(7, 0);
    b_sum += rbg.range(15, 8);
    if ((x & 0x7) == 7) {
        ap_uint<8> r = r_sum >> 3;
        ap_uint<8> g = g_sum >> 3;
        ap_uint<8> b = b_sum >> 3;
        uint8_t cx = x >> 3;
        if (line_boundary) {
            line_buffer[cx].r = r;
            line_buffer[cx].g = g;
            line_buffer[cx].b = b;
        } else {
            line_buffer[cx].r = (line_buffer[cx].r + r) / 2;
            line_buffer[cx].g = (line_buffer[cx].g + g) / 2;
            line_buffer[cx].b = (line_buffer[cx].b + b) / 2;
        }
        r_sum = 0;
        g_sum = 0;
        b_sum = 0;
    }    
}

void write_input_line(LineBuffer line_buffer[INPUT_SIZE], fifo<axis_data64>& yunet_ins) {
#pragma HLS inline

    axis_data64 pkt;
    for (uint8_t cx = 0; cx < INPUT_SIZE; cx++) {
#pragma HLS pipeline
        LineBuffer& buf = line_buffer[cx];
        ap_uint<1> b0 = 0;
        ap_uint<12> rgb = (b0, buf.r.range(7, 5), b0, buf.g.range(7, 5), b0, buf.b.range(7, 5));
        pkt.data = rgb.to_uint64();
        pkt.last = (cx == INPUT_SIZE - 1);
        yunet_ins.write(pkt);                
    }    
}

void write_input_padding(fifo<axis_data64>& yunet_ins) {
#pragma HLS inline

    axis_data64 pkt;
    for (uint8_t cy = 0; cy < 70; cy++) {
        for (uint8_t cx = 0; cx < INPUT_SIZE; cx++) {
#pragma HLS pipeline
            pkt.data = 0;
            pkt.last = (cx == INPUT_SIZE - 1);
            yunet_ins.write(pkt);            
        }
    }    
}

void select_line_sprites(const Detect detects[MAX_DETECTIONS], const uint8_t detect_count,
    const ap_uint<64> sprite_data[SPRITE_LINEBUF_SIZE * SPRITE_SIZE * SPRITE_FRAME_COUNT], const uint8_t frame,
    const uint16_t y, LineSprite line_sprites[MAX_LINE_SPRITES],
    ap_uint<2> sprite_buf[MAX_LINE_SPRITES][SPRITE_SIZE])
{
#pragma HLS inline

    int offset = frame * SPRITE_LINEBUF_SIZE * SPRITE_SIZE;
    int count = 0;
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        if (i < detect_count && count < MAX_LINE_SPRITES) {
            const Detect& detect = detects[i];
            if (detect.y1 <= y && y <= detect.y2) {
                LineSprite& sprite = line_sprites[count];
                uint16_t size = detect.y2 - detect.y1;
                uint16_t base = (y - detect.y1) * SPRITE_SIZE / size;
                uint16_t cx = (detect.x1 + detect.x2) / 2;
                sprite.x1 = cx - size / 2;
                sprite.x2 = cx + size / 2;
                for (int j = 0; j < SPRITE_LINEBUF_SIZE; j++) {
#pragma HLS pipeline
                    ap_uint<64> data = sprite_data[offset + base * SPRITE_LINEBUF_SIZE + j];
                    for (int k = 0; k < 32; k++) {
#pragma HLS unroll
                        sprite_buf[count][j * 32 + k] = data.range(k * 2 + 1, k * 2);
                    }
                }
                const uint32_t dx = (SPRITE_SIZE << 16) / size;
                sprite.src_x = -dx;
                sprite.src_dx = dx;
                sprite.base = 0xffff;
                sprite.enable = true;
                count++;
            }
        }
    }

    for (int i = count; i < MAX_LINE_SPRITES; i++) {
#pragma HLS unroll
        line_sprites[i].enable = false;
    }
}

void set_sprite_pixel(LineSprite line_sprites[MAX_LINE_SPRITES],
    const ap_uint<2> sprite_buf[MAX_LINE_SPRITES][SPRITE_SIZE],
    const uint16_t x, pixel_t& pix)
{
#pragma HLS inline

    for (int i = 0; i < MAX_LINE_SPRITES; i++) {
#pragma HLS unroll
        LineSprite& sprite = line_sprites[i];
        if (sprite.enable && sprite.x1 <= x && x <= sprite.x2) {
            sprite.src_x += sprite.src_dx;
            const uint16_t pos = sprite.src_x >> 16;
            ap_uint<2> color = sprite_buf[i][pos];
            if (color == 1) {
                pix.data = 0x08d64a;
            } else if (color == 2) {
                pix.data = 0xf7f7f7;
            }
        }
    }
}

void pattern_overlay(fifo<pixel_t>& pin, fifo<pixel_t>& pout,
    fifo<axis_data64>& yunet_ins, fifo<axis_data8>& yunet_outs,
    const ap_uint<64> params[PARAM_COUNT],
    const ap_uint<64> sprite_data[SPRITE_LINEBUF_SIZE * SPRITE_SIZE * SPRITE_FRAME_COUNT])
{
#pragma HLS interface axis port=pin
#pragma HLS interface axis port=pout
#pragma HLS interface axis port=yunet_ins
#pragma HLS interface axis port=yunet_outs
#pragma HLS interface m_axi port=params offset=slave bundle=gmem
#pragma HLS interface m_axi port=sprite_data offset=slave bundle=gmem
#pragma HLS interface s_axilite port=params bundle=ctrl
#pragma HLS interface s_axilite port=sprite_data bundle=ctrl
#pragma HLS interface s_axilite port=return bundle=ctrl

    static Detect detects[MAX_DETECTIONS];
    static uint8_t detect_count = 0;
    static uint8_t frame = 0;
// #pragma HLS bind_storage variable=detects type=ram_1p impl=lutram

    LineBuffer line_buffer[INPUT_SIZE];
    LineSprite line_sprites[MAX_LINE_SPRITES];
    ap_uint<2> sprite_buf[MAX_LINE_SPRITES][SPRITE_SIZE];
#pragma HLS bind_storage variable=sprite_buf type=ram_1p impl=lutram
#pragma HLS array_partition variable=line_sprites complete
#pragma HLS array_partition variable=sprite_buf complete dim=1

    bool line_boundary = true;
    for (uint16_t y = 0; y < HEIGHT; y++) {
        select_line_sprites(detects, detect_count, sprite_data, frame, y, line_sprites, sprite_buf);
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            pixel_t pix = pin.read();
            ap_uint<24> rbg = pix.data;
            set_sprite_pixel(line_sprites, sprite_buf, x, pix);
            pout.write(pix);
            update_line_buffer(line_buffer, x, line_boundary, rbg);
        }
        line_boundary = ((y & 0x07) == 7);
        if (line_boundary) {
            write_input_line(line_buffer, yunet_ins);
        }
    }

    write_input_padding(yunet_ins);

#pragma HLS dataflow

    write_params(params, yunet_ins);
    read_detects(yunet_outs, detects, detect_count);
    frame++;
    if (frame == SPRITE_FRAME_COUNT) {
        frame = 0;
    }
}
