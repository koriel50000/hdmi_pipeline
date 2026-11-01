#include "overlay.hpp"
#include <cstdint>

void write_params(const ap_uint<64> params[PARAM_COUNT], fifo<axis_data64>& yunet_ins) {
    int ptr = 0;
    for (int i = 0; i < PARAM_COUNT; i++) {
#pragma HLS pipeline 
        axis_data64 pkt;
        pkt.data = params[i];
        pkt.last = (i == PARAM_COUNT - 1);
        yunet_ins.write(pkt);
    }
}

void calculate_face_pitch(Detect& d) {
#pragma HLS inline

    int32_t lx = d.kps[0];
    int32_t ly = d.kps[1];
    int32_t rx = d.kps[2];
    int32_t ry = d.kps[3];

    int32_t mlx = d.kps[6];
    int32_t mly = d.kps[7];
    int32_t mrx = d.kps[8];
    int32_t mry = d.kps[9];

    if (lx > rx) {
        int32_t tmp = lx; lx = rx; rx = tmp;
        tmp = ly; ly = ry; ry = tmp;
        tmp = mlx; mlx = mrx; mrx = tmp;
        tmp = mly; mly = mry; mry = tmp;
    }

    int32_t ex = (lx + rx) >> 1;
    int32_t ey = (ly + ry) >> 1;
    int32_t mx = (mlx + mrx) >> 1;
    int32_t my = (mly + mry) >> 1;

    int32_t vx = mx - ex;
    int32_t vy = my - ey;

    int32_t nx = d.kps[4];
    int32_t ny = d.kps[5];

    int32_t expected_x = ex + (vx * 3) / 5;
    int32_t expected_y = ey + (vy * 3) / 5;

    int32_t dx = nx - expected_x;
    int32_t dy = ny - expected_y;

    int32_t v2 = vx * vx + vy * vy;

    if (v2 < 16) {
        d.pitch = 0;
        return;
    }

    int32_t pitch = ((dx * vx + dy * vy) << 8) / v2;

    if (pitch > 64) pitch = 64;
    if (pitch < -64) pitch = -64;

    d.pitch = pitch;
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
            detects[i].x1 = x1 * 5 + 240;
            detects[i].y1 = y1 * 9 / 2;
            detects[i].x2 = x2 * 5 + 240;
            detects[i].y2 = y2 * 9 / 2;
            ap_int<8> hi = outs.read().data;
            ap_int<8> lo = outs.read().data;
            // detects[i].score = (hi, lo);
            for (int k = 0; k < 10; k++) {
                ap_uint<8> kps = outs.read().data;
                detects[i].kps[k] = kps;
            }
            calculate_face_pitch(detects[i]);
        }
    }
}

void update_line_buffer(LineBuffer line_buf[INPUT_SIZE], const uint8_t cx, const ap_uint<24>& rbg)
{
#pragma HLS inline
    LineBuffer& buf = line_buf[cx];

    buf.r_sum += rbg.range(23, 16);
    buf.g_sum += rbg.range(7, 0);
    buf.b_sum += rbg.range(15, 8);
}

void write_input_line(fifo<axis_data64>& yunet_ins, LineBuffer line_buf[INPUT_SIZE]) {
    // const uint8_t cy, VideoBuffer& video_buf) {
#pragma HLS inline
    for (uint8_t cx = 0; cx < INPUT_SIZE; cx++) {
#pragma HLS pipeline
        LineBuffer& buf = line_buf[cx];
        ap_uint<12> rgb = 0;
        rgb.range(10, 8) = buf.r_sum >> (2 + 2 + 5);
        rgb.range(6, 4) = buf.g_sum >> (2 + 2 + 5);
        rgb.range(2, 0) = buf.b_sum >> (2 + 2 + 5);
        // video_buf[cy][cx] = rgb;

        axis_data64 pkt;
        pkt.data = rgb.to_uint64();
        pkt.last = (cx == INPUT_SIZE - 1);
        yunet_ins.write(pkt);                

        buf.r_sum = 0;
        buf.g_sum = 0;
        buf.b_sum = 0;
    }
}

// void set_video_pixel(VideoBuffer& video_buf, const uint8_t cx, const uint8_t cy, pixel_t& pix) {
// #pragma HLS inline
//     ap_uint<12> rgb = video_buf[cy][cx];
//     ap_uint<24> rbg = 0;
//     rbg.range(23, 21) = rgb.range(10, 8);
//     rbg.range(15, 13) = rgb.range(2, 0);
//     rbg.range( 7,  5) = rgb.range(6, 4);
//     pix.data = rbg;
// }

void select_line_sprites(const Detect detects[MAX_DETECTIONS], const uint8_t detect_count,
    const ap_uint<64> sprite_data[SPRITE_LINEBUF_SIZE * SPRITE_SIZE * SPRITE_FRAME_COUNT],
    const uint8_t frame, const uint16_t y,
    LineSprite line_sprites[MAX_LINE_SPRITES],
    ap_uint<2> sprite_buf[MAX_LINE_SPRITES][SPRITE_SIZE])
{
#pragma HLS inline

    uint32_t offset = frame * SPRITE_LINEBUF_SIZE * SPRITE_SIZE;
    int count = 0;

    for (int i = 0; i < MAX_DETECTIONS; i++) {
        if (i < detect_count && count < MAX_LINE_SPRITES) {
            const Detect& detect = detects[i];

            int32_t face_h = detect.y2 - detect.y1;
            if (face_h <= 0) continue;

            int32_t cx = (detect.x1 + detect.x2) >> 1;
            int32_t cy = (detect.y1 + detect.y2) >> 1;

            int32_t pitch = detect.pitch;

            if (pitch > 64) pitch = 64;
            if (pitch < -64) pitch = -64;

            int32_t top = cy - face_h / 2;
            int32_t bottom = top + face_h - 1;

            if (top <= y && y <= bottom) {
                LineSprite& sprite = line_sprites[count];

                // 顔の中心からの垂直距離
                int32_t row = 2 * (int32_t)y - top - bottom;

                // 上下方向の元画像参照位置
                int32_t base = ((int32_t)(y - top) * SPRITE_SIZE) / face_h
                    - ((int64_t)row * pitch * SPRITE_SIZE) / (face_h * 512);
                if (base < 0) base = 0;
                if (base >= SPRITE_SIZE) base = SPRITE_SIZE - 1;

                // Pitchによる横方向の拡縮
                int32_t width_scale = 256 + (row * pitch * 2) / face_h;

                int32_t line_width = (face_h * width_scale) >> 8;
                if (line_width < 1) line_width = 1;

                // 顔の中心を基準に横幅を変更
                int32_t raw_x1 = cx - line_width / 2;
                int32_t raw_x2 = raw_x1 + line_width - 1;

                if (raw_x2 >= 0 && raw_x1 < WIDTH) {
                    int32_t x1 = raw_x1 < 0 ? 0 : raw_x1;
                    int32_t x2 = raw_x2 >= WIDTH ? WIDTH - 1 : raw_x2;
                    sprite.x1 = x1;
                    sprite.x2 = x2;

                    // 横方向の元画像参照倍率
                    sprite.src_dx = ((uint64_t)SPRITE_SIZE << 16) / line_width;
                    sprite.src_x = (x1 - raw_x1) * (int32_t)sprite.src_dx - (int32_t)sprite.src_dx;
                    sprite.enable = true;

                    // 元画像の1ラインをDDRから読み込む
                    for (int j = 0; j < SPRITE_LINEBUF_SIZE; j++) {
#pragma HLS pipeline II=1
                        ap_uint<64> data =
                            sprite_data[offset + base * SPRITE_LINEBUF_SIZE + j];

                        for (int k = 0; k < 32; k++) {
#pragma HLS unroll
                            sprite_buf[count][j * 32 + k] =
                                data.range(k * 2 + 1, k * 2);
                        }
                    }
                    count++;
                }
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

            int32_t pos = sprite.src_x >> 16;

            if (pos >= 0 && pos < SPRITE_SIZE) {
                ap_uint<2> color = sprite_buf[i][pos];

                if (color == 1) {
                    pix.data = 0x08d64a;
                } else if (color == 2) {
                    pix.data = 0xf7f7f7;
                }
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

    static LineSprite line_sprites[MAX_LINE_SPRITES];
    static ap_uint<2> sprite_buf[MAX_LINE_SPRITES][SPRITE_SIZE];
#pragma HLS bind_storage variable=sprite_buf type=ram_1p impl=lutram
#pragma HLS array_partition variable=line_sprites complete
#pragma HLS array_partition variable=sprite_buf complete dim=1

    static LineBuffer line_buf[INPUT_SIZE] = { 0 };
#pragma HLS array_partition variable=line_buf complete
//     static VideoBuffer video_buf = { 0 };
// #pragma HLS array_partition variable=video_buf cyclic factor=INPUT_SIZE dim=2

    pixel_t pix = pin.read();
    do {
        pix = pin.read();
    } while (!pix.user);

    ap_uint<1> phase = 0;
    // uint8_t cy = 0;
    ap_uint<4> cy_count = 4;
    for (uint16_t y = 0; y < HEIGHT; y++) {
        uint8_t cx = 0;
        ap_uint<4> cx_count = 5;
        select_line_sprites(detects, detect_count, sprite_data, frame, y, line_sprites, sprite_buf);
        for (uint16_t x = 0; x < WIDTH; x++) {
#pragma HLS pipeline
            if (x != 0 || y != 0) {
                pix = pin.read();
            }
            if (240 <= x && x < 1040) {
                if (cy_count != 5 && cx_count != 5) {
                    update_line_buffer(line_buf, cx, pix.data);                    
                }
                if (--cx_count == 0) {
                    cx++;
                    cx_count = 5;
                }
            } else {
                pix.data.range(23, 16) = pix.data.range(7, 0); // g -> r
                pix.data.range(15, 8) = pix.data.range(7, 0); // g -> b
            }
            // if (y >= 560 && x >= 1120) {
            //     set_video_pixel(video_buf, x - 1120, y - 560, pix);
            // } else {
            set_sprite_pixel(line_sprites, sprite_buf, x, pix);
            pix.last = (x == WIDTH - 1);
            pix.user = (x == 0 && y == 0);
            pout.write(pix);
        }
        if (--cy_count == 0) {
            write_input_line(yunet_ins, line_buf);
            // cy++;
            phase ^= 1;
            cy_count = 4 | phase;
        }
    }

#pragma HLS dataflow

    write_params(params, yunet_ins);
    read_detects(yunet_outs, detects, detect_count);
    frame++;
    if (frame == SPRITE_FRAME_COUNT) {
        frame = 0;
    }
}
