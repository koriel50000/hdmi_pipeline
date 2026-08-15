# -*- coding: utf-8 -*-

from pynq import Overlay
from pynq import MMIO
from pynq import allocate # 利用可能な領域を確保するメソッドを利用可能にする

import numpy as np
from PIL import Image, ImageDraw
import time
import random

from image import *
from params import *

base = Overlay("./design_1.bit")
overlay = base.pattern_overlay_0
dma = base.axi_dma_0

t0 = time.time()

inbuf = allocate(shape=(160*160,), dtype=np.uint64)
outbuf = allocate(shape=(16,), dtype=np.uint8)


def send_images(images, size):
    assert len(images) == size, "size mismatch"
    inbuf[:] = np.array(images, dtype=np.uint64)
    for base in range(0, size, 160*20):
        dma.sendchannel.transfer(inbuf[base:base+160*20])
        dma.sendchannel.wait()


def send_params(weight, wisize, threshold, thrsize):
    buf = list(weight)
    buf.extend(threshold)
    size = wisize + thrsize
    assert len(buf) == size, "size mismatch"
    inbuf[:size] = np.array(buf, dtype=np.uint64) 
    dma.sendchannel.transfer(inbuf[:size])
    dma.sendchannel.wait()


def recv_output(size):
    dma.recvchannel.transfer(outbuf[:size])
    dma.recvchannel.wait()
    vals = [int(x) for x in outbuf[:size]]
    if size == 1:
        return vals[0]
    else:
        return vals
    # for i in range(size):
    #     assert outbuf[i] == output[i], f"Mismatch at index {i}: outbuf={outbuf[i]}, expected={output[i]}"


def jpg_to_rgb(path, do_crop=False):
    img = Image.open(path).convert("RGB")
    W, H = img.size

    x = random.randint(0, W - 160) if do_crop else 0
    y = random.randint(0, H - 160) if do_crop else 0
    crop = img.crop((x, y, x + 160, y + 160))
    arr = np.array(crop, dtype=np.uint8)

    out = []
    for y in range(160):
        for x in range(160):
            r, g, b = arr[y, x]
            r4 = r >> 5
            g4 = g >> 5
            b4 = b >> 5
            out.append((r4 << 8) | (g4 << 4) | b4)

    return out, crop


def draw_bboxes(img, size, bboxes, kps_list, save_path="output.jpg"):
    draw = ImageDraw.Draw(img)

    kps_colors = [
        (255, 0, 0),    # 左目: 赤
        (255, 0, 0),    # 右目: 赤
        (255, 255, 0),  # 鼻: 黄
        (0, 255, 255),  # 左口角: 水色
        (0, 255, 255),  # 右口角: 水色
    ]

    for i in range(size):
        x1, y1, x2, y2, score = bboxes[i]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        # draw.text((x1, y1), f"{score:.2f}", fill="lime")

        kps = kps_list[i]  # (10,)
        for j in range(5):
            px = kps[2*j]
            py = kps[2*j + 1]
            color = kps_colors[j]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), color, color)

    img.save(save_path)
    print("saved:", save_path)


# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def yunet(images):
    # torch.Size([1, 3, 160, 160])
    print("send_images: images")
    send_images(images, 160 * 160)

    # YuNetBackbone stage0
    # Conv_head
    print("send_params: backbone_model0_conv1_weight")
    send_params(
        backbone_model0_conv1_weight, 16 * 9,           # torch.Size([16, 3, 3, 3])
        backbone_model0_relu1_threshold, 16 * 4 # torch.Size([16, 7])
    )
    # Conv_head ConvDPUnit
    print("send_params: backbone_model0_conv2_conv1_weight")
    send_params(
        backbone_model0_conv2_conv1_weight, 16 * 1,             # torch.Size([16, 1, 1, 16])
        backbone_model0_conv2_quant1_threshold, 16 * 4  # torch.Size([16, 14])
    )
    send_params(
        backbone_model0_conv2_conv2_weight, 16 * 1,         # torch.Size([16, 1, 1, 9])
        backbone_model0_conv2_relu2_threshold, 16 * 4   # torch.Size([16, 7])
    )

    # YuNetBackbone stage1
    # YuNetBackbone Conv4layerBlock 1
    print("send_params: backbone_model1_conv1_conv1_weight")
    send_params(
        backbone_model1_conv1_conv1_weight, 16 * 1,             # torch.Size([16, 1, 1, 16])
        backbone_model1_conv1_quant1_threshold, 16 * 4  # torch.Size([16, 14])
    )
    send_params(
        backbone_model1_conv1_conv2_weight, 16 * 1,         # torch.Size([16, 1, 1, 9])
        backbone_model1_conv1_relu2_threshold, 16 * 4   # torch.Size([16, 7])
    )
    # YuNetBackbone Conv4layerBlock 2
    send_params(
        backbone_model1_conv2_conv1_weight, 64 * 1,             # torch.Size([64, 1, 1, 16])
        backbone_model1_conv2_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model1_conv2_conv2_weight, 64 * 1,         # torch.Size([16, 1, 1, 9])
        backbone_model1_conv2_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # YuNetBackbone stage2
    # YuNetBackbone Conv4layerBlock 1
    print("send_params: backbone_model2_conv1_conv1_weight")
    send_params(
        backbone_model2_conv1_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model2_conv1_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model2_conv1_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model2_conv1_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNetBackbone Conv4layerBlock 2
    send_params(
        backbone_model2_conv2_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model2_conv2_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model2_conv2_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model2_conv2_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # YuNetBackbone stage3
    # YuNetBackbone Conv4layerBlock 1
    print("send_params: backbone_model3_conv1_conv1_weight")
    send_params(
        backbone_model3_conv1_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model3_conv1_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model3_conv1_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model3_conv1_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNetBackbone Conv4layerBlock 2
    send_params(
        backbone_model3_conv2_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model3_conv2_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model3_conv2_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model3_conv2_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # YuNetBackbone stage4
    # YuNetBackbone Conv4layerBlock 1
    print("send_params: backbone_model4_conv1_conv1_weight")
    send_params(
        backbone_model4_conv1_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model4_conv1_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model4_conv1_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model4_conv1_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNetBackbone Conv4layerBlock 2
    send_params(
        backbone_model4_conv2_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model4_conv2_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model4_conv2_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model4_conv2_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # YuNetBackbone stage5
    # YuNetBackbone Conv4layerBlock 1
    print("send_params: backbone_model5_conv1_conv1_weight")
    send_params(
        backbone_model5_conv1_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model5_conv1_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model5_conv1_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model5_conv1_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNetBackbone Conv4layerBlock 2
    send_params(
        backbone_model5_conv2_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        backbone_model5_conv2_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        backbone_model5_conv2_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        backbone_model5_conv2_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # TFPN stride32
    # TFPN ConvDPUnit
    print("send_params: neck_lateral_convs_2_conv1_weight")
    send_params(
        neck_lateral_convs_2_conv1_weight, 64 * 1*4,            # torch.Size([64, 1, 1, 64])
        neck_lateral_convs_2_quant1_threshold, 64 * 4   # torch.Size([64, 14])
    )
    send_params(
        neck_lateral_convs_2_conv2_weight, 64 * 1,          # torch.Size([64, 1, 1, 9])
        neck_lateral_convs_2_relu2_threshold, 64 * 4    # torch.Size([64, 7])
    )
    # TFPN stride16
    # TFPN ConvDPUnit
    print("send_params: neck_lateral_convs_1_conv1_weight")
    send_params(
        neck_lateral_convs_1_conv1_weight, 64 * 1*4,            # torch.Size([64, 1, 1, 64])
        neck_lateral_convs_1_quant1_threshold, 64 * 4   # torch.Size([64, 14])
    )
    send_params(
        neck_lateral_convs_1_conv2_weight, 64 * 1,          # torch.Size([64, 1, 1, 9])
        neck_lateral_convs_1_relu2_threshold, 64 * 4    # torch.Size([64, 7])
    )
    # TFPN stride8
    # TFPN ConvDPUnit
    print("send_params: neck_lateral_convs_0_conv1_weight")
    send_params(
        neck_lateral_convs_0_conv1_weight, 64 * 1*4,            # torch.Size([64, 1, 1, 64])
        neck_lateral_convs_0_quant1_threshold, 64 * 4   # torch.Size([64, 14])
    )
    send_params(
        neck_lateral_convs_0_conv2_weight, 64 * 1,          # torch.Size([64, 1, 1, 9])
        neck_lateral_convs_0_relu2_threshold, 64 * 4    # torch.Size([64, 7])
    )

    # YuNet_Head stride8
    # YuNet_Head shared ConvDPUnit
    print("send_params: bbox_head_multi_level_share_convs_0_0_conv1_weight")
    send_params(
        bbox_head_multi_level_share_convs_0_0_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        bbox_head_multi_level_share_convs_0_0_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        bbox_head_multi_level_share_convs_0_0_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        bbox_head_multi_level_share_convs_0_0_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNet_Head stride16
    # YuNet_Head shared ConvDPUnit
    print("send_params: bbox_head_multi_level_share_convs_1_0_conv1_weight")
    send_params(
        bbox_head_multi_level_share_convs_1_0_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        bbox_head_multi_level_share_convs_1_0_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        bbox_head_multi_level_share_convs_1_0_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        bbox_head_multi_level_share_convs_1_0_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )
    # YuNet_Head stride32
    # YuNet_Head shared ConvDPUnit
    print("send_params: bbox_head_multi_level_share_convs_2_0_conv1_weight")
    send_params(
        bbox_head_multi_level_share_convs_2_0_conv1_weight, 64 * 1*4,           # torch.Size([64, 1, 1, 64])
        bbox_head_multi_level_share_convs_2_0_quant1_threshold, 64 * 4  # torch.Size([64, 14])
    )
    send_params(
        bbox_head_multi_level_share_convs_2_0_conv2_weight, 64 * 1,         # torch.Size([64, 1, 1, 9])
        bbox_head_multi_level_share_convs_2_0_relu2_threshold, 64 * 4   # torch.Size([64, 7])
    )

    # YuNet_Head cls ConvDPUnit
    # YuNet_Head stride8
    print("send_params: bbox_head_multi_level_cls_0_conv1_weight")
    send_params(
        bbox_head_multi_level_cls_0_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_cls_0_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_cls_0_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_cls_0_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )
    # YuNet_Head stride16
    print("send_params: bbox_head_multi_level_cls_1_conv1_weight")
    send_params(
        bbox_head_multi_level_cls_1_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_cls_1_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_cls_1_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_cls_1_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )
    # YuNet_Head stride32
    print("send_params: bbox_head_multi_level_cls_2_conv1_weight")
    send_params(
        bbox_head_multi_level_cls_2_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_cls_2_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_cls_2_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_cls_2_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )

    # YuNet_Head bbox ConvDPUnit
    # YuNet_Head stride8
    print("send_params: bbox_head_multi_level_bbox_0_conv1_weight")
    send_params(
        bbox_head_multi_level_bbox_0_conv1_weight, 4 * 1*4,             # torch.Size([4, 1, 1, 64])
        bbox_head_multi_level_bbox_0_quant1_threshold, 4 * 4    # torch.Size([4, 14])
    )
    send_params(
        bbox_head_multi_level_bbox_0_conv2_weight, 4 * 1,               # torch.Size([4, 1, 1, 9])
        bbox_head_multi_level_bbox_0_quant2_threshold, 4 * 4    # torch.Size([4, 14])
    )
    # YuNet_Head stride16
    print("send_params: bbox_head_multi_level_bbox_1_conv1_weight")
    send_params(
        bbox_head_multi_level_bbox_1_conv1_weight, 4 * 1*4,             # torch.Size([4, 1, 1, 64])
        bbox_head_multi_level_bbox_1_quant1_threshold, 4 * 4    # torch.Size([4, 14])
    )
    send_params(
        bbox_head_multi_level_bbox_1_conv2_weight, 4 * 1,               # torch.Size([4, 1, 1, 9])
        bbox_head_multi_level_bbox_1_quant2_threshold, 4 * 4    # torch.Size([4, 14])
    )
    # YuNet_Head stride32
    print("send_params: bbox_head_multi_level_bbox_2_conv1_weight")
    send_params(
        bbox_head_multi_level_bbox_2_conv1_weight, 4 * 1*4,             # torch.Size([4, 1, 1, 64])
        bbox_head_multi_level_bbox_2_quant1_threshold, 4 * 4    # torch.Size([4, 14])
    )
    send_params(
        bbox_head_multi_level_bbox_2_conv2_weight, 4 * 1,               # torch.Size([4, 1, 1, 9])
        bbox_head_multi_level_bbox_2_quant2_threshold, 4 * 4    # torch.Size([4, 14])
    )

    # YuNet_Head obj ConvDPUnit
    # YuNet_Head stride8
    print("send_params: bbox_head_multi_level_obj_0_conv1_weight")
    send_params(
        bbox_head_multi_level_obj_0_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_obj_0_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_obj_0_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_obj_0_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )
    # YuNet_Head stride16
    print("send_params: bbox_head_multi_level_obj_1_conv1_weight")
    send_params(
        bbox_head_multi_level_obj_1_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_obj_1_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_obj_1_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_obj_1_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )
    # YuNet_Head stride32
    print("send_params: bbox_head_multi_level_obj_2_conv1_weight")
    send_params(
        bbox_head_multi_level_obj_2_conv1_weight, 1 * 1*4,          # torch.Size([1, 1, 1, 64])
        bbox_head_multi_level_obj_2_quant1_threshold, 1 * 4     # torch.Size([1, 14])
    )
    send_params(
        bbox_head_multi_level_obj_2_conv2_weight, 1 * 1,            # torch.Size([1, 1, 1, 9])
        bbox_head_multi_level_obj_2_quant2_threshold, 1 * 4     # torch.Size([1, 14])
    )

    # YuNet_Head kps ConvDPUnit
    # YuNet_Head stride8
    print("send_params: bbox_head_multi_level_kps_0_conv1_weight")
    send_params(
        bbox_head_multi_level_kps_0_conv1_weight, 10 * 1*4,             # torch.Size([10, 1, 1, 64])
        bbox_head_multi_level_kps_0_quant1_threshold, 10 * 4    # torch.Size([10, 14])
    )
    send_params(
        bbox_head_multi_level_kps_0_conv2_weight, 10 * 1,               # torch.Size([10, 1, 1, 9])
        bbox_head_multi_level_kps_0_quant2_threshold, 10 * 4    # torch.Size([10, 14])
    )
    # YuNet_Head stride16
    print("send_params: bbox_head_multi_level_kps_1_conv1_weight")
    send_params(
        bbox_head_multi_level_kps_1_conv1_weight, 10 * 1*4,             # torch.Size([10, 1, 1, 64])
        bbox_head_multi_level_kps_1_quant1_threshold, 10 * 4    # torch.Size([10, 14])
    )
    send_params(
        bbox_head_multi_level_kps_1_conv2_weight, 10 * 1,               # torch.Size([10, 1, 1, 9])
        bbox_head_multi_level_kps_1_quant2_threshold, 10 * 4    # torch.Size([10, 14])
    )
    # YuNet_Head stride32
    print("send_params: bbox_head_multi_level_kps_2_conv1_weight")
    send_params(
        bbox_head_multi_level_kps_2_conv1_weight, 10 * 1*4,             # torch.Size([10, 1, 1, 64])
        bbox_head_multi_level_kps_2_quant1_threshold, 10 * 4    # torch.Size([10, 14])
    )
    send_params(
        bbox_head_multi_level_kps_2_conv2_weight, 10 * 1,               # torch.Size([10, 1, 1, 9])
        bbox_head_multi_level_kps_2_quant2_threshold, 10 * 4    # torch.Size([10, 14])
    )

    t1 = time.time()
    print("Elapsed time:", t1-t0)


def decode_bbox_kps(detects):
    x1 = detects[0]
    y1 = detects[1]
    x2 = detects[2]
    y2 = detects[3]
    score = (detects[4] << 8) | detects[5]
    kps = detects[6:16]  # [x0,y0,x1,y1,...,x4,y4]
    return [x1, y1, x2, y2, score], kps


def main():
    images, img = jpg_to_rgb('../data/largest_selfie_160x160.jpg')

    sender.register_map.CTRL.AP_START = 1

    yunet(images)
    size = recv_output(1)
    print('size=', size)
    bboxes = []
    kps_list = []
    for i in range(size):
        detects = recv_output(16)
        bbox, kps = decode_bbox_kps(detects)
        print('bbox=', bbox)
        print('kps=', kps)
        bboxes.append(bbox)
        kps_list.append(kps)

    while sender.register_map.CTRL.AP_DONE == 0:
        pass

    draw_bboxes(img, size, bboxes, kps_list, "output.jpg")


if __name__ == "__main__":
    main()
