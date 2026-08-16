# -*- coding: utf-8 -*-

from pynq import Overlay
from pynq import MMIO
from pynq import allocate # 利用可能な領域を確保するメソッドを利用可能にする

import numpy as np
from PIL import Image, ImageDraw
import time
import random

from params import *

base = Overlay("./design_1.bit")
overlay = base.pattern_overlay_0
dma = base.axi_dma_0

t0 = time.time()

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


def create_param_list():
    names = [
    # YuNetBackbone stage0
    # Conv_head
        'backbone_model0_conv1_weight',
        'backbone_model0_relu1_threshold',
    # Conv_head ConvDPUnit
        'backbone_model0_conv2_conv1_weight',
        'backbone_model0_conv2_quant1_threshold',
        'backbone_model0_conv2_conv2_weight',
        'backbone_model0_conv2_relu2_threshold',

    # YuNetBackbone stage1
    # YuNetBackbone Conv4layerBlock 1
        'backbone_model1_conv1_conv1_weight',
        'backbone_model1_conv1_quant1_threshold',
        'backbone_model1_conv1_conv2_weight',
        'backbone_model1_conv1_relu2_threshold',
    # YuNetBackbone Conv4layerBlock 2
        'backbone_model1_conv2_conv1_weight',
        'backbone_model1_conv2_quant1_threshold',
        'backbone_model1_conv2_conv2_weight',
        'backbone_model1_conv2_relu2_threshold',

    # YuNetBackbone stage2
    # YuNetBackbone Conv4layerBlock 1
        'backbone_model2_conv1_conv1_weight',
        'backbone_model2_conv1_quant1_threshold',
        'backbone_model2_conv1_conv2_weight',
        'backbone_model2_conv1_relu2_threshold',
    # YuNetBackbone Conv4layerBlock 2
        'backbone_model2_conv2_conv1_weight',
        'backbone_model2_conv2_quant1_threshold',
        'backbone_model2_conv2_conv2_weight',
        'backbone_model2_conv2_relu2_threshold',

    # YuNetBackbone stage3
    # YuNetBackbone Conv4layerBlock 1
        'backbone_model3_conv1_conv1_weight',
        'backbone_model3_conv1_quant1_threshold',
        'backbone_model3_conv1_conv2_weight',
        'backbone_model3_conv1_relu2_threshold',
    # YuNetBackbone Conv4layerBlock 2
        'backbone_model3_conv2_conv1_weight',
        'backbone_model3_conv2_quant1_threshold',
        'backbone_model3_conv2_conv2_weight',
        'backbone_model3_conv2_relu2_threshold',

    # YuNetBackbone stage4
    # YuNetBackbone Conv4layerBlock 1
        'backbone_model4_conv1_conv1_weight',
        'backbone_model4_conv1_quant1_threshold',
        'backbone_model4_conv1_conv2_weight',
        'backbone_model4_conv1_relu2_threshold',
    # YuNetBackbone Conv4layerBlock 2
        'backbone_model4_conv2_conv1_weight',
        'backbone_model4_conv2_quant1_threshold',
        'backbone_model4_conv2_conv2_weight',
        'backbone_model4_conv2_relu2_threshold',

    # YuNetBackbone stage5
    # YuNetBackbone Conv4layerBlock 1
        'backbone_model5_conv1_conv1_weight',
        'backbone_model5_conv1_quant1_threshold',
        'backbone_model5_conv1_conv2_weight',
        'backbone_model5_conv1_relu2_threshold',
    # YuNetBackbone Conv4layerBlock 2
        'backbone_model5_conv2_conv1_weight',
        'backbone_model5_conv2_quant1_threshold',
        'backbone_model5_conv2_conv2_weight',
        'backbone_model5_conv2_relu2_threshold',

    # TFPN stride32
    # TFPN ConvDPUnit
        'neck_lateral_convs_2_conv1_weight',
        'neck_lateral_convs_2_quant1_threshold',
        'neck_lateral_convs_2_conv2_weight',
        'neck_lateral_convs_2_relu2_threshold',
    # TFPN stride16
    # TFPN ConvDPUnit
        'neck_lateral_convs_1_conv1_weight',
        'neck_lateral_convs_1_quant1_threshold',
        'neck_lateral_convs_1_conv2_weight',
        'neck_lateral_convs_1_relu2_threshold',
    # TFPN stride8
    # TFPN ConvDPUnit
        'neck_lateral_convs_0_conv1_weight',
        'neck_lateral_convs_0_quant1_threshold',
        'neck_lateral_convs_0_conv2_weight',
        'neck_lateral_convs_0_relu2_threshold',

    # YuNet_Head stride8
    # YuNet_Head shared ConvDPUnit
        'bbox_head_multi_level_share_convs_0_0_conv1_weight',
        'bbox_head_multi_level_share_convs_0_0_quant1_threshold',
        'bbox_head_multi_level_share_convs_0_0_conv2_weight',
        'bbox_head_multi_level_share_convs_0_0_relu2_threshold',
    # YuNet_Head stride16
    # YuNet_Head shared ConvDPUnit
        'bbox_head_multi_level_share_convs_1_0_conv1_weight',
        'bbox_head_multi_level_share_convs_1_0_quant1_threshold',
        'bbox_head_multi_level_share_convs_1_0_conv2_weight',
        'bbox_head_multi_level_share_convs_1_0_relu2_threshold',
    # YuNet_Head stride32
    # YuNet_Head shared ConvDPUnit
        'bbox_head_multi_level_share_convs_2_0_conv1_weight',
        'bbox_head_multi_level_share_convs_2_0_quant1_threshold',
        'bbox_head_multi_level_share_convs_2_0_conv2_weight',
        'bbox_head_multi_level_share_convs_2_0_relu2_threshold',

    # YuNet_Head cls ConvDPUnit
    # YuNet_Head stride8
        'bbox_head_multi_level_cls_0_conv1_weight',
        'bbox_head_multi_level_cls_0_quant1_threshold',
        'bbox_head_multi_level_cls_0_conv2_weight',
        'bbox_head_multi_level_cls_0_quant2_threshold',
    # YuNet_Head stride16
        'bbox_head_multi_level_cls_1_conv1_weight',
        'bbox_head_multi_level_cls_1_quant1_threshold',
        'bbox_head_multi_level_cls_1_conv2_weight',
        'bbox_head_multi_level_cls_1_quant2_threshold',
    # YuNet_Head stride32
        'bbox_head_multi_level_cls_2_conv1_weight',
        'bbox_head_multi_level_cls_2_quant1_threshold',
        'bbox_head_multi_level_cls_2_conv2_weight',
        'bbox_head_multi_level_cls_2_quant2_threshold',

    # YuNet_Head bbox ConvDPUnit
    # YuNet_Head stride8
        'bbox_head_multi_level_bbox_0_conv1_weight',
        'bbox_head_multi_level_bbox_0_quant1_threshold',
        'bbox_head_multi_level_bbox_0_conv2_weight',
        'bbox_head_multi_level_bbox_0_quant2_threshold',
    # YuNet_Head stride16
        'bbox_head_multi_level_bbox_1_conv1_weight',
        'bbox_head_multi_level_bbox_1_quant1_threshold',
        'bbox_head_multi_level_bbox_1_conv2_weight',
        'bbox_head_multi_level_bbox_1_quant2_threshold',
    # YuNet_Head stride32
        'bbox_head_multi_level_bbox_2_conv1_weight',
        'bbox_head_multi_level_bbox_2_quant1_threshold',
        'bbox_head_multi_level_bbox_2_conv2_weight',
        'bbox_head_multi_level_bbox_2_quant2_threshold',

    # YuNet_Head obj ConvDPUnit
    # YuNet_Head stride8
        'bbox_head_multi_level_obj_0_conv1_weight',
        'bbox_head_multi_level_obj_0_quant1_threshold',
        'bbox_head_multi_level_obj_0_conv2_weight',
        'bbox_head_multi_level_obj_0_quant2_threshold',
    # YuNet_Head stride16
        'bbox_head_multi_level_obj_1_conv1_weight',
        'bbox_head_multi_level_obj_1_quant1_threshold',
        'bbox_head_multi_level_obj_1_conv2_weight',
        'bbox_head_multi_level_obj_1_quant2_threshold',
    # YuNet_Head stride32
        'bbox_head_multi_level_obj_2_conv1_weight',
        'bbox_head_multi_level_obj_2_quant1_threshold',
        'bbox_head_multi_level_obj_2_conv2_weight',
        'bbox_head_multi_level_obj_2_quant2_threshold',

    # YuNet_Head kps ConvDPUnit
    # YuNet_Head stride8
        'bbox_head_multi_level_kps_0_conv1_weight',
        'bbox_head_multi_level_kps_0_quant1_threshold',
        'bbox_head_multi_level_kps_0_conv2_weight',
        'bbox_head_multi_level_kps_0_quant2_threshold',
    # YuNet_Head stride16
        'bbox_head_multi_level_kps_1_conv1_weight',
        'bbox_head_multi_level_kps_1_quant1_threshold',
        'bbox_head_multi_level_kps_1_conv2_weight',
        'bbox_head_multi_level_kps_1_quant2_threshold',
    # YuNet_Head stride32
        'bbox_head_multi_level_kps_2_conv1_weight',
        'bbox_head_multi_level_kps_2_quant1_threshold',
        'bbox_head_multi_level_kps_2_conv2_weight',
        'bbox_head_multi_level_kps_2_quant2_threshold',
    ]

    params = []
    for name in names:
        params.extend(globals()[name])

    return np.array(params, dtype=np.uint64)


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

    param_list = create_param_list()
    param_size = len(param_list)

    params = pynq.allocate(shape=(param_size,), dtype=np.uint64)
    params[:] = np.array(param_list, dtype=np.uint64)

    overlay.params = params.physical_address
    overlay.params_size = param_size
    overlay.register_map.CTRL.AP_START = 1

    print('yunet(images)')
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

    while overlay.register_map.CTRL.AP_DONE == 0:
        pass

    draw_bboxes(img, size, bboxes, kps_list, "output.jpg")


if __name__ == "__main__":
    main()
