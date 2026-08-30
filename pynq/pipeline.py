# -*- coding: utf-8 -*-

from pynq import Overlay
from pynq import MMIO
from pynq import allocate

import sys
import time
import numpy as np

from params import *

base = Overlay("./design_1.bit")
pattern_overlay = base.pattern_overlay_0
vdma = base.axi_vdma_0


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


def initial_fbuf ():
    base = np.zeros((48, 64, 1), dtype=np.uint8)
    for y in range(24):
        for x in range(32):
            base[y + 24][x][0] = 0x80
            base[y][x + 32][0] = 0xc0
    return np.tile(base, (15, 20, 3))


def video_initialize (vdma):
    # video width and height
    VWIDTH = 1280
    VHEIGHT =720
    
    # frame buffers
    fbuf0 = allocate(shape=(VHEIGHT, VWIDTH, 3), dtype=np.uint8)
    fbuf1 = allocate(shape=(VHEIGHT, VWIDTH, 3), dtype=np.uint8)
    fbuf2 = allocate(shape=(VHEIGHT, VWIDTH, 3), dtype=np.uint8)
    fbuf_base = initial_fbuf()
    fbuf0[:] = fbuf_base
    fbuf1[:] = fbuf_base
    fbuf2[:] = fbuf_base
    
    # initialize VDMA
    vdma = base.axi_vdma_0
    vdma.write(0x30, 0x8b) # pattern write
    vdma.write(0xac, fbuf0.device_address)
    vdma.write(0xb0, fbuf1.device_address)
    vdma.write(0xb4, fbuf2.device_address)
    vdma.write(0xa8, VWIDTH * 3)
    vdma.write(0xa4, VWIDTH * 3)
    vdma.write(0xa0, VHEIGHT)
    vdma.write(0x00, 0x8b) # video read
    vdma.write(0x5c, fbuf0.device_address)
    vdma.write(0x60, fbuf1.device_address)
    vdma.write(0x64, fbuf2.device_address)
    vdma.write(0x58, VWIDTH * 3)
    vdma.write(0x54, VWIDTH * 3)
    vdma.write(0x50, VHEIGHT)
    
    return fbuf0, fbuf1, fbuf2


def video_finalize (vdma, fbuf0, fbuf1, fbuf2):
    # stop DMA
    vdma.write(0x30, 0x8a)
    while ((vdma.read(0x34) & 0x1) == 0):
        pass
    vdma.write(0x00, 0x8a)
    while ((vdma.read(0x04) & 0x1) == 0):
        pass
    
    # delete frame buffers
    fbuf0.freebuffer()
    fbuf1.freebuffer()
    fbuf2.freebuffer()


def main():
    param_list = create_param_list()
    param_size = len(param_list)
    print('param_size=', param_size)

    params = allocate(shape=(param_size,), dtype=np.uint64)
    params[:] = np.array(param_list, dtype=np.uint64)
    params.flush()

    # result = allocate(shape=(1 + 16 * 32,), dtype=np.uint64)

    print(pattern_overlay.register_map)
    pattern_overlay.register_map.params_1.params = params.physical_address
    # pattern_overlay.register_map.result_1.result = result.physical_address

    fbuf0, fbuf1, fbuf2 = video_initialize(vdma)

    start_time = time.time()
    frame_processed = 0

    try:
        while True:
            pattern_overlay.register_map.CTRL.AP_START = 1
            while pattern_overlay.register_map.CTRL.AP_DONE == 0:
                pass
            
            frame_processed += 1
            elapsed = time.time() - start_time
            fps = frame_processed / elapsed

            # if frame_processed == 1:
            #     result.invalidate()
            #     count = result[0]
            #     print(f"count: {count}")
            #     offset = 1
            #     for i in range(count):
            #         coord = result[offset : offset + 16]
            #         print(f"({coord[0]}, {coord[1]}, {coord[2]}, {coord[3]})")
            #         offset += 16                    

            sys.stdout.write(f"\rFPS: {fps:.2f} {frame_processed}")
            sys.stdout.flush()
            
            if elapsed > 60:
                break
    finally:
        video_finalize(vdma, fbuf0, fbuf1, fbuf2)
        print()


if __name__ == "__main__":
    main()
