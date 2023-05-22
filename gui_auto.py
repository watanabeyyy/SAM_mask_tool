"""MASK Predictor GUI using SAM"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import predictor_auto

#グローバル変数
mask = None #表示用マスク画像
sam_image = None #表示用SAM出力画像
drawing = False
mode = False

def draw(mask: np.ndarray, sam_image: np.ndarray, x: int, y: int, mode: int):
    color = sam_image[y,x]
    if mode == 1:#select
        mask[find_color_indices(sam_image, color)] = 255
    elif mode == 2:#unselect
        mask[find_color_indices(sam_image, color)] = 0
    elif mode == 4:#塗る
        cv2.circle(mask,(x,y),3,(255,255,255),-1)
    elif mode == 5:#消す
        cv2.circle(mask,(x,y),3,(0,0,0),-1)
    return mask, sam_image

# マウスイベントのコールバック関数
def callback(event, x, y, flags, param):
    global mask,sam_image,mode,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if(mode == 0):
            mode = 1
        elif(mode == 3):
            mode = 4
        mask, sam_image = draw(mask,sam_image,x,y, mode)
    elif event == cv2.EVENT_RBUTTONDOWN:#erase
        drawing = True
        if(mode == 0):
            mode = 2
        elif(mode == 3):
            mode = 5
        mask, sam_image = draw(mask,sam_image,x,y, mode)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            mask, sam_image = draw(mask,sam_image,x,y, mode)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mask, sam_image = draw(mask,sam_image,x,y, mode)
        if(mode < 3):
            mode = 0
        else:
            mode = 3
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        mask, sam_image = draw(mask,sam_image,x,y, mode)
        if(mode < 3):
            mode = 0
        else:
            mode = 3

def find_color_indices(image_array, color):
    indices = np.where(np.all(image_array == color, axis=-1))
    return indices

# ウィンドウを作成しマウスイベントを設定
cv2.namedWindow('sam_image')
cv2.namedWindow('mask')
cv2.namedWindow('image')
cv2.setMouseCallback('sam_image', callback)
cv2.setMouseCallback('mask', callback)
cv2.setMouseCallback('image', callback)

# メインループ
def main(impath: str, device: str) -> None:
    global mask, sam_image, mode

    # 画像の読み込み
    org_image = cv2.imread(impath)
    resize_mag = 800 / np.max(org_image.shape)#長辺のピクセル数を800に変更
    image = cv2.resize(org_image, (int(org_image.shape[1]*resize_mag), int(org_image.shape[0]*resize_mag)))
    #SAM推論用にRGBに変更
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #SAMマスク画像取得
    print("predicting...")
    mask_generator = predictor_auto.MaskGenerator(device)
    sam_image = mask_generator.pred(image)
    print("finish!")
    sam_tmp = np.copy(sam_image)
    #表示用にBGRに戻す
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #出力マスク画像
    mask = np.zeros_like(image, np.uint8)

    print("left click: select")
    print("right click: unselect")
    print("press s: save mask")
    print("press r: reset mask")
    print("press q: quit")

    while True:
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.imshow('sam_image', sam_image)

        key = -1
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            print("save")
            impath = Path(impath)
            maskpath = impath.parent.parent/ Path("mask") / (impath.stem + "_mask.jpg")
            cv2.imwrite(str(maskpath), mask)
        elif key & 0xFF == ord('r'):
            print("リセット")
            sam_image = np.copy(sam_tmp)
        elif key & 0xFF == ord('q'):
            print("終了")
            break
        elif key & 0xFF == ord('1'):
            print("選択モード")
            mode = 0
        elif key & 0xFF == ord('2'):
            print("塗り絵モード")
            mode = 3

    cv2.destroyAllWindows()

if __name__ == '__main__':
    impath = "img/test.png"
    device = "cuda"
    main(impath, device)