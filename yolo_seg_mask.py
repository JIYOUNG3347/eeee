import torch
import pyrealsense2 as rs
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Define the classes for 'cup' and 'wine glass'
classes = [40, 41]  # Class IDs in YOLO for 'wine glass' and 'cup'

# Load the YOLO model
model = YOLO('yolov8n-seg.pt')
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(10)

# # 정수 형태로 변환하기 위해 round
# w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap1.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적
#
# # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#
# # 1프레임과 다음 프레임 사이의 간격 설정
# out = cv2.VideoWriter('yolov8_detection.mp4', fourcc, fps, (w, h))

def COG(input_array):
    """
    :param input_array: 반드시 thresholding 된 array여야 함.
    :return:
    """
    global cx, cy
    contours, hierarchy = cv2.findContours(input_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        result_cx.append(cx)
        result_cy.append(cy)

    return result_cx, result_cy

def save_segmentation_masks(img_results, image_np):
    # 이미지 크기와 동일한 검은색 마스크 초기화
    mask_image = np.zeros(image_np.shape[:2], dtype=np.uint8)

    for result in img_results:
        # masks 또는 boxes가 None인 경우 continue
        if result.masks is None or result.boxes is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            polygon = np.array(mask, dtype=np.int32)

            # polygon 배열을 이용해 폴리곤을 그린 후 마스크 이미지를 생성
            cv2.fillPoly(mask_image, [polygon], 255)  # 255는 흰색, 폴리곤 내부를 채움

    return mask_image


while cap1.isOpened() and cap2.isOpened():
    # Read a frame from the video
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    if success1 and success2:
        # # Convert images to numpy array for masking
        left_image_np = np.array(frame1)
        right_image_np = np.array(frame2)

        # # Perform inference
        left_results = model(left_image_np, classes=classes, conf=0.6)
        right_results = model(right_image_np, classes=classes, conf=0.6)

        left_mask = save_segmentation_masks(left_results, left_image_np)
        right_mask = save_segmentation_masks(right_results, right_image_np)
        print(left_mask.shape)
        left_annotated_frame = left_results[0].plot()
        right_annotated_frame = right_results[0].plot()

        left = np.hstack([left_annotated_frame, np.repeat(left_mask[..., np.newaxis], 3, -1)])
        right = np.hstack([right_annotated_frame, np.repeat(right_mask[..., np.newaxis], 3, -1)])

        cv2.imshow("Left Inference", left)
        cv2.imshow("Right Inference", right)

        # out.write(left_annotated_frame)
        # out2.write(right_annotated_frame)

        # Break the loop if 'esc' is pressed
        if (cv2.waitKey(1) & 0xFF == 27):
            break
    else:
        # Break the loop if the end of the video is reached
        break