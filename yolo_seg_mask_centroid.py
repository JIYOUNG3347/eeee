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

def save_video(cap, name):

    # 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

    # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # 1프레임과 다음 프레임 사이의 간격 설정
    out = cv2.VideoWriter(name, fourcc, fps, (w, h))

    return out


def make_segmentation_masks(img_results, image_np):
    mask_image = np.zeros(image_np.shape[:2], dtype=np.uint8)

    for result in img_results:
        if result.masks is None or result.boxes is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            polygon = np.array(mask, dtype=np.int32)
            cv2.fillPoly(mask_image, [polygon], 255)

    return mask_image


# 컨투어에 대한 중심 찾기 및 그리기
def detect_centroid(rgb_image, mask_image, drawImg):
    centroids = []
    ret, binary = cv2.threshold(mask_image, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[0])
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))
        if drawImg:
            cv2.drawContours(rgb_image, contours[0], -1, (0, 255, 0), 3)
            cv2.circle(rgb_image, (cx, cy), 3, (255, 0, 255), 3)
            cv2.putText(rgb_image, text=f"centroid at {cx}, {cy}", org=(cx, cy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, color=(255, 0, 255))
    return np.hstack([rgb_image, np.repeat(mask_image[..., np.newaxis], 3, -1)]), centroids


# 이미지 모멘트 없이 중심 찾기 및 그리기
def compute_centroid_via_pixels_and_draw(rgb_image, mask_image, drawImg=True):
    Y, X = np.nonzero(mask_image)
    if len(X) == 0 or len(Y) == 0:
        return None
    centroid_x = int(np.mean(X))
    centroid_y = int(np.mean(Y))
    if drawImg:
        cv2.circle(rgb_image, (centroid_x, centroid_y), 5, (255, 0, 255), -1)
        cv2.putText(rgb_image, f"({centroid_x},{centroid_y})", (centroid_x, centroid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return np.hstack([rgb_image, np.repeat(mask_image[..., np.newaxis], 3, -1)]), (centroid_x, centroid_y)

# 바이너리 이미지의 바운딩 박스를 사용하여 중심 찾기 및 그리기
def compute_centroid_via_bbox_and_draw(rgb_image, mask_image, drawImg=True):
    ret, thresh = cv2.threshold(mask_image, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2
        centroids.append((cx, cy))
        if drawImg:
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(rgb_image, (cx, cy), 5, (255, 0, 255), -1)
            cv2.putText(rgb_image, f"({cx},{cy})", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return np.hstack([rgb_image, np.repeat(mask_image[..., np.newaxis], 3, -1)]), centroids


# OpenCV 함수 cv2.connectedComponentsWithStats() 사용하여 중심 찾기 및 그리기
def find_centroids_with_stats_and_draw(rgb_image, mask_image, drawImg=True):
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_image, connectivity=8, ltype=cv2.CV_32S)
    for i in range(1, num_labels):  # label 0 is the background
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        if drawImg:
            cv2.circle(rgb_image, (cx, cy), 5, (255, 0, 255), -1)
            cv2.putText(rgb_image, f"({cx},{cy})", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return np.hstack([rgb_image, np.repeat(mask_image[..., np.newaxis], 3, -1)]), centroids[1:]


if __name__ == "__main__":
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

            left_mask = make_segmentation_masks(left_results, left_image_np)
            right_mask = make_segmentation_masks(right_results, right_image_np)

            left_annotated_frame = left_results[0].plot()
            right_annotated_frame = right_results[0].plot()

            left, left_center = detect_centroid(left_image_np, left_mask, True)
            right, right_center = detect_centroid(right_image_np, right_mask, True)

            cv2.imshow("Left Inference", left)
            cv2.imshow("Right Inference", right)

            # save_video(cap1, "left_camera.mp4").write(left)
            # save_video(cap2, "right_camera.mp4").write(right)

            # Break the loop if 'esc' is pressed
            if (cv2.waitKey(1) & 0xFF == 27):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cv2.destroyAllWindows()
