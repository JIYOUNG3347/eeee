from ultralytics import YOLO
import numpy as np
import cv2


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

def drawImg(rgb_image, mask_image, center):
    mask_image = np.repeat(mask_image[..., np.newaxis], 3, -1)
    cx, cy = center
    if cx != 0 and cy != 0:
        cv2.circle(rgb_image, (cx, cy), 5, (255, 0, 255), -1)
        cv2.putText(rgb_image, text=f"centroid at {cx}, {cy}", org=(cx, cy),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(255, 0, 255))
        cv2.circle(mask_image, (cx, cy), 5, (255, 0, 255), -1)
        cv2.putText(mask_image, text=f"centroid at {cx}, {cy}", org=(cx, cy),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(255, 0, 255))
    return np.hstack([rgb_image, mask_image])



def detect_centroid(mask_image):
    centroids = (0, 0)
    if np.any(mask_image):
        contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids = (cx, cy)

    return centroids

# 이미지 모멘트 없이 중심 찾기 및 그리기
def compute_centroid_via_pixels_and_draw(mask_image):
    centroid_x = 0
    centroid_y = 0
    # 마스크 이미지가 전부 0인지 확인
    if np.any(mask_image):
        Y, X = np.nonzero(mask_image)
        if len(X) == 0 or len(Y) == 0:
            return None
        centroid_x = int(np.mean(X))
        centroid_y = int(np.mean(Y))
    return (centroid_x, centroid_y)


# 바이너리 이미지의 바운딩 박스를 사용하여 중심 찾기 및 그리기
def compute_centroid_via_bbox_and_draw(mask_image):
    centroids = (0, 0)
    if np.any(mask_image):
        ret, thresh = cv2.threshold(mask_image, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            centroids = (cx, cy)
    return centroids


# OpenCV 함수 cv2.connectedComponentsWithStats() 사용하여 중심 찾기 및 그리기
def find_centroids_with_stats_and_draw(mask_image):
    centroids = [0, 0, 0]
    center = (0, 0)
    if np.any(mask_image):
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_image, connectivity=8, ltype=cv2.CV_32S)
        for i in range(1, num_labels):  # label 0 is the background
            center = (int(centroids[i][0]), int(centroids[i][1]))
    return center


if __name__ == "__main__":
    # Define the classes for 'cup' and 'wine glass'
    classes = [40, 41]  # Class IDs in YOLO for 'wine glass' and 'cup'

    # Load the YOLO model
    model = YOLO('yolov8x-seg.pt')
    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(4)

    while cap1.isOpened():
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

            left_center1 = detect_centroid(left_mask)
            right_center1 = detect_centroid(right_mask)
            left1 = drawImg(left_annotated_frame, left_mask, left_center1)
            right1 = drawImg(right_annotated_frame, right_mask, right_center1)

            left_center2 = compute_centroid_via_pixels_and_draw(left_mask)
            right_center2 = compute_centroid_via_pixels_and_draw(right_mask)
            left2 = drawImg(left_annotated_frame, left_mask, left_center2)
            right2 = drawImg(right_annotated_frame, right_mask, right_center2)

            left_center3 = compute_centroid_via_bbox_and_draw(left_mask)
            right_center3 = compute_centroid_via_bbox_and_draw(right_mask)
            left3 = drawImg(left_annotated_frame, left_mask, left_center3)
            right3 = drawImg(right_annotated_frame, right_mask, right_center3)


            left_center4 = find_centroids_with_stats_and_draw(left_mask)
            right_center4 = find_centroids_with_stats_and_draw(right_mask)
            left4 = drawImg(left_annotated_frame, left_mask, left_center4)
            right4 = drawImg(right_annotated_frame, right_mask, right_center4)

            left = np.vstack([left1, left2, left3, left4])
            right = np.vstack([right1, right2, right3, right4])

            left_resize = cv2.resize(left, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            right_resize = cv2.resize(right, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.imshow("Left Inference", left_resize)
            cv2.imshow("Right Inference", right_resize)

            # save_video(cap1, "left_camera.mp4").write(left)
            # save_video(cap2, "right_camera.mp4").write(right)

            # Break the loop if 'esc' is pressed
            if (cv2.waitKey(1) & 0xFF == 27):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cv2.destroyAllWindows()