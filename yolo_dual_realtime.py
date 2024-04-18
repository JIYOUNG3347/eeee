from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pytransform3d.transform_manager import TransformManager
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
import time

def calculate_plane_angle(points_x,points_z):
    points = np.column_stack((points_x, points_z))

    """
    주어진 3D 점들의 평면 각도를 계산합니다.

    :param points: (N, 2) 크기의 배열. x, z 평면에 대한 점들.
    :return: 평면의 각도 (도).
    """
    # PCA 수행
    pca = PCA(n_components=2)
    pca.fit(points)

    # PCA 주성분
    principal_components = pca.components_

    # 평면의 법선 벡터
    normal_vector = principal_components[0]

    # 평면의 각도 계산 (법선 벡터와 y 축 사이의 각도)
    angle_rad = np.arccos(np.dot(normal_vector, [0, 1]) / (np.linalg.norm(normal_vector) * np.linalg.norm([0, 1])))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def plot_rotated_line_and_circle(ax, x, y, z, theta, diameter=0.07, color='blue'):
    # Starting point coordinates
    start_x = x
    start_y = y
    start_z = z

    # Plot the circle
    radius = diameter / 2  # Radius of the circle
    u = np.linspace(0, 2 * np.pi, 100)  # Parametric variable for the circle
    x_circle = start_x + radius * np.cos(u) * np.cos(np.radians(theta))  # Adjust x-coordinate based on theta
    y_circle = start_y + radius * np.sin(u)  # Y-coordinate remains the same
    z_circle = start_z + radius * np.cos(u) * np.sin(np.radians(theta))  # Adjust z-coordinate based on theta
    ax.plot(x_circle, y_circle, z_circle, color=color, linestyle='dotted')

    # Calculating diameter line points using the min and max values from x_circle and z_circle
    min_x = np.min(x_circle)
    max_x = np.max(x_circle)
    min_z = np.min(z_circle)
    max_z = np.max(z_circle)

    # Plotting the diameter
    ax.plot([min_x, max_x], [start_y, start_y], [min_z, max_z], color='green', linestyle='-', label='Diameter')


def degrees_to_rotation_matrix(theta_degrees):
    theta_radians = np.radians(theta_degrees)  # 도 단위를 라디안으로 변환
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)

    # y축 주위의 회전행렬 생성
    Ry = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # x축 주위의 90도 회전행렬
    Rx_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # 두 회전 행렬의 곱 (먼저 y축 회전, 그 다음 x축 회전)
    rotation_matrix = Rx_90 @ Ry

    return rotation_matrix

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

# Stereo Matching Q is ginven
# def calculateGraspingPose(left_image, right_image):
#     # 스테레오 매칭 설정
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=16*15,  # 수정 가능
#         blockSize=5,
#         P1=8 * 3 * 5**2,
#         P2=32 * 3 * 5**2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=0,
#         speckleRange=2,
#         preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#
#     # 변위 맵 계산
#     disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
#
#     # 3D 좌표 변환
#     points_3D = cv2.reprojectImageTo3D(disparity, Q)
#
#     return points_3D




def calculateGraspingPose(left_image, right_image):

    M_left = cv2.getRotationMatrix2D((320, 240), 45, 1.0)  # +45도 회전
    M_right = cv2.getRotationMatrix2D((320, 240), -45, 1.0)  # -45도 회전

    rotated_left_image = cv2.warpAffine(left_image, M_left, (640, 480))
    rotated_right_image = cv2.warpAffine(right_image, M_right, (640, 480))

    # 스테레오 매칭 파라미터 설정
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*15,  # 수정 가능
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 변위 맵 계산
    disparity = stereo.compute(rotated_left_image, rotated_right_image).astype(np.float32) / 16.0

    # 초점 거리 및 베이스라인 설정 (미터 단위로 전환)
    focal_length_mm = 1.93  # 초점 거리 (mm)
    sensor_width_mm = 5.7  # 센서 너비 (mm)
    image_width_px = 640   # 이미지 가로 크기 (픽셀)
    baseline = 0.8  # 베이스라인 (미터)
    cx = image_width_px / 2
    cy = 480 / 2

    # 초점 거리를 픽셀 단위로 환산
    focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px

    # 변환 행렬 Q 정의, X 축이 깊이
    Q = np.float32([
        [0, 0, -1 / baseline, 0],  # X-axis as depth
        [0, 1, 0, -cy],            # Y-axis (unchanged)
        [1, 0, 0, -cx],            # Z-axis swapped with X
        [0, 0, 0, focal_length_px]  # Disparity to depth conversion
    ])

    # 3D로 재투영
    points = cv2.reprojectImageTo3D(disparity, Q)

    # Masks for different visibility
    left_mask = (rotated_left_image > 0)
    right_mask = (rotated_right_image > 0)
    overlap_mask = left_mask & right_mask

    # Overlap points
    overlap_points = points[overlap_mask]

    x_coords = overlap_points[:, 0]
    y_coords = overlap_points[:, 1]
    z_coords = overlap_points[:, 2]

    # 중간에 위치하는 값 찾기
    mid_x = np.median(x_coords)
    mid_y = np.median(y_coords)
    mid_z = np.median(z_coords)

    rotationMatrix = degrees_to_rotation_matrix(180)

    return np.array([mid_x, mid_z , 0.66 - mid_y]), rotationMatrix


def update_plot(ax, center, rotationMatrix):

    ax.clear()
    # Transformation matrices
    B2A = pytr.transform_from(pyrot.matrix_from_axis_angle([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    C2B = pytr.transform_from(rotationMatrix, center)

    # Plot transformations
    pytr.plot_transform(ax, A2B=B2A, s=0.5)
    pytr.plot_transform(ax, A2B=C2B, s=0.5)

    # Coordinate axes settings
    ax.set_xlabel('X (depth)', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_title('3D Point Cloud of Cup in Meters', fontsize=16)
    ax.scatter(center[0], center[1], center[2], c='b', s=100, label='Cameras')

    ax.scatter([0.15, 0.15], [-0.66, 0.66], [0.66, 0.67], c='r', s=100, label='Cameras', marker='^')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    plt.draw()
    plt.pause(0.001)


if __name__=="__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Define the classes for 'cup' and 'wine glass'
    classes = [40, 41]  # Class IDs in YOLO for 'wine glass' and 'cup'

    # Load the YOLO model
    model = YOLO('yolov8n-seg.pt')
    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(10)

    while cap1.isOpened() and cap2.isOpened():
        # Read a frame from the video
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        if success1 and success2:
            # # Convert images to numpy array for masking
            left_image_np = np.array(frame1)
            right_image_np = np.array(frame2)

            # # Perform inference
            left_results = model(frame1, classes=classes, conf = 0.6)
            right_results = model(frame2, classes=classes, conf = 0.6)

            left_image = save_segmentation_masks(left_results, left_image_np)
            right_image = save_segmentation_masks(right_results, right_image_np)

            center, rotationMatrix = calculateGraspingPose(left_image, right_image)

            update_plot(ax, center, rotationMatrix)


            left_annotated_frame = left_results[0].plot()
            right_annotated_frame = right_results[0].plot()

            cv2.imshow("Left Inference", left_annotated_frame)
            cv2.imshow("Right Inference", right_annotated_frame)


            # out.write(left_annotated_frame)
            # out2.write(right_annotated_frame)

            # Break the loop if 'esc' is pressed
            if (cv2.waitKey(1) & 0xFF == 27):
                break
        else:
            # Break the loop if the end of the video is reached
            break



