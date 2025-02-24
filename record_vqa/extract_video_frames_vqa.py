import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse


def extract_frames_uniform_blur(video_path, output_dir, window_size=1, blur_threshold=100):
    """
    从视频中均匀提取帧，每个时间窗口选择模糊度分数最高的帧

    参数:
    video_path: 输入视频路径
    output_dir: 输出目录路径
    window_size: 时间窗口大小（秒）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    # 计算每个窗口的帧数
    frames_per_window = int(fps * window_size)

    # 视频名称（不包含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    window_frames = []
    window_scores = []

    with tqdm(total=int(duration), desc="处理视频") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算模糊度分数
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            window_frames.append(frame)
            window_scores.append(blur_score)

            if len(window_frames) == frames_per_window:
                # 选择模糊度分数最高的帧
                best_frame_index = np.argmax(window_scores)
                best_frame = window_frames[best_frame_index]
                best_score = window_scores[best_frame_index]
                # 保存最佳帧
                if blur_threshold is None or best_score >= blur_threshold:
                    output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:06d}.jpg")
                    cv2.imwrite(output_path, best_frame)

                # 重置窗口
                window_frames = []
                window_scores = []

                # 更新进度条
                pbar.update(window_size)
                frame_count += 1

    # 处理最后一个不完整的窗口（如果有）
    if window_frames:
        best_frame_index = np.argmax(window_scores)
        best_frame = window_frames[best_frame_index]
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, best_frame)

    # 释放资源
    cap.release()

    print(f"帧提取完成。共提取 {len(os.listdir(output_dir))} 帧。")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("output_dir")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--blur_threshold", type=int, default=30)
    args = parser.parse_args()

    extract_frames_uniform_blur(args.video_path, args.output_dir, window_size=args.window_size, blur_threshold=args.blur_threshold)
