"""
Real-time Traffic Sign Recognition
===================================
Nhận dạng biển báo giao thông real-time từ webcam.
Sử dụng OpenCV để capture video và detector để phát hiện biển báo.

Sử dụng:
    python -m src.detection.realtime
    python -m src.detection.realtime --model models/best_model.pth --camera 0
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import TrafficSignDetector, _put_unicode_text, _put_unicode_text_pil, _get_unicode_font
from PIL import Image, ImageDraw


class RealtimeRecognizer:
    """
    Nhận dạng biển báo giao thông real-time từ webcam.

    Args:
        model_path: Đường dẫn model
        model_type: Loại model
        camera_id: ID camera (mặc định 0)
        confidence_threshold: Ngưỡng confidence
    """

    def __init__(self, model_path="models/best_model.pth",
                 model_type="resnet18", camera_id=0,
                 confidence_threshold=0.7, frame_skip=2):
        self.camera_id = camera_id
        self.frame_skip = frame_skip  # Only run detection every N frames
        self.detector = TrafficSignDetector(
            model_path=model_path,
            model_type=model_type,
            confidence_threshold=confidence_threshold,
        )

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Cache last detections (reuse between skipped frames)
        self.last_detections = []

        # Detection history (để smooth kết quả)
        self.detection_history = []
        self.history_size = 5

    def run(self, display_width=1280, display_height=720):
        """
        Chạy nhận dạng real-time.

        Args:
            display_width: Chiều rộng cửa sổ
            display_height: Chiều cao cửa sổ
        """
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print("❌ Không thể mở camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize capture buffer lag

        print("=" * 50)
        print("🚦 REAL-TIME TRAFFIC SIGN RECOGNITION")
        print("=" * 50)
        print(f"📷 Camera: {self.camera_id}")
        print(f"🖥️ Resolution: {display_width}x{display_height}")
        print(f"⏩ Frame skip: {self.frame_skip} (detect every {self.frame_skip} frames)")
        print(f"⌨️ Nhấn 'q' để thoát | 's' để chụp ảnh | 'f' để lật camera")
        print("=" * 50)

        screenshot_count = 0
        flip_horizontal = True  # Lật ngang cho camera trước (mặc định bật)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame từ camera.")
                break

            # Lật ngang cho camera trước (fix ảnh bị ngược)
            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Only run detection every N frames (reuse last result otherwise)
            frame_idx += 1
            if frame_idx % self.frame_skip == 0:
                detections = self.detector.detect(frame, downscale_for_detection=640)
                self.last_detections = detections
            else:
                detections = self.last_detections

            # Vẽ detections lên frame
            annotated = self.detector.draw_detections(frame, detections)

            # Cập nhật FPS
            self._update_fps()

            # Vẽ thông tin HUD (optimized)
            annotated = self._draw_hud_fast(annotated, detections)

            # Hiển thị
            cv2.imshow("Traffic Sign Recognition", annotated)

            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Đã thoát.")
                break
            elif key == ord("s"):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"📸 Đã lưu: {filename}")
            elif key == ord("f"):
                flip_horizontal = not flip_horizontal
                state = "BẬT" if flip_horizontal else "TẮT"
                print(f"🔄 Lật camera: {state}")

        cap.release()
        cv2.destroyAllWindows()

    def _update_fps(self):
        """Cập nhật FPS counter."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def _draw_hud_fast(self, frame, detections):
        """
        Vẽ HUD (Head-Up Display) lên frame — optimized version.
        Batches all Unicode text into ONE PIL conversion.

        Args:
            frame: numpy array ảnh
            detections: List kết quả detection

        Returns:
            Frame đã vẽ HUD
        """
        h, w = frame.shape[:2]

        # Panel nền bán trong suốt
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS + Detected — fast ASCII text via OpenCV (no PIL needed)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {len(detections)} sign(s)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Q: Quit | S: Screenshot | F: Flip", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Panel kết quả bên phải — batch PIL render for Unicode
        if detections:
            panel_h = min(len(detections) * 40 + 20, h - 20)
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (w - 400, 10), (w - 10, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

            # Single PIL session for all detection labels
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            for i, det in enumerate(detections):
                y_pos = 35 + i * 40
                text = f"{det['class_name']}"
                conf = f"{det['confidence']:.0%}"

                _put_unicode_text_pil(draw, text, (w - 390, y_pos - 12),
                                      font_size=14, color=(255, 255, 255))
                _put_unicode_text_pil(draw, conf, (w - 60, y_pos - 12),
                                      font_size=14, color=(0, 255, 0))

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame

    def process_video(self, video_path, output_path=None):
        """
        Xử lý video file (không phải real-time).

        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn video output (nếu muốn lưu)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return

        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"🎥 Video: {video_path}")
        print(f"   Resolution: {w}x{h}, FPS: {fps}, Frames: {total_frames}")

        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            detections = self.detector.detect(frame)
            annotated = self.detector.draw_detections(frame, detections)
            annotated = self._draw_hud(annotated, detections)

            if writer:
                writer.write(annotated)

            if frame_idx % 30 == 0:
                print(f"   Processed: {frame_idx}/{total_frames} frames")

        cap.release()
        if writer:
            writer.release()

        print(f"✅ Hoàn tất xử lý video! ({frame_idx} frames)")
        if output_path:
            print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Nhận dạng biển báo real-time")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Đường dẫn model")
    parser.add_argument("--model-type", type=str, default="resnet18",
                        choices=["custom_cnn", "resnet18", "resnet50"])
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--confidence", type=float, default=0.7, help="Ngưỡng confidence")
    parser.add_argument("--skip", type=int, default=2,
                        help="Frame skip: chỉ detect mỗi N frame (1=mọi frame, 2=cách 1, 3=cách 2...)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--video", type=str, default=None, help="Video file (thay vì camera)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    args = parser.parse_args()

    recognizer = RealtimeRecognizer(
        model_path=args.model,
        model_type=args.model_type,
        camera_id=args.camera,
        confidence_threshold=args.confidence,
        frame_skip=args.skip,
    )

    if args.video:
        recognizer.process_video(args.video, args.output)
    else:
        recognizer.run(display_width=args.width, display_height=args.height)


if __name__ == "__main__":
    main()
