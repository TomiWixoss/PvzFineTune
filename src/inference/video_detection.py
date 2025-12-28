# -*- coding: utf-8 -*-
"""
Detect objects in video and output video with bounding boxes
"""

import cv2
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_MODEL_PATH, DEFAULT_CONFIDENCE
from inference.yolo_detector import YOLODetector


def detect_video(input_path: str, output_path: str = None, 
                 model_path: str = None,
                 conf: float = None, show: bool = False):
    """
    Run YOLO detection on video and save output with bounding boxes
    
    Args:
        input_path: Input video path
        output_path: Output video path (default: input_detected.mp4)
        model_path: YOLO model path
        conf: Confidence threshold
        show: Show video while processing
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"✗ Video not found: {input_path}")
        return
    
    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_detected.mp4"
    
    # Load model
    detector = YOLODetector(model_path or YOLO_MODEL_PATH)
    if not detector.load():
        return
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"✗ Cannot open video: {input_path}")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {input_path.name}")
    print(f"  Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
    print(f"Output: {output_path}")
    print(f"Processing...")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame, conf or DEFAULT_CONFIDENCE)
        
        # Draw
        frame = detector.draw_detections(frame, detections)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | Objects: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Show if requested
        if show:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Done! Output saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect objects in video')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('-m', '--model', default=None, help='Model path')
    parser.add_argument('-c', '--conf', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    
    args = parser.parse_args()
    
    detect_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        show=args.show
    )


if __name__ == "__main__":
    main()
