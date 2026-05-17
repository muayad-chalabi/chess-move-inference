import math
import os
import time

import cv2
from ultralytics import YOLOE


def detect_chess_pieces(
    model_to_use: str,
    model_path: str,
    video_path: str,
    classes_to_detect: list,
    max_frames_to_process: int = None,
):
    """
    General entry point to detect objects in a video using either YOLOe or SAM3.

    Args:
        model_to_use (str): Which model to use: "yoloe" or "sam3".
        model_path (str): Path to model weights (e.g. "yoloe-26l-seg.pt" or "sam3.pt").
        video_path (str): Path to the input video file.
        classes_to_detect (list): List of text prompts / class names to detect.
        max_frames_to_process (int|None): If set to a positive integer, the
            function will subsample frames so that no more than this number
            of frames are processed (inferenced). If None or <= 0, all frames
            are processed (subject to each model's internal behavior).
    """
    model_name = (model_to_use or "").lower()
    print(f"--- Starting Chess Piece Detector (model: {model_name}) ---")

    if model_name in ("yoloe", "yolo"):
        # ---------------- YOLOe workflow ----------------
        try:
            model = YOLOE(model_path)
        except Exception as e:
            print(f"Error initializing YOLOE model at '{model_path}': {e}")
            return

        print("Setting text prompts for YOLOe detection...")
        try:
            # Provide textual class prompts to YOLOE
            if classes_to_detect:
                model.set_classes(classes_to_detect)
        except Exception as e:
            print(f"Error setting classes on YOLOE model: {e}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'.")
            return

        # Try to get total frame count from metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            # Fallback: scan through the video to count frames (slower)
            print(
                "Warning: could not read total frame count from video file. Scanning to determine total frames..."
            )
            scanned = 0
            while True:
                ret_count, _ = cap.read()
                if not ret_count:
                    break
                scanned += 1
            total_frames = scanned
            # Reset to the beginning of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(f"Detected total frames by scanning: {total_frames}")

        # Compute subsampling stride so we never process more than max_frames_to_process
        skip = 1
        if (
            max_frames_to_process
            and max_frames_to_process > 0
            and total_frames > max_frames_to_process
        ):
            skip = math.ceil(total_frames / float(max_frames_to_process))
            print(
                f"Video has {total_frames} frames. Will process 1 every {skip} frames to limit processing to <= {max_frames_to_process} frames."
            )
        else:
            if max_frames_to_process and max_frames_to_process > 0:
                print(
                    f"Video has {total_frames} frames which is <= max_frames_to_process ({max_frames_to_process}). Processing all frames."
                )
            else:
                print("No maximum frames limit set; processing all frames.")

        total_read_frames = 0
        processed_frames = 0
        start_time = time.time()

        print("\nProcessing video frames with YOLOe. Detection started...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            total_read_frames += 1

            # Skip frames according to computed stride
            if skip > 1 and ((total_read_frames - 1) % skip) != 0:
                # Keep GUI responsive while skipping
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            processed_frames += 1
            print(
                f"\n[Frame {total_read_frames}] -> Processing ({processed_frames} processed so far)..."
            )

            try:
                # Run detection on the current frame (numpy array format)
                results = model.predict(frame, conf=0.5, verbose=False)

                # --- Result Processing & Visualization ---
                if results and len(results) > 0:
                    try:
                        annotated_frame = results[0].plot()
                    except Exception:
                        annotated_frame = frame

                    cv2.imshow("YOLOe Chess Detector", annotated_frame)

                    # Safely attempt to count boxes
                    try:
                        boxes_count = len(results[0].boxes)
                    except Exception:
                        boxes_count = 0

                    print(
                        f"Detection successful. Found {boxes_count} potential objects."
                    )
                else:
                    cv2.imshow("YOLOe Chess Detector", frame)
                    print("No chess pieces detected in this frame.")

            except Exception as e:
                print(
                    f"An error occurred during YOLOe inference on Frame {total_read_frames}: {e}"
                )
                # Continue to the next frame even if one fails

            # Allow manual break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Safety check - stop if we've reached the configured maximum
            if max_frames_to_process and processed_frames >= max_frames_to_process:
                print(
                    f"Reached the configured maximum of {max_frames_to_process} processed frames. Stopping early."
                )
                break

        # 4. Cleanup and stats
        end_time = time.time()
        elapsed = end_time - start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0
        print("\n--- Detection Finished (YOLOe) ---")
        print(f"Total frames read from video: {total_read_frames}")
        print(f"Total frames processed (inferenced): {processed_frames}")
        print(f"Processing time: {elapsed:.2f}s, Average processing FPS: {fps:.2f}")

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        return

    elif model_name in ("sam3", "sam"):
        # ---------------- SAM3 workflow ----------------
        # Note: SAM3's VideoSemanticPredictor provides a streaming API. We'll
        # iterate over its results and only "process" (show/save) a subset of
        # those frames according to the same subsampling strategy. Be aware
        # that depending on SAM3's internal implementation it may still run
        # inference on frames we choose to skip; skipping here ensures we do
        # not perform additional per-frame processing on the skipped frames.
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
        except Exception as e:
            print(f"Error importing SAM3VideoSemanticPredictor: {e}")
            return

        # Compute total frames using a lightweight cv2 probe (so we can decide skip)
        cap_probe = cv2.VideoCapture(video_path)
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            print(
                "Warning: could not read total frame count from video file. Scanning to determine total frames (this may be slow)..."
            )
            scanned = 0
            while True:
                ret_count, _ = cap_probe.read()
                if not ret_count:
                    break
                scanned += 1
            total_frames = scanned
            print(f"Detected total frames by scanning: {total_frames}")
        cap_probe.release()

        # Decide skip as before
        skip = 1
        if (
            max_frames_to_process
            and max_frames_to_process > 0
            and total_frames > max_frames_to_process
        ):
            skip = math.ceil(total_frames / float(max_frames_to_process))
            print(
                f"Video has {total_frames} frames. Will process 1 every {skip} frames to limit processing to <= {max_frames_to_process} frames."
            )
        else:
            if max_frames_to_process and max_frames_to_process > 0:
                print(
                    f"Video has {total_frames} frames which is <= max_frames_to_process ({max_frames_to_process}). Processing all frames."
                )
            else:
                print(
                    "No maximum frames limit set; will stream all frames from SAM3 predictor."
                )

        # Configure SAM3 predictor overrides; use provided model_path
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=640,
            model=model_path,
            half=True,
            save=False,
        )

        try:
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        except Exception as e:
            print(
                f"Error initializing SAM3VideoSemanticPredictor with model '{model_path}': {e}"
            )
            return

        # Start streaming results from SAM3
        processed_frames = 0
        total_streamed = 0
        start_time = time.time()

        print("\nStreaming results from SAM3 predictor...")
        try:
            results = predictor(source=video_path, text=classes_to_detect, stream=True)
            for r in results:
                total_streamed += 1

                # Skip according to computed stride (we still receive all streamed results)
                if skip > 1 and ((total_streamed - 1) % skip) != 0:
                    # Some result objects may hold resources; attempt to release if method exists
                    if hasattr(r, "close"):
                        try:
                            r.close()
                        except Exception:
                            pass
                    continue

                processed_frames += 1
                print(
                    f"\n[Streamed Frame {total_streamed}] -> Processing ({processed_frames} processed so far)..."
                )

                # Display / visualize the result. The SAM3 result object commonly
                # supports .show(); otherwise fall back to .plot() or .img if available.
                try:
                    if hasattr(r, "show"):
                        r.show()
                    elif hasattr(r, "plot"):
                        frame_img = r.plot()
                        cv2.imshow("SAM3 Chess Detector", frame_img)
                    elif hasattr(r, "img"):
                        cv2.imshow("SAM3 Chess Detector", r.img)
                    else:
                        print("Result object has no displayable image method.")
                except Exception as e:
                    print(
                        f"Error displaying SAM3 result for frame {total_streamed}: {e}"
                    )

                # Allow manual break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if max_frames_to_process and processed_frames >= max_frames_to_process:
                    print(
                        f"Reached the configured maximum of {max_frames_to_process} processed frames. Stopping early."
                    )
                    break

        except Exception as e:
            print(f"Error while streaming from SAM3 predictor: {e}")
        finally:
            # Attempt to clean up the predictor if it exposes a close method
            try:
                if hasattr(predictor, "close"):
                    predictor.close()
            except Exception:
                pass

        end_time = time.time()
        elapsed = end_time - start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0
        print("\n--- Detection Finished (SAM3) ---")
        print(f"Total frames streamed by predictor: {total_streamed}")
        print(f"Total frames processed (inferenced): {processed_frames}")
        print(f"Processing time: {elapsed:.2f}s, Average processing FPS: {fps:.2f}")

        cv2.destroyAllWindows()
        return

    else:
        print(
            f"Unknown model_to_use: {model_to_use}. Supported options: 'yoloe' or 'sam3'."
        )


if __name__ == "__main__":
    # ===============================================================
    # USER CONFIGURATION SECTION (*** EDIT THESE BEFORE RUNNING ***)
    # ===============================================================

    # Which model to use: set to 'yoloe' or 'sam3'
    MODEL_TO_USE = "sam3"

    # Path to your model weights (YOLOe or SAM3). You'll provide the correct file.
    MODEL_PATH = "D:\\.Personal projects\\chess-cv\\sam3.pt"

    # Path to the video input file
    VIDEO_PATH = "chessboards.mp4"

    # Text prompts for the pieces you want to detect. Examples: 'pawn', 'rook', ...
    CHESS_PIECES = [
        "pawn",
        "rook",
        "knight",
        "bishop",
        "queen",
        "king",
    ]
    # Add alternate phrasing to improve matching (optional)
    CHESS_PIECES = [f"{piece} chess piece" for piece in CHESS_PIECES]
    CHESS_PIECES = ["chess piece"] + CHESS_PIECES
    # Maximum number of frames to process. Set to a positive integer to limit processing
    # (script will subsample uniformly). Set to None or 0 to process every frame.
    MAX_FRAMES_TO_PROCESS = 500

    # ===============================================================

    detect_chess_pieces(
        MODEL_TO_USE, MODEL_PATH, VIDEO_PATH, CHESS_PIECES, MAX_FRAMES_TO_PROCESS
    )
