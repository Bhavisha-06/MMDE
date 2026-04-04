# zed_opencv_view.py
# Requires: ZED SDK + Stereolabs Python API (pyzed.sl), and OpenCV (cv2)
# Shows left and right images at the camera's native recording resolution.

import sys
import time
import numpy as np
import cv2
import pyzed.sl as sl

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    # Let the SDK pick camera resolution / fps by default so we match recording resolution.
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED:", err)
        zed.close()
        sys.exit(1)

    # Get camera info (optional) and print the native resolution the SDK reports.
    cam_info = zed.get_camera_information()
    res = cam_info.camera_configuration.resolution  # Resolution struct
    try:
        width = res.width
        height = res.height
    except Exception:
        # Fallback if attribute names differ
        width = int(getattr(res, 'width', 0) or getattr(res, 'w', 0))
        height = int(getattr(res, 'height', 0) or getattr(res, 'h', 0))

    print(f"ZED opened. Native resolution (per-eye): {width} x {height}")

    # Prepare sl.Mat to receive images
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    window_left = "ZED Left (native)"
    window_right = "ZED Right (native)"
    cv2.namedWindow(window_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_right, cv2.WINDOW_NORMAL)

    # Optionally set window size to native resolution (useful on HiDPI displays)
    cv2.resizeWindow(window_left, width, height)
    cv2.resizeWindow(window_right, width, height)

    print("Press 'q' or ESC to exit.")
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve left and right images (BGR)
                zed.retrieve_image(left_mat, sl.VIEW.LEFT)    # BGR image for left eye
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)  # BGR image for right eye

                # Convert to numpy (H x W x 3) contiguous array
                left_img = left_mat.get_data()   # already in BGR order compatible with cv2
                right_img = right_mat.get_data()

                # Sanity: ensure dtype is uint8 and shape matches reported resolution
                if left_img is None or right_img is None:
                    print("Warning: received empty image.")
                    continue

                # Show
                cv2.imshow(window_left, left_img)
                cv2.imshow(window_right, right_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            else:
                # grab failed / camera not ready; small sleep to avoid busy loop
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("ZED closed.")

if __name__ == "__main__":
    main()
