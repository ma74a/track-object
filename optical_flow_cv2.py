import cv2
import numpy as np

def main(input_path):
    video = cv2.VideoCapture(input_path) # read the video

    if not video.isOpened():
        raise RuntimeError("Could not open video")

    # read first frame
    success, frame1 = video.read() # get the first frame
    if not success: #check if could read the frame or not [true, false]
        raise RuntimeError("Could not read first frame")

    # create bbox around object
    bbox = cv2.selectROI(
        "select object",
        frame1,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Select Object")
    x, y, w, h = bbox
    #convert it into gray
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # get the roi
    roi = old_gray[y:y+h, x:x+w]
    # 4. Find good feature points
    old_points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=7
    )
    if old_points is None:
        raise RuntimeError("No feature points found")
    
    # Convert ROI coordinates to full-frame coordinates
    # make the points relative to the whole frame rather than the roi
    old_points[:, 0, 0] += x
    old_points[:, 0, 1] += y

    while True:
        success, frame = video.read()
        if not success:
            break

        # current grey frame
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            new_gray,
            old_points,
            None
        )
        if new_points is None:
            break

        # keep tracking
        good_old = old_points[status==1]
        good_new = new_points[status==1]

        if len(good_new) == 0:
            print("Lost all feature points")
            break

        # calculate the movement
        movement = good_new - good_old
        dx = np.median(movement[:, 0])
        dy = np.median(movement[:, 1])

        # update bbox
        x = int(x + dx)
        y = int(y + dy)

        # draw bbox
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # draw feature points
        for point in good_new:

            px, py = point

            cv2.circle(
                frame,
                (int(px), int(py)),
                3,
                (0, 0, 255),
                -1
            )
        cv2.imshow(
            "Optical Flow Tracker",
            frame
        )

        # prepare for the next frame
        old_gray = new_gray
        old_points = good_new.reshape(-1, 1, 2)

        # quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("input_videos/Seq01-1P-S0M1_CAM1 (online-video-cutter.com).mp4")