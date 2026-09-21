import cv2


def main(input_path, output_path, bbox=None):
    video = cv2.VideoCapture(input_path)
    success, frame = video.read()
    if not success:
        raise RuntimeError("Could not read video, check the path")

    # Read video properties so the output matches the input
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # size is (w, h)

    # If no bbox is given, select it once (this is the only step that needs a window)
    if bbox is None:
        bbox = cv2.selectROI("Select object", frame, False)
        cv2.destroyAllWindows()
        print("Use this bbox next time:", bbox)  # e.g. (312, 190, 64, 128)

    

    tracker = cv2.TrackerKCF_create()# or cv2.legacy.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Draw and write the first frame too, since it was already read
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    writer.write(frame)

    while True:
        success, frame = video.read()
        if not success:
            break

        found, bbox = tracker.update(frame)
        if found:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        writer.write(frame)  # save instead of imshow

    video.release()
    writer.release()  # important: without this the file may be corrupt

if __name__ == "__main__":
    main(
        "./input_videos/Seq01-1P-S0M1_CAM1 (online-video-cutter.com).mp4",
        "./output_videos/tracked_cv2_kfc.mp4",
        bbox=None,  # set to None to pick with the mouse
    )