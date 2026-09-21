from ultralytics import YOLO
import cv2


def iou(a, b):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-9)

def get_objects(result):
    """
    get the [track_id, bbox] for each object 
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()

    # Detections may exist without tracking IDs
    if boxes.id is None:
        return []
    ids = boxes.id.int().cpu().tolist()

    objects = []

    for track_id, box in zip(ids, xyxy):
        box = tuple(map(int, box))

        objects.append((track_id, box))

    return objects

def main(
        input_path,
        model_path,
        min_iou=0.2
    ):
    model = YOLO(model_path)

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise RuntimeError("Couldn't open the video")

    # read the first frame
    success, frame = video.read()
    if not success:
        raise RuntimeError("Could not read first frame")

    # let the user user select the bbox of the object
    bbox = cv2.selectROI(
        windowName="select object",
        img=frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Object")

    x, y, w, h = bbox
    # check if user select object or not
    if w == 0 or h == 0:
        raise RuntimeError("No object was selected")
    # (x, y) -> top-left
    # (x+w, y+h) -> bottom-right
    selected_bbox = (x, y, x+w, y+h)

    selected_id = None

    while True:
        success, frame = video.read()
        if not success:
            break

        results = model.track(
            frame,
            persist=True, # keep tracking state between frames
            tracker="bytetrack.yaml",  # tracking algorithm/configuration
            verbose=False # don't print information on terminal
        )

        result = results[0]

        objects = get_objects(result)
        chosen = None

        # try to find the current object
        if selected_id is not None:
            for track_id, box in objects:
                if track_id == selected_id:
                    chosen = box
                    break

        # if track is lost use iou to recover it
        if chosen == None:
            best = min_iou
            for track_id, box in objects:
                score = iou(box, selected_bbox)

                if score > best:
                    best = score
                    chosen = box
                    selected_id = track_id

        # draw selected object
        if chosen is not None:

            # update the last bbox
            selected_bbox = chosen

            x1, y1, x2, y2 = chosen

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"ID {selected_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # if the tracking is lost
        else:
            cv2.putText(
                frame,
                "Tracking lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )


        cv2.imshow("Tracking", frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("input_videos/Seq01-1P-S0M1_CAM1 (online-video-cutter.com).mp4")