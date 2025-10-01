import cv2 as cv
from PIL import Image
import numpy as np


def detection(DETECTION_INPUT_DIR, DETECTION_OUTPUT_DIR):
    image_files = sorted(DETECTION_INPUT_DIR.glob("*.png"))
    print(f"Starting color detection on {len(image_files)} images")

    for image_path in image_files:
        img = cv.imread(str(image_path))

        if img is None:
            print(f"Error reading {image_path}")
            continue

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        red_mask_1 = cv.inRange(hsv, np.array([0, 70, 90]), np.array([3, 255, 255]))
        red_mask_2 = cv.inRange(hsv, np.array([175, 70, 90]), np.array([180, 255, 255]))

        red_mask = cv.bitwise_or(red_mask_1, red_mask_2)

        blue_mask = cv.inRange(hsv, np.array([100, 70, 90]), np.array([130, 255, 255]))
        cyan_mask = cv.inRange(hsv, np.array([80, 70, 90]), np.array([100, 255, 255]))
        green_mask = cv.inRange(hsv, np.array([40, 70, 90]), np.array([80, 255, 255]))

        masks = {
            "red": red_mask,
            "blue": blue_mask,
            "cyan": cyan_mask,
            "green": green_mask,
        }
        
        # calculating complementary colors for display purpose
        complementary_colors = {}
        for name, mask in masks.items():
            mean_val = cv.mean(img, mask)
            assert isinstance(mean_val, tuple), "cv.mean should return a tuple"
            complementary_colors[name] = (int(255 - mean_val[0]), int(255 - mean_val[1]), int(255 - mean_val[2]))

        combined_mask = cv.bitwise_or(
            cv.bitwise_or(red_mask, blue_mask), cv.bitwise_or(cyan_mask, green_mask)
        )

        # kernel = np.ones((3, 3), dtype=np.uint8)
        # red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
        # red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(combined_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        moment_list = [cv.moments(contour) for contour in contours]
        centroid_list = [
            (M["m10"] / (M["m00"] + 1e-5), M["m01"] / (M["m00"] + 1e-5))
            for M in moment_list
        ]

        processed_img = img

        for i in range(len(contours)):
            x, y = int(contours[i][0][0][0]), int(contours[i][0][0][1])
            color = next(
                (
                    complementary_colors[name]
                    for name, mask in masks.items()
                    if mask[y, x] > 0
                ),
                (0, 0, 0),
            )

            cv.drawContours(processed_img, contours, i, color, 1)
            cv.circle(
                processed_img,
                (int(centroid_list[i][0]), int(centroid_list[i][1])),
                1,
                color,
                -1,
            )

        processed_img = cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)
        processed_img = Image.fromarray(processed_img)
        processed_img_path = DETECTION_OUTPUT_DIR / image_path.name
        processed_img.save(processed_img_path)
        print(f"Image saved : {processed_img_path}")

    print(f"Detection finished, results saved at {DETECTION_OUTPUT_DIR}")
    return DETECTION_OUTPUT_DIR
