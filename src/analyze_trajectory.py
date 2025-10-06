import json
import math
import cv2 as cv
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def analyze_trajectory(REFERENCE_IMAGE_PATH, ANALYSIS_INPUT_DIR, ANALYSIS_OUTPUT_DIR):
    with open(ANALYSIS_INPUT_DIR / "trajectory.json", 'r', encoding="utf-8") as f:
        trajectory = json.load(f)
        times = [None] * (len(trajectory) - 1)
        velocities = [None] * (len(trajectory) - 1)

        reference_img = cv.imread(str(REFERENCE_IMAGE_PATH))

        if reference_img is None: 
            print("Error loading reference img")
            return


        for i in range(len(trajectory) - 1):
            img_data1 = trajectory[i]
            img_data2 = trajectory[i + 1]

            x1, y1 = img_data1['centroid']
            x2, y2 = img_data2['centroid']

            t1, t2 = img_data1['timestamp'], img_data2['timestamp']
            
            velocities[i] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / (t2 - t1) 
            times[i] = (t1 + t2) / 2 

            cv.line(reference_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

            if i == len(trajectory) - 2:
                cv.circle(reference_img, (x1, y1), radius=1, color=(0, 0, 255), thickness=-1)
                cv.circle(reference_img, (x2, y2), radius=1, color=(0, 0, 255), thickness=-1)
            else : 
                cv.circle(reference_img, (x1, y1), radius=1, color=(0, 0, 255), thickness=-1)
            
        cv.imwrite(str(ANALYSIS_OUTPUT_DIR / 'trajectory_visualization.png'), reference_img)

        print("Trajectory visualization saved !")

        plt.figure(figsize=(10, 6))
        plt.plot(times, velocities, marker='o', linestyle='-', linewidth=1, markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Instantaneous velocity (px/s)')
        plt.title('Instantaneous velocity = f(Time)')
        plt.grid(True, alpha=0.3)
        plt.savefig(ANALYSIS_OUTPUT_DIR / 'velocity_plot.png', dpi=300)

        print("Velocity graph saved !")

    return ANALYSIS_OUTPUT_DIR
