from pathlib import Path
from src.mujoco_sim import mujoco_sim
from src.detection import detection
from src.analyze_trajectory import analyze_trajectory

MUJOCO_XML_PATH = Path(__file__).parent / "assets" / "conveyor.xml"
MUJOCO_OUTPUT_DIR = Path(__file__).parent / "outputs" / "camera_img"
DETECTION_OUTPUT_DIR = Path(__file__).parent / "outputs" / "detection"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent / "outputs" / "analysis"
REFERENCE_IMAGE_PATH = Path(__file__).parent / "outputs" / "reference_img" / "reference_img.png"

def main():
    MUJOCO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DETECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if mujoco_sim(MUJOCO_XML_PATH, MUJOCO_OUTPUT_DIR, duration=10, realtime=False) is not None:
        detection(MUJOCO_OUTPUT_DIR, DETECTION_OUTPUT_DIR)
        analyze_trajectory(REFERENCE_IMAGE_PATH, DETECTION_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR)
    else: 
        print("Sim failed, skipping detection")


if __name__ == "__main__":
    main()
