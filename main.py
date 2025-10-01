from pathlib import Path
from src.mujoco_sim import mujoco_sim
from src.detection import detection

MUJOCO_XML_PATH = Path(__file__).parent / "assets" / "scene.xml"
MUJOCO_OUTPUT_DIR = Path(__file__).parent / "outputs" / "camera_img"
DETECTION_OUTPUT_DIR = Path(__file__).parent / "outputs" / "detection"

def main():
    MUJOCO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DETECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if mujoco_sim(MUJOCO_XML_PATH, MUJOCO_OUTPUT_DIR) is not None:
        detection(MUJOCO_OUTPUT_DIR, DETECTION_OUTPUT_DIR)
    else: 
        print("Sim failed, skipping detection")


if __name__ == "__main__":
    main()
