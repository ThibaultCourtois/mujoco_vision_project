from pathlib import Path
from src.mujoco_sim import mujoco_sim
from src.detection import detection

MUJOCO_XML_PATH = Path(__file__).parent / "assets" / "scene.xml"
MUJOCO_OUTPUT_DIR = Path(__file__).parent / "outputs" / "camera_img"
DETECTION_OUTPUT_DIR = Path(__file__).parent / "outputs" / "detection"


def main():
    mujoco_sim(MUJOCO_XML_PATH, MUJOCO_OUTPUT_DIR)
    detection(MUJOCO_OUTPUT_DIR, DETECTION_OUTPUT_DIR)


if __name__ == "__main__":
    main()
