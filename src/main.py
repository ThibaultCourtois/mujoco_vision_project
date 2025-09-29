import mujoco as mj
import mujoco.viewer as mjv
from pathlib import Path
from PIL import Image

XML_FILE_PATH = Path(__file__).parent.parent / "assets" / "scene.xml" 
OUTPUTS_FIlE_PATH = Path(__file__).parent.parent / "outputs"

def main():
    try:
        m = mj.MjModel.from_xml_path(str(XML_FILE_PATH))
    except Exception as e:
        print(f"Error while loading XML model: {e}")
        return 1 

    d = mj.MjData(m) 
    r = mj.Renderer(m, height=480, width=640)
    cam_id = m.camera('camera').id

    frame_count = 0
    last_picture_time = -1.0

    with mjv.launch_passive(m, d) as viewer :  
        while viewer.is_running() and d.time < 10:
            mj.mj_step(m, d)
            viewer.sync()
            if int(d.time) > last_picture_time:
                last_picture_time = int(d.time)
                r.update_scene(d, camera=cam_id)
                image = r.render()
                img = Image.fromarray(image)
                img_path = OUTPUTS_FIlE_PATH / f"frame_{frame_count:04d}_t{d.time:.2f}s.png"
                img.save(img_path)
                print(f'Image saved : {img_path}')           
                frame_count += 1
    print(f"Simulation ended. {frame_count} images saved at {OUTPUTS_FIlE_PATH}")
    return 0



if __name__ == "__main__":
    main()
