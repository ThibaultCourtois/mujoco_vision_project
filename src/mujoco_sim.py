import mujoco as mj
import mujoco.viewer as mjv
from PIL import Image


def mujoco_sim(MUJOCO_XML_PATH, MUJOCO_OUTPUT_DIR):
    try:
        m = mj.MjModel.from_xml_path(str(MUJOCO_XML_PATH))
    except Exception as e:
        print(f"Error while loading XML model: {e}")
        return None

    d = mj.MjData(m)
    r = mj.Renderer(m, height=480, width=640)
    cam_id = m.camera("camera").id

    frame_count = 0
    last_picture_time = -1.0

    with mjv.launch_passive(m, d) as viewer:
        while viewer.is_running() and d.time < 10:
            mj.mj_step(m, d)
            viewer.sync()
            if int(d.time) > last_picture_time:
                last_picture_time = int(d.time)
                r.update_scene(d, camera=cam_id)
                image = r.render()
                img = Image.fromarray(image)
                img_path = (
                    MUJOCO_OUTPUT_DIR / f"frame_{frame_count:04d}_t{d.time:.2f}s.png"
                )
                img.save(img_path)
                print(f"Image saved : {img_path}")
                frame_count += 1
    print(f"Simulation ended. {frame_count} images saved at {MUJOCO_OUTPUT_DIR}")
    return MUJOCO_OUTPUT_DIR
