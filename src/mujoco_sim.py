import mujoco as mj
import mujoco.viewer as viewer
from PIL import Image
import time

CONVEYOR_SPEED = 5 
DECAY_TIME = 2.0

last_direction_modification_time = -float('inf')

def mujoco_sim(MUJOCO_XML_PATH, MUJOCO_OUTPUT_DIR, duration, realtime):
    global last_direction_modification_time
    try:
        model = mj.MjModel.from_xml_path(str(MUJOCO_XML_PATH))
    except Exception as e:
        print(f"Error while loading XML model: {e}")
        return None

    data = mj.MjData(model)
    
    # variable reset
    last_direction_modification_time = -float('inf')

    renderer = mj.Renderer(model, height=480, width=640)
    cam_id = model.camera("top_conveyor").id
    frame_count = 0
    last_picture_time = -1.0

    if realtime:
        with mj.viewer.launch_passive(model, data) as viewer:
            sim_time = 0
            dt = model.opt.timestep
            start_realtime_time = time.time()

            while viewer.is_running() and sim_time < duration:
                conveyor_control(model, data, sim_time)
                mj.mj_step(model, data)
                sim_time += dt

                if sim_time > last_picture_time:
                    last_picture_time = sim_time + 200*dt
                    renderer.update_scene(data, camera=cam_id)
                    image = renderer.render()
                    img = Image.fromarray(image)
                    img_path = (
                        MUJOCO_OUTPUT_DIR / f"frame_{frame_count:04d}_t{data.time:.2f}s.png"
                    )
                    img.save(img_path)
                    frame_count += 1

                # realtime sync
                elapsed_real_time = time.time() - start_realtime_time
                if sim_time > elapsed_real_time:
                    time.sleep(sim_time - elapsed_real_time)
                viewer.sync()

            print(f"Simulation ended. {frame_count} images saved at {MUJOCO_OUTPUT_DIR}")
            return MUJOCO_OUTPUT_DIR
    else:
        sim_time = 0
        dt = model.opt.timestep

        while sim_time < duration:
            conveyor_control(model, data, sim_time)
            mj.mj_step(model, data)
            sim_time += dt

            if sim_time > last_picture_time:
                last_picture_time = sim_time + 200*dt
                renderer.update_scene(data, camera=cam_id)
                image = renderer.render()
                img = Image.fromarray(image)
                img_path = (
                    MUJOCO_OUTPUT_DIR / f"frame_{frame_count:04d}_t{data.time:.2f}s.png"
                )
                img.save(img_path)
                frame_count += 1

        print(f"Simulation ended. {frame_count} images saved at {MUJOCO_OUTPUT_DIR}")
        return MUJOCO_OUTPUT_DIR


def conveyor_control(model, data, sim_time):
    global last_direction_modification_time

    if data.sensordata[3] > 0.5: # red_bloc on the belt

        collision_detected = data.sensordata[6] != 0 or data.sensordata[9] != 0
        time_since_last_change = sim_time - last_direction_modification_time

        if data.ctrl[0] == 0:
            data.ctrl[0] = CONVEYOR_SPEED

        if time_since_last_change > DECAY_TIME:
            if collision_detected:
                data.ctrl[0] = -data.ctrl[0]
                last_direction_modification_time = sim_time
        data.ctrl[1] = -0.05 * data.ctrl[0]
