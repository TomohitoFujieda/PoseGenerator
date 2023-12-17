import modules.scripts as scripts
import gradio as gr
import numpy as np
import torch
import os
import cv2
import math
import json

from modules import script_callbacks
from scripts.posegen.pose import Pose
from scripts.posegen.pose import Pose_angle
from scripts.sentiment_analysis.Text_analyser import Text_analyser

WIDTH = 512
HEIGHT = 512

pose_model_root = os.path.join(scripts.basedir(), "models", "model_norm_weight")
pose_angle_model_root = os.path.join(scripts.basedir(), "models", "model_angle_tanh_weight")
text_analyser_model_root = os.path.join(scripts.basedir(), "models")
image_folder = os.path.join(scripts.basedir(), "outputs")

pose_estimator = Pose(pose_model_root)
pose_estimator_angle = Pose_angle(pose_angle_model_root)
text_analyser = Text_analyser(text_analyser_model_root)
generated_keypoints = []

def generate_pose(text):
    emo_dict = text_analyser.analyse(text)
    emo_lst = list(emo_dict.values())
    emo = torch.tensor([emo_lst], dtype=torch.float32)
    keypoints = pose_estimator.estimate(emo)
    return keypoints

def generate_pose_v2(text):
    emo_dict = text_analyser.analyse(text)
    emo_lst = list(emo_dict.values())
    emo = torch.tensor([emo_lst], dtype=torch.float32)
    keypoints = pose_estimator_angle.estimate(emo)
    return keypoints

def analyze_sentiment(text):
    sentiment = text_analyser.analyse(text)
    res = []
    for k, v in sentiment.items():
        res.append([k, float(v)])
    return res

def draw_pose(text):
    global generated_keypoints
    generated_keypoints = []
    canvas = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
    coordinates = generate_pose(text)
    stickwidth = 4
    limbs = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [2, 6], [3, 7], [6, 8], [7, 9],
        [4, 10], [5, 11], [10, 12], [11, 13],
        [1, 14], [1, 15], [14, 16], [15, 17]
    ]

    keypoint_colors = [
        [0, 85, 255], [0, 0, 255], [0, 170, 255], [0, 255, 85],
        [170, 255, 0], [255, 85, 0], [0, 255, 255], [0, 255, 0],
        [0, 255, 170], [85, 255, 0], [255, 255, 0], [255, 0, 0],
        [255, 170, 0], [255, 0, 85], [255, 0, 170], [255, 0, 255],
        [170, 0, 255], [85, 0, 255]
    ]
    limb_colors = [
        [153, 0, 0], [0, 0, 153], [0, 51, 153], [0, 153, 0],
        [153, 153, 0], [0, 102, 153], [0, 153, 102], [0, 153, 153],
        [0, 153, 51], [51, 153, 0], [153, 102, 0], [102, 153, 0],
        [153, 51, 0], [153, 0, 51], [153, 0, 153],
        [153, 0, 102], [102, 0, 153]
    ]

    for i in range(len(coordinates)):
        c = coordinates[i] * 192 + torch.tensor([[256, 128]])
        c_np = c.cpu().detach().numpy().copy()
        c_x = c_np[0][0]
        c_y = c_np[0][1]
        cv2.circle(canvas, (int(c_x), int(c_y)), 4, keypoint_colors[i], thickness=-1)
        generated_keypoints.append([int(c_x), int(c_y)])

    for i in range(len(limbs)):
        frm = coordinates[limbs[i][0]]
        dst = coordinates[limbs[i][1]]

        frm = frm * 192 + torch.tensor([[256, 128]])
        frm_np = frm.cpu().detach().numpy().copy()
        frm_x = frm_np[0][0]
        frm_y = frm_np[0][1]

        dst = dst * 192 + torch.tensor([[256, 128]])
        dst_np = dst.cpu().detach().numpy().copy()
        dst_x = dst_np[0][0]
        dst_y = dst_np[0][1]

        m_x = (frm_x + dst_x) / 2
        m_y = (frm_y + dst_y) / 2

        length = ((frm_x - dst_x) ** 2 + (frm_y - dst_y) ** 2) ** 0.5
        angle = math.degrees(math.atan2(frm_y - dst_y, frm_x - dst_x))

        cur_canvas = canvas.copy()

        polygon = cv2.ellipse2Poly((int(m_x), int(m_y)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas, save_json()

def draw_pose_v2(text):
    global generated_keypoints
    generated_keypoints = []
    canvas = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
    coordinates = generate_pose_v2(text)
    stickwidth = 4
    limbs = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [2, 6], [3, 7], [6, 8], [7, 9],
        [4, 10], [5, 11], [10, 12], [11, 13],
        [1, 14], [1, 15], [14, 16], [15, 17]
    ]

    keypoint_colors = [
        [0, 85, 255], [0, 0, 255], [0, 170, 255], [0, 255, 85],
        [170, 255, 0], [255, 85, 0], [0, 255, 255], [0, 255, 0],
        [0, 255, 170], [85, 255, 0], [255, 255, 0], [255, 0, 0],
        [255, 170, 0], [255, 0, 85], [255, 0, 170], [255, 0, 255],
        [170, 0, 255], [85, 0, 255]
    ]
    limb_colors = [
        [153, 0, 0], [0, 0, 153], [0, 51, 153], [0, 153, 0],
        [153, 153, 0], [0, 102, 153], [0, 153, 102], [0, 153, 153],
        [0, 153, 51], [51, 153, 0], [153, 102, 0], [102, 153, 0],
        [153, 51, 0], [153, 0, 51], [153, 0, 153],
        [153, 0, 102], [102, 0, 153]
    ]

    for i in range(len(coordinates)):
        c = coordinates[i] * 192 + torch.tensor([[256, 128]])
        c_np = c.cpu().detach().numpy().copy()
        c_x = c_np[0][0]
        c_y = c_np[0][1]
        cv2.circle(canvas, (int(c_x), int(c_y)), 4, keypoint_colors[i], thickness=-1)
        generated_keypoints.append([int(c_x), int(c_y)])

    for i in range(len(limbs)):
        frm = coordinates[limbs[i][0]]
        dst = coordinates[limbs[i][1]]

        frm = frm * 192 + torch.tensor([[256, 128]])
        frm_np = frm.cpu().detach().numpy().copy()
        frm_x = frm_np[0][0]
        frm_y = frm_np[0][1]

        dst = dst * 192 + torch.tensor([[256, 128]])
        dst_np = dst.cpu().detach().numpy().copy()
        dst_x = dst_np[0][0]
        dst_y = dst_np[0][1]

        m_x = (frm_x + dst_x) / 2
        m_y = (frm_y + dst_y) / 2

        length = ((frm_x - dst_x) ** 2 + (frm_y - dst_y) ** 2) ** 0.5
        angle = math.degrees(math.atan2(frm_y - dst_y, frm_x - dst_x))

        cur_canvas = canvas.copy()

        polygon = cv2.ellipse2Poly((int(m_x), int(m_y)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas, save_json()

def save_json():
    global generated_keypoints
    target_idx = [1, 0, 2, 6, 8, 3, 7, 9, 4, 10, 12, 5, 11, 13, 14, 15, 16, 17]
    keypoints = []
    for i in target_idx:
        keypoints.append(generated_keypoints[i])
    res = {"width": WIDTH, "height": HEIGHT, "keypoints": keypoints}
    # with open(os.path.join(image_folder, "output.json"), "w") as f:
    #     json.dump(res, f)

    return res

def image_save(img):
    cv2.imwrite(os.path.join(image_folder, "output.png"), img)

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column():
                text = gr.Text(label="words")
                with gr.Row():
                    # send_button = gr.Button(value="Estimate Pose")
                    send_button_v2 = gr.Button(value="Estimate Pose")
                    analyze_button = gr.Button(value="Analyze Text")
                emo_output = gr.DataFrame(headers=["emotions", "probs"])

            with gr.Column():
                with gr.Row():
                    img_output = gr.Image(label="Result")
                # with gr.Row():
                    json_output = gr.JSON(label="JSON", visible=False)
                with gr.Row():
                    save_button = gr.Button(value="Save Output as Image")
                    save_json_button = gr.Button(value="Save Output as JSON")

        # send_button.click(fn=draw_pose, inputs=text, outputs=[img_output, json_output])
        send_button_v2.click(fn=draw_pose_v2, inputs=text, outputs=[img_output, json_output])
        analyze_button.click(fn=analyze_sentiment, inputs=text, outputs=emo_output)
        save_button.click(fn=None, inputs=img_output, _js="downloadPNG")
        # save_button.click(fn=image_save, inputs=img_output)
        save_json_button.click(fn=None, inputs=[json_output], _js="downloadJSON")
        # save_json_button.click(fn=save_json, outputs=json_output)

    return [(ui_component, "Pose Generator", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)