import numpy as np
import torch
import os

from .model import keypoint_estimator
from .model_angle import keypoint_estimator_angle

keypoint_names = [
    "nose",
    "right_shoulder",
    "left_shoulder",
    "right_thigh",
    "left_thigh",
    "right_elbow",
    "left_elbow",
    "right_hand",
    "left_hand",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear"
]

emotion_names = ['joy', 'sadness', 'excited', 'surprise', 'anger', 'fear', 'disgust', 'trust']

angle_data = ["cosine", "sine"]

def index_parser(i):
  if 0 <= i and i <= 4: # nose, right_shoulder, left_shoulder, right_thigh, left_thigh
    return 0
  elif 13 <= i and i <= 14: # right_eye, left_eye
    return 1

  elif i == 5: # right_elbow
    return 2
  elif i == 6: # left_elbow
    return 3

  elif i == 7: # right_hand
    return 6
  elif i == 8: # left_ hand
    return 7

  elif i == 9: # right_knee
    return 4
  elif i == 10: # left_knee
    return 5

  elif i == 11: # right_ankle
    return 10
  elif i == 12: # left_ankle
    return 11

  elif i == 15:
    return 14
  elif i == 16:
    return 15

class Pose(object):
    def __init__(self, root_folder):
        self.models = []
        self.root_folder = root_folder
        for name in keypoint_names:
            model = keypoint_estimator()
            model.load_state_dict(torch.load(os.path.join(root_folder, name + '_est_mdl.pth')))
            model.eval()
            self.models.append(model)

    def length(self):
        return len(self.models)

    def estimate(self, emo):
        neck_c = torch.tensor([[0, 0]], dtype=torch.float32)
        keypoints = [neck_c]

        for i in range(len(self.models)):
            idx = index_parser(i)
            res = self.models[i](keypoints[idx], emo)
            keypoints.append(res)

        return keypoints


class Pose_angle(object):
    def __init__(self, root_folder):
        self.models = []
        self.root_folder = root_folder

        for a in range(len(angle_data)):
            self.models.append([])
            for e in range(len(emotion_names)):
                self.models[a].append([])
                for i in range(len(keypoint_names)):
                    model_name = keypoint_names[i] +"_est_mdl.pth"
                    model_path = os.path.join(self.root_folder, angle_data[a], emotion_names[e], model_name)
                    model = keypoint_estimator_angle()
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    self.models[a][e].append(model)

    def estimate(self, emo, top_emo=2, btm_emo=2):
        def get_index_of_max_emo(emotion, n):
            res = sorted(emotion, reverse=True)[0:n]
            res_idx = []
            for i in res:
              res_idx.append(emotion.index(i))
            return res_idx

        def get_index_of_min_emo(emotion, n):
            res = sorted(emotion)[0:n]
            res_idx = []
            for i in res:
              res_idx.append(emotion.index(i))
            return res_idx

        limb_length = [
            0.25, 0.25, 0.25, 0.8, 0.8,
            0.4, 0.4, 0.4, 0.4,
            0.5, 0.5, 0.5, 0.5,
            0.1, 0.1, 0.1, 0.1
        ]

        neck_c = torch.tensor([[0, 0]], dtype=torch.float32)
        keypoints = [neck_c]
        emo_lst = emo.squeeze().tolist()
        max_idx_lst = get_index_of_max_emo(emo_lst, top_emo)
        min_idx_lst = get_index_of_min_emo(emo_lst, btm_emo)
        n = len(max_idx_lst + min_idx_lst)

        for i in range(len(keypoint_names)):
            cosine_tensor = torch.tensor([[0]], dtype=torch.float32)
            sine_tensor = torch.tensor([[0]], dtype=torch.float32)
            for j in (max_idx_lst + min_idx_lst):
                emo_val = emo_lst[j]
                emo_tensor = torch.tensor([[emo_val]])

                # cos
                cosine_tensor += self.models[0][j][i](emo_tensor)
                # sin
                sine_tensor += self.models[1][j][i](emo_tensor)

            cosine_tensor /= n
            sine_tensor /= n

            res_x = cosine_tensor[0][0] * limb_length[i]
            res_y = sine_tensor[0][0] * limb_length[i]

            bfr_i = index_parser(i)
            res = [keypoints[bfr_i][0][0] + res_x, keypoints[bfr_i][0][1] + res_y]
            keypoints.append(torch.tensor([res], dtype=torch.float32))

        return keypoints
