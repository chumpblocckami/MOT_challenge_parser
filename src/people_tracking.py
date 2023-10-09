import glob
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.schema import MOTChallenge


class PeopleTracker():
    def __init__(self, model_url, model_save_path):
        self.model_url = model_url
        self.model_save_path = model_save_path
        self.model = self.load_model()
        self.results = []

    def load_model(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        splitted = self.model_url.rsplit('/', 1)
        path = f"{self.model_save_path}/{splitted[len(splitted) - 1]}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(self.model_url, path)
        return YOLO(path)

    def show_points(self, image, bboxes):
        plt.figure()
        plt.imshow(image)
        plt.scatter(bboxes[0], bboxes[1], s=100, c='blue', marker='o')
        plt.annotate('bbox[0], bbox[1]', xy=(bboxes[0], bboxes[1]))
        plt.scatter(bboxes[2], bboxes[3], s=100, c='blue', marker='o')
        plt.annotate('bbox[2], bbox[3]', xy=(bboxes[2], bboxes[3]))

        plt.annotate(f'width; {bboxes[2] - bboxes[0]}', xy=(bboxes[0], bboxes[3]), c="red")
        plt.annotate(f'height: {bboxes[3] - bboxes[1]}', xy=(bboxes[0], bboxes[3] + 10), c="red")
        plt.show()

    def __call__(self, input_folder, output_folder, frame_rate, challenge_name, frame_format):

        os.makedirs(f"{output_folder}/{challenge_name}-train/{challenge_name}/gt", exist_ok=True)
        frame_format if frame_format.startswith('.') else '.' + frame_format
        wh = []
        total_frames = glob.glob(f"{input_folder}/*{frame_format}")
        for n, frame in tqdm(enumerate(total_frames), desc="Extracting dataset"):
            predictions = self.model(frame, verbose=False)
            for id, prediction in enumerate(predictions):
                wh.append(prediction.orig_shape)
                # https://github.com/matterport/Mask_RCNN/issues/327
                bboxes = np.array(prediction.boxes.xyxy.cpu()).flatten()
                try:
                    # self.show_points(image=prediction.orig_img, bboxes=bboxes)
                    data = MOTChallenge(**{"frame": n,
                                           "id": id,
                                           "bb_left": bboxes[0],
                                           "bb_top": bboxes[1],
                                           "bb_height": bboxes[3] - bboxes[1],
                                           "bb_width": bboxes[2] - bboxes[0]})
                except Exception as e:
                    # print(f"{e}")
                    data = MOTChallenge(**{"frame": n,
                                           "id": id,
                                           "bb_left": 0,
                                           "bb_top": 0,
                                           "bb_width": 0,
                                           "bb_height": 0})
                self.results.append(data)

        heights, widths = set([x[0] for x in wh]), set([x[1] for x in wh])
        if len(heights) != 1 or len(widths) != 1:
            raise AssertionError("Different height and width in frames!")
        height, width = list(heights)[0], list(widths)[0]

        with open(f"{output_folder}/{challenge_name}-train/{challenge_name}/gt/gt.txt", "w+") as file:
            for record in self.results:
                line = ", ".join([str(x) for x in list(record.dict().values())])
                file.write(line + "\n")

        with open(f"{output_folder}/{challenge_name}-train/{challenge_name}/seqinfo.ini", "w+") as file:
            file.write("[Sequence]\n")
            file.write(f"name={challenge_name}\n")
            file.write(f"imDir={input_folder}\n")
            file.write(f"frameRate={frame_rate}\n")
            file.write(f"seqLength={len(total_frames)}\n")
            file.write(f"imWidth={width}\n")
            file.write(f"imHeight={height}\n")
            file.write(f"imExt={frame_format}\n")

        with open(f"{output_folder}/{challenge_name}/{challenge_name}-train.txt", "w+") as file:
            file.write("name\n")
            file.write(f"{challenge_name}")
