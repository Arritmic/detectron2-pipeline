# Function created by Tino Álvarez (University of Oulu)
# Purpose: Print the detectron2 results in a .csv or .txt file

# Import packages.
import csv
import torch
import tqdm

# importing the numpy and collections for tracking the "main" box
import numpy as np
import collections

from detectron2.data import MetadataCatalog
from pipeline.pipeline import Pipeline


class SaveAnnotations(Pipeline):
    """Pipeline task for saving annotations."""

    def __init__(self, src, metadata_name, filename, number_of_frames, fps=25):
        self.src = src
        self.metadata_name = metadata_name
        self.metadata = MetadataCatalog.get(self.metadata_name)
        self.file_name = filename
        self.buffer_center_position =  collections.deque(maxlen=60) # Circular Buffer for storing the center of the "main" box object.
        self.number_of_keypoints = 17
        self.cpu_device = torch.device("cpu")
        self.number_of_frames = number_of_frames
        self.fps = fps

        # Header for the CSV file based on the number of keypoints of COCO dataset. (17 keypoints)
        # Each row is as follow:
        # [frame number][timestamp][x1][y1][prob1][x2][y2][prob2].......[x15][y15][prob15][x16][y16][prob16][detection score]
        self.header = []
        self.header.append((str("frame_number")))
        self.header.append((str("timestamp")))
        for i in range(self.number_of_keypoints):
            self.header.extend(("x"+str(i), "y"+str(i),"prob"+str(i)))
        self.header.append((str("detection_score")))

        with open(self.file_name, mode='w') as annotations_file:
            annotations_writer = csv.writer(annotations_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            annotations_writer.writerow([h for h in self.header])

        # For printing the logging messages in the same line everytime.
        self.file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')

        super().__init__()

    def map(self, data):
        src_image = data["image"].copy()
        data[self.src] = src_image

        self.save_results_in_csv(data)

        return data

    # Function for injecting the logging messages in the tqdm class
    def print_status_annotations(self, message):
        self.file_log.set_description_str(f'{message}')

    # Function for tracking the box/keypoints/object is being recorded.
    def track_recorded_box(self, box):
        x0, y0, x1, y1 = box
        x_center = (x1 + x0) / 2
        y_center = (y1 + y0) / 2

        self.buffer_center_position.append([x_center,y_center])
        numpy_array = np.array(self.buffer_center_position)
        mean = numpy_array.mean(axis=0)
        variance = numpy_array.var(axis=0)

        same_center = False
        if (all([x_center,y_center] <= mean + variance) and all([x_center,y_center] >= mean - variance)):
            same_center = True

        return same_center


    # Function for checking which is the biggest detected box.
    # Constrain assumed that the interested object in the scene is the biggest one.
    def check_box_size(self, boxes):
        num_of_boxes = len(boxes)

        bigger_box_index = 0
        x0, y0, x1, y1 = boxes[0]
        width = x1 - x0
        height = y1 - y0
        max_area = width*height

        for i in range(num_of_boxes):
            x0, y0, x1, y1 = boxes[i]
            width = x1 - x0
            height = y1 - y0
            area = width * height
            if area >= max_area:
                bigger_box_index = i

        if self.track_recorded_box(boxes[bigger_box_index]):
            return bigger_box_index
        else:
            return None


    # Function for printing Zeros in the CSV
    def print_zeros_array_in_csv(self):
        with open(self.file_name, mode='a') as annotations_file:
            annotations_writer = csv.writer(annotations_file, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
            vector_zeros = np.zeros(self.number_of_keypoints * 3 + 3) # 17 keypoints + frame_number + timestamp + detection score
            annotations_writer.writerow(vector_zeros)


    # Function for storing keypoints in the .CSV file
    def save_results_in_csv(self,data):
        if "predictions" not in data:
            return self.print_status_annotations("Save annotations: No predictions. Storage skipped...")

        predictions = data["predictions"]
        if "instances" not in predictions:
                return self.print_status_annotations("Save annotations: No keypoints information. Storage skipped...")

        # instances
        instances = predictions["instances"]
        frame_idx = data["frame_num"]

        if not(instances.has("pred_keypoints")):
                return self.print_status_annotations("Save annotations: No keypoints information. Storage skipped...")


        # Number of predictions.
        num_instances = len(predictions)

        # If there are predictions --> Check the boxes ---> Select the biggest box ---> save data
        if num_instances != 0:

            boxes = instances.to(self.cpu_device).pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
            scores = instances.to(self.cpu_device).scores if instances.has("scores") else None
            classes = instances.to(self.cpu_device).pred_classes.numpy() if instances.has("pred_classes") else None
            keypoints = instances.to(self.cpu_device).pred_keypoints if instances.has("pred_keypoints") else None

            index = self.check_box_size(boxes) # Track if the box we are recording is the same. If it loses the track, write zero vector for the current vector.
            if index != None:
                with open(self.file_name, mode='a') as annotations_file:
                    annotations_writer = csv.writer(annotations_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    row = []
                    row.append((int(frame_idx)))     # Number of frame
                    row.append((int(frame_idx*1000/self.fps))) # TimeStamp
                    # save keypoint
                    for idx, keypoint in enumerate(keypoints[index]):
                        x, y, prob = keypoint
                        row.extend((float("{0:.3f}".format(x)),float("{0:.3f}".format(y)),float("{0:.4f}".format(prob))))
                    row.append((float("{0:.4f}".format(scores[index])))) # Detection score for the whole object.
                    annotations_writer.writerow(row)

                return  self.print_status_annotations("Save annotations: Data for frame " + str(frame_idx) + " saved.")
            else:
                # If the object detected as main box is not the tracked one..
                self.print_zeros_array_in_csv()
                return self.print_status_annotations("Save annotations: Zero data for frame " + str(frame_idx) + " saved.")

        else:
            # If there is not prediction ---> Store array of zeros in that line correspond to the current processed frame.
            # If any object detected in the line of that frame is written zeros.
            self.print_zeros_array_in_csv()
            return self.print_status_annotations("Save annotations: Zero data for frame " + str(frame_idx) + " saved.")