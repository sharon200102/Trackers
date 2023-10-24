import cv2 as cv
import numpy as np
import yaml
from pybboxes import BoundingBox
import pybboxes as pbx 
import pandas as pd
def create_video_writer_from_capture(video_capture, output_video_path):
    frame_rate = video_capture.get(cv.CAP_PROP_FPS)
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width)
    new_height = int(height)
    size = (new_width, new_height)

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video_writer = cv.VideoWriter(str(output_video_path), fourcc, frame_rate, size)
    return video_writer

# function that generates random colors, the color will be attached to each track
def generate_color():
  color = np.random.choice(range(256), size=3)
  return tuple(int(num) for num in color)

def load_yaml(path):
    with open(path, "r") as stream:
        try:
            content = yaml.load(stream, Loader=yaml.FullLoader)
            return content
        except yaml.YAMLError as exc:
            print(exc)


def draw_video_from_bool_csv(video, df,bbox_cols_names, output_video_path,frame_col_name='frame_num',class_col_name=None,confidence_col_name=None,bbox_foramt='coco',id_col_name=None,frame_limit=None):
    
    writer = create_video_writer_from_capture(video, output_video_path)
    limit_flag = False
    color_map = {}
    default_box_color = (0, 255, 0)
    while True:
        frame_num = video.get(cv.CAP_PROP_POS_FRAMES)

        current_df = df[df[frame_col_name] == frame_num]
        current_bboxes = current_df[bbox_cols_names]
        current_classes = current_df[class_col_name] if class_col_name is not None else pd.Series([''] * len(current_df))
        current_confidences = current_df[confidence_col_name] if confidence_col_name is not None else pd.Series([''] * len(current_df))
        current_ids = current_df[id_col_name] if id_col_name is not None else pd.Series([''] * len(current_df))


        ret, frame = video.read()
        if frame_limit is not None:
            limit_flag = frame_num>frame_limit
            
        if not ret or limit_flag:
            break

        for i, bbox in enumerate(current_bboxes.values):
            x, y, width, height = BoundingBox.from_coco(*pbx.convert_bbox(bbox,from_type=bbox_foramt,to_type='coco')).raw_values
            bbox_cls = current_classes.iloc[i]
            bbox_confidence = current_confidences.iloc[i]
            bbox_id = current_ids.iloc[i]
            if  id_col_name is not None:
              if bbox_id not in color_map:
                color_map[bbox_id] = generate_color()
              bbox_color =  color_map[bbox_id]
            else:
              bbox_color = default_box_color

            frame = cv.rectangle(frame, (x,y), (x+width,y+height), color=bbox_color, thickness=2)
            
            cv.putText(frame, f'{bbox_cls}{bbox_confidence}{bbox_id}'.strip(), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


        writer.write(frame)

    video.release()
    writer.release()
