from argparse import ArgumentParser
import pandas as pd
import cv2 as cv
from data import Sample,Detections,ExtractBboxFromTracks
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
from SoiUtils.load import load_yaml
from SoiUtils.video_manipulations import draw_video_from_bool_csv
import logging
parser = ArgumentParser()
parser.add_argument('--video_path',type=str)
parser.add_argument('--bboxes_path',type=str)
parser.add_argument('--bbox_col_names',nargs='+',default=['x','y','width','height'])
parser.add_argument('--bbox_format',type=str,default='coco')
parser.add_argument('--frame_col_name',type=str,default='frame_num')
parser.add_argument('--class_col_name',type=str,default=None)
parser.add_argument('--confidence_col_name',type=str,default=None)
parser.add_argument('--ids_col_name',type=str,default='ids')
parser.add_argument('--config_path',type=str,default=Path('configs/default.yaml'))
parser.add_argument('--frame_limit',default=None)
parser.add_argument('--rendered_video_save_path',type=str,default=None)
parser.add_argument('--bboxes_save_path',type=str,default=None)


logging.basicConfig(level=logging.DEBUG)
args = parser.parse_args()
detections_df = pd.read_csv(args.bboxes_path,index_col=0)
if args.class_col_name is None:
    class_col_name = 'cls'
    detections_df = detections_df.assign(**{class_col_name:'IRRELVANT'})
else:
    class_col_name = args.class_col_name

if args.confidence_col_name is None:
    confidence_col_name = 'confidence'
    detections_df = detections_df.assign(**{confidence_col_name:1})

else:
    confidence_col_name = args.confidence_col_name


cap = cv.VideoCapture(str(args.video_path))
sample = Sample(cap,Detections.from_dataframe(detections_df,args.frame_col_name))
config_params = load_yaml(args.config_path)


# info upon the tracker can be found here https://github.com/levan92/deep_sort_realtime
tracker = DeepSort(**config_params.get('tracker_params',{}))
bboxes_extractor = ExtractBboxFromTracks(args.ids_col_name,args.bbox_format,args.bbox_col_names,args.frame_col_name
,args.class_col_name,args.confidence_col_name)
aggregated_tracks = []
for frame_num,(frame,detections) in enumerate(sample):
    if args.frame_limit is not None and frame_num > args.frame_limit:
        break
    tracks = tracker.update_tracks([d.to_deepsort_format(lambda metadata: metadata[[confidence_col_name,class_col_name]]) for d in detections], frame=frame)
    frame_tracked_bboxes = bboxes_extractor.extract_from_deep_sort_format(tracks,frame_num,frame.shape[::-1][1:]) # Transform h,w,c to w,h
    if len(frame_tracked_bboxes)>0:
        aggregated_tracks.append(frame_tracked_bboxes)
    logging.debug(f'Finished processing frame number {frame_num}')

cap.release()


# Render the tracks upon the video using a unique color for each track.
# This should become a function.
tracks_df = pd.concat(aggregated_tracks)

if args.rendered_video_save_path is not None:
    cap = cv.VideoCapture(str(args.video_path))
    draw_video_from_bool_csv(cap,tracks_df,args.bbox_col_names,args.rendered_video_save_path,args.frame_col_name,
    args.class_col_name,args.confidence_col_name,args.bbox_format,args.ids_col_name,args.frame_limit)
    cap.release()

if args.bboxes_save_path is not None:
    tracks_df.to_csv(args.bboxes_save_path)















