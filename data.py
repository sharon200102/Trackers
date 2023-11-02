from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2
from pybboxes import BoundingBox
import pybboxes as pbx
import logging
# The following class is a data-class that defines the structure of the ouput of the VMD for each video seperatley.
@dataclass
class Detections:
  # For each frame save the detections it contained
  # The tracker requests the detections to be of the following format: ( [left,top,w,h] , confidence, detection_class)
  # Currently,our vmd algorithm is unsupervised,therefore, it has no confidence and detection class, so I assign confidence=1 and class IRRELEVANT for each detectino.
  full_detections: dict #{frame_i: [detection_0,detection_1,..,detection_n]}


  @classmethod
  def from_dataframe(cls,df:pd.DataFrame,frame_col_name:str,extract_bbox_fn = None, extract_metadata_fn = None,mode='coco'):
    if extract_bbox_fn is None:
      extract_bbox_fn = lambda row: row.iloc[0:4]
    if extract_metadata_fn is None:
      extract_metadata_fn = lambda row: row.iloc[4:]

    # group the detetions upon frame number.
    detections_dict = {}
    detections_grouped_by_frame = df.groupby(by=frame_col_name)
    # iterate over each frame.
    for frame_num,detections_in_specifc_frame in detections_grouped_by_frame:
      # for each frame initialize its detections.
        detections_dict[frame_num] = []
        num_of_detections_in_frame = len(detections_in_specifc_frame)
        # For each detection in the frame create a Detection object
        for i in range(num_of_detections_in_frame):
          detection_series = detections_in_specifc_frame.iloc[i]
          bbox = BoundingBox.from_coco(*pbx.convert_bbox(list(extract_bbox_fn(detection_series)),from_type=mode,to_type='coco'))
          detections_dict[frame_num].append(Detection(bbox,extract_metadata_fn(detection_series)))

    return cls(full_detections = detections_dict)

# Define a sample class, that contains a video and its corresponding detections.
""" Due to the fact that I would like to read one frame at a time and not to load the whole video,
 I'm saving the VideoCapture of the video and not an array of the video."""

@dataclass
class Sample:
  video:cv2.VideoCapture
  detections:Detections
  def __iter__(self):
        return _SampleIter(self)

# A class that defines how to iterate over the sample class
class _SampleIter:
  def __init__(self,sample:Sample):
    self.sample = sample
    self._current_index = 0

  def __iter__(self):
        return self
  def __next__(self):
    # Stop iterating only if the frame counter exceded the frame limit or if it reached the end of the video.
      video_keep_reading_flag, frame = self.sample.video.read()
      if not video_keep_reading_flag:
        raise StopIteration

       # read one frame from the 'capture' object; img is (H, W, C)
      ret = frame,self.sample.detections.full_detections.get(self._current_index,[])
      self._current_index+=1
      return ret


# A class that structures a detection in a way that the tracker can handle.
# A refactor should be considered, create a generic class of Detection and a method that can output it in the tracker format.
@dataclass
class Detection:
  detection:BoundingBox
  metadata: pd.Series = None

  def to_deepsort_format(self,metadata_extraction_fn=None):
    requested_metadata_columns = ['confidence','detection_class']
    if metadata_extraction_fn is None:
      metadata_extraction_fn = lambda metadata: metadata[requested_metadata_columns]

    return [self.detection.to_coco().raw_values,*(metadata_extraction_fn(self.metadata))]


class ExtractBboxFromTracks():
  def __init__(self,ids_save_name = 'ids',bbox_save_format='coco',bbox_save_names = None,frame_num_save_name=None,cls_save_name=None,confidence_save_name=None):
    self.ids_save_name = ids_save_name
    self.bbox_save_format = bbox_save_format
    self.bbox_save_names = bbox_save_names
    self.cls_save_name = cls_save_name
    self.confidence_save_name = confidence_save_name
    self.frame_num_save_name = frame_num_save_name

  def extract_from_deep_sort_format(self,tracks,frame_num=None):
    tracks_records_in_current_frame = []
    for track in tracks:
      if not track.is_confirmed():
          continue

      # The Kalman filter estimation can return negative values for each one of x,y,w,h, if so don't include the detection in the reporter
      track_bbox = track.to_ltwh(orig=True,orig_strict=False)
      # pbx.convert_bbox rounds the bboxes to intergers hence even values smaller then 1 will fail
      if (track_bbox < 1).any():
        logging.debug(f'Track number {track.track_id} had corrupted estimation {track_bbox} therefore is discarded ')
        continue

      track_record_in_current_frame = {self.ids_save_name:track.track_id}
      # convert the current track bbox to the desired format
      track_bbox = pbx.convert_bbox(track_bbox, from_type="coco", to_type=self.bbox_save_format)
      track_record_in_current_frame.update({bbox_field:bbox_field_value for bbox_field, bbox_field_value in zip(self.bbox_save_names,track_bbox)})
      
      if self.cls_save_name is not None:
        track_record_in_current_frame[self.cls_save_name] = track.get_det_class() 
      
      if self.confidence_save_name is not None:
        track_record_in_current_frame[self.confidence_save_name] = track.get_det_conf()
      
      if self.frame_num_save_name is not None and frame_num is not None:
        track_record_in_current_frame[self.frame_num_save_name] = frame_num

      tracks_records_in_current_frame.append(track_record_in_current_frame)
    return pd.DataFrame.from_dict(tracks_records_in_current_frame)


# The trakcer outputs tracks for each frame but doesn't aggregate the tracks for the whole video, therefore I built the TrackingSummarizer
class TrackingSummarizer:
  def __init__(self,ids_save_name = 'ids',bbox_save_format='coco',bbox_save_names = None,frame_num_save_name='frame_num',cls_save_name=None,confidence_save_name=None):
    self.tracking_summary = []
    self._frame_count = 0
    self.ids_save_name = ids_save_name
    self.bbox_save_format = bbox_save_format
    self.bbox_save_names = bbox_save_names if bbox_save_names is not None else ['x','y','width','height']
    self.frame_num_save_name = frame_num_save_name
    self.cls_save_name = cls_save_name
    self.confidence_save_name = confidence_save_name

  def update(self,tracks):
    # return the current frame tracks as a dataframe and also aggregates them into the whole video tracks. 
    tracks_records_in_current_frame = []
    for track in tracks:
      if not track.is_confirmed():
          continue

      # The Kalman filter estimation can return negative values for each one of x,y,w,h, if so don't include the detection in the reporter
      track_bbox = track.to_ltwh(orig=True,orig_strict=False)
      # pbx.convert_bbox rounds the bboxes to intergers hence even values smaller then 1 will fail
      if (track_bbox < 1).any():
        logging.debug(f'Track number {track.track_id} had corrupted estimation {track_bbox} therefore is discarded ')
        continue

      track_record_in_current_frame = {self.ids_save_name:track.track_id,self.frame_num_save_name:self._frame_count}
      # convert the current track bbox to the desired format
      track_bbox = pbx.convert_bbox(track_bbox, from_type="coco", to_type=self.bbox_save_format)
      track_record_in_current_frame.update({bbox_field:bbox_field_value for bbox_field, bbox_field_value in zip(self.bbox_save_names,track_bbox)})
      
      if self.cls_save_name is not None:
        track_record_in_current_frame[self.cls_save_name] = track.get_det_class() 
      
      if self.confidence_save_name is not None:
        track_record_in_current_frame[self.confidence_save_name] = track.get_det_conf()

      self.tracking_summary.append(track_record_in_current_frame)
      tracks_records_in_current_frame.append(track_record_in_current_frame)      
    self._frame_count+=1
    return pd.DataFrame.from_dict(tracks_records_in_current_frame)
  
  def to_dataframe(self):
    return pd.DataFrame.from_dict(self.tracking_summary)


