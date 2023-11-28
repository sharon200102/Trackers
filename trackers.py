from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.tracker import Tracker
from SoiUtils.interfaces import  Resetable, Updatable 

class ExtendedDeepSort(DeepSort,Resetable,Updatable):

    def update(self,max_iou_distance,max_age,n_init,gating_only_position,**kwargs):
        """
        Currently we only support modifications in the tracker object inside the DeepSort,
        In addition we decided that each update should begin with a reset of all existing tracks. 
        """
        self.reset()
        self.tracker.max_iou_distance = max_iou_distance
        self.tracker.max_age = max_age
        self.tracker.n_init = n_init
        self.gating_only_position.n_init = gating_only_position
        


        

    def reset(self):
        self.delete_all_tracks()
