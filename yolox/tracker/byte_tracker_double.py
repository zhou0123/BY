import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman0 = KalmanFilter()
    shared_kalman1 = KalmanFilter()
    def __init__(self, tlwhs0,tlwhs1=None,I0=False,I1=False):

        # wait activate
        if I0:

            self._tlwh0 = np.asarray(tlwhs0, dtype=np.float)
            self._tlwh1 = None
            self.kalman_filter0,self.kalman_filter1 = None,None
            self.mean0, self.covariance0,self.mean1, self.covariance1 = None, None,None, None
            self.is_activated = False
            self.tracklet_len = 0
        if I1:

            self._tlwh1 = np.asarray(tlwhs1, dtype=np.float)
            self._tlwh0 = None
            self.kalman_filter0,self.kalman_filter1 = None,None
            self.mean0, self.covariance0,self.mean1, self.covariance1 = None, None,None, None
            self.is_activated = False
            self.tracklet_len = 0
        else:
            self._tlwh0 = np.asarray(tlwhs0, dtype=np.float)
            self._tlwh1 = np.asarray(tlwhs1, dtype=np.float)
            self.kalman_filter0,self.kalman_filter1 = None,None
            self.mean0, self.covariance0,self.mean1, self.covariance1 = None, None,None, None
            self.is_activated = False

            self.tracklet_len = 0

    def predict(self):
        mean_state0 = self.mean0.copy()
        mean_state1 = self.mean1.copy()
        if self.state != TrackState.Tracked:
            mean_state0[7] = 0
            mean_state1[7] = 0
        self.mean0, self.covariance0 = self.kalman_filter0.predict(mean_state0, self.covariance0)
        self.mean1, self.covariance1 = self.kalman_filter1.predict(mean_state1, self.covariance1)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean0 = np.asarray([st.mean0.copy() for st in stracks])
            multi_mean1 = np.asarray([st.mean1.copy() for st in stracks])
            multi_covariance0 = np.asarray([st.covariance0 for st in stracks])
            multi_covariance1 = np.asarray([st.covariance1 for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean0[i][7] = 0
                    multi_mean1[i][7] = 0
            multi_mean0, multi_covariance0 = STrack.shared_kalman0.multi_predict(multi_mean0, multi_covariance0)
            multi_mean1, multi_covariance1 = STrack.shared_kalman1.multi_predict(multi_mean1, multi_covariance1)
            for i, (mean0, cov0,mean1, cov1) in enumerate(zip(multi_mean0, multi_covariance0,multi_mean1, multi_covariance1)):
                stracks[i].mean0 = mean0
                stracks[i].mean1 = mean1
                stracks[i].covariance0 = cov0
                stracks[i].covariance1 = cov1

    def activate(self, kalman_filters, frame_id):
        """Start a new tracklet"""
        self.kalman_filter0,self.kalman_filter1 = kalman_filters[0],kalman_filters[1]
        self.track_id = self.next_id()
        self.mean0, self.covariance0 = self.kalman_filter0.initiate(self.tlwh_to_xyah(self._tlwh0))
        self.mean1, self.covariance1 = self.kalman_filter1.initiate(self.tlwh_to_xyah(self._tlwh1))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate_and(self, new_track, frame_id, new_id=False):
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(new_track.tlwh0)
        )
        self.mean1, self.covariance1 = self.kalman_filter.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(new_track.tlwh1)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
    def re_activate_I0(self, new_track, frame_id, new_id=False):
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(new_track.tlwh0)
        )
        self.mean1, self.covariance1 = self.kalman_filter.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(self.tlwh1)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
    def re_activate_I1(self, new_track, frame_id, new_id=False):
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(self.tlwh0)
        )
        self.mean1, self.covariance1 = self.kalman_filter.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(new_track.tlwh1)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update_and(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh0, new_tlwh1= new_track.tlwh0,new_track.tlwh1
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(new_tlwh0))
        self.mean1, self.covariance1 = self.kalman_filter1.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(new_tlwh1))
        self.state = TrackState.Tracked
        self.is_activated = True

    def update_I0(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh0= new_track.tlwh0
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(new_tlwh0))
        self.mean1, self.covariance1 = self.kalman_filter1.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(self.tlwh1))
        self.state = TrackState.Tracked
        self.is_activated = True
    def update_I1(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh1= new_track.tlwh1
        self.mean0, self.covariance0 = self.kalman_filter0.update(
            self.mean0, self.covariance0, self.tlwh_to_xyah(self.tlwh0))
        self.mean1, self.covariance1 = self.kalman_filter1.update(
            self.mean1, self.covariance1, self.tlwh_to_xyah(new_tlwh1))
        self.state = TrackState.Tracked
        self.is_activated = True

    @property
    # @jit(nopython=True)
    def tlwh0(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean0 is None:
            return self._tlwh0.copy()
        ret = self.mean0[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    @property
    # @jit(nopython=True)
    def tlwh1(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean1 is None:
            return self._tlwh1.copy()
        ret = self.mean1[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr0(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh0.copy()
        ret[2:] += ret[:2]
        return ret
    @property
    # @jit(nopython=True)
    def tlbr1(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh1.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah0(self):
        return self.tlwh_to_xyah(self.tlwh0)
    def to_xyah1(self):
        return self.tlwh_to_xyah(self.tlwh1)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter0 = KalmanFilter()
        self.kalman_filter1 = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        #[N,10]
        output_results = output_results.cpu().numpy()
        #AND matching
        bboxes = output_results[:,:8]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        score0 = output_results[:,8]
        score1 = output_results[:,9]

        remain_inds = (score0>self.args.track_thresh)&(score1>self.arg.track_thresh)
        inds_low = (score0>0.1)&(score1>0.1)
        inds_high = (score0<self.args.track_thresh)&(score1<self.arg.track_thresh)

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep0,scores_keep1 = score0[remain_inds],score1[remain_inds]
        scores_second0,scores_second1 = score0[inds_second],score1[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:])) for
            (tlbr) in zip(dets)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance_and(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update_and(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_and(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:])) for
            (tlbr) in zip(dets_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance_and(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_and(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_and(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        
      

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance_and(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update_and(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate([self.kalman_filter0,self.kalman_filter1],self.frame_id)
            activated_starcks.append(track)
        

        """ I0 update"""
        r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        remain_inds = score0>self.args.track_thresh - (score0>self.args.track_thresh)&(score1>self.arg.track_thresh)
        inds_low = score0>0.1
        inds_high = score0<self.args.track_thresh

        inds_second_ = np.logical_and(inds_low, inds_high) - inds_second
        dets_second = bboxes[inds_second_]
        dets = bboxes[remain_inds]
        scores_keep0 = score0[remain_inds]
        scores_second0 = score0[inds_second_]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:]),I0=True) for
            (tlbr) in zip(dets)]
        else:
            detections = []

        dists = matching.iou_distance_I0(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update_I0(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_I0(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:]),I0=True) for
            (tlbr) in zip(dets_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance_I0(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_I0(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_I0(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        

        """ I1 update"""


        r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        remain_inds = score1>self.args.track_thresh - (score0>self.args.track_thresh)&(score1>self.arg.track_thresh)
        inds_low = score1>0.1
        inds_high = score1<self.args.track_thresh


        inds_second_ = np.logical_and(inds_low, inds_high) - inds_second
        dets_second = bboxes[inds_second_]
        dets = bboxes[remain_inds]
        scores_keep1 = score1[remain_inds]
        scores_second1 = score1[inds_second_]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:]),I1=True) for
            (tlbr) in zip(dets)]
        else:
            detections = []
        
        dists = matching.iou_distance_I1(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update_I1(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_I1(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr[:4])\
            ,STrack.tlbr_to_tlwh(tlbr[4:]),I0=True) for
            (tlbr) in zip(dets_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance_I1(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_I1(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate_I1(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


       
       
       
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        """ Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
