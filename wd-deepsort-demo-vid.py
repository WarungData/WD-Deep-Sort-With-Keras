#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Untuk back compability ke python 2
from __future__ import division, print_function, absolute_import

import warnings
warnings.filterwarnings("ignore")

import os
from timeit import time
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

# Untuk algoritma Deep Sort yang diambil
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

#
# Deteksi
#
def deteksi_ds(yolo):

   # Definisikan Parameternya
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    tulis_video_output = True

   # Model DEEP SORT diambil di sini..
    namafile_model = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(namafile_model, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    video_capture = cv2.VideoCapture("demo-1.mp4")
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    #
    # Kalau ingin menulis hasil rekaman videonya buat VideoWriter object dgn codec-nya
    # sekalian tulis juga hasil deteksi ke file txt
    #
    if tulis_video_output:

        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('hasil_output2.avi', fourcc, 15, (w, h))
        list_file = open('hasil_deteksi.txt', 'w')
        frame_index = -1

    # --------------------
    # MULAI CAPTURE VIDEO
    # --------------------
    # FPS awal
    fps = 0.0
    jum_track = set()

    while True:
        ret, frame = video_capture.read()

        # Warna Gray
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # baca frame
        if ret != True:
            break

        # siapkan hitungan waktu
        t1 = time.time()

        # konversi bgr ke rgb
        image = Image.fromarray(frame[...,::-1])
        kotak = yolo.detect_image(image)

        # print("[-] Kotak : ",len(kotak))
        features = encoder(frame,kotak)

        # score to 1.0 here).
        deteksi_box = [Detection(bbox, 1.0, feature) for bbox, feature in zip(kotak, features)]

        # panggil non-maxima suppression
        boxes = np.array([d.tlwh for d in deteksi_box])
        scores = np.array([d.confidence for d in deteksi_box])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        deteksi_box = [deteksi_box[i] for i in indices]

        # Panggil class tracker
        tracker.predict()
        tracker.update(deteksi_box)

        # Tracking hasil deteksi
        for track in tracker.tracks:
            # apakah berhasil di track?
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # buat bounding box-nya
            bbox = track.to_tlbr()

            # box
            # Ini adalah prediction box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

            # teks untuk box_id
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),1)

            jum_track.add(track.track_id)

            cv2.putText(frame, "> Jumlah Orang : " + str(max(jum_track)), (10, 25), cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,255),1)

            print(">> Hits (Manusia) :", max(jum_track))

        #
        # pastikan box deteksi ada terus
        # ini adalah Detection Box
        for det in deteksi_box:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # tampilkan - DEBUG
        cv2.imshow('Test Video', frame)

        #
        # Jika tulis video?
        #
        if tulis_video_output:

            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(kotak) != 0:
                for i in range(0,len(kotak)):
                    list_file.write(str(kotak[i][0]) + ' '+str(kotak[i][1]) + ' '+str(kotak[i][2]) + ' '+str(kotak[i][3]) + ' ')
            list_file.write('\n')

        # tampilan fps
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print(">> fps= %f"%(fps))

        # tekan Q untuk stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # bersihkan video_capture
    video_capture.release()
    if tulis_video_output:
        # release video
        out.release()
        # tutup file
        list_file.close()
    # release semua
    cv2.destroyAllWindows()

# ---------------
# Start di sini
# ---------------
if __name__ == '__main__':
    deteksi_ds(YOLO())
