#!/usr/bin/env python
# coding: utf-8


import cv2 as cv
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader as sl

class YOLO_PRED():
    def __init__(self, onnx_model, data_yaml):
        
        # 1. Load the YAML file ----file data.yaml
        with open(data.yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=sl)
        # call names and number of classes from the data.yaml file -- this is what will be required 
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']


        # 1.1 Load YOLO Model
        self.yolo = cv.dnn.readNetFromONNX(onnx_model)
        # Set the backend ---since we trained in GPU environment so we need to specify that we need to use target CPU
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        
    def predictions(self, image):
        
        # 2. Load the image for testing the model
                 
        rows,cols,d = self.image.shape        

        # 2.1 Get the yolo predictions from the image 

        # Step 1:convert the image into square image (array)
        max_rc = max(rows,cols)
        input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)    
        input_image[0:rows,0:cols] = self.image
        INPUT_WH_YOLO = 640
        # find blob from image 
        blob = cv.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)      
        # set the yolo input 
        self.yolo.setInput(blob)
        # predictions from yolo model
        preds = self.yolo.forward()


        #3. NON MAXIMUM SUPPRESSION:

        #step 1: filter detection based on confidence score (0.4) and probability score (0.25)
                # also take values of center_X, center_y,w,h and reconvert that into original values xmin,xmax,ymin,ymax
        # Flatten the preds 
        detections = preds[0]
        # empty list to store the bounding boxes info, confidence score and prob score of each class
        boxes = []
        confidences = []
        classes = []

        # width and height of the image(input)
        image_w, image_h = input_image.shape[:2]
        # using the input image to get the x factor --factor with which to multiply the bounding box 
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        # filter 
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]       # confidence score for detecting an object 
            if confidence > 0.4:
                class_score = row[5:].max()    
                class_id = row[5:].argmax()     

                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    # construct bounding box from teh above four values 
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left, top, width, height])
                    # append values into the list 
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)


        # Cleaning 
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS --- non maximum suppression  -- params (boxes, score, score_threshold, nms_threshold)
        index = cv.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()        

        # from the index position that we get from above 
        # 4. Draw the bounding boxes 

        for i in index:
            # Extract bounding boxes
            x,y,w,h = boxes_np[i]
            # Extractthe confidences 
            bb_conf = int(confidences_np[i]*100)
            # extract the class
            class_id = classes[i]
            class_name = self.labels[class_id]
            colors = self.generate_colors(class_id)

            text = f'{class_name}: {bb_conf}%'
            print(text)

            cv.rectangle(self.image,(x,y),(x+w,y+h),colors,2)
            cv.rectangle(self.image,(x,y-30),(x+w,y),colors,-1)
            cv.putText(self.image,text,(x,y-10),cv.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
            
        return self.image
    
    def generate_colors(self,ID):
        np.random.seed(10)
        # number of colors to generate is equal to the number of classes --- information from data_yaml file 
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
                                   



