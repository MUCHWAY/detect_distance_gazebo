"""
An example that uses TensorRT's Python api to make inferences.
"""
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy

import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import queue
import time
import math
from sort import *


CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process2(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[j])], result_scores[j]
                    ),
                )
        return batch_image_raw, end - start

    def img_infer(self, raw_image):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # t1=time.time()
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image)
        # print("pre time:",time.time()-t1)

        start = time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # end = time.time()
        # print("infer_time:",end-start)
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        # Do postprocess
        for i in range(self.batch_size):
            boxes = self.post_process(output, origin_h, origin_w)

            # Draw rectangles and labels on the original image
            #for j in range(len(result_boxes)):
            #    box = result_boxes[j]
            #    plot_one_box(box,raw_image,label="{}:{:.2f}".format(categories[int(result_classid[j])], result_scores[j]),)

        return boxes

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        t=time.time()
        #image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image = image_raw[: , : , ::-1]
        # print("BGR2RGB:",time.time()-t)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        #print("image_shape",image.shape)
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        #image /= 255.0
        image=cv2.normalize(image,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
    def post_process2(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def post_process(self, output, origin_h, origin_w):

        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        pred[:, :4] = self.xywh2xyxy(origin_h, origin_w, pred[:, :4])
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        #result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        #result_scores = boxes[:, 4] if len(boxes) else np.array([])
        #result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return boxes

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        #boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        # print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))

class videoThread:
    def __init__(self, video_path):

        self.capture=cv2.VideoCapture(video_path)

        # self.capture.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
        # self.capture.set(3,1920)
        # self.capture.set(4,1080)
        # self.capture.set(5,30)

        print("vedio:",self.capture.get(3),self.capture.get(4),self.capture.get(5))
        
        self.fourcc=cv2.VideoWriter_fourcc(*'XVID')
        self.out=cv2.VideoWriter('output.avi',self.fourcc,10,(960,540))
        self.q = queue.Queue()

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        if self.capture.isOpened():
            while True:
                ret,img = self.capture.read()
                # img= img[0:1280,0:1280]

                self.ret_flag = ret

                if not ret:
                    break
                else:
                    self.shape = img.shape
                      
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.q.put(img)
        else :
            print('vedio open failed')
    
    def read(self):
        return self.q.get()
    
    def write(self,img):
        self.out.write(img)

class crop_img():
    def __init__(self):
        pass

    def start_points(self,size,split_size, overlap=0.5):
        points = [0]
        temp=float(size)/split_size
        if size % split_size !=0:
            up=math.ceil(temp)
        else:
            up=temp+1
        down=int(temp)
        overlap=((float(split_size*up)% size)/down)/split_size
        stride = math.ceil(split_size * (1 - overlap))
        if overlap<0.08:
            up=up+1
            down=down+1
            overlap=((float(split_size*up)% size)/down)/split_size
            stride = math.ceil(split_size * (1 - overlap))
        counter = 1
        # print("overlap",overlap)
        while True:
            pt = stride * counter
            if pt + stride <= size:
                points.append(pt)
            else:
                break
            counter += 1
        return points
    
    def focus(self,box,img,size):
        xc=(box[0]+box[2])/2
        yc=(box[1]+box[3])/2
        if xc-size/2<0:
            if yc-size/2<0:
                #target_img=img(0:s,0:s)
                a=0
                b=0
            elif yc+size/2>img.shape[0]:
                #target_img=img(0:s,img.shape[0]-size:img.shape[0])
                a=0
                b=img.shape[0]-size
            else:
                #target_img=img(0:s,yc-size/2:yc+size/2)
                a=0
                b=yc-size/2
        elif xc+size/2>img.shape[1]:
            if yc-size/2<0:
                #target_img=img(img.shape[1]-size:img.shape[1],0:s)
                a=img.shape[1]-size
                b=0
            elif yc+size/2>img.shape[0]:
                #target_img=img(img.shape[1]-size:img.shape[1],img.shape[0]-size:img.shape[0])
                a=img.shape[1]-size
                b=img.shape[0]-size
            else:
                #target_img=img(img.shape[1]-size:img.shape[1],yc-size/2:yc+size/2)
                a=img.shape[1]-size
                b=yc-size/2
        else:
            if yc-size/2<0:
                #target_img=img(xc-size/2:xc+size/2,0:s)
                a=xc-size/2
                b=0
            elif yc+size/2>img.shape[0]:
                #target_img=img(xc-size/2:xc+size/2,img.shape[0]-size:img.shape[0])
                a=xc-size/2
                b=img.shape[0]-size
            else:
                #target_img=img(xc-size/2:xc+size/2,yc-size/2:yc+size/2)
                a=xc-size/2
                b=yc-size/2
        return int(a),int(b)

class Distance:
    def __init__(self):
        self.dis_x=0
        self.dis_y=0

        self.line_distance=0

        self.camera_y_angle=18
        self.camera_x_angle=30

        self.img_y_size=1080
        self.img_x_size=1920

        self.target_y_angle=0
        self.target_x_angle=0

    def distance(self,h,pitch,img_xy):
        self.target_y_angle=math.atan(img_xy[1]/(self.img_y_size/2)*math.tan(self.camera_y_angle/2/57.3))*57.3
        self.dis_y=h*math.tan((90-(pitch+self.target_y_angle))/57.3)

        self.dis_x= (math.sqrt(math.pow(h,2)+math.pow(self.dis_y,2))*img_xy[0]*math.tan(self.camera_x_angle/2/57.3))/(self.img_x_size/2)

        self.line_distance=math.sqrt(math.pow(math.sqrt( math.pow(h,2) + math.pow(self.dis_y,2) ),2)+math.pow(self.dis_x,2))


def uwb_sub(data):
    print(data)

if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "./build0127/libmyplugins.so"
    engine_file_path = "./build0127/0126best.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    categories = ["car","carb"]

    # if os.path.exists('output/'):
    #     shutil.rmtree('output/')
    # os.makedirs('output/')
    
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)

    dis=Distance()

    mot_track=Sort()

    try:
        print('batch size is', yolov5_wrapper.batch_size)
        
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()


        # cap=videoThread("./detect/test_vedio/MOT.avi")
        # cap=videoThread(0)
        # cv2.namedWindow("detect")

        vedio_=cv2.VideoCapture("./detect/test_vedio/MOT.avi")

        crop=crop_img()

        split_height = 1024
        split_width = 1024
        focus_size=1024

        # X_points = crop.start_points(1920,1024)
        # Y_points = crop.start_points(1080,1024)
        # print(X_points,Y_points) 

        X_points = [0,840]
        Y_points = [0]

        pic=0

        find_4K= not_find_1024=0

        while not rospy.is_shutdown():
            start = time.time()

            # frame = cap.read()
            # if not cap.ret_flag:
            #     break 

            rec,frame=vedio_.read()
            if not rec:
                break

            pre_combine = np.empty((0,6))  # save box in whole image

            # if(find_4K<3):
            if(1):
                for i in Y_points:
                    for j in X_points:
                        split = frame[i:i + split_height, j:j + split_width] 
                        boxes = yolov5_wrapper.img_infer(split)

                        if boxes.any():
                            boxes[:,0] += j
                            boxes[:,2] += j
                            boxes[:,1] += i
                            boxes[:,3] += i
                            pre_combine=np.vstack((pre_combine,boxes))
        
                # print("pre_shape:",pre_combine.shape,'\n')
                # print("pre:",pre_combine)

                if len(pre_combine):                
                    final_boxes=yolov5_wrapper.non_max_suppression( pre_combine , frame.shape[0],  frame.shape[1], conf_thres=0.5, nms_thres=0.4)

                    # for l in range(len(final_boxes)):

                    # track_box=np.array([[110.22, 150.23,  130.12, 170.345, 0.8] , [50.12, 30.65, 70.65, 50.43, 0.84]])
                    track_box = np.zeros(shape=final_boxes.shape)
                    track_box = final_boxes[:,0:6]

                    print('track_box: ',track_box)
                    mot_box=mot_track.update(track_box[:,0:5]) 
                    print("mot_box:",mot_box,'\n')

                    # print("final_boxes:",final_boxes)
                    target_box=final_boxes[0,:]
                    find_4K+=1

                    for j in range(len(final_boxes)):
                        # print("final_box:",final_boxes[j,])
                        img_xy=final_boxes[j,0:2]
                        # print(img_xy)
                        # dis.distance(15,25,img_xy)
                        plot_one_box(final_boxes[j,0:4],frame,label="{} {:.2f} {:.2f}m".format(categories[int(final_boxes[j,5])], final_boxes[j,4],dis.line_distance,))
                        
            # else:
            #     x1,y1=crop.focus(target_box,frame,focus_size)

            #     img=frame[y1:y1+focus_size,x1:x1+focus_size]

            #     new_boxes = yolov5_wrapper.img_infer(img)
                
            #     cv2.rectangle(frame, (x1, y1), (x1+focus_size, y1+focus_size), (0, 0, 255), 5)

            #     if new_boxes.any():
            #         new_boxes[:,0] += x1
            #         new_boxes[:,2] += x1
            #         new_boxes[:,1] += y1
            #         new_boxes[:,3] += y1

            #         target_box=new_boxes[0,:]
            #         not_find_1024=0

            #         # print("new_boxes:",new_boxes)

            #         for j in range(len(new_boxes)):
            #             # print("new_box:",new_boxes[j,])
            #             img_xy=new_boxes[j,0:2]
            #             # print(img_xy)
            #             # dis.distance(15,25,img_xy)
            #             plot_one_box(new_boxes[j,0:4],frame,label="{} {:.2f} {:.2f}m".format(categories[int(new_boxes[j,5])], new_boxes[j,4],dis.line_distance,))
            #     else:
            #         not_find_1024+=1
                
            #     if not_find_1024>4:
            #         find_4K=0 

            # im=cv2.resize(frame, (960, 540))
            # cap.write(im)
            cv2.imshow("detect",frame)
            cv2.waitKey(1)
            end = time.time()
            # print("FPS:",int(1/(end-start)))
            
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
