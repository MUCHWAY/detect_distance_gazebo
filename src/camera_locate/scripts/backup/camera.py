import cv2
import datetime
import numpy as np
import threading
from time import sleep

FPS=12
fourcc = cv2.VideoWriter_fourcc(*"mp4v")         #MPEG-4
OUTPATH=''
INPATH=0

mtx=np.mat([[2.70657664e+03,0.00000000e+00,1.95982888e+03],
            [0.00000000e+00,2.78588012e+03,1.15432068e+03],
            [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist=np.mat([[ 0.0262969,-0.04752638,0.00634423,-0.01886641,0.11200369]])
newcameramtx=np.mat([[2.79422363e+03,0.00000000e+00,1.92286828e+03],
                    [0.00000000e+00,2.96108154e+03,1.23570756e+03],
                    [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

def open_cam_rtsp(uri, width, height, latency):
    import subprocess
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink max_buffers=0 drop=true').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

#cap=cv2.VideoCapture(INPATH)
#cap=cv2.VideoCapture('rtsp://admin:admin@192.168.42.108:554/cam/realmonitor?channel=1&subtype=0',cv2.CAP_FFMPEG)
cap=open_cam_rtsp('rtsp://admin:admin@192.168.42.108:554/cam/realmonitor?channel=1&subtype=0',4096,2160,0)
ret,frame=cap.read()
if not ret:
    print('Unable to capture video')
    exit()
height,width,_=frame.shape
start=datetime.datetime.now()

isRecording=False
isWriting=False
isEnding=False
video=None

print('Press R to start/stop a record')
print('Press Q to quit')

def runByThread():
    global isRecording
    global isEnding
    global video
    while True:
        frameCp=frame.copy()
        #frameCp=cv2.undistort(frameCp, mtx, dist, None, newcameramtx)

        if isRecording:
            secs=int((datetime.datetime.now()-start).total_seconds())
            cv2.putText(frameCp,('@' if (secs % 2)==0 else ''),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            cv2.putText(frameCp,'REC',(70,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            cv2.putText(frameCp,str(datetime.datetime.now()-start).split('.')[0],(10,height-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)

        show=cv2.resize(frameCp,(960,540))
        cv2.imshow("camera",show)
        
        key=cv2.waitKey(5) & 0xff
        if key==ord('q') or key==ord('Q'):
            if isRecording:
                while isWriting:
                    sleep(0.1)
                video.release()
                print(' --stop <saved as '+OUTPATH+str(start).replace(':','-')+'.mp4'+'>')
            print(' --quit')
            isEnding=True
            break
        elif key==ord('r') or key==ord('R'):
            if isRecording:
                isRecording=False
                while isWriting:
                    sleep(0.01)
                video.release()
                print(' --stop <saved as '+OUTPATH+str(start).replace(':','-')+'.mp4'+'>')
            else:
                start=datetime.datetime.now()
                video = cv2.VideoWriter(OUTPATH+str(start).replace(':','-')+'.mp4', fourcc, FPS, (width,height))
                print('Recording...')
                isRecording=True

#tr = threading.Thread(target=runByThread, args=())
#tr.setDaemon(True)
#tr.start()

while not isEnding:
    _,frame=cap.read()
    if isRecording:
        isWriting=True
        img=np.uint8(frame)
        video.write(img)
        isWriting=False

cap.release()
cv2.destroyAllWindows()
