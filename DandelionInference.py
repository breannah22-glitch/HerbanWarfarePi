import numpy as np
import cv2
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics
from flask import Flask, Response
import threading
import time
import serial
import socket
import glob

time.sleep(2)

latest_frame = None
COOLDOWN = 3.0
SPRAY_DURATION = 2.5
spraying = False
last_spray_time = 0.0

current_cmd = "STOP"
last_cmd_time = 0
CMD_TIMEOUT = 0.3

### LINES 29 - 53: TELEOP CODE (TEENSY REQUIRED)

# def teleop_server():
# 	global current_cmd, last_cmd_time
	
# 	UDP_PORT = 5005
# 	BAUD = 115200
	
# 	ports = glob.glob("/dev/ttyACM*")
# 	if not ports:
# 		raise Exception("No Teensy found!")
		
	
# 	ser = serial.Serial(ports[0], BAUD, timeout=0.1)
	
# 	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 	sock.bind(("0.0.0.0", UDP_PORT))
	
# 	print("Teleop server running")
	
# 	while True:
# 		data, adr = sock.recvfrom(1024)
# 		cmd = data.decode().strip()
		
# 		current_cmd = cmd
# 		last_cmd_time = time.time()
		
# 		ser.write((cmd + "\n").encode())

class YOLO(Model):
	
	def __init__(self):
		super().__init__(
			model_file="best_Jan29_imx_model/packerOut.zip",
			model_type=MODEL_TYPE.CONVERTED,
			color_format=COLOR_FORMAT.RGB,
			preserve_aspect_ratio=False,
		)
		
		self.labels = np.genfromtxt(
			"best_Jan29_imx_model/labels.txt",
			dtype=str,
			delimiter="\n",
		)
		
	def post_process(self, output_tensors):
		return pp_od_yolo_ultralytics(output_tensors)
		
app = Flask(__name__)

def generate_frames():
	global latest_frame
	while True:
		if latest_frame is None:
			continue
		ret, buffer = cv2.imencode('.jpg',latest_frame)
		if not ret:
			continue
			
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
		
@app.route('/video')
def video_feed():
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
		
device = AiCamera(frame_rate=15, image_size=(1080, 720)) # Initialize framerate and resolution
model = YOLO()
device.deploy(model, overwrite=False)

annotator = Annotator()

def start_server():
	app.run(host='0.0.0.0', port=8080, threaded=True)
	
threading.Thread(target=start_server, daemon=True).start()


#teleop_thread = threading.Thread(target=teleop_server, daemon=True) # Streams to browser
#teleop_thread.start()

with device as stream: # INFERENCE HAPPENS HERE
	for frame in stream:

		detections = frame.detections[frame.detections.confidence > 0.85]
		labels = [f"dandelion: {score:0.2f}" for _, score, class_id, _ in detections]
		
		annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10) # DRAWS BOUNDING BOX (/herbie/home/.local/lib/modlib/apps/annotator.py)
		
		
		latest_frame = cv2.rotate(frame.image.copy(), cv2.ROTATE_180) # FLIPS CAMERA FEED 180 DEGREES
		
		if time.time() - last_cmd_time > CMD_TIMEOUT:
			current_cmd = "STOP"
		
		#frame.display()


### UNCOMMENT FOR CAMERA HOLD ON WEED DETECTION -> DRIVE MOTORS

#with device as stream:
#	for frame in stream:
#		current_time = time.time()
#		
#		if current_time - last_spray_time > SPRAY_DURATION + COOLDOWN:
#			
#			detections = frame.detections[frame.detections.confidence > 0.85]
#			labels = [f"dandelion: {score:0.2f}" for _, score, class_id, _ in detections]
#		
#			annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
#		
#		
#			latest_frame = frame.image.copy()
#			
#			if len(detections) > 0:
#				print("Dandelion detected! Sending SPRAY command...")
#				teensy.write(b'SPRAY\n')
#				last_spray_time = time.time()
				
