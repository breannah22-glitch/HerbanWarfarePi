# THE FOLLOWING CODE RUNS INFERENCE ON THE RASPBERRY PI AI CAMERA
# AND STREAMS THE OUTPUT TO AN EXTERNAL BROWSER

import numpy as np
import cv2
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics
from flask import Flask, Response
import threading

latest_frame = None

# -----------------------
# YOLO MODEL
# -----------------------

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
		
# -----------------------
# FLASK SERVER STREAM
# -----------------------
		
app = Flask(__name__)

def generate_frames():
	global latest_frame
	while True:
		if latest_frame is None:
			continue
		ret, buffer = cv2.imencode('.jpg',latest_frame)
		if not ret:
			continue
			
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
		buffer.tobytes() + 
		b'\r\n')
		
@app.route('/video')
def video_feed():
	return Response(generate_frames(), 
					mimetype='multipart/x-mixed-replace; boundary=frame')
					
def start_server():
	app.run(host='0.0.0.0', port=8080, threaded=True)
	
threading.Thread(target=start_server, daemon=True).start()


# -----------------------
# CAMERA + INFERENCE LOOP
# -----------------------
		
device = AiCamera(frame_rate=15, image_size=(1080, 720))
model = YOLO()
device.deploy(model, overwrite=False)

annotator = Annotator()

with device as stream:
	for frame in stream:

		detections = frame.detections[frame.detections.confidence > 0.85]
		
		## LINES 88-90 draws bounding boxes and flips feed 180 degrees
		## Does NOT flip the bounding boxes => DONE IN ANNOTATE.PY
		
		labels = [f"dandelion: {score:0.2f}" for _, score, class_id, _ in detections]
		
		annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
		
		latest_frame = cv2.rotate(frame.image, cv2.ROTATE_180)	
		
