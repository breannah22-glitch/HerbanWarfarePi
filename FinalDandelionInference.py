# # THE FOLLOWING CODE RUNS INFERENCE ON THE RASPBERRY PI AI CAMERA
# # AND STREAMS THE OUTPUT TO AN EXTERNAL BROWSER

####### TEENSY INCORPORATION #######
############# WORKING ##############


# THE FOLLOWING CODE RUNS INFERENCE ON THE RASPBERRY PI AI CAMERA
# AND STREAMS THE OUTPUT TO AN EXTERNAL BROWSER
# MOTION AND SPRAY COMMANDS ARE SENT TO TEENSY

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
# TEENSY INIT.
# -----------------------

try: # GROUND the Teensy
	TEENSY_PORT = "/dev/ttyAMA0" # change to /dev/tty/AMA0 to detect serial
	BAUD = 9600
	ser = serial.Serial(TEENSY_PORT, BAUD, timeout=0.1)
	time.sleep(2)
	print("Teensy connected on ", TEENSY_PORT)
except Exception as e:
	print("No Teensy found, running in camera-only mode: ", e)
	ser = None
	
# -----------------------
# MOTION HELPER COMMANDS
#
# send_velocity() sends linear and angular commands
# ---Maps velocity to a value between 0 and 255
# ---Converts both values into integers
# send_spray() sends spray command
# -----------------------

def send_velocity(linear,angular):
	linear = np.interp(linear, [0, 0.28], [0, 255])
	linear = int(linear)
	angular = int(angular)
	if ser:
		ser.write(f"LIN {linear} ".encode())
		ser.write(f"ANG {angular}\n".encode())
		print(f"[SER] VEL linear={linear:.3f}, angular={angular:.3f}")
	else:
		None
		print(f"[SIM] VEL linear={linear:.3f}, angular={angular:.3f}")

def send_spray():
	if ser:
		ser.write(b"PUMP\n")
	else:
		None
		#print("[SIM] SPRAY command")
		
# -----------------------
# FLASK SERVER STREAM
#
# Streams camera feed to browser with format:
# http://<IP ADDRESS>:8080/video
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
#
# Outputs angle/velocity (no teensy, only prints to command window)
# ---5 frames before accepting detection
# ---Will not spray unless the weed is centered, at the bottom of the
#    frame, and the bounding box width is a certain value
# -----------------------

fps = 0
prev_time = time.time()
		
device = AiCamera(frame_rate=15, image_size=(1080, 720))
model = YOLO()
device.deploy(model, overwrite=False)

detection_count = 0
DETECTION_THRESHOLD = 5
SPRAY_DURATION = 1.0
COOLDOWN = 0.0
spray_start_time = 0.0
spray_active = False
stop_sent = False

DANDELION_CLASS_ID = 0

# time between serial communications
VELOCITY_INTERVAL = 0.125  # seconds
last_velocity_time = 0

# tells when to set angle sent to Teensy to 0
ANGLE_THRESHOLD = 5

annotator = Annotator()


# Actual inference code below
with device as stream:
	for frame in stream:
		
		current_time = time.time()
		dt = current_time - prev_time
		prev_time = current_time
		
		if dt > 0:
			fps = 1.0/dt

		detections = frame.detections[frame.detections.confidence > 0.85]
		
		current_time = time.time()
	
		
		if spray_active: # Cooldown timer during and after spray command
			time_since_spray = current_time - spray_start_time
			if current_time - spray_start_time >= SPRAY_DURATION + COOLDOWN:
				spray_active = False
				print("[INFO] Ready for next detection")
			else: # stops movement and detections for spray
				linear = 0.0
				angular = 0.0
				send_velocity(linear,angular)
				ser = None # stops sending serial data for duration of spray
				time.sleep(1.5)
				ser = serial.Serial(TEENSY_PORT, BAUD, timeout=0.1)
				# if current_time - last_velocity_time >= VELOCITY_INTERVAL:
					# if angle_x > ANGLE_THRESHOLD:
						# send_velocity(0.0,angular)
					# else:
						# send_velocity(linear,0.0)
				print(f"[COOLDOWN] {SPRAY_DURATION + COOLDOWN - time_since_spray:.2f}s remaining")
				detections = detections[0:0]
			
		
		elif len(detections) > 0:
			detection_count += 1
			
			# ----------------------------
			# TARGETING (RUNS EVERY FRAME)
			# ----------------------------
			frame_height, frame_width = frame.image.shape[:2]
			x1, y1, x2, y2 = detections.bbox[0]
			
			x1 = int(x1*frame_width)
			y1 = int(y1*frame_height)
			x2 = int(x2*frame_width)
			y2 = int(y2*frame_height)
			
			cx = (x1+x2)/2
			cy = (y1+y2)/2
			
			error_x = cx-(frame_width/2)
			error_y = cy
			
			fov_x = 66.3 # horizontal field of view
			angle_x = (error_x/(frame_width/2))*(fov_x/2) # maps pixel position to angle
			
			# velocity decreases as the weed nears bottom of frame
			MIN_LINEAR = 0.08 # minimum speed
			MAX_LINEAR = 0.29
			
			normalized_y = error_y/frame_height
			#print("Normalized y = ", normalized_y)
			linear = MAX_LINEAR*(2*normalized_y)
			if linear < MIN_LINEAR: #ensures velocity never drops below a certain value
				linear = MIN_LINEAR
			elif linear > MAX_LINEAR: #ensures velocity never reaches above a certain value
				linear = MAX_LINEAR
			
			
			angular = round((np.radians(angle_x))*1000)
			#angular = max(min(angular,0.5), -0.5)
			linear = max(min(linear, 0.3), 0)
			
			# pause for spray
			if error_y > (frame_height-100):
				linear = 0.0
				
			centered = abs(error_x) < 20
			if centered:
				bbox_width = (x2-x1)
				#print(f"width={bbox_width}")
			else:
				bbox_width = None
			
			# range of bbox_width values
			MIN_WIDTH = 270
			MAX_WIDTH = 290
			
			# SPRAY if detection count reached, centered, at bottom of frame
			# and if the bounding box is a certain size (distance)
			if detection_count >= DETECTION_THRESHOLD and abs(error_x) < 50 and error_y < 100: # and MIN_WIDTH <= bbox_width <= MAX_WIDTH:
				# send_velocity(0.0,0.0)
				send_spray()
				spray_active = True
				spray_start_time = current_time
				detection_count = 0
			else:
				if current_time - last_velocity_time >= VELOCITY_INTERVAL:
					if angle_x > ANGLE_THRESHOLD:
						send_velocity(0.0,angular)
					else:
						send_velocity(linear,0.0)
					last_velocity_time = current_time
			
			#print(f"[DEBUG] angle_x={angle_x:.2f}, linear={linear:.2f}")
			cv2.circle(frame.image, (int(cx), int(cy)), 5, (0, 255, 0), -1)
		else:
			detection_count = 0
			if current_time - last_velocity_time >= VELOCITY_INTERVAL:
					send_velocity(0.29,0.0)
					last_velocity_time = current_time #roaming velocity
			
		## Draws bounding boxes and flips feed 180 degrees
		## Does NOT flip the bounding boxes => DONE IN ANNOTATE.PY
		
		labels = [f"dandelion: {score:0.2f}" for _, score, class_id, _ in detections]
		
		annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
		
		font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
		fps_text = f"FPS: {fps:.1f}"

		(text_w, text_h), bl = cv2.getTextSize(fps_text, font, scale, thick)

		patch = np.zeros((text_h + bl, text_w, 3), dtype=np.uint8)

		cv2.putText(patch, fps_text, (0, text_h), font, scale, (255, 255, 255), thick)

		# flip it BEFORE putting on frame
		flipped_patch = cv2.rotate(patch, cv2.ROTATE_180)

		# place it where it will appear top-left AFTER rotation
		h, w, _ = frame.image.shape
		tx = w - text_w - 10
		ty = h - 10

		try:
		    frame.image[ty-(text_h+bl):ty, tx:tx+text_w] = flipped_patch
		except ValueError:
		    pass
		
		latest_frame = cv2.rotate(frame.image, cv2.ROTATE_180)
