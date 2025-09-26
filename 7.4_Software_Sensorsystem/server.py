from picamera2 import Picamera2
import cv2
import time
import numpy as np
from scipy.signal import find_peaks
from time import gmtime, strftime
from flask import Flask, render_template, Response, jsonify, request
from gpiozero import CPUTemperature
import os
import RPi.GPIO as GPIO

# Kamera initialisieren
camheight = 4056	# 4056
camwidth = 900	# 3040
fps_faktor = 2
picam2 = Picamera2()
print("---")
for i, mode in enumerate(picam2.sensor_modes):
	print(f"Mode {i}: {mode}")
	print("---")
mode = picam2.sensor_modes[0]

size = mode['size']

# Kamera Preset High Quality
hq_config = picam2.create_video_configuration(
	main={
		"size": (camheight,camwidth),
		"format": "RGB888"
	},
	controls={
		"FrameRate": 120,
		"ExposureTime": 3000,
		"AnalogueGain": 10.0,
	},
	sensor={
		"bit_depth": 12,
		"output_size": (camheight,camwidth),
	}
)

# Kamera Preset High FPS
fps_config = picam2.create_video_configuration(
	main={
		"size": (int(camheight/fps_faktor),int(camwidth/fps_faktor)),
		"format": "RGB888"
	},
	controls={
		"FrameRate": 120,
		"ExposureTime": 3000,
		"AnalogueGain": 10.0,
	},
	sensor={
		"bit_depth": 12,
		"output_size": (int(camheight/fps_faktor),int(camwidth/fps_faktor)),
	}
)

picam2.configure(hq_config)
picam2.start()

# GPIO fuer Laser initialisieren
Laserpin = 22
GPIO.setmode(GPIO.BCM)
GPIO.setup(Laserpin, GPIO.OUT)

# Flask-App
app = Flask(__name__)

# Variablen
framerate = 42
temperatur = 0
text = "Online"
zaehler = 0
laser_status = False
record_status = False
fps_status = False
fps_change = False
video_writer = None
detect_width = 0
detect_baseheight = 0
detect_angle = 0
save_height = 0
baseheight_pixel = 0

yposition = np.zeros(camwidth, dtype=int)

height = camheight
width = camwidth


def generate_frames():
	"""Frames fuer den Stream generieren"""
	global framerate, temperatur, record_status, video_writer, height, width, detect_width, detect_baseheight, detect_angle, fps_change, save_height, baseheight_pixel
	print(height, width)
	#Programmvariablen
	kernel = np.ones((11,11),np.uint8)
	kernel_XL = np.ones((171,171),np.uint8)
	edge_red = np.zeros((height, width,3),np.uint8)
	delta = 40
 
	left_pos, right_pos = None, None
 
	start_time = time.time()

	timestamp = "999"
	
	ordner = strftime("%d %b %Y", gmtime())
	if not os.path.exists(os.path.join("videos", ordner)):
		os.makedirs(os.path.join("videos", ordner))

	while True:
		# Frame von der Kamera holen
		frame = picam2.capture_array("main")
  		  
    	# Frame rotieren und Masse extrahieren
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		height, width, _ = frame.shape

		# Bei FPS-Wechsel den edge-Array zuruecksetzen
		if fps_change:
			edge_red = np.zeros((height, width,3),np.uint8)
			fps_change = False

		# Einzelne Farbkanaele extrahieren
		red = frame[:,:,2]
		grn = frame[:,:,1]
		blu = frame[:,:,0]

		# Gruen von Rot subtrahieren
		sub = cv2.subtract(red, grn)

		# OTSU-Threshold anwenden
		ret3, thresh3 = cv2.threshold(sub, 0, 255, cv2.THRESH_OTSU)
  
		# Rauschen unterdruecken durch kleines Opening
		opening = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
  
		# Grosse Regionen rund um erkannte Kanten definieren
		dilation = cv2.dilate(opening,kernel_XL,1)

		# Vom Bild nur diese Regionen behalten
		filtered = np.where(dilation == 255, sub, 0)

		# Kante erkennen und fuer Darstellung aufdicken
		edge = find_mean_position_weighted(filtered)
		edge_thick = cv2.dilate(edge,kernel,0)
		edge_red[:,:,0] = edge_thick

		# Winkelmessung der Linie
		detect_angle = measure_angle(edge)

		result = cv2.add(frame,edge_red)

		counts, bin_edges = np.histogram(yposition, bins=int(height/50))
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
		padded_counts = np.pad(counts, pad_width=1, mode='constant', constant_values=0)
		peaks, properties = find_peaks(padded_counts, prominence=10)
		adjusted_peaks = peaks - 1
		valid_peaks = adjusted_peaks[(adjusted_peaks >= 0) & (adjusted_peaks < len(bin_centers))]

		peak_vals = bin_centers[valid_peaks]
	
		merged_peaks = merge_peaks(peak_vals,60)
		
		baseheight_pixel = find_base_height(yposition,int(merged_peaks[0]),10)
		# FPS-Modus beruecksichtigen!
		detect_baseheight = round(calculate_height(baseheight_pixel, save_height),2)
		#print(yposition)

		print(merged_peaks)

		if len(merged_peaks) > 1:
			center_peak = merged_peaks[0]
			mask = np.abs(yposition - center_peak) <= 30
			indices_near_peak = np.where(mask)[0]
			xpos_centerpeak = int(np.mean(indices_near_peak))
			cv2.line(result, (xpos_centerpeak, 0), (xpos_centerpeak, height), (100,100,100), 3)
			left_pos, right_pos = find_first_out(yposition, center_peak, xpos_centerpeak, delta)
			cv2.line(result, (left_pos, 0), (left_pos, height), (100,255,100), 3)
			cv2.line(result, (right_pos, 0), (right_pos, height), (100,255,100), 3)

		for i in merged_peaks:
			cv2.line(result, (0, int(i)), (width, int(i)), (100,255,100), 3)
   
		if right_pos is not None and left_pos is not None:
			width_pixels = right_pos - left_pos
		else:
			width_pixels = 0

		detect_width = round(calculate_width(width_pixels),2)

		# Aufnahme und Speichern des Kamerabildes
		if record_status:
			if video_writer is None:
				timestamp = time.asctime()
				video_path = os.path.join("videos", ordner, timestamp + '.avi')
				height, width, _ = frame.shape
				fourcc = cv2.VideoWriter_fourcc(*'XVID')
				video_writer = cv2.VideoWriter(video_path, fourcc, framerate, (width, height))
			video_writer.write(frame)	
		else:
			if video_writer is not None:
				video_writer.release()
				video_writer = None

   		# Frame zu JPEG kodieren
		ret, buffer = cv2.imencode('.jpg', result)
		result = buffer.tobytes()

		# Framerate berechnen
		elapsed = time.time() - start_time
		start_time = time.time()
		framerate = round(1 / elapsed,1)

		# Frame fuer HTTP-Stream formatieren
		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n')

def calculate_height(height_raw,zeropos):
	global fps_status
	if fps_status:
		faktor = -0.0032
	else:
		faktor = -0.0064
	# y = -0,0064x + 14,048
	# Wenn gleich, dann relative Distanz 0
	if (height_raw == zeropos):
		height_in_mm = 0
	else:
		delta = height_raw - zeropos
		# Linear -> Aufloesung auf Halber Strecke
		resolution = faktor*(zeropos+(delta/2)) + 14.048
		#print(resolution)
		height_in_mm = delta / resolution
	return height_in_mm

def calculate_width(width_raw):
	global fps_status
	if fps_status:
		faktor = 0.0464
	else:
		faktor = 0.0232
	# x = 0,0232x
	width_in_mm = width_raw * faktor
	return width_in_mm

def find_base_height(ypositions, height, thresh):
	i = 0
	base_height = 0
	for val in ypositions:
		if abs(val - height) < thresh:
			base_height = base_height + val
			i = i + 1
	if (i == 0):
		print("i=0")
		i = 1
	base_height = round(base_height/i,2)
	return base_height

def find_mean_position_weighted(array):
	global yposition
	new_array = np.zeros_like(array)
	height, width = array.shape
 
	for col in range(width):
		column = array[:, col].astype(np.float32)
		weights = column
		positions = np.arange(height)

		weight_sum = np.sum(weights)
		if weight_sum > 0:
			weighted_mean = int(np.round(np.sum(positions * weights) / weight_sum))
			if 0 <= weighted_mean < height:
				yposition[col] = weighted_mean
				new_array[weighted_mean, col] = 255
	return new_array

def measure_angle(array):
	y_coords, x_coords = np.where(array == 255)
	if y_coords.size > 0:
		slope, intercept = np.polyfit(x_coords, y_coords, 1)
		angle_rad = np.arctan(slope)
		angle_deg = np.degrees(angle_rad)
		angle_deg = round(angle_deg,2)
	else:
		angle_deg = "---"
	return angle_deg

def merge_peaks(peaks, distance):
	if len(peaks) == 0:
		return np.array([])
	
	peaks = np.sort(peaks)
	merged_peaks = []
	group = [peaks[0]]
	
	for p in peaks[1:]:
		if p - group[-1] <= distance:
			group.append(p)
		else:
			merged_peaks.append(np.mean(group))
			group = [p]
			
	merged_peaks.append(np.mean(group))
	
	return np.array(merged_peaks)

def find_first_out(array,ypos,mid,delta):
	mid
	
	first_left = None
	for i in range(mid, -1, -1):
		if abs(array[i] - ypos) > delta:
			first_left = i
			break
	
	first_right = None
	for i in range(mid, len(array)):
		if abs(array[i] - ypos) > delta:
			first_right = i
			break
	return first_left, first_right

def toggle_laser():
	global laser_status, Laserpin
	if laser_status:
		GPIO.output(Laserpin,False)
		laser_status = False
	else:
		GPIO.output(Laserpin,True)
		laser_status = True
	print(laser_status)

def toggle_record():
	global record_status
	if record_status:
		record_status = False
	else:
		record_status = True
	print(record_status)
 
def toggle_fps():
	global fps_status, fps_change
	fps_change = True
	if fps_status:
		picam2.stop()
		picam2.configure(hq_config)
		picam2.start()
		fps_status = False
	else:
		picam2.stop()
		picam2.configure(fps_config)
		picam2.start()
		fps_status = True
	print(fps_status)

def save_hoehe():
	global baseheight_pixel, save_height
	save_height = baseheight_pixel
    
@app.route('/')
def index():
	"""Hauptseite mit Video"""
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	"""Video-Stream"""
	return Response(generate_frames(),
				mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/variables')
def get_variables():
	global zaehler, temperatur, laser_status, record_status, detect_width, detect_baseheight, detect_angle, fps_status
	zaehler +=1

	cpu = CPUTemperature()
	temperatur = round(cpu.temperature,1)

	return jsonify({
		'framerate': framerate,
		'temperatur': temperatur,
		'text': text,
		'zaehler': zaehler,
		'laser_status': laser_status,
		'record_status': record_status,
		'fps_status': fps_status,
		'timestamp': time.time(),
		'detect_width': detect_width,
		'detect_baseheight': detect_baseheight,
		'detect_angle': detect_angle
	})

@app.route('/api/button', methods=['POST'])
def button_pressed_handler():
	global text, laser_status, record_status, fps_status

	print("Button_Handler gestartet")
	try:
		data = request.get_json()
		print(data)
		if data and 'action' in data:
			if data['action'] == 'press':
				text = "Button gedrueckt"
				toggle_laser()
				print("Button wurde gedrueckt")
				return jsonify({
					'status': 'success',
					'message': 'Button wurde registriert',
					'laser_status': laser_status,
				})
    
			elif data['action'] == 'record':
				text = "Record gedrueckt"
				toggle_record()
				print("Record wurde gedrueckt")
				return jsonify({
					'status': 'success',
					'message': 'Button wurde registriert',
					'record_status': record_status,
				})
			elif data['action'] == 'fps':
				text = "fps gedrueckt"
				toggle_fps()
				print("fps wurde gedrueckt")
				return jsonify({
					'status': 'success',
					'message': 'Button wurde registriert',
					'fps_status': fps_status,
				})

			elif data['action'] == 'hoehe':
				text = "hoehe gedrueckt"
				save_hoehe()
				print("hoehe wurde gedrueckt")
				return jsonify({
					'status': 'success',
					'message': 'Button wurde registriert',
				})

		return jsonify({'status': 'error', 'message': 'Ungueltige Aktion'})

	except Exception as e:
		print(f"Fehler beim Button-Handler: {e}")
		return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
	print("Starte einfachen Kamera-Server...")
	print("IP-Adresse + Port: http://192.168.4.1:5000")

	# Server starten
	app.run(host='0.0.0.0', port=5000, debug=False)