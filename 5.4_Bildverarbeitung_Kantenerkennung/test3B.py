from picamera2 import Picamera2
import numpy as np
import time
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"format": 'RGB888', "size": (4056, 500)}))
#Originalformat 4056x3040
picam2.set_controls({"ExposureTime": 45000, "AnalogueGain": 8.0})
picam2.start()

cv2.startWindowThread()
im = picam2.capture_array("main")


height, width, depth = im.shape
edges_red = np.zeros((height,width, 3), dtype=np.uint8)


# Funktionen
def normal_distance(point, line_point, line_angle_deg):
    """
    Berechnet den normalen Abstand eines Punktes von einer Geraden,
    die durch 'line_point' verläuft und den Winkel 'line_angle_deg' hat.
    """
    angle_rad = np.deg2rad(line_angle_deg)
    # Richtungsvektor der Linie
    line_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    # Normalenvektor (senkrecht auf line_dir)
    normal_vec = np.array([-line_dir[1], line_dir[0]])
    
    vec = np.array(point) - np.array(line_point)
    distance = abs(np.dot(vec, normal_vec))
    return distance

def merge_lines(lines, glob_angle, max_distance):
	# Liste der gemergten Linien
	merged = []
	# Checkliste, ob Linie verwendet wurde (standardmäßig erstmal alle Nein)
	used = [False] * len(lines)
	
	
	for i, line_i in enumerate(lines):
		# Wenn Linie schon gemerged wurde, dann skippen
		if used[i]:
			continue
		
		# Aufpunkt definieren: Startpunkt der Basis-Linie
		ref_point = np.array(line_i['start'])
		# Liste aller Punkte, erstmal nur Aufpunkt
		group_points = [ref_point]
		# Basis-Linie auf Verarbeitet markieren
		used[i] = True
		
		for j, line_j in enumerate(lines):
			if used[j]:
				continue
			
			point_j = np.array(line_j['start'])
			dist = normal_distance(point_j, ref_point, mean_angle)
			
			if dist < max_distance:
				group_points.append(point_j)
				used[j] = True
			
		new_point = np.mean(group_points, axis=0)
		
		merged.append({
			'start': tuple(new_point),
			'angle': mean_angle
		})
	return merged

# Fadenkreuz Parameter
mid_x = width // 2
mid_y = height // 2

thickness = 3
color = (128,128,128)

print(im.shape)

threshold1 = 100
threshold2 = 180
Exposure = 30000
min_length = 60

start_time_total = time.time()

zaehler = 0
# Maximaler Winkelunterschied zum mergen
max_deviation = 2
# Maximaler Linienabstand zum mergen
max_dist = 15

messung = 0

while True:
	zaehler += 1
	start_time = time.time()
	# Bild aufnehmen
	im = picam2.capture_array("main")
	# Bild drehen
	im = cv2.rotate(im,cv2.ROTATE_180)

	# Kantenerkennung über Bild laufen lassen
	edges = cv2.Canny(im, threshold1, threshold2)

	# Kanten als rot in RGB umwandeln
	edges_red[:,:,2] = edges
	
	# Linienerkennung mit HoughLines
	lines_raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
	lines = []

	# Neue Linien-Liste anlegen mit Linienwinkel
	if lines_raw is not None:
		for line in lines_raw:
			x1, y1, x2, y2 = line[0]
			angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
			lines.append({
				'start': (x1, y1),
				'end': (x2, y2),
				'angle': angle
			})
			
	# Mittelwert der Winkel bestimmen
	angles = np.array([line['angle'] for line in lines])
	mean_angle = np.mean(angles)

	# Linien mit großer Winkelabweichung rausfiltern
	filtered_lines = [line for line in lines if abs(line['angle'] - mean_angle) <= max_deviation]
	if filtered_lines is not None:
		for line in filtered_lines:
			cv2.line(im, line['start'], line['end'], (255, 0, 0), 2)

	# Mittelwert Winkel nur von gefilterten Linien neu bestimmen
	angles = np.array([line['angle'] for line in filtered_lines])
	mean_angle_new = np.mean(angles)
	
	
	# Linien mergen
	while True:
		new_lines = merge_lines(lines, mean_angle_new, max_dist)
		if len(new_lines) == len(lines):
			break
		lines = new_lines
		
	lenght = 1000
		
	# Linien einzeichnen
	for line in new_lines:
		start = np.array(line['start'])
		angle_rad = np.deg2rad(line['angle'])
		direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
		
		end = start + direction * lenght
		start_point = tuple(start.astype(int))
		end_point = tuple(end.astype(int))
		
		cv2.line(im, start_point, end_point, (0, 255, 0), 2)

	
	# Farbbild und rote Kanten addieren
	result = cv2.add(im,edges_red)
	
	# Fadenkreuz einzeichnen
	cv2.line(result, (mid_x,0), (mid_x, height), color, thickness)
	cv2.line(result, (0, mid_y), (width, mid_y), color, thickness)
	
#	result = cv2.medianBlur(result, 5)

	if len(new_lines) == 2:
		messung = normal_distance(new_lines[1]['start'],new_lines[0]['start'],new_lines[0]['angle'])
		aktuell = True
	else:
		aktuell = False
		
	
	text = f"Abstand: {messung:.2f}°"

	if aktuell:
		Farbe = (0,0,0)
	else:
		Farbe = (0,0,255)

	# Text ins Bild zeichnen
	cv2.putText(
		result,						# das Bild
		text,						# der darzustellende Text
		(10, 50),					# Position (x, y) in Pixeln
		cv2.FONT_HERSHEY_SIMPLEX,	# Schriftart
		2,							# Schriftgröße (Scale)
		Farbe,						# Textfarbe (BGR – hier Rot)
		2,							# Dicke der Schrift
		cv2.LINE_AA					# Antialiasing für glatteren Text
		)
	
	# Bild anzeigen
	cv2.imshow("Test", result)
	
	
	#cv2.waitKey(0) ### Am Ende "0" (Null) als Taste drücken zum Beenden
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord('q'):
		print("Programm beendet")
		break
		
	elif key == ord('i'):
		threshold1 +=1
		print("Threshold1 ++")
		print(threshold1)
	elif key == ord('k'):
		threshold1 -=1
		print("Threshold1 --")
		print(threshold1)	
	elif key == ord('o'):
		threshold2 +=1
		print("Threshold2 ++")
		print(threshold2)
	elif key == ord('l'):
		threshold2 -=1
		print("Threshold2 --")
		print(threshold2)	
	elif key == ord('u'):
		Exposure +=1000
		picam2.set_controls({"ExposureTime": Exposure})
		print("Exposure ++")
		print(Exposure)
	elif key == ord('j'):
		Exposure -=1000
		picam2.set_controls({"ExposureTime": Exposure})
		print("Exposure --")
		print(Exposure)
	elif key == ord('z'):
		min_length +=1
		print("Min Length ++")
		print(min_length)
	elif key == ord('h'):
		min_length -=1
		print("Min Length --")
		print(min_length)
		
	end_time = time.time()
	duration = end_time - start_time

end_time_total = time.time()

freq_total = zaehler / (end_time_total-start_time_total) 

print(zaehler)

print(lines)

print(f"Frequenz total: {freq_total:.4f} Hz")
#np.savetxt('output.csv', edges, delimiter=',', fmt='%d')

cv2.destroyAllWindows()
