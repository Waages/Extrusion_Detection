### Triangulation

from picamera2 import Picamera2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"format": 'RGB888', "size": (4056, 1000)}))
#Originalformat 4056x3040
picam2.set_controls({"ExposureTime": 30000, "AnalogueGain": 15.0})
picam2.start()

cv2.startWindowThread()
im = picam2.capture_array("main")
im = cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)

height, width, depth = im.shape

save = False

yposition = np.zeros(width, dtype=int)

pixels = height*width

threshold = 70

prozent = 2

kernel = np.ones((11,11),np.uint8)
kernel_XL = np.ones((171,171),np.uint8)

edge_red = np.zeros_like(im);

right_pos = None
left_pos = None

# Fadenkreuz Parameter
mid_x = width // 2
mid_y = height // 2

thickness = 1
color = (255,128,128)

delta = 50

### 
def find_mean_position_binary(array):
	new_array = np.zeros_like(array)
	for col in range(array.shape[1]):
		positions = np.where(array[:, col] == 255)[0]
		if len(positions) > 0:
			mean_pos = int(np.round(np.median(positions)))
			
			if 0 < mean_pos < array.shape[0]:
				yposition[col] = mean_pos
				new_array[mean_pos, col] = 255
	return new_array


def find_mean_position_weighted(array):
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



while True:
	
	im = picam2.capture_array("main")
	im = cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
	
	if save:
		cv2.imwrite("images/im.png", im)
 
	red = im[:,:,2]
	grn = im[:,:,1]
	blu = im[:,:,0]

	if save:
		cv2.imwrite("images/red.png", red)
		cv2.imwrite("images/grn.png", grn)

	red16 = red.astype(np.int16)
	grn16 = grn.astype(np.int16)
	blu16 = blu.astype(np.int16)

	subtest16 = red16 - grn16# - blu16
	subtest = np.clip(subtest16, 0, 255).astype(np.uint8)
	
	if save:
		cv2.imwrite("images/sub.png", subtest)


	ret, thresh = cv2.threshold(subtest,threshold,255, cv2.THRESH_TOZERO)
	
	
	ret3, thresh3 = cv2.threshold(subtest, 0,255,cv2.THRESH_OTSU)
	
	if save:
		cv2.imwrite("images/thresh3.png", thresh3)

	opening = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
	
	dilation = cv2.dilate(opening,kernel_XL,1)
	
	filtered = np.where(dilation == 255, subtest, 0)
	
	edge = find_mean_position_weighted(filtered)
	
	edge_thick = cv2.dilate(edge,kernel,0)
	
	edge_red[:,:,0] = edge_thick
	
	im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	im_new = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
	
	
	result = cv2.add(im,edge_red)

	if save:
		cv2.imwrite("images/opening.png", opening)
		cv2.imwrite("images/dilation.png", dilation)
		cv2.imwrite("images/filtered.png", filtered)
		cv2.imwrite("images/edge.png", edge)
		cv2.imwrite("images/edge_red.png", edge_red)
		erfolg = cv2.imwrite("images/Linie.png", result)
		print("Save Result:", erfolg)

	# Fadenkreuz einzeichnen
	cv2.line(result, (mid_x,0), (mid_x, height), color, thickness)
	cv2.line(result, (0, mid_y), (width, mid_y), color, thickness)
	
	yposcenter = yposition[int(width/2)]

	# Peaks finden
	counts, bin_edges = np.histogram(yposition, bins=int(height/40))
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	padded_counts = np.pad(counts, pad_width=1, mode='constant', constant_values=0)
	peaks, properties = find_peaks(padded_counts, prominence=10)
	
	adjusted_peaks = peaks - 1
	valid_peaks = adjusted_peaks[(adjusted_peaks >= 0) & (adjusted_peaks < len(bin_centers))]

	peak_vals = bin_centers[valid_peaks]
	
	merged_peaks = merge_peaks(peak_vals,60)
 
	# Histogramm der Peaks als csv speichern
	if save:
		np.savetxt("images/histogram.csv", np.column_stack((bin_centers, counts)), delimiter=",", header="Wert,Häufigkeit", comments="")
		print("Histogramm gespeichert")
	
	for i in merged_peaks:
		cv2.line(result, (0, int(i)), (width, int(i)), (100,255,100), 3)

	if save:
		cv2.imwrite("images/peaks.png", result)

    # Nur wenn es mehr als einen Peak gibt den Obersten auswerten
	if len(merged_peaks) > 1:
		center_peak = merged_peaks[0]
		mask = np.abs(yposition - center_peak) <= 30
		indices_near_peak = np.where(mask)[0]
		xpos_centerpeak = int(np.mean(indices_near_peak))
		print(xpos_centerpeak)
		cv2.line(result, (xpos_centerpeak, 0), (xpos_centerpeak, height), (100,100,100), 3)
		left_pos, right_pos = find_first_out(yposition, center_peak, xpos_centerpeak, delta)
		print(left_pos,right_pos)
		cv2.line(result, (left_pos, 0), (left_pos, height), (100,255,100), 3)
		cv2.line(result, (right_pos, 0), (right_pos, height), (100,255,100), 3)

	if save:
		cv2.imwrite("images/breite.png", result)
	
	if right_pos is not None and left_pos is not None:
		lenght = right_pos - left_pos
	else:
		lenght = 0
	
	text = f"Mass: {lenght:.2f}°"
	
	# Text ins Bild zeichnen
	cv2.putText(
		result,						# das Bild
		text,						# der darzustellende Text
		(10, 50),					# Position (x, y) in Pixeln
		cv2.FONT_HERSHEY_SIMPLEX,	# Schriftart
		2.0,						# Schriftgröße (Scale)
		(100,255,100),				# Textfarbe (BGR – hier Rot)
		2,							# Dicke der Schrift
		cv2.LINE_AA					# Antialiasing für glatteren Text
		)
	
	cv2.imshow('Threshold', result)

	print(save)
	save = False
	
	#cv2.waitKey(0) ### Am Ende "0" (Null) als Taste drücken zum Beenden
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elif cv2.waitKey(1) & 0xFF == ord('s'):
		save = True
		print("Save:", save)

# Plot zur Veranschaulichung
plt.plot(bin_centers, counts)
plt.plot(bin_centers[valid_peaks], counts[valid_peaks], "x")
plt.xlabel("Wert")
plt.ylabel("Häufigkeit")
plt.title("Histogramm mit Peaks")
plt.show()

cv2.destroyAllWindows()

