import RPi.GPIO as GPIO
import time
from gpiozero import CPUTemperature
import os

fanpin = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(fanpin, GPIO.OUT)

pwm = GPIO.PWM(fanpin, 100)  # Set frequency to 100Hz
pwm.start(0)  # Start PWM with 0% duty cycle


solltemp = 55  # Desired temperature in Celsius

# Short Fan Burst to indicate funcionality
pwm.ChangeDutyCycle(100)
time.sleep(2)
pwm.ChangeDutyCycle(0)
print("CPU Fan Control gestartet, Solltemp:", solltemp, "°C")

# Beginn Temp controller
while True:
	cputemp = CPUTemperature().temperature
	#print(f"CPU Temperature: {cputemp}°C")
	diff = cputemp - solltemp
	#print("\r                                                                    ", end="", flush=True)
	if diff > 0:
		duty_cycle = min(100, max(15, diff * 10))
		pwm.ChangeDutyCycle(duty_cycle)
		#print(f"\rCPU Temperature: {cputemp}°C, Fan speed: {round(duty_cycle,2)}%", end="", flush=True)
	else:
		pwm.ChangeDutyCycle(0)
		#print(f"\rCPU Temperature: {cputemp}°C, Fan off", end="", flush=True)
	time.sleep(0.2)
