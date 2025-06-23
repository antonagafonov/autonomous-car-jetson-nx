#!/usr/bin/env python
import RPi.GPIO as GPIO
import time

# Test both sides with digital HIGH/LOW first
ENA = 32  # Right PWM
IN1 = 35  # Right dir 1
IN2 = 37  # Right dir 2

ENB = 33  # Left PWM
IN3 = 18  # Left dir 1
IN4 = 16  # Left dir 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT, initial=GPIO.LOW)

try:
    print("Testing RIGHT side with digital HIGH (not PWM)...")
    GPIO.output(ENA, GPIO.HIGH)  # Digital HIGH, not PWM
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(3)
    
    print("Right side stop...")
    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN1, GPIO.LOW)
    time.sleep(2)
    
    print("Testing LEFT side with digital HIGH...")
    GPIO.output(ENB, GPIO.HIGH)  # Digital HIGH, not PWM
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(3)
    
    print("Left side stop...")
    GPIO.output(ENB, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    time.sleep(2)

finally:
    GPIO.cleanup()