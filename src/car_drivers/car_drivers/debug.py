#!/usr/bin/env python
import RPi.GPIO as GPIO
import time

# Right side motors (Motor B) - UPDATED PINS
ENA = 32  # Right side PWM enable (Pin 32 - true PWM pin)
IN1 = 35  # Right side direction 1
IN2 = 37  # Right side direction 2

# Left side motors (Motor A) - UNCHANGED  
ENB = 15  # Left side PWM enable (Pin 33 - works fine)
IN3 = 18  # Left side direction 1
IN4 = 16  # Left side direction 2

# Set pin numbers to the board's
GPIO.setmode(GPIO.BOARD)
time.sleep(2)

# Initialize all pins
print("Setting up GPIO pins...")
GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)  # Right enable (Pin 32)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)  # Right dir 1
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)  # Right dir 2
GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)  # Left enable (Pin 33)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)  # Left dir 1
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)  # Left dir 2

# Create PWM objects at 25Hz frequency
pwm_right = GPIO.PWM(ENA, 25)  # Pin 32 - hardware PWM
pwm_left = GPIO.PWM(ENB, 25)   # Pin 33 - hardware PWM
pwm_right.start(0)  # Start with 0% duty cycle
pwm_left.start(0)   # Start with 0% duty cycle

try:
    print("\nTesting LEFT SIDE motors...")
    # Enable left side with 25% PWM
    pwm_left.ChangeDutyCycle(25)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(2)
    
    # Left side forward
    print("Left side FORWARD for 5 seconds...")
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(5)
    
    # Left side stop
    print("Left side STOP...")
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(2)
    
    # Left side backward
    print("Left side BACKWARD for 5 seconds...")
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    time.sleep(5)
    
    # Left side stop and disable
    print("Left side STOP and DISABLE...")
    pwm_left.ChangeDutyCycle(0)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(2)
    
    print("\nTesting RIGHT SIDE motors with Pin 32 PWM...")
    # Enable right side with 25% PWM (now using Pin 32)
    pwm_right.ChangeDutyCycle(25)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(2)
    
    # Right side forward
    print("Right side FORWARD for 5 seconds...")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(5)
    
    # Right side stop
    print("Right side STOP...")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(2)
    
    # Right side backward
    print("Right side BACKWARD for 5 seconds...")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    time.sleep(5)
    
    # Right side stop and disable
    print("Right side STOP and DISABLE...")
    pwm_right.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(2)
    
    print("\nTesting BOTH SIDES together...")
    # Enable both sides with 25% PWM
    pwm_right.ChangeDutyCycle(25)  # Pin 32 - 25%
    pwm_left.ChangeDutyCycle(25)   # Pin 33 - 25%
    time.sleep(1)
    
    # Both forward
    print("BOTH sides FORWARD for 5 seconds...")
    GPIO.output(IN1, GPIO.HIGH)  # Right forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)  # Left forward
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(5)
    
    # Both stop
    print("BOTH sides STOP...")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(2)
    
    # Both backward
    print("BOTH sides BACKWARD for 5 seconds...")
    GPIO.output(IN1, GPIO.LOW)   # Right backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)   # Left backward
    GPIO.output(IN4, GPIO.HIGH)
    time.sleep(5)
    
    # Turn left (right side forward, left side stop)
    print("TURN LEFT for 3 seconds...")
    GPIO.output(IN1, GPIO.HIGH)  # Right forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)   # Left stop
    GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)   # Left PWM off
    time.sleep(3)
    
    # Turn right (left side forward, right side stop)
    print("TURN RIGHT for 3 seconds...")
    GPIO.output(IN1, GPIO.LOW)   # Right stop
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)  # Left forward
    GPIO.output(IN4, GPIO.LOW)
    pwm_right.ChangeDutyCycle(0)  # Right PWM off
    pwm_left.ChangeDutyCycle(25)  # Left PWM on
    time.sleep(3)

finally:
    # Final stop and cleanup
    print("Final STOP and cleanup...")
    pwm_right.ChangeDutyCycle(0)
    pwm_left.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(1)
    
    # Stop PWM objects
    pwm_right.stop()
    pwm_left.stop()
    GPIO.cleanup()
    print("Test completed!")