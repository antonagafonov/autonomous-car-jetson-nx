#!/usr/bin/env python3
"""
Motor Test Script for Jetson Xavier NX
Tests each motor independently to diagnose hardware/software issues
"""

import RPi.GPIO as GPIO
import time
import sys

class MotorTester:
    def __init__(self):
        # GPIO pin assignments (BOARD mode)
        self.in1_left = 18   # Left side direction 1 (Pin 18)
        self.in2_left = 16   # Left side direction 2 (Pin 16)  
        self.en_left = 15    # Left side PWM enable (Pin 15)
        self.in1_right = 35  # Right side direction 1 (Pin 35)
        self.in2_right = 37  # Right side direction 2 (Pin 37)
        self.en_right = 32   # Right side PWM enable (Pin 32)
        
        self.setup_gpio()
    
    def setup_gpio(self):
        """Setup GPIO pins"""
        print("Setting up GPIO pins...")
        GPIO.setmode(GPIO.BOARD)
        
        try:
            # Setup all pins as outputs, initially LOW
            GPIO.setup([self.in1_left, self.in2_left, self.en_left, 
                       self.in1_right, self.in2_right, self.en_right], 
                      GPIO.OUT, initial=GPIO.LOW)
            
            # Initialize PWM at 100Hz
            self.pwm_left = GPIO.PWM(self.en_left, 100)
            self.pwm_right = GPIO.PWM(self.en_right, 100)
            self.pwm_left.start(0)
            self.pwm_right.start(0)
            
            print("GPIO setup completed successfully")
            print(f"Left side pins: IN1={self.in1_left}, IN2={self.in2_left}, PWM={self.en_left}")
            print(f"Right side pins: IN1={self.in1_right}, IN2={self.in2_right}, PWM={self.en_right}")
            
        except Exception as e:
            print(f"GPIO setup failed: {e}")
            sys.exit(1)
    
    def test_motor_side(self, side, speed=50, duration=2):
        """Test one side of motors"""
        print(f"\n{'='*50}")
        print(f"Testing {side.upper()} MOTOR")
        print(f"{'='*50}")
        
        if side == 'left':
            in1_pin = self.in1_left
            in2_pin = self.in2_left
            pwm_control = self.pwm_left
        else:
            in1_pin = self.in1_right
            in2_pin = self.in2_right
            pwm_control = self.pwm_right
        
        # Test Forward
        print(f"1. {side.upper()} MOTOR FORWARD at {speed}% for {duration}s")
        pwm_control.ChangeDutyCycle(speed)
        GPIO.output(in1_pin, GPIO.HIGH)
        GPIO.output(in2_pin, GPIO.LOW)
        print(f"   IN1={GPIO.input(in1_pin)}, IN2={GPIO.input(in2_pin)}, PWM={speed}%")
        time.sleep(duration)
        
        # Stop
        print(f"   Stopping {side} motor...")
        pwm_control.ChangeDutyCycle(0)
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)
        time.sleep(1)
        
        # Test Backward
        print(f"2. {side.upper()} MOTOR BACKWARD at {speed}% for {duration}s")
        pwm_control.ChangeDutyCycle(speed)
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.HIGH)
        print(f"   IN1={GPIO.input(in1_pin)}, IN2={GPIO.input(in2_pin)}, PWM={speed}%")
        time.sleep(duration)
        
        # Stop
        print(f"   Stopping {side} motor...")
        pwm_control.ChangeDutyCycle(0)
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)
        time.sleep(1)
        
        # Test different speeds
        print(f"3. {side.upper()} MOTOR SPEED TEST (Forward)")
        for test_speed in [25, 50, 75, 100]:
            print(f"   Testing at {test_speed}%...")
            pwm_control.ChangeDutyCycle(test_speed)
            GPIO.output(in1_pin, GPIO.HIGH)
            GPIO.output(in2_pin, GPIO.LOW)
            time.sleep(1)
        
        # Stop
        pwm_control.ChangeDutyCycle(0)
        GPIO.output(in1_pin, GPIO.LOW)
        GPIO.output(in2_pin, GPIO.LOW)
        print(f"   {side.upper()} motor tests completed")
    
    def test_both_motors_differential(self):
        """Test differential drive (like your original issue)"""
        print(f"\n{'='*50}")
        print("TESTING DIFFERENTIAL DRIVE")
        print(f"{'='*50}")
        
        test_cases = [
            ("Both Forward", 50, 50),
            ("Both Backward", -50, -50),
            ("Turn Right (Left faster)", 50, 25),
            ("Turn Left (Right faster)", 25, 50),
            ("Pivot Right (Left forward, Right back)", 50, -50),
            ("Pivot Left (Right forward, Left back)", -50, 50),
        ]
        
        for test_name, left_speed, right_speed in test_cases:
            print(f"\n{test_name}:")
            print(f"  Left: {left_speed}%, Right: {right_speed}%")
            
            # Control left motor
            self.control_motor(left_speed, 'left')
            
            # Control right motor
            self.control_motor(right_speed, 'right')
            
            time.sleep(2)
            
            # Stop both
            self.stop_all_motors()
            time.sleep(1)
    
    def control_motor(self, speed, side):
        """Control one motor with signed speed (-100 to +100)"""
        if side == 'left':
            in1_pin = self.in1_left
            in2_pin = self.in2_left
            pwm_control = self.pwm_left
        else:
            in1_pin = self.in1_right
            in2_pin = self.in2_right
            pwm_control = self.pwm_right
        
        # Set PWM duty cycle
        pwm_control.ChangeDutyCycle(abs(speed))
        
        # Control direction
        if speed > 0:
            # Forward
            GPIO.output(in1_pin, GPIO.HIGH)
            GPIO.output(in2_pin, GPIO.LOW)
            direction = "FWD"
        elif speed < 0:
            # Backward
            GPIO.output(in1_pin, GPIO.LOW)
            GPIO.output(in2_pin, GPIO.HIGH)
            direction = "BWD"
        else:
            # Stop
            GPIO.output(in1_pin, GPIO.LOW)
            GPIO.output(in2_pin, GPIO.LOW)
            direction = "STOP"
        
        print(f"  {side.upper()}: {direction} {abs(speed)}% (IN1={GPIO.input(in1_pin)}, IN2={GPIO.input(in2_pin)})")
    
    def stop_all_motors(self):
        """Stop all motors"""
        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)
        GPIO.output([self.in1_left, self.in2_left, self.in1_right, self.in2_right], GPIO.LOW)
    
    def gpio_info_test(self):
        """Display GPIO information"""
        print(f"\n{'='*50}")
        print("GPIO INFORMATION")
        print(f"{'='*50}")
        try:
            print(f"GPIO Model: {GPIO.model}")
            print(f"GPIO Mode: {GPIO.getmode()}")
            print(f"GPIO Version: {GPIO.VERSION}")
        except Exception as e:
            print(f"Error getting GPIO info: {e}")
    
    def pin_connectivity_test(self):
        """Test if pins are properly connected"""
        print(f"\n{'='*50}")
        print("PIN CONNECTIVITY TEST")
        print(f"{'='*50}")
        
        # Test direction pins only (not PWM pins)
        direction_pins = [
            ("Left IN1", self.in1_left),
            ("Left IN2", self.in2_left),
            ("Right IN1", self.in1_right),
            ("Right IN2", self.in2_right),
        ]
        
        for pin_name, pin_num in direction_pins:
            print(f"Testing {pin_name} (Pin {pin_num})...")
            try:
                # Test HIGH
                GPIO.output(pin_num, GPIO.HIGH)
                state = GPIO.input(pin_num)
                print(f"  Set HIGH, Read: {state}")
                
                # Test LOW
                GPIO.output(pin_num, GPIO.LOW)
                state = GPIO.input(pin_num)
                print(f"  Set LOW, Read: {state}")
                
            except Exception as e:
                print(f"  ERROR testing {pin_name}: {e}")
            
            time.sleep(0.1)
        
        # Test PWM pins separately (they're already configured with PWM objects)
        print(f"\nTesting PWM pins:")
        try:
            print(f"Left PWM (Pin {self.en_left}): Testing PWM functionality...")
            self.pwm_left.ChangeDutyCycle(25)
            time.sleep(0.5)
            self.pwm_left.ChangeDutyCycle(0)
            print(f"  Left PWM: OK")
        except Exception as e:
            print(f"  Left PWM ERROR: {e}")
        
        try:
            print(f"Right PWM (Pin {self.en_right}): Testing PWM functionality...")
            self.pwm_right.ChangeDutyCycle(25)
            time.sleep(0.5)
            self.pwm_right.ChangeDutyCycle(0)
            print(f"  Right PWM: OK")
        except Exception as e:
            print(f"  Right PWM ERROR: {e}")
    
    def cleanup(self):
        """Clean up GPIO"""
        print("\nCleaning up GPIO...")
        try:
            self.stop_all_motors()
            self.pwm_left.stop()
            self.pwm_right.stop()
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except Exception as e:
            print(f"GPIO cleanup error: {e}")

def main():
    print("Motor Test Script for Jetson Xavier NX")
    print("This script will test each motor independently")
    print("Make sure your motors are connected and powered!")
    
    response = input("\nPress Enter to continue or 'q' to quit: ")
    if response.lower() == 'q':
        sys.exit(0)
    
    tester = MotorTester()
    
    try:
        # Display GPIO information
        tester.gpio_info_test()
        
        # Test pin connectivity
        tester.pin_connectivity_test()
        
        input("\nPress Enter to test LEFT motor...")
        tester.test_motor_side('left', speed=50, duration=2)
        
        input("\nPress Enter to test RIGHT motor...")
        tester.test_motor_side('right', speed=50, duration=2)
        
        input("\nPress Enter to test differential drive...")
        tester.test_both_motors_differential()
        
        print("\n" + "="*50)
        print("TESTING COMPLETE!")
        print("="*50)
        print("Check the results above:")
        print("- Did both motors work in forward direction?")
        print("- Did both motors work in backward direction?")
        print("- Did pivot movements work correctly?")
        print("- Are there any error messages?")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()