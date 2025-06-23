#!/usr/bin/env python3
"""
Motor Debug Script for Jetson Xavier NX
Tests each motor pin individually to verify connections
"""

import Jetson.GPIO as GPIO
import time
import sys

class MotorDebugger:
    def __init__(self):
        self.setup_complete = False
        
        # Pin definitions (BOARD numbering) - 4 Motor Setup
        self.pins = {
            'Left Side Motors': {
                'in1_left': 18,    # Left side direction 1
                'in2_left': 16,    # Left side direction 2  
                'en_left': 33      # Left side PWM speed
            },
            'Right Side Motors': {
                'in1_right': 37,   # Right side direction 1
                'in2_right': 35,   # Right side direction 2
                'en_right': 32     # Right side PWM speed
            }
        }
        
        self.pwm_objects = {}
        
    def setup_gpio(self):
        """Initialize GPIO pins"""
        try:
            print("🔧 Setting up GPIO...")
            print(f"📋 Board model: {GPIO.model}")
            
            GPIO.setmode(GPIO.BOARD)
            
            # Setup all pins as outputs
            all_pins = []
            for motor, pins in self.pins.items():
                all_pins.extend(pins.values())
            
            GPIO.setup(all_pins, GPIO.OUT, initial=GPIO.LOW)
            
            # Create PWM objects
            self.pwm_objects['Left Side Motors'] = GPIO.PWM(self.pins['Left Side Motors']['en_left'], 100)  # 100Hz
            self.pwm_objects['Right Side Motors'] = GPIO.PWM(self.pins['Right Side Motors']['en_right'], 100)  # 100Hz
            
            # Start PWM with 0% duty cycle
            for pwm in self.pwm_objects.values():
                pwm.start(0)
            
            print("✅ GPIO setup completed successfully!")
            print(f"📍 Left Side Motors pins: {list(self.pins['Left Side Motors'].values())}")
            print(f"📍 Right Side Motors pins: {list(self.pins['Right Side Motors'].values())}")
            print(f"🚗 Setup: 2 motors on left side, 2 motors on right side")
            print()
            
            self.setup_complete = True
            
        except Exception as e:
            print(f"❌ GPIO setup failed: {e}")
            print("💡 Make sure you have GPIO permissions: sudo usermod -a -G gpio $USER")
            return False
        
        return True
    
    def test_individual_pins(self):
        """Test each pin individually"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("🔍 INDIVIDUAL PIN TEST")
        print("=" * 50)
        
        for motor_name, pins in self.pins.items():
            print(f"\n🔧 Testing {motor_name}:")
            
            for pin_name, pin_number in pins.items():
                print(f"  📌 Testing Pin {pin_number} ({pin_name})...")
                
                if 'en_' in pin_name:  # PWM pin
                    # Test PWM at different duty cycles
                    pwm_obj = self.pwm_objects[motor_name]
                    for duty in [25, 50, 75]:
                        print(f"    ⚡ PWM {duty}% for 1 second")
                        pwm_obj.ChangeDutyCycle(duty)
                        time.sleep(1)
                    pwm_obj.ChangeDutyCycle(0)
                    print(f"    ⏹️  PWM stopped")
                else:  # Direction pin
                    print(f"    🔴 HIGH for 1 second")
                    GPIO.output(pin_number, GPIO.HIGH)
                    time.sleep(1)
                    print(f"    ⚫ LOW for 1 second")
                    GPIO.output(pin_number, GPIO.LOW)
                    time.sleep(1)
                
                input(f"    ❓ Did you see any response from {motor_name} {pin_name}? Press Enter to continue...")
    
    def test_motor_directions(self):
        """Test motor directions systematically"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("\n🔄 MOTOR DIRECTION TEST")
        print("=" * 50)
        
        # Test Left Side Motors
        print(f"\n🚗 Testing Left Side Motors (2 motors)")
        self.test_single_motor_group('Left Side Motors', 25)  # 25% speed
        
        print(f"\n🚗 Testing Right Side Motors (2 motors)")
        self.test_single_motor_group('Right Side Motors', 25)  # 25% speed
    
    def test_single_motor_group(self, motor_group_name, speed_percent):
        """Test a group of motors (left side or right side) in both directions"""
        pins = self.pins[motor_group_name]
        pwm = self.pwm_objects[motor_group_name]
        
        print(f"  🔧 {motor_group_name} at {speed_percent}% speed")
        
        # Set PWM speed
        pwm.ChangeDutyCycle(speed_percent)
        
        # Test Forward
        print(f"  ▶️  Forward for 2 seconds...")
        if 'Left' in motor_group_name:
            GPIO.output(pins['in1_left'], GPIO.HIGH)
            GPIO.output(pins['in2_left'], GPIO.LOW)
        else:
            GPIO.output(pins['in1_right'], GPIO.HIGH)
            GPIO.output(pins['in2_right'], GPIO.LOW)
        
        time.sleep(2)
        
        # Stop
        print(f"  ⏸️  Stop for 1 second...")
        if 'Left' in motor_group_name:
            GPIO.output(pins['in1_left'], GPIO.LOW)
            GPIO.output(pins['in2_left'], GPIO.LOW)
        else:
            GPIO.output(pins['in1_right'], GPIO.LOW)
            GPIO.output(pins['in2_right'], GPIO.LOW)
        
        time.sleep(1)
        
        # Test Reverse
        print(f"  ◀️  Reverse for 2 seconds...")
        if 'Left' in motor_group_name:
            GPIO.output(pins['in1_left'], GPIO.LOW)
            GPIO.output(pins['in2_left'], GPIO.HIGH)
        else:
            GPIO.output(pins['in1_right'], GPIO.LOW)
            GPIO.output(pins['in2_right'], GPIO.HIGH)
        
        time.sleep(2)
        
        # Stop
        print(f"  ⏹️  Final stop...")
        if 'Left' in motor_group_name:
            GPIO.output(pins['in1_left'], GPIO.LOW)
            GPIO.output(pins['in2_left'], GPIO.LOW)
        else:
            GPIO.output(pins['in1_right'], GPIO.LOW)
            GPIO.output(pins['in2_right'], GPIO.LOW)
        
        pwm.ChangeDutyCycle(0)
        
        # Get user feedback
        response = input(f"  ❓ Did {motor_group_name} move forward then reverse? (y/n): ").lower()
        if response != 'y':
            print(f"  ⚠️  Issue detected with {motor_group_name}")
            self.diagnose_motor_issue(motor_group_name)
        else:
            print(f"  ✅ {motor_group_name} working correctly!")
    
    def test_differential_drive(self):
        """Test both motors together for differential drive"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("\n🚗 DIFFERENTIAL DRIVE TEST")
        print("=" * 50)
        
        speed = 30  # 30% speed for safety
        
        tests = [
            ("Forward", (speed, speed)),
            ("Backward", (-speed, -speed)),
            ("Turn Left", (speed//2, speed)),
            ("Turn Right", (speed, speed//2)),
            ("Spin Left", (-speed//2, speed//2)),
            ("Spin Right", (speed//2, -speed//2))
        ]
        
        for test_name, (left_speed, right_speed) in tests:
            print(f"\n  🧪 Test: {test_name}")
            print(f"     Left motor: {left_speed}%, Right motor: {right_speed}%")
            
            # Set motor speeds and directions
            self.set_motor_speed('Left Side Motors', left_speed)
            self.set_motor_speed('Right Side Motors', right_speed)
            
            print(f"     ⏳ Running for 3 seconds...")
            time.sleep(3)
            
            # Stop motors
            self.set_motor_speed('Left Side Motors', 0)
            self.set_motor_speed('Right Side Motors', 0)
            
            response = input(f"     ❓ Did the car {test_name.lower()} correctly? (y/n): ").lower()
            if response != 'y':
                print(f"     ⚠️  Issue with {test_name} movement")
            else:
                print(f"     ✅ {test_name} working correctly!")
            
            time.sleep(1)
    
    def set_motor_speed(self, motor_group_name, speed_percent):
        """Set motor group speed and direction"""
        pins = self.pins[motor_group_name]
        pwm = self.pwm_objects[motor_group_name]
        
        # Set PWM duty cycle
        pwm.ChangeDutyCycle(abs(speed_percent))
        
        # Set direction
        if 'Left' in motor_group_name:
            if speed_percent > 0:
                GPIO.output(pins['in1_left'], GPIO.HIGH)
                GPIO.output(pins['in2_left'], GPIO.LOW)
            elif speed_percent < 0:
                GPIO.output(pins['in1_left'], GPIO.LOW)
                GPIO.output(pins['in2_left'], GPIO.HIGH)
            else:
                GPIO.output(pins['in1_left'], GPIO.LOW)
                GPIO.output(pins['in2_left'], GPIO.LOW)
        else:  # Right Side Motors
            if speed_percent > 0:
                GPIO.output(pins['in1_right'], GPIO.HIGH)
                GPIO.output(pins['in2_right'], GPIO.LOW)
            elif speed_percent < 0:
                GPIO.output(pins['in1_right'], GPIO.LOW)
                GPIO.output(pins['in2_right'], GPIO.HIGH)
            else:
                GPIO.output(pins['in1_right'], GPIO.LOW)
                GPIO.output(pins['in2_right'], GPIO.LOW)
    
    def diagnose_motor_issue(self, motor_group_name):
        """Help diagnose motor connection issues"""
        print(f"\n🔍 DIAGNOSING {motor_group_name} ISSUE")
        print("-" * 40)
        
        pins = self.pins[motor_group_name]
        
        print(f"📋 Expected connections for {motor_group_name}:")
        for pin_name, pin_number in pins.items():
            print(f"   Jetson Pin {pin_number} → Motor Driver {pin_name.upper()}")
        
        print(f"\n🔧 Check these items:")
        print(f"   1. All 3 control wires connected to motor driver")
        print(f"   2. Motor driver has power (VCC and GND)")
        print(f"   3. Both motors on this side connected to driver output")
        print(f"   4. Motors wired in parallel (same polarity)")
        print(f"   5. No loose connections")
        print(f"   6. Motor driver enable jumper (if applicable)")
        print(f"   7. Sufficient power supply for 2 motors per side")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.setup_complete:
            print("\n🧹 Cleaning up GPIO...")
            try:
                for pwm in self.pwm_objects.values():
                    pwm.stop()
                GPIO.cleanup()
                print("✅ GPIO cleanup completed")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    def run_full_test(self):
        """Run complete test sequence"""
        print("🚗 MOTOR DEBUG SCRIPT FOR JETSON XAVIER NX")
        print("=" * 50)
        print("This script will test your motor connections step by step.")
        print("Make sure your car is on blocks or can move safely!")
        print()
        
        if not self.setup_gpio():
            return
        
        try:
            # Test menu
            while True:
                print("\n📋 SELECT TEST:")
                print("1. Individual pin test")
                print("2. Motor direction test") 
                print("3. Differential drive test")
                print("4. Run all tests")
                print("5. Exit")
                
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    self.test_individual_pins()
                elif choice == '2':
                    self.test_motor_directions()
                elif choice == '3':
                    self.test_differential_drive()
                elif choice == '4':
                    self.test_individual_pins()
                    self.test_motor_directions()
                    self.test_differential_drive()
                elif choice == '5':
                    break
                else:
                    print("❌ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Test interrupted by user")
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("⚠️  SAFETY WARNING:")
    print("   - Make sure your car can move safely")
    print("   - Keep your hand near the power switch")
    print("   - Start with low speeds")
    print()
    
    response = input("Ready to start motor testing? (y/n): ").lower()
    if response == 'y':
        debugger = MotorDebugger()
        debugger.run_full_test()
    else:
        print("Test cancelled. Stay safe! 👍")