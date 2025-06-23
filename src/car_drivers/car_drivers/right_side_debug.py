#!/usr/bin/env python3
"""
Right Side Motors Test for Jetson Xavier NX
Tests only pins 32, 35, 37 (right side motor driver)
"""

import Jetson.GPIO as GPIO
import time

class RightMotorTest:
    def __init__(self):
        # Right side motor pins (BOARD numbering)
        self.in1_right = 37   # Right side direction 1
        self.in2_right = 35   # Right side direction 2
        self.en_right = 32    # Right side PWM (enable)
        
        self.pwm_right = None
        self.setup_complete = False
    
    def setup_gpio(self):
        """Setup GPIO for right side motors only"""
        try:
            print("🔧 Setting up RIGHT SIDE MOTORS (Pins 32, 35, 37)")
            print(f"📋 Board model: {GPIO.model}")
            
            # Setup GPIO mode
            GPIO.setmode(GPIO.BOARD)
            
            # Setup pins as outputs
            GPIO.setup([self.in1_right, self.in2_right, self.en_right], GPIO.OUT, initial=GPIO.LOW)
            
            # Create PWM object for enable pin (pin 32)
            self.pwm_right = GPIO.PWM(self.en_right, 100)  # 100Hz frequency
            self.pwm_right.start(0)  # Start with 0% duty cycle
            
            print("✅ GPIO setup completed successfully!")
            print(f"📍 Right motor pins configured:")
            print(f"   Pin {self.in1_right} (IN1) - Direction control 1")
            print(f"   Pin {self.in2_right} (IN2) - Direction control 2")
            print(f"   Pin {self.en_right} (PWM) - Speed control")
            print()
            
            self.setup_complete = True
            return True
            
        except Exception as e:
            print(f"❌ GPIO setup failed: {e}")
            print("💡 Try running with: sudo python3 right_motor_test.py")
            return False
    
    def test_pwm_only(self):
        """Test PWM pin (32) at 100% duty cycle only"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("⚡ PWM TEST - Pin 32 at 100% duty cycle")
        print("=" * 50)
        
        # Keep direction pins LOW (no direction set)
        GPIO.output(self.in1_right, GPIO.LOW)
        GPIO.output(self.in2_right, GPIO.LOW)
        
        print("📌 Direction pins (35, 37) set to LOW")
        print("⚡ Setting PWM pin 32 to 100% duty cycle for 5 seconds...")
        print("🔍 You should hear/see the motor driver activate")
        print("⚠️  Motors may not spin without direction set")
        
        # Set PWM to 100%
        self.pwm_right.ChangeDutyCycle(100)
        
        # Run for 5 seconds
        for i in range(5, 0, -1):
            print(f"   ⏰ {i} seconds remaining...")
            time.sleep(1)
        
        # Stop PWM
        self.pwm_right.ChangeDutyCycle(0)
        print("⏹️  PWM stopped")
        
        response = input("\n❓ Did you hear/see any response from the motor driver? (y/n): ").lower()
        if response == 'y':
            print("✅ PWM pin 32 is working!")
        else:
            print("⚠️  No response - check connections to pin 32")
    
    def test_direction_pins(self):
        """Test direction pins (35, 37) individually"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("\n📍 DIRECTION PINS TEST")
        print("=" * 50)
        
        # Test Pin 37 (IN1)
        print(f"🔴 Testing Pin {self.in1_right} (IN1) - HIGH for 2 seconds")
        GPIO.output(self.in1_right, GPIO.HIGH)
        GPIO.output(self.in2_right, GPIO.LOW)
        time.sleep(2)
        GPIO.output(self.in1_right, GPIO.LOW)
        
        input(f"❓ Did you see any LED/indicator on motor driver for pin {self.in1_right}? Press Enter...")
        
        # Test Pin 35 (IN2)
        print(f"🔴 Testing Pin {self.in2_right} (IN2) - HIGH for 2 seconds")
        GPIO.output(self.in1_right, GPIO.LOW)
        GPIO.output(self.in2_right, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.in2_right, GPIO.LOW)
        
        input(f"❓ Did you see any LED/indicator on motor driver for pin {self.in2_right}? Press Enter...")
    
    def test_combined_forward(self):
        """Test PWM + Direction for forward movement"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("\n▶️  COMBINED TEST - Forward Direction")
        print("=" * 50)
        
        print("Setting up forward direction:")
        print(f"   Pin {self.in1_right} (IN1) = HIGH")
        print(f"   Pin {self.in2_right} (IN2) = LOW")
        print(f"   Pin {self.en_right} (PWM) = 100%")
        
        # Set forward direction
        GPIO.output(self.in1_right, GPIO.HIGH)
        GPIO.output(self.in2_right, GPIO.LOW)
        
        print("▶️  Running motors FORWARD for 3 seconds...")
        self.pwm_right.ChangeDutyCycle(100)
        
        for i in range(3, 0, -1):
            print(f"   ⏰ {i} seconds remaining...")
            time.sleep(1)
        
        # Stop
        self.pwm_right.ChangeDutyCycle(0)
        GPIO.output(self.in1_right, GPIO.LOW)
        GPIO.output(self.in2_right, GPIO.LOW)
        print("⏹️  Motors stopped")
        
        response = input("\n❓ Did the RIGHT SIDE motors spin forward? (y/n): ").lower()
        if response == 'y':
            print("✅ Right side motors working correctly!")
        else:
            print("⚠️  Issue with right side motors - check power/connections")
    
    def test_combined_reverse(self):
        """Test PWM + Direction for reverse movement"""
        if not self.setup_complete:
            print("❌ GPIO not set up properly")
            return
        
        print("\n◀️  COMBINED TEST - Reverse Direction")
        print("=" * 50)
        
        print("Setting up reverse direction:")
        print(f"   Pin {self.in1_right} (IN1) = LOW")
        print(f"   Pin {self.in2_right} (IN2) = HIGH")
        print(f"   Pin {self.en_right} (PWM) = 100%")
        
        # Set reverse direction
        GPIO.output(self.in1_right, GPIO.LOW)
        GPIO.output(self.in2_right, GPIO.HIGH)
        
        print("◀️  Running motors REVERSE for 3 seconds...")
        self.pwm_right.ChangeDutyCycle(100)
        
        for i in range(3, 0, -1):
            print(f"   ⏰ {i} seconds remaining...")
            time.sleep(1)
        
        # Stop
        self.pwm_right.ChangeDutyCycle(0)
        GPIO.output(self.in1_right, GPIO.LOW)
        GPIO.output(self.in2_right, GPIO.LOW)
        print("⏹️  Motors stopped")
        
        response = input("\n❓ Did the RIGHT SIDE motors spin reverse? (y/n): ").lower()
        if response == 'y':
            print("✅ Right side motors reverse working correctly!")
        else:
            print("⚠️  Issue with reverse direction - check motor wiring")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.setup_complete and self.pwm_right:
            print("\n🧹 Cleaning up GPIO...")
            try:
                self.pwm_right.ChangeDutyCycle(0)
                self.pwm_right.stop()
                GPIO.output([self.in1_right, self.in2_right], GPIO.LOW)
                GPIO.cleanup()
                print("✅ GPIO cleanup completed")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    def run_all_tests(self):
        """Run complete test sequence for right side motors"""
        print("🚗 RIGHT SIDE MOTORS TEST (Pins 32, 35, 37)")
        print("=" * 55)
        print("This will test your right side motor driver step by step")
        print("⚠️  Make sure your car can move safely!")
        print()
        
        if not self.setup_gpio():
            return
        
        try:
            while True:
                print("\n📋 SELECT TEST:")
                print("1. PWM test only (Pin 32 at 100%)")
                print("2. Direction pins test (Pins 35, 37)")
                print("3. Combined forward test")
                print("4. Combined reverse test")
                print("5. Run all tests")
                print("6. Exit")
                
                choice = input("\nEnter choice (1-6): ").strip()
                
                if choice == '1':
                    self.test_pwm_only()
                elif choice == '2':
                    self.test_direction_pins()
                elif choice == '3':
                    self.test_combined_forward()
                elif choice == '4':
                    self.test_combined_reverse()
                elif choice == '5':
                    self.test_pwm_only()
                    self.test_direction_pins()
                    self.test_combined_forward()
                    self.test_combined_reverse()
                elif choice == '6':
                    break
                else:
                    print("❌ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Test interrupted by user")
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("⚠️  SAFETY WARNING:")
    print("   - This tests RIGHT SIDE motors only (pins 32, 35, 37)")
    print("   - Make sure your car is secured and can move safely")
    print("   - Keep your hand near the power switch")
    print("   - PWM will run at 100% - be ready to stop!")
    print()
    
    response = input("Ready to test RIGHT SIDE motors? (y/n): ").lower()
    if response == 'y':
        tester = RightMotorTest()
        tester.run_all_tests()
    else:
        print("Test cancelled. Stay safe! 👍")