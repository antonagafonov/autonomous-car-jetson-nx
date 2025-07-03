#!/usr/bin/env python3
"""
PWM Pin Checker for Jetson Xavier NX
Checks which pins support PWM functionality
"""

import RPi.GPIO as GPIO
import time
import sys

def check_pwm_pins():
    """Check if specific pins support PWM"""
    print("Checking PWM pin availability on Jetson Xavier NX...")
    
    # Common PWM pins on Jetson Xavier NX
    potential_pwm_pins = [
        (12, "Pin 12 (PWM0)"),
        (13, "Pin 13 (PWM1)"), 
        (15, "Pin 15 (GPIO14)"),  # This is your problematic pin
        (18, "Pin 18 (PWM2)"),
        (32, "Pin 32 (PWM3)"),   # This is your right PWM pin
        (33, "Pin 33 (PWM4)"),
    ]
    
    GPIO.setmode(GPIO.BOARD)
    working_pwm_pins = []
    
    for pin_num, pin_desc in potential_pwm_pins:
        print(f"\nTesting {pin_desc}...")
        try:
            # Setup pin as output
            GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)
            
            # Try to create PWM object
            pwm = GPIO.PWM(pin_num, 100)  # 100Hz
            pwm.start(0)
            
            # Test PWM functionality
            print(f"  PWM object created successfully")
            
            # Test different duty cycles
            for duty in [25, 50, 75]:
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.2)
            
            pwm.ChangeDutyCycle(0)
            pwm.stop()
            working_pwm_pins.append((pin_num, pin_desc))
            print(f"  ✓ {pin_desc} - PWM WORKS")
            
        except Exception as e:
            print(f"  ✗ {pin_desc} - PWM FAILED: {e}")
        
        finally:
            try:
                GPIO.cleanup(pin_num)
            except:
                pass
    
    print(f"\n{'='*50}")
    print("PWM PIN TEST RESULTS")
    print(f"{'='*50}")
    
    if working_pwm_pins:
        print("Working PWM pins:")
        for pin_num, pin_desc in working_pwm_pins:
            print(f"  ✓ {pin_desc}")
    else:
        print("No working PWM pins found!")
    
    return working_pwm_pins

def test_alternative_pins():
    """Test alternative pin configurations"""
    print(f"\n{'='*50}")
    print("TESTING ALTERNATIVE PIN CONFIGURATIONS")
    print(f"{'='*50}")
    
    # Alternative configurations
    configs = [
        {
            'name': 'Config 1: Use Pin 12 for Left PWM',
            'left_pwm': 12,
            'right_pwm': 32,
        },
        {
            'name': 'Config 2: Use Pin 18 for Left PWM',
            'left_pwm': 18,
            'right_pwm': 32,
        },
        {
            'name': 'Config 3: Use Pin 33 for Left PWM',
            'left_pwm': 33,
            'right_pwm': 32,
        }
    ]
    
    GPIO.setmode(GPIO.BOARD)
    
    for config in configs:
        print(f"\n{config['name']}:")
        
        left_works = test_single_pwm_pin(config['left_pwm'], "Left")
        right_works = test_single_pwm_pin(config['right_pwm'], "Right")
        
        if left_works and right_works:
            print(f"  ✓ RECOMMENDED: Use pins {config['left_pwm']} and {config['right_pwm']}")
        else:
            print(f"  ✗ Configuration not suitable")

def test_single_pwm_pin(pin_num, side_name):
    """Test a single PWM pin"""
    try:
        GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)
        pwm = GPIO.PWM(pin_num, 100)
        pwm.start(0)
        
        # Quick test
        pwm.ChangeDutyCycle(50)
        time.sleep(0.1)
        pwm.ChangeDutyCycle(0)
        
        pwm.stop()
        GPIO.cleanup(pin_num)
        
        print(f"    {side_name} PWM (Pin {pin_num}): ✓ WORKS")
        return True
        
    except Exception as e:
        print(f"    {side_name} PWM (Pin {pin_num}): ✗ FAILED - {e}")
        try:
            GPIO.cleanup(pin_num)
        except:
            pass
        return False

def main():
    print("PWM Pin Checker for Jetson Xavier NX")
    print("This will help identify which pins work for PWM")
    
    try:
        # Check PWM pins
        working_pins = check_pwm_pins()
        
        # Test alternative configurations
        test_alternative_pins()
        
        print(f"\n{'='*50}")
        print("RECOMMENDATIONS")
        print(f"{'='*50}")
        
        if len(working_pins) >= 2:
            print("You have multiple working PWM pins!")
            print("Recommended pin configuration for your motor controller:")
            
            # Find best pins (avoid pin 15 if it's problematic)
            good_pins = [pin for pin, desc in working_pins if pin != 15]
            
            if len(good_pins) >= 2:
                print(f"  Left PWM: Pin {good_pins[0]}")
                print(f"  Right PWM: Pin {good_pins[1]} (or keep Pin 32 if it works)")
            else:
                print(f"  Limited options - use pins: {[pin for pin, _ in working_pins]}")
                
        else:
            print("Limited PWM pins available. You may need to:")
            print("1. Check your device tree configuration")
            print("2. Enable PWM in jetson-io")
            print("3. Use software PWM as a fallback")
        
        print(f"\nTo modify your motor controller, change these lines:")
        print(f"  self.en_left = <new_left_pwm_pin>")
        print(f"  self.en_right = <new_right_pwm_pin>")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        try:
            GPIO.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()