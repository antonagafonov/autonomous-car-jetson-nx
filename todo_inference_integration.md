# To-Do List: Inference Node Testing & Integration

## 🧪 Testing the Modified Node
- [ ] **Test inference node standalone** - verify queue behavior works correctly
- [ ] **Check queue override logic** - ensure new predictions clear old ones
- [ ] **Monitor publishing rate** - confirm 20Hz publishing from queue
- [ ] **Validate inference rate** - verify 5Hz inference generation
- [ ] **Test queue size topic** - check `/car/queue_size` publishing

## 🔗 Car Bringup Integration  
- [ ] **Add inference node to car_bringup launch file**
- [ ] **Configure topic remapping** - ensure `/camera/image_raw` connects properly
- [ ] **Set model path parameter** - point to your trained model checkpoint
- [ ] **Test with actual camera feed** - verify end-to-end pipeline

## ✅ Validation & Debugging
- [ ] **Monitor log outputs** - check for confidence, queue size, error messages
- [ ] **Test autonomous enable/disable** - verify `/car/autonomous/enable` topic
- [ ] **Safety testing** - confirm stop commands when queue empty or low confidence
- [ ] **Performance check** - ensure no lag or memory issues

## 🚗 Final Integration Test
- [ ] **Full system test** - camera → inference → queue → cmd_vel at 20Hz
- [ ] **Emergency stop verification** - test safety mechanisms work

## 🛠️ Debug Commands for Tomorrow

### Monitor Topics
```bash
# Monitor autonomous commands output
ros2 topic echo /cmd_vel_autonomous

# Monitor prediction queue size
ros2 topic echo /car/queue_size

# Monitor angular predictions
ros2 topic echo /car/angular_prediction

# Monitor confidence scores
ros2 topic echo /car/inference_confidence

# Monitor inference status
ros2 topic echo /car/inference_status
```

### Run Nodes
```bash
# Run inference node standalone
ros2 run car_inference inference_node

# Run camera node with specific settings
ros2 run car_perception camera_node --ros-args -p output_width:=640 -p output_height:=480 -p framerate:=60

# Monitor system performance
jtop
```

### Quick Test Sequence
```bash
# Terminal 1: Start camera
ros2 run car_perception camera_node --ros-args -p output_width:=640 -p output_height:=480 -p framerate:=60

# Terminal 2: Start inference
ros2 run car_inference inference_node

# Terminal 3: Monitor commands
ros2 topic echo /cmd_vel_autonomous

# Terminal 4: Monitor queue
ros2 topic echo /car/queue_size
```