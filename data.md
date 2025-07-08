#### Extract Complete Training Data

Use the comprehensive bag data extractor to convert ROS2 bags into ML-ready datasets with all sensor modalities:

```bash
# Extract complete dataset (images + commands + joystick data)
python3 extract_complete_bag.py ~/training_data/drive_session_001

# Extract from compressed bag files
python3 extract_complete_bag.py ~/training_data/behavior_20250627_174455

# Process multiple sessions
for bag in ~/training_data/drive_session_*; do
    python3 extract_complete_bag.py "$bag"
done
```

#### Complete Dataset Structure

The complete extractor creates a comprehensive training dataset:

```
drive_session_001_extracted/
├── images/                    # Camera images (640x480 PNG)
│   ├── image_000000.png       # Sequential images for visual input
│   ├── image_000001.png
│   └── ...
└── data/                      # Multi-modal sensor data
    ├── images.csv             # Image metadata with timestamps
    ├── cmd_vel_manual.csv     # Manual control commands (training labels)
    ├── cmd_vel.csv            # Processed velocity commands
    ├── joy.csv                # Raw joystick inputs (axes + buttons)
    ├── synchronized_dataset.csv  # ⭐ TIME-ALIGNED COMPLETE DATASET
    └── *.json                 # Same data in JSON format
```

#### Multi-Modal Data Formats

**Manual Commands (cmd_vel_manual.csv) - Primary Training Labels:**
```csv
timestamp,seq,linear_x,linear_y,linear_z,angular_x,angular_y,angular_z
1751035499659895179,0,0.2,0.0,0.0,0.0,0.0,0.1
1751035499759895179,1,0.15,0.0,0.0,0.0,0.0,0.05
```

**Joystick Raw Data (joy.csv) - Input Device States:**
```csv
timestamp,seq,axis_0,axis_1,axis_2,axis_3,button_0,button_1,button_2
1751035499659895179,0,0.2,-0.1,0.0,0.0,0,0,1
1751035499759895179,1,0.15,0.0,0.0,0.0,0,1,0
```

**Synchronized Dataset (synchronized_dataset.csv) - Complete Training Data:**
```csv
image_filename,image_timestamp,manual_linear_x,manual_angular_z,cmd_linear_x,cmd_angular_z,joy_axis_0,joy_axis_1,joy_button_0,manual_time_diff,cmd_time_diff,joy_time_diff
image_000000.png,1751035499659895179,0.2,0.1,0.2,0.1,0.2,-0.1,0,0.001,0.002,0.001
image_000001.png,1751035499759895179,0.15,0.05,0.15,0.05,0.15,0.0,1,0.001,0.001,0.002
```

#### Multi-Modal Learning Applications

The complete dataset enables advanced ML approaches:

**🎯 Behavior Cloning (Image → Commands):**
- Input: `image_filename` (camera images)
- Labels: `manual_linear_x`, `manual_angular_z` (human driver commands)
- Perfect for end-to-end autonomous driving

**🕹️ Joystick-to-Command Learning:**
- Input: `joy_axis_*`, `joy_button_*` (raw joystick)
- Labels: `manual_linear_x`, `manual_angular_z` (processed commands)
- Learn human control preprocessing

**🔄 Multi-Modal Fusion:**
- Combine camera images + joystick state for robust control
- Cross-modal learning between vision and control inputs
- Sensor fusion for improved reliability

**📊 Control Analysis:**
- Compare `manual_*` vs `cmd_*` to understand control processing
- Analyze `*_time_diff` for synchronization quality
- Study human driving patterns and reaction times

#### Advanced Training Examples

**PyTorch Behavior Cloning Setup:**
```python
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class AutonomousCarDataset(Dataset):
    def __init__(self, data_dir):
        self.df = pd.read_csv(f"{data_dir}/data/synchronized_dataset.csv")
        self.images_dir = f"{data_dir}/images"
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = f"{self.images_dir}/{row['image_filename']}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get control commands
        linear_x = row['manual_linear_x']
        angular_z = row['manual_angular_z']
        
        return {
            'image': torch.FloatTensor(image).permute(2,0,1),
            'commands': torch.FloatTensor([linear_x, angular_z])
        }
```

**Multi-Modal Learning:**
```python
# Load complete synchronized dataset
df = pd.read_csv("data/synchronized_dataset.csv")

# Vision + Joystick fusion
vision_features = load_images(df['image_filename'])
joystick_features = df[['joy_axis_0', 'joy_axis_1', 'joy_button_0']].values
control_labels = df[['manual_linear_x', 'manual_angular_z']].values

# Train multi-modal network
model = MultiModalNetwork(vision_dim=640*480*3, joystick_dim=3)
```

#### Dataset Quality Metrics

The extractor provides synchronization quality metrics:

```bash
# Check time alignment quality
python3 -c "
import pandas as pd
df = pd.read_csv('data/synchronized_dataset.csv')
print(f'Average sync quality:')
print(f'  Manual commands: {df[\"manual_time_diff\"].mean():.3f}s')
print(f'  Velocity commands: {df[\"cmd_time_diff\"].mean():.3f}s') 
print(f'  Joystick inputs: {df[\"joy_time_diff\"].mean():.3f}s')
"
```

**Good synchronization:** < 0.1s time differences  
**Excellent synchronization:** < 0.05s time differences

#### Data Collection Best Practices

**For High-Quality Training Data:**

1. **Smooth Driving Patterns:**
   ```bash
   # Use conservative speed settings for training data
   ros2 launch car_bringup car_full_system.launch.py \
     max_linear_speed:=0.6 base_speed_scale:=50
   ```

2. **Diverse Scenarios:**
   - Record multiple sessions in different environments
   - Include various turning patterns and speeds
   - Capture edge cases and recovery maneuvers

3. **Data Validation:**
   ```bash
   # Check dataset completeness
   python3 -c "
   import pandas as pd
   df = pd.read_csv('data/synchronized_dataset.csv')
   print(f'Dataset size: {len(df)} samples')
   print(f'Speed range: {df[\"manual_linear_x\"].min():.2f} to {df[\"manual_linear_x\"].max():.2f} m/s')
   print(f'Turn range: {df[\"manual_angular_z\"].min():.2f} to {df[\"manual_angular_z\"].max():.2f} rad/s')
   "
   ```

4. **Balanced Data:**
   - Equal amounts of left/right turns
   - Various speed ranges represented
   - Include stopped/slow sections for completeness