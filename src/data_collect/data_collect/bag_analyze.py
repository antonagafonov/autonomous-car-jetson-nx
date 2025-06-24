#!/usr/bin/env python3
"""
ROS2 Bag Analyzer for Behavior Cloning Dataset Analysis

This tool provides comprehensive analysis of recorded bag files:
- Dataset statistics and quality metrics
- Topic synchronization analysis
- Data visualization and reports
- Dataset validation for ML training
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

# ROS2 bag reading
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message

# Message types
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist


class BagAnalyzer(Node):
    """Comprehensive bag file analyzer for behavior cloning datasets."""
    
    def __init__(self):
        super().__init__('bag_analyzer')
        self.get_logger().info("📊 Bag Analyzer initialized")
    
    def analyze_dataset(self, dataset_path: str, output_dir: Optional[str] = None) -> Dict:
        """Analyze complete dataset directory or single bag."""
        
        if os.path.isfile(dataset_path):
            # Single bag file
            return self.analyze_single_bag(dataset_path, output_dir)
        elif os.path.isdir(dataset_path):
            # Dataset directory
            return self.analyze_dataset_directory(dataset_path, output_dir)
        else:
            raise ValueError(f"Invalid path: {dataset_path}")
    
    def analyze_dataset_directory(self, dataset_dir: str, output_dir: Optional[str] = None) -> Dict:
        """Analyze entire dataset directory containing multiple sessions."""
        
        self.get_logger().info(f"📁 Analyzing dataset directory: {dataset_dir}")
        
        # Find all bag directories
        bag_dirs = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains bag files
                if any(f.endswith('.db3') or f.endswith('.mcap') for f in os.listdir(item_path)):
                    bag_dirs.append(item_path)
        
        if not bag_dirs:
            raise ValueError("No bag files found in dataset directory")
        
        self.get_logger().info(f"📋 Found {len(bag_dirs)} recording sessions")
        
        # Analyze each bag
        session_results = {}
        overall_stats = defaultdict(list)
        
        for bag_dir in sorted(bag_dirs):
            session_name = os.path.basename(bag_dir)
            self.get_logger().info(f"🔍 Analyzing session: {session_name}")
            
            try:
                result = self.analyze_single_bag(bag_dir)
                session_results[session_name] = result
                
                # Accumulate stats
                for key, value in result['statistics'].items():
                    if isinstance(value, (int, float)):
                        overall_stats[key].append(value)
                        
            except Exception as e:
                self.get_logger().error(f"❌ Failed to analyze {session_name}: {e}")
                continue
        
        # Compute overall statistics
        overall_summary = self.compute_overall_stats(overall_stats)
        
        # Generate reports
        report = {
            'dataset_info': {
                'dataset_path': dataset_dir,
                'analysis_time': datetime.now().isoformat(),
                'total_sessions': len(session_results),
                'successful_sessions': len([r for r in session_results.values() if r['valid']])
            },
            'overall_statistics': overall_summary,
            'session_results': session_results,
            'quality_assessment': self.assess_dataset_quality(session_results),
            'recommendations': self.generate_recommendations(session_results)
        }
        
        # Save report
        if output_dir:
            self.save_analysis_report(report, output_dir, 'dataset_analysis')
            self.generate_visualizations(report, output_dir)
        
        return report
    
    def analyze_single_bag(self, bag_path: str, output_dir: Optional[str] = None) -> Dict:
        """Analyze single bag file or directory."""
        
        self.get_logger().info(f"🔍 Analyzing bag: {bag_path}")
        
        try:
            # Setup bag reader
            storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            
            # Get topic metadata
            topic_metadata = reader.get_all_topics_and_types()
            
            # Initialize analysis data
            analysis_data = {
                'messages': defaultdict(list),
                'timestamps': defaultdict(list),
                'message_counts': defaultdict(int),
                'topic_info': {},
                'time_range': {'start': None, 'end': None}
            }
            
            # Process messages
            self.get_logger().info("📖 Reading messages...")
            
            message_count = 0
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                # Get message type
                topic_type = next((t.type for t in topic_metadata if t.name == topic), None)
                if not topic_type:
                    continue
                
                # Deserialize message
                try:
                    msg_type = get_message(topic_type)
                    msg = deserialize_message(data, msg_type)
                    
                    # Store data
                    analysis_data['messages'][topic].append(msg)
                    analysis_data['timestamps'][topic].append(timestamp)
                    analysis_data['message_counts'][topic] += 1
                    
                    # Update time range
                    if analysis_data['time_range']['start'] is None:
                        analysis_data['time_range']['start'] = timestamp
                    analysis_data['time_range']['end'] = timestamp
                    
                    message_count += 1
                    
                    if message_count % 1000 == 0:
                        self.get_logger().info(f"📊 Processed {message_count} messages...")
                        
                except Exception as e:
                    self.get_logger().warn(f"⚠️  Failed to deserialize message: {e}")
                    continue
            
            reader.close()
            
            # Compute statistics
            statistics = self.compute_bag_statistics(analysis_data)
            
            # Validate data quality
            validation = self.validate_bag_data(analysis_data)
            
            # Load metadata if available
            metadata = self.load_session_metadata(bag_path)
            
            result = {
                'bag_path': bag_path,
                'analysis_time': datetime.now().isoformat(),
                'valid': validation['valid'],
                'statistics': statistics,
                'validation': validation,
                'metadata': metadata,
                'topic_info': {t.name: t.type for t in topic_metadata}
            }
            
            # Save individual report
            if output_dir:
                session_name = os.path.basename(bag_path)
                self.save_analysis_report(result, output_dir, f'session_{session_name}')
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to analyze bag: {e}")
            return {
                'bag_path': bag_path,
                'valid': False,
                'error': str(e)
            }
    
    def compute_bag_statistics(self, analysis_data: Dict) -> Dict:
        """Compute comprehensive statistics for a bag."""
        
        stats = {}
        
        # Time statistics
        if analysis_data['time_range']['start'] and analysis_data['time_range']['end']:
            duration_ns = analysis_data['time_range']['end'] - analysis_data['time_range']['start']
            duration_s = duration_ns / 1e9
            stats['duration_seconds'] = duration_s
            stats['duration_minutes'] = duration_s / 60.0
        else:
            stats['duration_seconds'] = 0.0
            stats['duration_minutes'] = 0.0
        
        # Message counts
        stats['total_messages'] = sum(analysis_data['message_counts'].values())
        stats['message_counts_by_topic'] = dict(analysis_data['message_counts'])
        
        # Topic-specific statistics
        for topic, messages in analysis_data['messages'].items():
            topic_key = topic.replace('/', '_').lstrip('_')
            
            if topic == '/camera/image_raw':
                stats[f'{topic_key}_count'] = len(messages)
                if stats['duration_seconds'] > 0:
                    stats[f'{topic_key}_fps'] = len(messages) / stats['duration_seconds']
                
                # Image statistics
                if messages:
                    first_img = messages[0]
                    stats[f'{topic_key}_width'] = first_img.width
                    stats[f'{topic_key}_height'] = first_img.height
                    stats[f'{topic_key}_encoding'] = first_img.encoding
                    
            elif topic == '/cmd_vel_manual' or topic == '/cmd_vel':
                stats[f'{topic_key}_count'] = len(messages)
                
                # Command statistics
                if messages:
                    linear_x = [msg.linear.x for msg in messages]
                    angular_z = [msg.angular.z for msg in messages]
                    
                    stats[f'{topic_key}_linear_mean'] = np.mean(linear_x)
                    stats[f'{topic_key}_linear_std'] = np.std(linear_x)
                    stats[f'{topic_key}_linear_max'] = np.max(np.abs(linear_x))
                    
                    stats[f'{topic_key}_angular_mean'] = np.mean(angular_z)
                    stats[f'{topic_key}_angular_std'] = np.std(angular_z)
                    stats[f'{topic_key}_angular_max'] = np.max(np.abs(angular_z))
                    
                    # Movement statistics
                    moving_count = sum(1 for msg in messages 
                                     if abs(msg.linear.x) > 0.1 or abs(msg.angular.z) > 0.1)
                    stats[f'{topic_key}_movement_ratio'] = moving_count / len(messages)
                    
            elif topic == '/joy':
                stats[f'{topic_key}_count'] = len(messages)
                
                # Joystick activity statistics
                if messages:
                    active_count = 0
                    for msg in messages:
                        if any(abs(axis) > 0.1 for axis in msg.axes) or any(msg.buttons):
                            active_count += 1
                    
                    stats[f'{topic_key}_activity_ratio'] = active_count / len(messages)
                    
                    # Button press statistics
                    if messages and len(messages[0].buttons) > 0:
                        button_presses = defaultdict(int)
                        for msg in messages:
                            for i, button in enumerate(msg.buttons):
                                if button:
                                    button_presses[f'button_{i}'] += 1
                        stats[f'{topic_key}_button_presses'] = dict(button_presses)
        
        return stats
    
    def validate_bag_data(self, analysis_data: Dict) -> Dict:
        """Validate bag data quality for ML training."""
        
        validation = {'valid': True, 'issues': [], 'warnings': []}
        
        # Check required topics
        required_topics = ['/camera/image_raw', '/cmd_vel_manual', '/joy']
        missing_topics = [topic for topic in required_topics 
                         if topic not in analysis_data['message_counts']]
        
        if missing_topics:
            validation['valid'] = False
            validation['issues'].append(f"Missing required topics: {missing_topics}")
        
        # Check minimum duration
        if analysis_data['time_range']['start'] and analysis_data['time_range']['end']:
            duration_s = (analysis_data['time_range']['end'] - 
                         analysis_data['time_range']['start']) / 1e9
            
            if duration_s < 10.0:
                validation['warnings'].append(f"Short recording: {duration_s:.1f}s")
            
            # Check message rates
            for topic in required_topics:
                if topic in analysis_data['message_counts']:
                    count = analysis_data['message_counts'][topic]
                    rate = count / duration_s if duration_s > 0 else 0
                    
                    if topic == '/camera/image_raw' and rate < 10.0:
                        validation['warnings'].append(f"Low camera rate: {rate:.1f} fps")
                    elif topic == '/joy' and rate < 20.0:
                        validation['warnings'].append(f"Low joystick rate: {rate:.1f} Hz")
        
        # Check synchronization
        sync_issues = self.check_synchronization(analysis_data)
        if sync_issues:
            validation['warnings'].extend(sync_issues)
        
        # Check data diversity
        diversity_issues = self.check_data_diversity(analysis_data)
        if diversity_issues:
            validation['warnings'].extend(diversity_issues)
        
        return validation
    
    def check_synchronization(self, analysis_data: Dict) -> List[str]:
        """Check temporal synchronization between topics."""
        issues = []
        
        if ('/camera/image_raw' in analysis_data['timestamps'] and 
            '/cmd_vel_manual' in analysis_data['timestamps']):
            
            img_times = np.array(analysis_data['timestamps']['/camera/image_raw']) / 1e9
            cmd_times = np.array(analysis_data['timestamps']['/cmd_vel_manual']) / 1e9
            
            if len(img_times) > 1 and len(cmd_times) > 1:
                # Check time alignment
                img_start, img_end = img_times[0], img_times[-1]
                cmd_start, cmd_end = cmd_times[0], cmd_times[-1]
                
                # Check if time ranges overlap significantly
                overlap_start = max(img_start, cmd_start)
                overlap_end = min(img_end, cmd_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                total_duration = max(img_end, cmd_end) - min(img_start, cmd_start)
                overlap_ratio = overlap_duration / total_duration if total_duration > 0 else 0
                
                if overlap_ratio < 0.8:
                    issues.append(f"Poor time synchronization: {overlap_ratio:.1%} overlap")
        
        return issues
    
    def check_data_diversity(self, analysis_data: Dict) -> List[str]:
        """Check diversity of recorded data."""
        issues = []
        
        # Check command diversity
        if '/cmd_vel_manual' in analysis_data['messages']:
            commands = analysis_data['messages']['/cmd_vel_manual']
            
            if commands:
                linear_values = [msg.linear.x for msg in commands]
                angular_values = [msg.angular.z for msg in commands]
                
                # Check for movement diversity
                linear_range = max(linear_values) - min(linear_values)
                angular_range = max(angular_values) - min(angular_values)
                
                if linear_range < 0.5:
                    issues.append(f"Limited linear movement diversity: {linear_range:.2f}")
                
                if angular_range < 1.0:
                    issues.append(f"Limited angular movement diversity: {angular_range:.2f}")
                
                # Check for static periods
                static_count = sum(1 for msg in commands 
                                 if abs(msg.linear.x) < 0.05 and abs(msg.angular.z) < 0.05)
                static_ratio = static_count / len(commands)
                
                if static_ratio > 0.5:
                    issues.append(f"High static ratio: {static_ratio:.1%}")
        
        return issues
    
    def compute_overall_stats(self, stats_by_session: Dict) -> Dict:
        """Compute overall statistics across all sessions."""
        overall = {}
        
        for key, values in stats_by_session.items():
            if values:
                overall[f'{key}_mean'] = np.mean(values)
                overall[f'{key}_std'] = np.std(values)
                overall[f'{key}_min'] = np.min(values)
                overall[f'{key}_max'] = np.max(values)
                overall[f'{key}_total'] = np.sum(values) if key.endswith('_count') else np.mean(values)
        
        return overall
    
    def assess_dataset_quality(self, session_results: Dict) -> Dict:
        """Assess overall dataset quality for ML training."""
        
        valid_sessions = [r for r in session_results.values() if r.get('valid', False)]
        total_sessions = len(session_results)
        
        quality = {
            'overall_score': 0.0,
            'valid_sessions': len(valid_sessions),
            'total_sessions': total_sessions,
            'validity_ratio': len(valid_sessions) / total_sessions if total_sessions > 0 else 0,
            'quality_metrics': {},
            'grade': 'F'
        }
        
        if not valid_sessions:
            return quality
        
        # Compute quality metrics
        total_duration = sum(s['statistics'].get('duration_seconds', 0) for s in valid_sessions)
        total_images = sum(s['statistics'].get('camera_image_raw_count', 0) for s in valid_sessions)
        total_commands = sum(s['statistics'].get('cmd_vel_manual_count', 0) for s in valid_sessions)
        
        quality['quality_metrics'] = {
            'total_duration_minutes': total_duration / 60.0,
            'total_images': total_images,
            'total_commands': total_commands,
            'average_session_duration': total_duration / len(valid_sessions),
            'data_density': total_images / (total_duration / 60.0) if total_duration > 0 else 0
        }
        
        # Calculate quality score (0-100)
        score = 0
        
        # Duration score (30 points max)
        if total_duration > 1800:  # 30 minutes
            score += 30
        else:
            score += (total_duration / 1800) * 30
        
        # Data quantity score (25 points max)
        if total_images > 10000:
            score += 25
        else:
            score += (total_images / 10000) * 25
        
        # Session count score (20 points max)
        if len(valid_sessions) > 10:
            score += 20
        else:
            score += (len(valid_sessions) / 10) * 20
        
        # Validity ratio score (15 points max)
        score += quality['validity_ratio'] * 15
        
        # Diversity score (10 points max) - simplified
        avg_movement_ratio = np.mean([
            s['statistics'].get('cmd_vel_manual_movement_ratio', 0) 
            for s in valid_sessions
        ])
        score += min(avg_movement_ratio * 2, 1.0) * 10
        
        quality['overall_score'] = min(score, 100)
        
        # Assign grade
        if score >= 90:
            quality['grade'] = 'A'
        elif score >= 80:
            quality['grade'] = 'B'
        elif score >= 70:
            quality['grade'] = 'C'
        elif score >= 60:
            quality['grade'] = 'D'
        else:
            quality['grade'] = 'F'
        
        return quality
    
    def generate_recommendations(self, session_results: Dict) -> List[str]:
        """Generate recommendations for improving dataset quality."""
        recommendations = []
        
        valid_sessions = [r for r in session_results.values() if r.get('valid', False)]
        
        if not valid_sessions:
            recommendations.append("❌ No valid sessions found - check recording setup")
            return recommendations
        
        # Check total duration
        total_duration = sum(s['statistics'].get('duration_seconds', 0) for s in valid_sessions)
        if total_duration < 1800:  # 30 minutes
            recommendations.append(
                f"⏱️  Collect more data: {total_duration/60:.1f} min total "
                f"(recommend 30+ min for good results)"
            )
        
        # Check session diversity
        if len(valid_sessions) < 5:
            recommendations.append(
                "🔄 Record more diverse sessions (different lighting, paths, speeds)"
            )
        
        # Check movement diversity
        avg_movement = np.mean([
            s['statistics'].get('cmd_vel_manual_movement_ratio', 0) 
            for s in valid_sessions
        ])
        if avg_movement < 0.3:
            recommendations.append(
                f"🚗 Increase movement variety: {avg_movement:.1%} movement ratio "
                f"(recommend 40%+ active driving)"
            )
        
        # Check data quality issues
        issue_count = sum(len(s.get('validation', {}).get('issues', [])) for s in valid_sessions)
        warning_count = sum(len(s.get('validation', {}).get('warnings', [])) for s in valid_sessions)
        
        if issue_count > 0:
            recommendations.append(f"⚠️  Fix {issue_count} critical data issues")
        
        if warning_count > len(valid_sessions) * 2:
            recommendations.append(f"⚠️  Address {warning_count} data quality warnings")
        
        # Check frame rate
        avg_fps = np.mean([
            s['statistics'].get('camera_image_raw_fps', 0) 
            for s in valid_sessions
        ])
        if avg_fps < 20:
            recommendations.append(
                f"📸 Improve camera frame rate: {avg_fps:.1f} fps (recommend 25+ fps)"
            )
        
        return recommendations
    
    def load_session_metadata(self, bag_path: str) -> Optional[Dict]:
        """Load session metadata if available."""
        try:
            metadata_path = os.path.join(bag_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.get_logger().warn(f"⚠️  Failed to load metadata: {e}")
        return None
    
    def save_analysis_report(self, report: Dict, output_dir: str, filename: str):
        """Save analysis report to JSON file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            report_path = os.path.join(output_dir, f'{filename}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.get_logger().info(f"💾 Report saved: {report_path}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save report: {e}")
    
    def generate_visualizations(self, report: Dict, output_dir: str):
        """Generate visualization plots for the dataset analysis."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create figures directory
            figures_dir = os.path.join(output_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            # 1. Session duration distribution
            self.plot_session_durations(report, figures_dir)
            
            # 2. Data quality metrics
            self.plot_quality_metrics(report, figures_dir)
            
            # 3. Timeline visualization
            self.plot_recording_timeline(report, figures_dir)
            
            # 4. Command distribution
            self.plot_command_distributions(report, figures_dir)
            
            self.get_logger().info(f"📊 Visualizations saved to: {figures_dir}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to generate visualizations: {e}")
    
    def plot_session_durations(self, report: Dict, output_dir: str):
        """Plot session duration distribution."""
        session_results = report.get('session_results', {})
        valid_sessions = [r for r in session_results.values() if r.get('valid', False)]
        
        if not valid_sessions:
            return
        
        durations = [s['statistics'].get('duration_minutes', 0) for s in valid_sessions]
        
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(durations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(durations):.1f} min')
        plt.xlabel('Session Duration (minutes)')
        plt.ylabel('Number of Sessions')
        plt.title('Distribution of Recording Session Durations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'session_durations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_metrics(self, report: Dict, output_dir: str):
        """Plot dataset quality metrics."""
        quality = report.get('quality_assessment', {})
        metrics = quality.get('quality_metrics', {})
        
        if not metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall score gauge
        score = quality.get('overall_score', 0)
        grade = quality.get('grade', 'F')
        
        ax = axes[0, 0]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        values = [20, 20, 20, 20, 20]
        start_angle = 180
        
        wedges, texts = ax.pie(values, startangle=start_angle, colors=colors,
                              counterclock=False, wedgeprops=dict(width=0.3))
        
        # Add score needle
        angle = start_angle - (score / 100) * 180
        ax.annotate('', xy=(0.7*np.cos(np.radians(angle)), 0.7*np.sin(np.radians(angle))), 
                   xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        
        ax.text(0, -0.3, f'Score: {score:.1f}\nGrade: {grade}', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title('Dataset Quality Score')
        
        # Data metrics
        ax = axes[0, 1]
        metric_names = ['Duration (min)', 'Images (k)', 'Commands (k)', 'Sessions']
        metric_values = [
            metrics.get('total_duration_minutes', 0),
            metrics.get('total_images', 0) / 1000,
            metrics.get('total_commands', 0) / 1000,
            quality.get('valid_sessions', 0)
        ]
        
        bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'coral', 'gold'])
        ax.set_title('Dataset Size Metrics')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        
        # Validity ratio
        ax = axes[1, 0]
        valid_ratio = quality.get('validity_ratio', 0)
        ax.pie([valid_ratio, 1-valid_ratio], labels=['Valid', 'Invalid'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        ax.set_title('Session Validity')
        
        # Data density
        ax = axes[1, 1]
        density = metrics.get('data_density', 0)
        ax.bar(['Data Density'], [density], color='mediumpurple')
        ax.set_ylabel('Images per Minute')
        ax.set_title('Data Collection Density')
        ax.text(0, density + density*0.05, f'{density:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_recording_timeline(self, report: Dict, output_dir: str):
        """Plot recording timeline."""
        session_results = report.get('session_results', {})
        
        # Extract session times and durations
        sessions_data = []
        for name, result in session_results.items():
            if result.get('valid', False):
                metadata = result.get('metadata', {})
                session_info = metadata.get('session_info', {})
                
                start_time = session_info.get('start_time')
                duration = result['statistics'].get('duration_minutes', 0)
                
                if start_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        sessions_data.append({
                            'name': name,
                            'start': start_dt,
                            'duration': duration
                        })
                    except:
                        pass
        
        if not sessions_data:
            return
        
        sessions_data.sort(key=lambda x: x['start'])
        
        plt.figure(figsize=(12, 6))
        
        for i, session in enumerate(sessions_data):
            start_time = session['start']
            duration_td = timedelta(minutes=session['duration'])
            end_time = start_time + duration_td
            
            plt.barh(i, session['duration'], left=session['start'], 
                    alpha=0.7, color=plt.cm.Set3(i % 12))
            
            # Add session name
            plt.text(start_time + duration_td/2, i, session['name'], 
                    ha='center', va='center', fontsize=8, rotation=0)
        
        plt.xlabel('Recording Time')
        plt.ylabel('Session')
        plt.title('Recording Timeline')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        if sessions_data:
            min_time = min(s['start'] for s in sessions_data)
            max_time = max(s['start'] + timedelta(minutes=s['duration']) for s in sessions_data)
            plt.xlim(min_time, max_time)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recording_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_command_distributions(self, report: Dict, output_dir: str):
        """Plot command distribution analysis."""
        session_results = report.get('session_results', {})
        valid_sessions = [r for r in session_results.values() if r.get('valid', False)]
        
        if not valid_sessions:
            return
        
        # Collect command statistics
        linear_means = [s['statistics'].get('cmd_vel_manual_linear_mean', 0) for s in valid_sessions]
        angular_means = [s['statistics'].get('cmd_vel_manual_angular_mean', 0) for s in valid_sessions]
        movement_ratios = [s['statistics'].get('cmd_vel_manual_movement_ratio', 0) for s in valid_sessions]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Linear velocity distribution
        ax = axes[0, 0]
        ax.hist(linear_means, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(linear_means), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(linear_means):.2f}')
        ax.set_xlabel('Mean Linear Velocity (m/s)')
        ax.set_ylabel('Number of Sessions')
        ax.set_title('Linear Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angular velocity distribution  
        ax = axes[0, 1]
        ax.hist(angular_means, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(np.mean(angular_means), color='red', linestyle='--',
                  label=f'Mean: {np.mean(angular_means):.2f}')
        ax.set_xlabel('Mean Angular Velocity (rad/s)')
        ax.set_ylabel('Number of Sessions')
        ax.set_title('Angular Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Movement ratio distribution
        ax = axes[1, 0]
        ax.hist(movement_ratios, bins=15, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(movement_ratios), color='red', linestyle='--',
                  label=f'Mean: {np.mean(movement_ratios):.2f}')
        ax.set_xlabel('Movement Ratio')
        ax.set_ylabel('Number of Sessions')
        ax.set_title('Movement Activity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot: linear vs angular
        ax = axes[1, 1]
        scatter = ax.scatter(linear_means, angular_means, 
                           c=movement_ratios, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Mean Linear Velocity (m/s)')
        ax.set_ylabel('Mean Angular Velocity (rad/s)')
        ax.set_title('Linear vs Angular Velocity')
        plt.colorbar(scatter, ax=ax, label='Movement Ratio')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'command_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for the bag analyzer."""
    parser = argparse.ArgumentParser(description='Analyze ROS2 bags for behavior cloning dataset')
    parser.add_argument('path', help='Path to bag file or dataset directory')
    parser.add_argument('-o', '--output', help='Output directory for analysis results')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        analyzer = BagAnalyzer()
        
        # Set default output directory
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"analysis_{timestamp}"
        
        # Analyze dataset
        analyzer.get_logger().info(f"🔍 Starting analysis of: {args.path}")
        
        result = analyzer.analyze_dataset(args.path, args.output)
        
        # Print summary
        if 'quality_assessment' in result:
            quality = result['quality_assessment']
            analyzer.get_logger().info("📊 Analysis Summary:")
            analyzer.get_logger().info(f"  Quality Score: {quality['overall_score']:.1f}/100 (Grade: {quality['grade']})")
            analyzer.get_logger().info(f"  Valid Sessions: {quality['valid_sessions']}/{quality['total_sessions']}")
            
            metrics = quality.get('quality_metrics', {})
            if metrics:
                analyzer.get_logger().info(f"  Total Duration: {metrics.get('total_duration_minutes', 0):.1f} minutes")
                analyzer.get_logger().info(f"  Total Images: {metrics.get('total_images', 0):,}")
                analyzer.get_logger().info(f"  Total Commands: {metrics.get('total_commands', 0):,}")
        
        # Print recommendations
        if 'recommendations' in result and result['recommendations']:
            analyzer.get_logger().info("💡 Recommendations:")
            for rec in result['recommendations']:
                analyzer.get_logger().info(f"  {rec}")
        
        analyzer.get_logger().info(f"✅ Analysis complete. Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    finally:
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())