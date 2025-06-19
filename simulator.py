import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import random
import math


class EmployeeActivitySimulator:
    """
    Simulates realistic employee digital activity patterns across multiple data sources.
    Generates data compatible with the AllMetrics class output format.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the simulator with configurable parameters."""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Metric indices matching AllMetrics output
        self.metric_indices = {
            'email_count': 0, 'email_length': 1, 'email_frr': 2, 'email_recipients': 3,
            'meeting_count': 4, 'meeting_duration': 5, 'meeting_cancel_rate': 6,
            'meeting_virtual_ratio': 7, 'meeting_btb_ratio': 8,
            'teams_messages': 9, 'teams_msg_length': 10, 'teams_channels': 11,
            'docs_accessed': 12, 'docs_created': 13, 'docs_edited': 14, 'docs_shared': 15,
            'dropbox_added': 16, 'dropbox_edited': 17
        }
        
        # Department profiles with characteristic patterns
        self.department_profiles = {
            'engineering': {
                'email_volume': 'low', 'meeting_volume': 'medium', 'teams_activity': 'high',
                'doc_activity': 'high', 'work_hours': 'flexible', 'weekend_work': 0.3,
                'base_metrics': {
                    'email_count': (5, 15), 'email_length': (150, 350), 'email_frr': (0.6, 0.8),
                    'email_recipients': (1, 3), 'meeting_count': (2, 5), 'meeting_duration': (180, 360),
                    'meeting_cancel_rate': (0.05, 0.15), 'meeting_virtual_ratio': (0.7, 0.9),
                    'meeting_btb_ratio': (0.2, 0.4), 'teams_messages': (20, 50),
                    'teams_msg_length': (30, 80), 'teams_channels': (3, 8),
                    'docs_accessed': (10, 25), 'docs_created': (1, 4), 'docs_edited': (3, 10),
                    'docs_shared': (1, 3), 'dropbox_added': (2, 8), 'dropbox_edited': (3, 10)
                }
            },
            'sales': {
                'email_volume': 'high', 'meeting_volume': 'high', 'teams_activity': 'medium',
                'doc_activity': 'medium', 'work_hours': 'standard', 'weekend_work': 0.1,
                'base_metrics': {
                    'email_count': (20, 60), 'email_length': (100, 250), 'email_frr': (0.4, 0.6),
                    'email_recipients': (2, 8), 'meeting_count': (4, 10), 'meeting_duration': (240, 600),
                    'meeting_cancel_rate': (0.1, 0.25), 'meeting_virtual_ratio': (0.5, 0.7),
                    'meeting_btb_ratio': (0.3, 0.6), 'teams_messages': (10, 30),
                    'teams_msg_length': (40, 100), 'teams_channels': (2, 5),
                    'docs_accessed': (5, 15), 'docs_created': (1, 3), 'docs_edited': (2, 6),
                    'docs_shared': (2, 5), 'dropbox_added': (1, 4), 'dropbox_edited': (1, 4)
                }
            },
            'management': {
                'email_volume': 'high', 'meeting_volume': 'very_high', 'teams_activity': 'medium',
                'doc_activity': 'medium', 'work_hours': 'extended', 'weekend_work': 0.4,
                'base_metrics': {
                    'email_count': (25, 70), 'email_length': (200, 500), 'email_frr': (0.5, 0.7),
                    'email_recipients': (3, 15), 'meeting_count': (6, 15), 'meeting_duration': (360, 900),
                    'meeting_cancel_rate': (0.1, 0.2), 'meeting_virtual_ratio': (0.4, 0.6),
                    'meeting_btb_ratio': (0.5, 0.8), 'teams_messages': (15, 40),
                    'teams_msg_length': (50, 150), 'teams_channels': (4, 10),
                    'docs_accessed': (15, 35), 'docs_created': (2, 5), 'docs_edited': (5, 15),
                    'docs_shared': (3, 8), 'dropbox_added': (2, 6), 'dropbox_edited': (3, 8)
                }
            },
            'hr': {
                'email_volume': 'medium', 'meeting_volume': 'high', 'teams_activity': 'low',
                'doc_activity': 'high', 'work_hours': 'standard', 'weekend_work': 0.05,
                'base_metrics': {
                    'email_count': (15, 40), 'email_length': (250, 600), 'email_frr': (0.3, 0.5),
                    'email_recipients': (2, 10), 'meeting_count': (3, 8), 'meeting_duration': (180, 480),
                    'meeting_cancel_rate': (0.05, 0.15), 'meeting_virtual_ratio': (0.3, 0.5),
                    'meeting_btb_ratio': (0.2, 0.5), 'teams_messages': (5, 20),
                    'teams_msg_length': (60, 200), 'teams_channels': (1, 4),
                    'docs_accessed': (20, 50), 'docs_created': (2, 6), 'docs_edited': (8, 20),
                    'docs_shared': (2, 6), 'dropbox_added': (3, 10), 'dropbox_edited': (5, 15)
                }
            },
            'finance': {
                'email_volume': 'medium', 'meeting_volume': 'medium', 'teams_activity': 'low',
                'doc_activity': 'very_high', 'work_hours': 'standard', 'weekend_work': 0.2,
                'base_metrics': {
                    'email_count': (10, 30), 'email_length': (200, 400), 'email_frr': (0.5, 0.7),
                    'email_recipients': (2, 5), 'meeting_count': (2, 6), 'meeting_duration': (120, 360),
                    'meeting_cancel_rate': (0.05, 0.1), 'meeting_virtual_ratio': (0.4, 0.6),
                    'meeting_btb_ratio': (0.1, 0.3), 'teams_messages': (5, 15),
                    'teams_msg_length': (40, 120), 'teams_channels': (1, 3),
                    'docs_accessed': (25, 60), 'docs_created': (3, 8), 'docs_edited': (10, 30),
                    'docs_shared': (1, 4), 'dropbox_added': (5, 15), 'dropbox_edited': (8, 25)
                }
            }
        }
        
        # Anomaly patterns for injection
        self.anomaly_patterns = {
            'data_exfiltration': {
                'description': 'Sudden spike in document access/sharing and dropbox activity',
                'metrics_affected': ['docs_accessed', 'docs_shared', 'dropbox_added', 'email_count'],
                'multipliers': {'docs_accessed': 5.0, 'docs_shared': 8.0, 'dropbox_added': 10.0, 'email_count': 2.0}
            },
            'burnout': {
                'description': 'Gradual decline in all activities',
                'metrics_affected': 'all',
                'multipliers': 0.3  # Reduces all metrics to 30% of normal
            },
            'overwork': {
                'description': 'Excessive activity across all metrics',
                'metrics_affected': 'all',
                'multipliers': 2.5
            },
            'disengagement': {
                'description': 'Reduced communication and collaboration',
                'metrics_affected': ['email_count', 'teams_messages', 'meeting_count', 'docs_shared'],
                'multipliers': {'email_count': 0.2, 'teams_messages': 0.1, 'meeting_count': 0.3, 'docs_shared': 0.1}
            },
            'unusual_hours': {
                'description': 'Activity at unusual times (simulated as weekend spike)',
                'metrics_affected': 'all',
                'multipliers': 3.0,
                'weekend_only': True
            },
            'communication_spike': {
                'description': 'Sudden increase in all communication channels',
                'metrics_affected': ['email_count', 'teams_messages', 'meeting_count', 'teams_channels'],
                'multipliers': {'email_count': 4.0, 'teams_messages': 5.0, 'meeting_count': 3.0, 'teams_channels': 2.0}
            },
            'policy_violation': {
                'description': 'Unusual file sharing and external communication',
                'metrics_affected': ['email_recipients', 'docs_shared', 'dropbox_added', 'meeting_virtual_ratio'],
                'multipliers': {'email_recipients': 5.0, 'docs_shared': 10.0, 'dropbox_added': 8.0, 'meeting_virtual_ratio': 0.1}
            }
        }
        
        # Seasonal patterns
        self.seasonal_patterns = {
            'quarter_end': {'week_numbers': [12, 13, 25, 26, 38, 39, 51, 52], 'multiplier': 1.3},
            'summer_vacation': {'week_numbers': list(range(26, 35)), 'multiplier': 0.7},
            'holiday_season': {'week_numbers': [51, 52, 1], 'multiplier': 0.5},
            'new_year_surge': {'week_numbers': [2, 3, 4], 'multiplier': 1.2}
        }
        
        # Inter-metric correlations
        self.metric_correlations = {
            'meeting_count': {
                'email_count': 0.6,  # More meetings → more emails
                'teams_messages': 0.4,  # Some Teams discussion around meetings
                'docs_accessed': 0.5  # Meeting materials
            },
            'docs_created': {
                'docs_edited': 0.8,  # Created docs are often edited
                'docs_shared': 0.6,  # New docs are shared
                'email_count': 0.3   # Emails about new documents
            },
            'teams_messages': {
                'teams_channels': 0.7,  # More messages → more channels used
                'email_count': -0.3  # Teams can replace email
            }
        }
        
        # Store simulated data
        self.simulated_data = {}
        self.employee_profiles = {}
        
    def create_employee_profile(self, employee_id: str, department: str, 
                               seniority: str = 'mid', personality: Optional[Dict] = None) -> Dict:
        """Create an individual employee profile with personal variations."""
        if department not in self.department_profiles:
            raise ValueError(f"Unknown department: {department}")
        
        base_profile = self.department_profiles[department].copy()
        
        # Apply seniority modifiers
        seniority_modifiers = {
            'junior': {'activity_level': 0.8, 'weekend_work': 0.5},
            'mid': {'activity_level': 1.0, 'weekend_work': 1.0},
            'senior': {'activity_level': 1.2, 'weekend_work': 1.5},
            'executive': {'activity_level': 1.5, 'weekend_work': 2.0}
        }
        
        modifier = seniority_modifiers.get(seniority, seniority_modifiers['mid'])
        
        # Create personalized metrics with random variations
        personal_metrics = {}
        for metric, (low, high) in base_profile['base_metrics'].items():
            # Add personal variation (±20%)
            personal_variation = np.random.uniform(0.8, 1.2)
            personal_low = low * modifier['activity_level'] * personal_variation
            personal_high = high * modifier['activity_level'] * personal_variation
            personal_metrics[metric] = (personal_low, personal_high)
        
        # Personal work patterns
        profile = {
            'employee_id': employee_id,
            'department': department,
            'seniority': seniority,
            'base_metrics': personal_metrics,
            'weekend_work_probability': base_profile['weekend_work'] * modifier['weekend_work'],
            'work_hours': base_profile['work_hours'],
            'personality': personality or self._generate_personality()
        }
        
        self.employee_profiles[employee_id] = profile
        return profile
    
    def _generate_personality(self) -> Dict:
        """Generate random personality traits affecting work patterns."""
        return {
            'consistency': np.random.uniform(0.7, 1.0),  # How consistent their patterns are
            'morning_person': np.random.uniform(0, 1),   # Affects activity distribution
            'collaborative': np.random.uniform(0.3, 1.0), # Affects teams/meeting metrics
            'detail_oriented': np.random.uniform(0.5, 1.0), # Affects email length, doc edits
            'responsive': np.random.uniform(0.5, 1.0)    # Affects reply ratios
        }
    
    def _apply_weekday_pattern(self, base_value: float, weekday: int, metric_name: str) -> float:
        """Apply day-of-week patterns to metrics."""
        # Monday (0) to Friday (4) patterns
        weekday_multipliers = {
            0: 1.2,  # Monday - catch up from weekend
            1: 1.0,  # Tuesday - normal
            2: 1.0,  # Wednesday - normal
            3: 1.0,  # Thursday - normal
            4: 0.9,  # Friday - wind down
            5: 0.1,  # Saturday - minimal
            6: 0.05  # Sunday - very minimal
        }
        
        # Some metrics have different patterns
        if metric_name in ['meeting_count', 'meeting_duration']:
            # Fewer meetings on Monday/Friday
            weekday_multipliers[0] = 0.8
            weekday_multipliers[4] = 0.7
        
        return base_value * weekday_multipliers.get(weekday, 1.0)
    
    def _apply_seasonal_pattern(self, base_value: float, date: datetime) -> float:
        """Apply seasonal variations to metrics."""
        week_num = date.isocalendar()[1]
        
        for pattern_name, pattern in self.seasonal_patterns.items():
            if week_num in pattern['week_numbers']:
                return base_value * pattern['multiplier']
        
        return base_value
    
    def _apply_correlations(self, metrics: np.ndarray, primary_metric: str, 
                           primary_value: float) -> np.ndarray:
        """Apply inter-metric correlations for realism."""
        if primary_metric not in self.metric_correlations:
            return metrics
        
        correlations = self.metric_correlations[primary_metric]
        primary_idx = self.metric_indices[primary_metric]
        primary_deviation = (primary_value - metrics[primary_idx]) / max(metrics[primary_idx], 1)
        
        for correlated_metric, correlation_strength in correlations.items():
            if correlated_metric in self.metric_indices:
                idx = self.metric_indices[correlated_metric]
                # Apply correlation-based adjustment
                adjustment = primary_deviation * correlation_strength * 0.3
                metrics[idx] *= (1 + adjustment)
        
        return metrics
    
    def _generate_daily_metrics(self, profile: Dict, date: datetime) -> np.ndarray:
        """Generate metrics for a single day for an employee."""
        metrics = np.zeros(18)
        weekday = date.weekday()
        
        # Check if it's a weekend and if employee works weekends
        if weekday >= 5:  # Weekend
            if np.random.random() > profile['weekend_work_probability']:
                # No weekend work
                return metrics
        
        # Generate each metric
        for metric_name, (low, high) in profile['base_metrics'].items():
            idx = self.metric_indices[metric_name]
            
            # Base value with personality influence
            personality = profile['personality']
            base_value = np.random.uniform(low, high)
            
            # Apply personality modifiers
            if 'email' in metric_name and metric_name != 'email_frr':
                base_value *= (0.7 + 0.6 * personality['responsive'])
            elif 'teams' in metric_name:
                base_value *= (0.5 + personality['collaborative'])
            elif metric_name == 'email_length':
                base_value *= (0.6 + 0.8 * personality['detail_oriented'])
            
            # Apply patterns
            value = self._apply_weekday_pattern(base_value, weekday, metric_name)
            value = self._apply_seasonal_pattern(value, date)
            
            # Add daily random variation
            daily_variation = np.random.normal(1.0, 0.1 * (2 - personality['consistency']))
            value *= max(0, daily_variation)
            
            # Special handling for ratios (keep between 0 and 1)
            if metric_name in ['email_frr', 'meeting_cancel_rate', 'meeting_virtual_ratio', 'meeting_btb_ratio']:
                value = np.clip(value, 0, 1)
            
            metrics[idx] = max(0, value)
        
        # Apply inter-metric correlations
        # Randomly select a primary driver metric
        driver_metrics = ['meeting_count', 'docs_created', 'teams_messages']
        for driver in driver_metrics:
            if np.random.random() < 0.3:  # 30% chance to apply each correlation
                idx = self.metric_indices[driver]
                metrics = self._apply_correlations(metrics, driver, metrics[idx])
        
        return np.round(metrics, 2)
    
    def simulate_employees(self, employee_configs: List[Dict], 
                          start_date: str, end_date: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Simulate activity data for multiple employees over a date range.
        
        Args:
            employee_configs: List of dicts with keys 'id', 'department', 'seniority'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary matching AllMetrics output format
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create employee profiles
        for config in employee_configs:
            if config['id'] not in self.employee_profiles:
                self.create_employee_profile(
                    config['id'], 
                    config['department'],
                    config.get('seniority', 'mid'),
                    config.get('personality', None)
                )
        
        # Generate daily metrics
        self.simulated_data = {}
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            
            for config in employee_configs:
                emp_id = config['id']
                profile = self.employee_profiles[emp_id]
                
                if emp_id not in self.simulated_data:
                    self.simulated_data[emp_id] = {}
                
                daily_metrics = self._generate_daily_metrics(profile, current_date)
                self.simulated_data[emp_id][date_str] = daily_metrics
            
            current_date += timedelta(days=1)
        
        return self.simulated_data
    
    def inject_anomaly(self, employee_id: str, anomaly_type: str, 
                      start_date: str, end_date: Optional[str] = None,
                      intensity: float = 1.0) -> None:
        """
        Inject anomalous patterns into existing simulated data.
        
        Args:
            employee_id: Employee to affect
            anomaly_type: Type of anomaly from self.anomaly_patterns
            start_date: Start date of anomaly
            end_date: End date (if None, single day anomaly)
            intensity: Multiplier for anomaly strength (default 1.0)
        """
        if employee_id not in self.simulated_data:
            raise ValueError(f"Employee {employee_id} not found in simulated data")
        
        if anomaly_type not in self.anomaly_patterns:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        pattern = self.anomaly_patterns[anomaly_type]
        affected_metrics = pattern['metrics_affected']
        multipliers = pattern['multipliers']
        
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else start
        
        # Apply anomaly
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            if date_str in self.simulated_data[employee_id]:
                metrics = self.simulated_data[employee_id][date_str].copy()
                
                # Check weekend-only anomalies
                if pattern.get('weekend_only', False) and current.weekday() < 5:
                    current += timedelta(days=1)
                    continue
                
                # Apply multipliers
                if affected_metrics == 'all':
                    # Apply to all metrics
                    if isinstance(multipliers, dict):
                        for metric, mult in multipliers.items():
                            if metric in self.metric_indices:
                                idx = self.metric_indices[metric]
                                metrics[idx] *= mult * intensity
                    else:
                        metrics *= multipliers * intensity
                else:
                    # Apply to specific metrics
                    for metric in affected_metrics:
                        if metric in self.metric_indices:
                            idx = self.metric_indices[metric]
                            if isinstance(multipliers, dict):
                                metrics[idx] *= multipliers.get(metric, 1.0) * intensity
                            else:
                                metrics[idx] *= multipliers * intensity
                
                # Ensure ratios stay in bounds
                ratio_indices = [self.metric_indices[m] for m in 
                               ['email_frr', 'meeting_cancel_rate', 'meeting_virtual_ratio', 'meeting_btb_ratio']]
                for idx in ratio_indices:
                    metrics[idx] = np.clip(metrics[idx], 0, 1)
                
                self.simulated_data[employee_id][date_str] = np.round(metrics, 2)
            
            current += timedelta(days=1)
        
        print(f"Injected {anomaly_type} anomaly for {employee_id} from {start_date} to {end_date}")
    
    def inject_gradual_anomaly(self, employee_id: str, anomaly_type: str,
                             start_date: str, peak_date: str, end_date: str,
                             peak_intensity: float = 1.0) -> None:
        """
        Inject gradually developing anomaly (e.g., burnout, slow data exfiltration).
        
        The anomaly grows from 0 to peak_intensity at peak_date, then optionally declines.
        """
        if employee_id not in self.simulated_data:
            raise ValueError(f"Employee {employee_id} not found")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        peak = datetime.strptime(peak_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate phases
        growth_days = (peak - start).days
        decline_days = (end - peak).days
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Calculate intensity for this day
            if current <= peak:
                # Growth phase
                days_from_start = (current - start).days
                intensity = (days_from_start / growth_days) * peak_intensity if growth_days > 0 else peak_intensity
            else:
                # Decline phase (optional)
                days_from_peak = (current - peak).days
                intensity = peak_intensity * (1 - days_from_peak / decline_days) if decline_days > 0 else peak_intensity
            
            # Apply anomaly with calculated intensity
            if date_str in self.simulated_data[employee_id]:
                self.inject_anomaly(employee_id, anomaly_type, date_str, date_str, intensity)
            
            current += timedelta(days=1)
    
    def add_team_correlation(self, team_members: List[str], correlation_strength: float = 0.5,
                           start_date: str = None, end_date: str = None) -> None:
        """
        Add correlation between team members' activities (e.g., shared meetings, projects).
        """
        if not all(emp in self.simulated_data for emp in team_members):
            raise ValueError("All team members must exist in simulated data")
        
        # Get date range
        all_dates = set()
        for emp in team_members:
            all_dates.update(self.simulated_data[emp].keys())
        
        dates_to_process = sorted(all_dates)
        if start_date:
            dates_to_process = [d for d in dates_to_process if d >= start_date]
        if end_date:
            dates_to_process = [d for d in dates_to_process if d <= end_date]
        
        # Apply correlations
        correlation_metrics = ['meeting_count', 'teams_channels', 'docs_shared']
        
        for date in dates_to_process:
            # Calculate team average for correlation metrics
            team_averages = {}
            for metric in correlation_metrics:
                idx = self.metric_indices[metric]
                values = [self.simulated_data[emp][date][idx] 
                         for emp in team_members 
                         if date in self.simulated_data[emp]]
                if values:
                    team_averages[metric] = np.mean(values)
            
            # Adjust each member toward team average
            for emp in team_members:
                if date in self.simulated_data[emp]:
                    metrics = self.simulated_data[emp][date].copy()
                    
                    for metric, avg in team_averages.items():
                        idx = self.metric_indices[metric]
                        current = metrics[idx]
                        # Move toward team average by correlation strength
                        metrics[idx] = current + (avg - current) * correlation_strength
                    
                    self.simulated_data[emp][date] = np.round(metrics, 2)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for all simulated employees."""
        summary_data = []
        
        for emp_id, emp_data in self.simulated_data.items():
            if not emp_data:
                continue
            
            # Convert to array for easier computation
            all_metrics = np.array(list(emp_data.values()))
            
            profile = self.employee_profiles.get(emp_id, {})
            
            summary = {
                'employee_id': emp_id,
                'department': profile.get('department', 'unknown'),
                'seniority': profile.get('seniority', 'unknown'),
                'total_days': len(emp_data),
                'weekend_days_active': sum(1 for date in emp_data.keys() 
                                         if datetime.strptime(date, '%Y-%m-%d').weekday() >= 5)
            }
            
            # Add metric averages
            metric_names = list(self.metric_indices.keys())
            for i, metric in enumerate(metric_names):
                summary[f'avg_{metric}'] = np.mean(all_metrics[:, i])
                summary[f'std_{metric}'] = np.std(all_metrics[:, i])
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def export_for_anomaly_detector(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Export data in format expected by AnomalyDetector class."""
        return self.simulated_data.copy()
    
    def visualize_employee_timeline(self, employee_id: str, 
                                  metrics_to_plot: Optional[List[str]] = None) -> None:
        """Create a simple text visualization of employee metrics over time."""
        if employee_id not in self.simulated_data:
            print(f"Employee {employee_id} not found")
            return
        
        if not metrics_to_plot:
            metrics_to_plot = ['email_count', 'meeting_count', 'teams_messages', 'docs_accessed']
        
        print(f"\nActivity Timeline for {employee_id}")
        print(f"Department: {self.employee_profiles[employee_id]['department']}")
        print(f"Seniority: {self.employee_profiles[employee_id]['seniority']}")
        print("-" * 80)
        
        dates = sorted(self.simulated_data[employee_id].keys())[-30:]  # Last 30 days
        
        for metric in metrics_to_plot:
            if metric not in self.metric_indices:
                continue
            
            idx = self.metric_indices[metric]
            values = [self.simulated_data[employee_id][date][idx] for date in dates]
            max_val = max(values) if values else 1
            
            print(f"\n{metric}:")
            for date, value in zip(dates, values):
                bar_length = int((value / max_val) * 50) if max_val > 0 else 0
                bar = "█" * bar_length
                print(f"{date}: {bar} {value:.1f}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize simulator
    simulator = EmployeeActivitySimulator()
    
    # Define employees
    employees = [
        {'id': 'john.doe@company.com', 'department': 'engineering', 'seniority': 'senior'},
        {'id': 'jane.smith@company.com', 'department': 'sales', 'seniority': 'mid'},
        {'id': 'bob.johnson@company.com', 'department': 'management', 'seniority': 'executive'},
        {'id': 'alice.brown@company.com', 'department': 'hr', 'seniority': 'senior'},
        {'id': 'charlie.wilson@company.com', 'department': 'engineering', 'seniority': 'junior'},
    ]
    
    # Simulate 90 days of normal activity
    print("Simulating normal activity...")
    data = simulator.simulate_employees(employees, '2024-09-01', '2024-11-30')
    
    # Add team correlation for engineering team
    print("\nAdding team correlation for engineering...")
    simulator.add_team_correlation(
        ['john.doe@company.com', 'charlie.wilson@company.com'], 
        correlation_strength=0.6
    )
    
    # Inject various anomalies
    print("\nInjecting anomalies...")
    
    # Data exfiltration - sudden spike
    simulator.inject_anomaly(
        'john.doe@company.com', 
        'data_exfiltration', 
        '2024-11-15', 
        '2024-11-17',
        intensity=1.5
    )
    
    # Burnout - gradual pattern
    simulator.inject_gradual_anomaly(
        'jane.smith@company.com',
        'burnout',
        '2024-10-15',
        '2024-11-01',
        '2024-11-20',
        peak_intensity=1.0
    )
    
    # Unusual hours - weekend activity
    simulator.inject_anomaly(
        'charlie.wilson@company.com',
        'unusual_hours',
        '2024-11-09',  # Saturday
        '2024-11-10',  # Sunday
        intensity=1.0
    )
    
    # Print summary
    print("\nSimulation Summary:")
    summary = simulator.get_summary_statistics()
    print(summary[['employee_id', 'department', 'total_days', 'weekend_days_active']].to_string())
    
    # Visualize one employee
    print("\nVisualizing anomalous activity for john.doe@company.com (last 30 days):")
    simulator.visualize_employee_timeline('john.doe@company.com')
    
    # Export for anomaly detector
    final_data = simulator.export_for_anomaly_detector()
    print(f"\nExported data for {len(final_data)} employees")
    print(f"Ready for use with AnomalyDetector class!")