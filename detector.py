import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from functools import lru_cache


class AnomalyDetector:
    """
    Fixed anomaly detector that properly handles weekday/weekend patterns.
    
    Key fixes:
    1. Proper weekday-aware baselines to prevent weekend contamination
    2. Correct historical data filtering (was including future data)
    3. Better handling of insufficient data cases
    4. More sensitive weekend detection
    """
    
    def __init__(self, unified_data: Dict[str, Dict[str, np.ndarray]], 
                 min_baseline_days: int = 30,
                 lookback_window: int = 90,
                 threshold_personal: float = 3.0,
                 threshold_group: float = 3.0,
                 enable_group_detection: bool = False,
                 metric_weights: Optional[Dict[str, float]] = None,
                 strict_weekday_separation: bool = True,
                 weekend_threshold_factor: float = 0.7,
                 drop_threshold: float = 0.8):
        """
        Initialize anomaly detector with fixed weekday handling and asymmetric detection.
        
        Args:
            unified_data: Dictionary of user -> date -> metrics array
            min_baseline_days: Minimum days needed for baseline
            lookback_window: Days to look back for baseline
            threshold_personal: Z-score threshold for anomalies
            threshold_group: Group comparison threshold
            enable_group_detection: Enable peer comparison
            metric_weights: Custom weights for metrics
            strict_weekday_separation: If True, compare weekdays only to same weekday type
            weekend_threshold_factor: Multiply threshold by this for weekends (lower = more sensitive)
            drop_threshold: Percentage drop (0.8 = 80%) in count metrics to flag as anomaly
        """
        self.unified_data = unified_data
        self.min_baseline_days = min_baseline_days
        self.lookback_window = lookback_window
        self.threshold_personal = threshold_personal
        self.threshold_group = threshold_group
        self.enable_group_detection = enable_group_detection
        self.strict_weekday_separation = strict_weekday_separation
        self.weekend_threshold_factor = weekend_threshold_factor
        self.drop_threshold = drop_threshold
        self.num_metrics = 18
        
        # Metric information
        self.metric_names = [
            'Email_Count', 'Email_Avg_Length', 'Email_FRR', 'Email_Recipients',
            'Meeting_Count', 'Meeting_Duration', 'Meeting_Cancel_Rate', 
            'Meeting_Virtual_Ratio', 'Meeting_BTB_Ratio',
            'Teams_Messages', 'Teams_Median_Length', 'Teams_Channels',
            'Docs_Accessed', 'Docs_Created', 'Docs_Edited', 'Docs_Shared',
            'Dropbox_Added', 'Dropbox_Edited'
        ]
        
        # Metric types for different statistical handling
        self.metric_types = {
            'count': [0, 3, 4, 9, 11, 12, 13, 14, 15, 16, 17],
            'continuous': [1, 5, 10],
            'ratio': [2, 6, 7, 8]
        }
        
        # Metric indices for quick lookup
        self.metric_indices = {
            'email_count': 0, 'email_length': 1, 'email_frr': 2, 'email_recipients': 3,
            'meeting_count': 4, 'meeting_duration': 5, 'meeting_cancel_rate': 6,
            'meeting_virtual_ratio': 7, 'meeting_btb_ratio': 8,
            'teams_messages': 9, 'teams_msg_length': 10, 'teams_channels': 11,
            'docs_accessed': 12, 'docs_created': 13, 'docs_edited': 14, 'docs_shared': 15,
            'dropbox_added': 16, 'dropbox_edited': 17
        }
        
        # Set and normalize metric weights
        if metric_weights is None:
            self.metric_weights = np.ones(self.num_metrics)
        else:
            self.metric_weights = np.array([
                metric_weights.get(name, 1.0) for name in self.metric_names
            ])
        self.metric_weights = self.metric_weights / np.sum(self.metric_weights) * self.num_metrics
        
        # Pre-compute data structures
        self._preprocess_data()
        
        # Results storage
        self.anomaly_results = {}
        self.baseline_quality = {}
        
        # Process data
        self._validate_data()
        self._compute_anomalies()
    
    def _preprocess_data(self):
        """Pre-process data into efficient structures."""
        print("Preprocessing data structures...")
        
        self.user_data_arrays = {}
        self.user_dates = {}
        self.date_to_weekday = {}
        self.date_to_timestamp = {}
        self.user_weekday_dates = {}
        
        for user, user_data in self.unified_data.items():
            dates = sorted(user_data.keys())
            self.user_dates[user] = dates
            
            # Stack all metrics into a single array
            metrics_array = np.array([user_data[date] for date in dates])
            self.user_data_arrays[user] = metrics_array
            
            # Organize dates by weekday
            weekday_dates = defaultdict(list)
            
            # Cache date conversions
            for date in dates:
                if date not in self.date_to_weekday:
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    weekday = dt.weekday()
                    self.date_to_weekday[date] = weekday
                    self.date_to_timestamp[date] = dt.timestamp()
                    
                weekday_dates[self.date_to_weekday[date]].append(date)
            
            self.user_weekday_dates[user] = dict(weekday_dates)
        
        # Pre-compute global date mappings
        all_dates = set()
        for dates in self.user_dates.values():
            all_dates.update(dates)
        
        self.all_dates_sorted = sorted(all_dates)
        self.date_to_index = {date: i for i, date in enumerate(self.all_dates_sorted)}
    
    def _validate_data(self):
        """Validate input data and filter users with insufficient data."""
        print("Validating data...")
        
        valid_users = {}
        for user, user_data in self.unified_data.items():
            if len(user_data) >= self.min_baseline_days:
                valid_users[user] = user_data
            else:
                print(f"Warning: User {user} has only {len(user_data)} days of data (minimum: {self.min_baseline_days})")
        
        self.unified_data = valid_users
        self.user_data_arrays = {k: v for k, v in self.user_data_arrays.items() if k in valid_users}
        self.user_dates = {k: v for k, v in self.user_dates.items() if k in valid_users}
        
        print(f"Processing {len(self.unified_data)} users with sufficient data")
    
    @lru_cache(maxsize=1000)
    def _get_weekday(self, date_str: str) -> int:
        """Cached weekday computation."""
        return self.date_to_weekday.get(date_str, datetime.strptime(date_str, '%Y-%m-%d').weekday())
    
    @lru_cache(maxsize=1000)
    def _is_weekend(self, date_str: str) -> bool:
        """Cached weekend check."""
        return self._get_weekday(date_str) >= 5
    
    def _get_historical_data_weekday_aware(self, user: str, target_date: str) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Get historical data with proper weekday separation.
        
        Returns dictionary with:
        - 'same_weekday': Only the same day of week
        - 'all_weekdays': All Mon-Fri
        - 'weekends': All Sat-Sun
        - 'all': Everything
        """
        if user not in self.user_dates:
            empty = (np.array([]), [])
            return {'same_weekday': empty, 'all_weekdays': empty, 'weekends': empty, 'all': empty}
        
        dates = self.user_dates[user]
        metrics_array = self.user_data_arrays[user]
        target_timestamp = self.date_to_timestamp[target_date]
        target_weekday = self._get_weekday(target_date)
        
        # Initialize collections
        historical = {
            'same_weekday': ([], []),
            'all_weekdays': ([], []),
            'weekends': ([], []),
            'all': ([], [])
        }
        
        # Collect historical data
        for i, date in enumerate(dates):
            # CRITICAL FIX: Skip if date is on or after target date
            if date >= target_date:
                continue
                
            date_timestamp = self.date_to_timestamp[date]
            days_diff = (target_timestamp - date_timestamp) / (24 * 3600)
            
            # Only use data within lookback window
            if 1 <= days_diff <= self.lookback_window:
                weekday = self._get_weekday(date)
                metrics = metrics_array[i]
                
                # Add to 'all'
                historical['all'][0].append(metrics)
                historical['all'][1].append(date)
                
                # Add to specific categories
                if weekday == target_weekday:
                    historical['same_weekday'][0].append(metrics)
                    historical['same_weekday'][1].append(date)
                
                if weekday < 5:  # Weekday
                    historical['all_weekdays'][0].append(metrics)
                    historical['all_weekdays'][1].append(date)
                else:  # Weekend
                    historical['weekends'][0].append(metrics)
                    historical['weekends'][1].append(date)
        
        # Convert to numpy arrays
        for key in historical:
            if historical[key][0]:
                historical[key] = (np.array(historical[key][0]), historical[key][1])
            else:
                historical[key] = (np.array([]), [])
        
        return historical
    
    def _robust_baseline_stats(self, data: np.ndarray, data_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute robust baseline statistics with type-aware handling.
        
        Args:
            data: Historical data array
            data_type: Type of data ('same_weekday', 'weekends', 'mixed')
        """
        if len(data) == 0:
            return np.zeros(self.num_metrics), np.full(self.num_metrics, 1e-6), 0
        
        # Compute medians
        medians = np.median(data, axis=0)
        
        # Compute MAD
        abs_deviations = np.abs(data - medians)
        mads = np.median(abs_deviations, axis=0)
        
        # Apply type-specific minimum MAD values
        if data_type == 'weekends':
            # For weekends, use tighter bounds (more sensitive)
            for metric_idx in self.metric_types['count']:
                mads[metric_idx] = max(mads[metric_idx], 0.5)
            for metric_idx in self.metric_types['ratio']:
                mads[metric_idx] = max(mads[metric_idx], 0.05)
        elif data_type == 'same_weekday':
            # For same weekday comparison, use moderate bounds
            for metric_idx in self.metric_types['count']:
                mads[metric_idx] = max(mads[metric_idx], 1.0)
            for metric_idx in self.metric_types['ratio']:
                mads[metric_idx] = max(mads[metric_idx], 0.1)
        else:
            # For mixed data, use standard bounds
            for metric_idx in self.metric_types['count']:
                mads[metric_idx] = max(mads[metric_idx], 2.0)
            for metric_idx in self.metric_types['ratio']:
                mads[metric_idx] = max(mads[metric_idx], 0.1)
        
        # Ensure no zero MADs
        mads = np.maximum(mads, 1e-6)
        
        return medians, mads, len(data)
    
    def _compute_seasonal_adjustments(self, user: str, target_date: str, 
                                    historical_data: Dict) -> np.ndarray:
        """
        Compute seasonal adjustments based on weekday patterns.
        """
        target_weekday = self._get_weekday(target_date)
        
        # If we have enough same-weekday data, compute adjustment
        same_weekday_data, _ = historical_data.get('same_weekday', (np.array([]), []))
        all_data, _ = historical_data.get('all', (np.array([]), []))
        
        if len(same_weekday_data) >= 3 and len(all_data) >= 10:
            # Compute weekday-specific median
            weekday_median = np.median(same_weekday_data, axis=0)
            # Compute overall median
            overall_median = np.median(all_data, axis=0)
            # Adjustment is the difference
            adjustment = weekday_median - overall_median
            return adjustment
        else:
            return np.zeros(self.num_metrics)
    
    def _compute_personal_anomaly(self, user: str, date: str) -> Dict:
        """
        Compute personal anomaly with weekday-aware baselines and asymmetric bounds.
        """
        metrics = self.unified_data[user][date]
        weekday = self._get_weekday(date)
        is_weekend = weekday >= 5
        
        # Get historical data
        historical = self._get_historical_data_weekday_aware(user, date)
        
        # Select appropriate baseline based on strict_weekday_separation
        if self.strict_weekday_separation and is_weekend:
            # For weekends, use weekend-only data if available
            baseline_data, baseline_dates = historical['weekends']
            baseline_type = 'weekends'
            
            # If very few weekend samples, flag as minimal weekend worker
            if len(baseline_data) < 4:
                baseline_type = 'minimal_weekend'
        elif self.strict_weekday_separation and not is_weekend:
            # For weekdays, prefer same weekday data
            if len(historical['same_weekday'][0]) >= 4:
                baseline_data, baseline_dates = historical['same_weekday']
                baseline_type = 'same_weekday'
            else:
                baseline_data, baseline_dates = historical['all_weekdays']
                baseline_type = 'all_weekdays'
        else:
            # Non-strict mode: use all data
            baseline_data, baseline_dates = historical['all']
            baseline_type = 'all'
        
        # Check if we have enough data
        if len(baseline_data) < self.min_baseline_days and baseline_type == 'all':
            return {
                'personal_z_scores': np.zeros(self.num_metrics),
                'seasonal_adjusted_z_scores': np.zeros(self.num_metrics),
                'baseline_quality': 'insufficient_data',
                'baseline_days': len(baseline_data),
                'baseline_type': baseline_type,
                'overall_personal_score': 0.0,
                'is_personal_anomaly': False,
                'seasonal_adjustments': np.zeros(self.num_metrics)
            }
        
        # For minimal weekend workers, any significant weekend activity is anomalous
        if baseline_type == 'minimal_weekend':
            total_activity = np.sum(metrics[[0, 4, 9, 12]])  # Email, meetings, teams, docs
            if total_activity > 5:
                # Create artificial high z-scores for weekend activity
                z_scores = np.where(metrics > 0, 5.0, 0.0)  # High z-score for any activity
                return {
                    'personal_z_scores': z_scores,
                    'seasonal_adjusted_z_scores': z_scores,
                    'baseline_quality': 'minimal_weekend',
                    'baseline_days': len(baseline_data),
                    'baseline_type': baseline_type,
                    'overall_personal_score': np.mean(np.abs(z_scores) * self.metric_weights),
                    'is_personal_anomaly': True,
                    'seasonal_adjustments': np.zeros(self.num_metrics),
                    'anomaly_reason': 'unusual_weekend_activity'
                }
        
        # Compute baseline statistics
        medians, mads, sample_size = self._robust_baseline_stats(baseline_data, baseline_type)
        
        # Compute seasonal adjustments
        seasonal_adj = self._compute_seasonal_adjustments(user, date, historical)
        
        # Compute Z-scores with asymmetric handling
        scaled_mads = mads * 1.4826
        personal_z_scores = np.zeros_like(metrics)
        
        for i in range(self.num_metrics):
            if scaled_mads[i] > 0:
                z_score = (metrics[i] - medians[i]) / scaled_mads[i]
                personal_z_scores[i] = z_score
                
                # For count metrics, also check relative drop
                if i in self.metric_types['count'] and medians[i] > 5:
                    # If metric dropped by more than 80% from median, it's anomalous
                    relative_drop = (medians[i] - metrics[i]) / medians[i]
                    if relative_drop > 0.8:  # 80% drop
                        # Amplify the z-score to ensure detection
                        personal_z_scores[i] = min(personal_z_scores[i], -3.5)
                    elif relative_drop > 0.9:  # 90% drop
                        personal_z_scores[i] = min(personal_z_scores[i], -5.0)
        
        # Apply seasonal adjustments
        adjusted_expected = medians + seasonal_adj
        seasonal_z_scores = np.zeros_like(metrics)
        
        for i in range(self.num_metrics):
            if scaled_mads[i] > 0:
                z_score = (metrics[i] - adjusted_expected[i]) / scaled_mads[i]
                seasonal_z_scores[i] = z_score
                
                # Apply same relative drop check for seasonal scores
                if i in self.metric_types['count'] and adjusted_expected[i] > 5:
                    relative_drop = (adjusted_expected[i] - metrics[i]) / adjusted_expected[i]
                    if relative_drop > 0.8:
                        seasonal_z_scores[i] = min(seasonal_z_scores[i], -3.5)
                    elif relative_drop > 0.9:
                        seasonal_z_scores[i] = min(seasonal_z_scores[i], -5.0)
        
        # Overall anomaly score with special handling for extreme drops
        weighted_abs_z_scores = np.zeros_like(seasonal_z_scores)
        
        for i in range(self.num_metrics):
            # For count metrics, use asymmetric scoring
            if i in self.metric_types['count']:
                if seasonal_z_scores[i] < 0:  # Drop in activity
                    # More sensitive to drops - use squared z-score for drops
                    weighted_abs_z_scores[i] = abs(seasonal_z_scores[i]) * self.metric_weights[i] * 1.5
                else:  # Increase in activity
                    weighted_abs_z_scores[i] = abs(seasonal_z_scores[i]) * self.metric_weights[i]
            else:
                # For other metrics, use standard absolute z-score
                weighted_abs_z_scores[i] = abs(seasonal_z_scores[i]) * self.metric_weights[i]
        
        overall_score = np.mean(weighted_abs_z_scores)
        
        # Additional check for severe drops in key metrics
        key_activity_metrics = [
            self.metric_indices.get('email_count', 0),
            self.metric_indices.get('meeting_count', 4),
            self.metric_indices.get('teams_messages', 9),
            self.metric_indices.get('docs_accessed', 12)
        ]
        
        severe_drop_detected = False
        drop_reasons = []
        
        for idx in key_activity_metrics:
            if idx < len(metrics) and medians[idx] > 5:  # Only check if baseline is meaningful
                current_val = metrics[idx]
                expected_val = adjusted_expected[idx]
                
                # Check for severe drops
                if current_val < expected_val * 0.2:  # Less than 20% of expected
                    severe_drop_detected = True
                    metric_name = self.metric_names[idx]
                    drop_pct = (1 - current_val / expected_val) * 100
                    drop_reasons.append(f'{metric_name}_dropped_{drop_pct:.0f}%')
        
        # Determine threshold based on day type
        if is_weekend:
            effective_threshold = self.threshold_personal * self.weekend_threshold_factor
        else:
            effective_threshold = self.threshold_personal
        
        # Determine baseline quality
        if baseline_type in ['same_weekday', 'weekends']:
            if sample_size >= 8:
                quality = 'high'
            elif sample_size >= 4:
                quality = 'medium'
            else:
                quality = 'low'
        else:
            if sample_size >= self.lookback_window * 0.5:
                quality = 'high'
            elif sample_size >= self.min_baseline_days:
                quality = 'medium'
            else:
                quality = 'low'
        
        # Determine if it's an anomaly
        is_anomaly = False
        anomaly_reasons = []
        
        # Standard threshold check
        if overall_score > effective_threshold:
            is_anomaly = True
            anomaly_reasons.append('statistical_deviation')
        
        # Severe drop check (overrides threshold)
        if severe_drop_detected:
            is_anomaly = True
            anomaly_reasons.extend(drop_reasons)
            # Boost the score to reflect severity
            overall_score = max(overall_score, effective_threshold + 1.0)
        
        # Existing anomaly reason from minimal weekend detection
        if 'anomaly_reason' in locals():
            anomaly_reasons.append(locals()['anomaly_reason'])
        
        return {
            'personal_z_scores': personal_z_scores,
            'seasonal_adjusted_z_scores': seasonal_z_scores,
            'baseline_quality': quality,
            'baseline_days': sample_size,
            'baseline_type': baseline_type,
            'overall_personal_score': overall_score,
            'is_personal_anomaly': is_anomaly,
            'seasonal_adjustments': seasonal_adj,
            'effective_threshold': effective_threshold,
            'anomaly_reasons': anomaly_reasons,
            'severe_drop_detected': severe_drop_detected
        }
    
    def _compute_anomalies(self):
        """Main anomaly computation with fixed logic."""
        print("Computing anomalies...")
        
        total_users = len(self.unified_data)
        processed_users = 0
        
        for user in self.unified_data:
            self.anomaly_results[user] = {}
            
            for date in self.user_dates[user]:
                metrics = self.unified_data[user][date]
                
                # Compute personal anomaly
                personal_result = self._compute_personal_anomaly(user, date)
                
                # Check for weekend skip
                weekend_skip = False
                if self._is_weekend(date) and personal_result['baseline_type'] != 'minimal_weekend':
                    # Check if this user typically works weekends
                    weekend_dates = [d for d in self.user_dates[user] if self._is_weekend(d)]
                    if len(weekend_dates) > 5:
                        weekend_metrics = [self.unified_data[user][d] for d in weekend_dates[:5]]
                        weekend_activity = np.mean([np.sum(m[:4]) for m in weekend_metrics])
                        if weekend_activity < 0.1:
                            weekend_skip = True
                
                # Store result with all required fields
                result = {
                    'date': date,
                    'user': user,
                    'metrics': metrics,
                    'is_weekend': self._is_weekend(date),
                    'weekend_skip': weekend_skip,
                    'is_anomaly': personal_result.get('is_personal_anomaly', False),  # Ensure is_anomaly is set
                    'overall_anomaly_score': personal_result.get('overall_personal_score', 0.0),
                    **personal_result
                }
                
                # Add anomaly reasons if present
                if 'anomaly_reasons' in personal_result:
                    result['anomaly_reasons'] = personal_result['anomaly_reasons']
                
                self.anomaly_results[user][date] = result
            
            processed_users += 1
            if processed_users % 10 == 0 or processed_users == total_users:
                print(f"Processed {processed_users}/{total_users} users...")
        
        # Add group detection if enabled
        if self.enable_group_detection:
            self._compute_group_anomalies()
        
        total_observations = sum(len(user_data) for user_data in self.anomaly_results.values())
        print(f"Anomaly detection complete. Processed {total_observations} observations for {total_users} users.")
    
    def _compute_group_anomalies(self):
        """Compute group anomalies with weekday awareness."""
        print("Computing group anomalies...")
        
        # Organize all user data by date
        data_by_date = defaultdict(list)
        for user, results in self.anomaly_results.items():
            for date, result in results.items():
                data_by_date[date].append((user, result['metrics']))
        
        # Compute group baselines
        for user, results in self.anomaly_results.items():
            for date, result in results.items():
                target_weekday = self._get_weekday(date)
                
                # Collect peer data
                peer_data = []
                for hist_date in sorted(data_by_date.keys()):
                    if hist_date >= date:  # Don't use future data
                        continue
                    
                    hist_weekday = self._get_weekday(hist_date)
                    
                    # Use same logic as personal baselines
                    if self.strict_weekday_separation:
                        if target_weekday < 5 and hist_weekday < 5:  # Both weekdays
                            peer_data.extend([m for _, m in data_by_date[hist_date]])
                        elif target_weekday >= 5 and hist_weekday >= 5:  # Both weekends
                            peer_data.extend([m for _, m in data_by_date[hist_date]])
                    else:
                        peer_data.extend([m for _, m in data_by_date[hist_date]])
                
                if len(peer_data) >= 20:  # Need sufficient peer data
                    peer_array = np.array(peer_data)
                    peer_medians, peer_mads, _ = self._robust_baseline_stats(
                        peer_array, 
                        'weekends' if target_weekday >= 5 else 'weekdays'
                    )
                    
                    # Compute group z-scores
                    scaled_mads = peer_mads * 1.4826
                    group_z_scores = np.divide(
                        result['metrics'] - peer_medians, 
                        scaled_mads,
                        out=np.zeros_like(result['metrics']), 
                        where=scaled_mads!=0
                    )
                    
                    # Overall group score
                    weighted_abs_z_scores = np.abs(group_z_scores) * self.metric_weights
                    overall_group_score = np.mean(weighted_abs_z_scores)
                    
                    # Update result with group detection info
                    result['group_z_scores'] = group_z_scores
                    result['overall_group_score'] = overall_group_score
                    result['is_group_anomaly'] = overall_group_score > self.threshold_group
                    result['group_baseline_quality'] = 'sufficient'
                else:
                    result['group_z_scores'] = np.zeros(self.num_metrics)
                    result['overall_group_score'] = 0.0
                    result['is_group_anomaly'] = False
                    result['group_baseline_quality'] = 'insufficient_group_data'
                
                # Update overall anomaly flag (personal OR group)
                result['is_anomaly'] = result.get('is_personal_anomaly', False) or result.get('is_group_anomaly', False)
                result['overall_anomaly_score'] = max(
                    result.get('overall_personal_score', 0.0), 
                    result.get('overall_group_score', 0.0)
                )
    
    # Keep all original interface methods
    def get_anomalies(self, user_email: Optional[str] = None, 
                     min_quality: str = 'low',
                     exclude_weekends: bool = False) -> Dict:
        """Get detected anomalies with quality filtering."""
        quality_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_quality_level = quality_levels.get(min_quality, 0)
        
        def quality_filter(result):
            result_quality = quality_levels.get(result['baseline_quality'], 0)
            return result_quality >= min_quality_level
        
        if user_email:
            user_results = self.anomaly_results.get(user_email.lower(), {})
            return {
                date: result for date, result in user_results.items()
                if (result['is_anomaly'] and 
                    quality_filter(result) and
                    (not exclude_weekends or not result['is_weekend']) and
                    not result.get('weekend_skip', False))
            }
        else:
            anomalies = {}
            for user, user_results in self.anomaly_results.items():
                user_anomalies = {
                    date: result for date, result in user_results.items()
                    if (result['is_anomaly'] and 
                        quality_filter(result) and
                        (not exclude_weekends or not result['is_weekend']) and
                        not result.get('weekend_skip', False))
                }
                if user_anomalies:
                    anomalies[user] = user_anomalies
            return anomalies
    
    def get_anomaly_summary(self, min_quality: str = 'low') -> Dict:
        """Get comprehensive summary of anomaly detection results."""
        total_observations = 0
        total_anomalies = 0
        quality_counts = {'low': 0, 'medium': 0, 'high': 0, 'insufficient_data': 0, 'minimal_weekend': 0}
        baseline_type_counts = defaultdict(int)
        anomaly_scores = []
        weekend_anomalies = 0
        
        for user, user_results in self.anomaly_results.items():
            for date, result in user_results.items():
                total_observations += 1
                quality_counts[result.get('baseline_quality', 'unknown')] = \
                    quality_counts.get(result.get('baseline_quality', 'unknown'), 0) + 1
                baseline_type_counts[result.get('baseline_type', 'unknown')] += 1
                
                if result['overall_anomaly_score'] > 0:
                    anomaly_scores.append(result['overall_anomaly_score'])
                
                if result['is_anomaly']:
                    total_anomalies += 1
                    if result['is_weekend']:
                        weekend_anomalies += 1
        
        return {
            'total_observations': total_observations,
            'total_anomalies': total_anomalies,
            'anomaly_rate': total_anomalies / total_observations if total_observations > 0 else 0,
            'weekend_anomalies': weekend_anomalies,
            'baseline_quality_distribution': dict(quality_counts),
            'baseline_type_distribution': dict(baseline_type_counts),
            'threshold_personal': self.threshold_personal,
            'threshold_group': self.threshold_group,
            'group_detection_enabled': self.enable_group_detection,
            'strict_weekday_separation': self.strict_weekday_separation,
            'weekend_threshold_factor': self.weekend_threshold_factor,
            'avg_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0,
            'median_anomaly_score': np.median(anomaly_scores) if anomaly_scores else 0,
            'max_anomaly_score': np.max(anomaly_scores) if anomaly_scores else 0,
            'min_baseline_days': self.min_baseline_days,
            'lookback_window': self.lookback_window
        }
    
    def get_top_anomalies(self, n: int = 10, min_quality: str = 'medium') -> List[Tuple[str, str, float, str]]:
        """Get top N anomalies by score with quality information."""
        all_anomalies = []
        quality_levels = {'low': 0, 'medium': 1, 'high': 2, 'minimal_weekend': 1}
        min_quality_level = quality_levels.get(min_quality, 0)
        
        for user, user_results in self.anomaly_results.items():
            for date, result in user_results.items():
                if (result['is_anomaly'] and 
                    quality_levels.get(result['baseline_quality'], 0) >= min_quality_level):
                    all_anomalies.append((
                        user, date, 
                        result['overall_anomaly_score'], 
                        result['baseline_quality']
                    ))
        
        all_anomalies.sort(key=lambda x: x[2], reverse=True)
        return all_anomalies[:n]
    
    def get_metric_contributions(self, user_email: str, date: str) -> Dict:
        """Get detailed breakdown of metric contributions to anomaly."""
        if user_email.lower() not in self.anomaly_results:
            return {}
        
        if date not in self.anomaly_results[user_email.lower()]:
            return {}
        
        result = self.anomaly_results[user_email.lower()][date]
        
        contributions = []
        for i, name in enumerate(self.metric_names):
            contributions.append({
                'metric': name,
                'value': result['metrics'][i],
                'personal_z_score': result['personal_z_scores'][i],
                'seasonal_adjusted_z_score': result['seasonal_adjusted_z_scores'][i],
                'group_z_score': result.get('group_z_scores', np.zeros(self.num_metrics))[i],
                'weight': self.metric_weights[i],
                'weighted_contribution': abs(result['seasonal_adjusted_z_scores'][i]) * self.metric_weights[i],
                'seasonal_adjustment': result.get('seasonal_adjustments', np.zeros(self.num_metrics))[i]
            })
        
        contributions.sort(key=lambda x: x['weighted_contribution'], reverse=True)
        
        return {
            'user': user_email,
            'date': date,
            'overall_anomaly_score': result['overall_anomaly_score'],
            'is_anomaly': result['is_anomaly'],
            'baseline_quality': result['baseline_quality'],
            'baseline_type': result['baseline_type'],
            'baseline_days': result['baseline_days'],
            'is_weekend': result['is_weekend'],
            'effective_threshold': result.get('effective_threshold', self.threshold_personal),
            'metric_contributions': contributions
        }
    
    def export_results_to_dataframe(self, include_non_anomalies: bool = False) -> pd.DataFrame:
        """Export results to pandas DataFrame."""
        rows = []
        
        for user, user_results in self.anomaly_results.items():
            for date, result in user_results.items():
                if not include_non_anomalies and not result['is_anomaly']:
                    continue
                
                row = {
                    'user': user,
                    'date': date,
                    'is_anomaly': result['is_anomaly'],
                    'overall_anomaly_score': result['overall_anomaly_score'],
                    'baseline_quality': result['baseline_quality'],
                    'baseline_type': result['baseline_type'],
                    'baseline_days': result['baseline_days'],
                    'is_weekend': result['is_weekend'],
                    'weekend_skip': result.get('weekend_skip', False),
                    'effective_threshold': result.get('effective_threshold', self.threshold_personal)
                }
                
                # Add metric values and scores
                for i, metric_name in enumerate(self.metric_names):
                    row[f'{metric_name}_value'] = result['metrics'][i]
                    row[f'{metric_name}_personal_z'] = result['personal_z_scores'][i]
                    row[f'{metric_name}_seasonal_z'] = result['seasonal_adjusted_z_scores'][i]
                    if self.enable_group_detection:
                        row[f'{metric_name}_group_z'] = result.get('group_z_scores', np.zeros(self.num_metrics))[i]
                
                rows.append(row)
        
        return pd.DataFrame(rows)