import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from functools import lru_cache


class AnomalyDetector:
    def __init__(self, unified_data: Dict[str, Dict[str, np.ndarray]], 
                 min_baseline_days: int = 30,
                 lookback_window: int = 90,
                 threshold_personal: float = 3.0,
                 threshold_group: float = 3.0,
                 enable_group_detection: bool = False,
                 metric_weights: Optional[Dict[str, float]] = None):
        """
        Optimized anomaly detector with improved performance.
        """
        self.unified_data = unified_data
        self.min_baseline_days = min_baseline_days
        self.lookback_window = lookback_window
        self.threshold_personal = threshold_personal
        self.threshold_group = threshold_group
        self.enable_group_detection = enable_group_detection
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
        
        # Set and normalize metric weights
        if metric_weights is None:
            self.metric_weights = np.ones(self.num_metrics)
        else:
            self.metric_weights = np.array([
                metric_weights.get(name, 1.0) for name in self.metric_names
            ])
        self.metric_weights = self.metric_weights / np.sum(self.metric_weights) * self.num_metrics
        
        # Pre-compute data structures for faster access
        self._preprocess_data()
        
        # Results storage
        self.anomaly_results = {}
        self.baseline_quality = {}
        
        # Process data
        self._validate_data()
        self._compute_anomalies()
    
    def _preprocess_data(self):
        """Pre-process data into more efficient structures."""
        print("Preprocessing data structures...")
        
        # Convert to numpy arrays and cache date conversions
        self.user_data_arrays = {}
        self.user_dates = {}
        self.date_to_weekday = {}
        self.date_to_timestamp = {}
        
        for user, user_data in self.unified_data.items():
            dates = sorted(user_data.keys())
            self.user_dates[user] = dates
            
            # Stack all metrics into a single array
            metrics_array = np.array([user_data[date] for date in dates])
            self.user_data_arrays[user] = metrics_array
            
            # Cache date conversions
            for date in dates:
                if date not in self.date_to_weekday:
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    self.date_to_weekday[date] = dt.weekday()
                    self.date_to_timestamp[date] = dt.timestamp()
        
        # Pre-compute global date mappings for faster lookups
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
        
        # Update preprocessed data to only include valid users
        self.user_data_arrays = {k: v for k, v in self.user_data_arrays.items() if k in valid_users}
        self.user_dates = {k: v for k, v in self.user_dates.items() if k in valid_users}
        
        print(f"Processing {len(self.unified_data)} users with sufficient data")
    
    @lru_cache(maxsize=1000)
    def _get_weekday(self, date_str: str) -> int:
        """Cached weekday computation."""
        return self.date_to_weekday[date_str]
    
    @lru_cache(maxsize=1000)
    def _is_weekend(self, date_str: str) -> bool:
        """Cached weekend check."""
        return self.date_to_weekday[date_str] >= 5
    
    def _get_historical_data_vectorized(self, user: str, target_date: str, 
                                      exclude_weekends: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Vectorized version of historical data extraction."""
        if user not in self.user_dates:
            return np.array([]), []
        
        dates = self.user_dates[user]
        metrics_array = self.user_data_arrays[user]
        target_timestamp = self.date_to_timestamp[target_date]
        
        # Vectorized date filtering
        valid_indices = []
        valid_dates = []
        
        for i, date in enumerate(dates):
            date_timestamp = self.date_to_timestamp[date]
            days_diff = (target_timestamp - date_timestamp) / (24 * 3600)  # Convert to days
            
            if 1 <= days_diff <= self.lookback_window:
                if exclude_weekends and self._is_weekend(date):
                    continue
                valid_indices.append(i)
                valid_dates.append(date)
        
        if not valid_indices:
            return np.array([]), []
        
        return metrics_array[valid_indices], valid_dates
    
    def _robust_baseline_stats_vectorized(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Vectorized computation of baseline statistics for all metrics at once."""
        if len(data) == 0:
            return np.zeros(self.num_metrics), np.full(self.num_metrics, 1e-6), 0
        
        # Compute medians for all metrics at once
        medians = np.median(data, axis=0)
        
        # Compute MAD for all metrics
        abs_deviations = np.abs(data - medians)
        mads = np.median(abs_deviations, axis=0)
        
        # Apply metric-specific minimum MAD values vectorized
        for metric_idx in self.metric_types['count']:
            mads[metric_idx] = max(mads[metric_idx], 1.0)
        for metric_idx in self.metric_types['ratio']:
            mads[metric_idx] = max(mads[metric_idx], 0.1)
        
        # Ensure no zero MADs
        mads = np.maximum(mads, 1e-6)
        
        return medians, mads, len(data)
    
    def _compute_seasonal_adjustments_batch(self, user: str, target_dates: List[str]) -> Dict[str, np.ndarray]:
        """Batch computation of seasonal adjustments for multiple dates."""
        if user not in self.user_dates:
            return {date: np.zeros(self.num_metrics) for date in target_dates}
        
        dates = self.user_dates[user]
        metrics_array = self.user_data_arrays[user]
        
        # Group historical data by weekday
        weekday_data = defaultdict(list)
        weekday_indices = defaultdict(list)
        
        for i, date in enumerate(dates):
            weekday = self._get_weekday(date)
            weekday_data[weekday].append(metrics_array[i])
            weekday_indices[weekday].append(i)
        
        # Convert to numpy arrays for each weekday
        weekday_arrays = {}
        weekday_medians = {}
        
        for weekday, data_list in weekday_data.items():
            if len(data_list) >= 3:
                weekday_arrays[weekday] = np.array(data_list)
                weekday_medians[weekday] = np.median(weekday_arrays[weekday], axis=0)
            else:
                weekday_medians[weekday] = np.zeros(self.num_metrics)
        
        # Compute overall median
        if len(metrics_array) >= 10:
            overall_median = np.median(metrics_array, axis=0)
        else:
            overall_median = np.zeros(self.num_metrics)
        
        # Compute adjustments for each target date
        adjustments = {}
        for target_date in target_dates:
            target_weekday = self._get_weekday(target_date)
            if target_weekday in weekday_medians and len(weekday_data[target_weekday]) >= 3:
                adjustments[target_date] = weekday_medians[target_weekday] - overall_median
            else:
                adjustments[target_date] = np.zeros(self.num_metrics)
        
        return adjustments
    
    def _precompute_group_baselines(self):
        """Pre-compute group baselines for all dates to avoid redundant computation."""
        if not self.enable_group_detection:
            return
        
        print("Pre-computing group baselines...")
        self.group_baselines_cache = {}
        
        # Get all unique dates and sort them
        all_dates = set()
        for dates in self.user_dates.values():
            all_dates.update(dates)
        
        sorted_dates = sorted(all_dates)
        
        for target_date in sorted_dates:
            target_timestamp = self.date_to_timestamp[target_date]
            target_weekday = self._get_weekday(target_date)
            
            # Collect all data for this weekday within lookback window
            group_data = []
            
            for user in self.user_dates:
                dates = self.user_dates[user]
                metrics_array = self.user_data_arrays[user]
                
                for i, date in enumerate(dates):
                    date_timestamp = self.date_to_timestamp[date]
                    days_diff = (target_timestamp - date_timestamp) / (24 * 3600)
                    
                    if (1 <= days_diff <= self.lookback_window and 
                        self._get_weekday(date) == target_weekday):
                        group_data.append(metrics_array[i])
            
            if len(group_data) >= 10:
                group_matrix = np.array(group_data)
                medians, mads, sample_size = self._robust_baseline_stats_vectorized(group_matrix)
                
                self.group_baselines_cache[target_date] = {
                    'medians': medians,
                    'mads': mads,
                    'sample_size': sample_size
                }
    
    def _compute_personal_anomaly_batch(self, user: str) -> Dict:
        """Compute personal anomalies for all dates of a user in batch."""
        if user not in self.user_dates:
            return {}
        
        dates = self.user_dates[user]
        metrics_array = self.user_data_arrays[user]
        results = {}
        
        # Pre-compute seasonal adjustments for all dates
        seasonal_adjustments = self._compute_seasonal_adjustments_batch(user, dates)
        
        for i, date in enumerate(dates):
            metrics = metrics_array[i]
            
            # Get historical data
            hist_data, hist_dates = self._get_historical_data_vectorized(user, date)
            
            if len(hist_data) < self.min_baseline_days:
                results[date] = {
                    'personal_z_scores': np.zeros(self.num_metrics),
                    'seasonal_adjusted_z_scores': np.zeros(self.num_metrics),
                    'baseline_quality': 'insufficient_data',
                    'baseline_days': len(hist_data),
                    'overall_personal_score': 0.0,
                    'is_personal_anomaly': False,
                    'seasonal_adjustments': np.zeros(self.num_metrics)
                }
                continue
            
            # Compute baseline statistics (vectorized)
            medians, mads, sample_size = self._robust_baseline_stats_vectorized(hist_data)
            
            # Compute Z-scores (vectorized)
            scaled_mads = mads * 1.4826
            personal_z_scores = np.divide(metrics - medians, scaled_mads, 
                                        out=np.zeros_like(metrics), where=scaled_mads!=0)
            
            # Apply seasonal adjustments
            seasonal_adj = seasonal_adjustments[date]
            adjusted_expected = medians + seasonal_adj
            seasonal_z_scores = np.divide(metrics - adjusted_expected, scaled_mads,
                                        out=np.zeros_like(metrics), where=scaled_mads!=0)
            
            # Overall anomaly score (vectorized)
            weighted_abs_z_scores = np.abs(seasonal_z_scores) * self.metric_weights
            overall_score = np.mean(weighted_abs_z_scores)
            
            # Determine baseline quality
            if sample_size >= self.lookback_window * 0.8:
                quality = 'high'
            elif sample_size >= self.min_baseline_days * 2:
                quality = 'medium'
            else:
                quality = 'low'
            
            results[date] = {
                'personal_z_scores': personal_z_scores,
                'seasonal_adjusted_z_scores': seasonal_z_scores,
                'baseline_quality': quality,
                'baseline_days': len(hist_data),
                'overall_personal_score': overall_score,
                'is_personal_anomaly': overall_score > self.threshold_personal,
                'seasonal_adjustments': seasonal_adj
            }
        
        return results
    
    def _compute_group_anomaly_batch(self, user: str) -> Dict:
        """Compute group anomalies for all dates of a user in batch."""
        if not self.enable_group_detection or user not in self.user_dates:
            return {}
        
        dates = self.user_dates[user]
        metrics_array = self.user_data_arrays[user]
        results = {}
        
        for i, date in enumerate(dates):
            metrics = metrics_array[i]
            
            if date not in self.group_baselines_cache:
                results[date] = {
                    'group_z_scores': np.zeros(self.num_metrics),
                    'overall_group_score': 0.0,
                    'is_group_anomaly': False,
                    'group_baseline_quality': 'insufficient_group_data'
                }
                continue
            
            baseline = self.group_baselines_cache[date]
            scaled_mads = baseline['mads'] * 1.4826
            
            # Vectorized group Z-score computation
            group_z_scores = np.divide(metrics - baseline['medians'], scaled_mads,
                                     out=np.zeros_like(metrics), where=scaled_mads!=0)
            
            # Overall group anomaly score
            weighted_abs_z_scores = np.abs(group_z_scores) * self.metric_weights
            overall_group_score = np.mean(weighted_abs_z_scores)
            
            results[date] = {
                'group_z_scores': group_z_scores,
                'overall_group_score': overall_group_score,
                'is_group_anomaly': overall_group_score > self.threshold_group,
                'group_baseline_quality': 'sufficient'
            }
        
        return results
    
    def _compute_anomalies(self):
        """Optimized batch computation of anomalies."""
        print("Computing anomalies (optimized)...")
        
        # Pre-compute group baselines if needed
        if self.enable_group_detection:
            self._precompute_group_baselines()
        
        total_users = len(self.unified_data)
        processed_users = 0
        
        for user in self.unified_data:
            # Batch process personal anomalies for this user
            personal_results = self._compute_personal_anomaly_batch(user)
            
            # Batch process group anomalies for this user
            group_results = self._compute_group_anomaly_batch(user) if self.enable_group_detection else {}
            
            # Combine results
            self.anomaly_results[user] = {}
            dates = self.user_dates[user]
            metrics_array = self.user_data_arrays[user]
            
            for i, date in enumerate(dates):
                metrics = metrics_array[i]
                
                # Check for weekend skip
                weekend_skip = False
                if self._is_weekend(date):
                    weekend_data = []
                    for j, d in enumerate(dates):
                        if self._is_weekend(d):
                            weekend_data.append(metrics_array[j])
                    
                    if len(weekend_data) > 5:
                        weekend_activity = np.mean([np.sum(v[:4]) for v in weekend_data])
                        if weekend_activity < 0.1:
                            weekend_skip = True
                
                # Combine personal and group results
                personal_result = personal_results.get(date, {})
                group_result = group_results.get(date, {
                    'group_z_scores': np.zeros(self.num_metrics),
                    'overall_group_score': 0.0,
                    'is_group_anomaly': False,
                    'group_baseline_quality': 'disabled'
                })
                
                combined_result = {
                    'date': date,
                    'user': user,
                    'metrics': metrics,
                    'is_weekend': self._is_weekend(date),
                    'weekend_skip': weekend_skip,
                    **personal_result,
                    **group_result
                }
                
                # Overall anomaly determination
                combined_result['is_anomaly'] = personal_result.get('is_personal_anomaly', False)
                combined_result['overall_anomaly_score'] = personal_result.get('overall_personal_score', 0.0)
                
                self.anomaly_results[user][date] = combined_result
            
            processed_users += 1
            if processed_users % 10 == 0 or processed_users == total_users:
                print(f"Processed {processed_users}/{total_users} users...")
        
        total_observations = sum(len(user_data) for user_data in self.anomaly_results.values())
        print(f"Anomaly detection complete. Processed {total_observations} observations for {total_users} users.")
    
    # Keep all the original interface methods unchanged for compatibility
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
        quality_counts = {'low': 0, 'medium': 0, 'high': 0, 'insufficient_data': 0}
        anomaly_scores = []
        weekend_anomalies = 0
        
        for user, user_results in self.anomaly_results.items():
            for date, result in user_results.items():
                total_observations += 1
                quality_counts[result['baseline_quality']] += 1
                
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
            'baseline_quality_distribution': quality_counts,
            'threshold_personal': self.threshold_personal,
            'threshold_group': self.threshold_group,
            'group_detection_enabled': self.enable_group_detection,
            'avg_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0,
            'median_anomaly_score': np.median(anomaly_scores) if anomaly_scores else 0,
            'max_anomaly_score': np.max(anomaly_scores) if anomaly_scores else 0,
            'min_baseline_days': self.min_baseline_days,
            'lookback_window': self.lookback_window
        }
    
    def get_top_anomalies(self, n: int = 10, min_quality: str = 'medium') -> List[Tuple[str, str, float, str]]:
        """Get top N anomalies by score with quality information."""
        all_anomalies = []
        quality_levels = {'low': 0, 'medium': 1, 'high': 2}
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
            'baseline_days': result['baseline_days'],
            'is_weekend': result['is_weekend'],
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
                    'baseline_days': result['baseline_days'],
                    'is_weekend': result['is_weekend'],
                    'weekend_skip': result.get('weekend_skip', False)
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
    
    def enable_group_detection_later(self):
        """Enable group detection and recompute group-based scores."""
        if self.enable_group_detection:
            print("Group detection already enabled")
            return
        
        print("Enabling group detection and recomputing scores...")
        self.enable_group_detection = True
        
        # Pre-compute group baselines
        self._precompute_group_baselines()
        
        # Recompute only group-related scores for all users
        for user in self.anomaly_results:
            group_results = self._compute_group_anomaly_batch(user)
            
            for date, result in self.anomaly_results[user].items():
                if date in group_results:
                    result.update(group_results[date])
        
        print("Group detection enabled and scores updated.")