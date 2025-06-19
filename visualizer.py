import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class AnomalyVisualizer:
    """
    Comprehensive visualization class for anomaly detection validation.
    
    Provides various plots to:
    1. Validate simulated data patterns
    2. Verify anomaly injection
    3. Analyze detection results
    4. Compare weekday vs weekend patterns
    """
    
    def __init__(self, simulator=None, detector=None, figsize_default=(12, 6)):
        """
        Initialize visualizer with optional simulator and detector instances.
        
        Args:
            simulator: EmployeeActivitySimulator instance
            detector: AnomalyDetector instance
            figsize_default: Default figure size for plots
        """
        self.simulator = simulator
        self.detector = detector
        self.figsize_default = figsize_default
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Metric groupings for visualization
        self.metric_groups = {
            'communication': ['email_count', 'teams_messages', 'meeting_count'],
            'documents': ['docs_accessed', 'docs_created', 'docs_edited', 'docs_shared'],
            'storage': ['dropbox_added', 'dropbox_edited'],
            'email_details': ['email_length', 'email_frr', 'email_recipients'],
            'meeting_details': ['meeting_duration', 'meeting_cancel_rate', 'meeting_virtual_ratio', 'meeting_btb_ratio'],
            'teams_details': ['teams_msg_length', 'teams_channels']
        }
        
    def plot_employee_timeline(self, employee_id: str, metrics: List[str], 
                             date_range: Optional[Tuple[str, str]] = None,
                             highlight_anomalies: bool = True,
                             show_thresholds: bool = True) -> None:
        """
        Plot timeline of specified metrics for an employee.
        
        Args:
            employee_id: Employee email ID
            metrics: List of metric names to plot
            date_range: Optional (start_date, end_date) tuple
            highlight_anomalies: Whether to highlight detected anomalies
            show_thresholds: Whether to show detection thresholds
        """
        if not self.simulator or employee_id not in self.simulator.simulated_data:
            print(f"No data found for {employee_id}")
            return
        
        emp_data = self.simulator.simulated_data[employee_id]
        dates = sorted(emp_data.keys())
        
        # Filter date range if specified
        if date_range:
            dates = [d for d in dates if date_range[0] <= d <= date_range[1]]
        
        # Create subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3*len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # Get anomalies if detector is available
        anomalies = {}
        if self.detector and highlight_anomalies:
            user_anomalies = self.detector.get_anomalies(user_email=employee_id)
            anomalies = {date: result for date, result in user_anomalies.items() if date in dates}
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric not in self.simulator.metric_indices:
                print(f"Unknown metric: {metric}")
                continue
                
            idx = self.simulator.metric_indices[metric]
            values = [emp_data[date][idx] for date in dates]
            
            # Convert dates to datetime objects for better plotting
            date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            
            # Main plot
            axes[i].plot(date_objs, values, marker='o', linewidth=2, markersize=4, 
                        label=metric, color='darkblue')
            
            # Highlight weekends
            for j, (date_obj, date_str) in enumerate(zip(date_objs, dates)):
                if date_obj.weekday() >= 5:  # Weekend
                    axes[i].axvspan(date_obj - timedelta(hours=12), 
                                   date_obj + timedelta(hours=12), 
                                   alpha=0.1, color='gray', zorder=0)
            
            # Highlight anomalies
            if anomalies:
                anomaly_dates = []
                anomaly_values = []
                for j, (date_obj, date_str) in enumerate(zip(date_objs, dates)):
                    if date_str in anomalies:
                        anomaly_dates.append(date_obj)
                        anomaly_values.append(values[j])
                
                if anomaly_dates:
                    axes[i].scatter(anomaly_dates, anomaly_values, 
                                  color='red', s=100, zorder=5, 
                                  label='Anomaly', edgecolors='darkred', linewidth=2)
            
            # Show baseline and thresholds if detector available
            if self.detector and show_thresholds and len(values) > 30:
                # Calculate rolling statistics
                window = 14  # 2-week window
                if len(values) >= window:
                    rolling_median = pd.Series(values).rolling(window=window, center=True).median()
                    rolling_std = pd.Series(values).rolling(window=window, center=True).std()
                    
                    # Plot baseline
                    axes[i].plot(date_objs, rolling_median, 'g--', alpha=0.5, 
                               label='14-day median')
                    
                    # Plot threshold bands (3 sigma)
                    upper_bound = rolling_median + 3 * rolling_std
                    lower_bound = rolling_median - 3 * rolling_std
                    axes[i].fill_between(date_objs, lower_bound, upper_bound, 
                                       alpha=0.1, color='green', 
                                       label='±3σ band')
            
            # Formatting
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
            
            # Format x-axis
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # Overall title
        profile = self.simulator.employee_profiles.get(employee_id, {})
        dept = profile.get('department', 'Unknown')
        seniority = profile.get('seniority', 'Unknown')
        
        fig.suptitle(f'Activity Timeline: {employee_id}\nDepartment: {dept.title()}, Seniority: {seniority.title()}', 
                    fontsize=16, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_weekday_patterns(self, employee_id: str, metric: str) -> None:
        """
        Plot weekday vs weekend patterns for an employee.
        
        Shows distribution of values by day of week.
        """
        if not self.simulator or employee_id not in self.simulator.simulated_data:
            print(f"No data found for {employee_id}")
            return
        
        emp_data = self.simulator.simulated_data[employee_id]
        idx = self.simulator.metric_indices.get(metric)
        
        if idx is None:
            print(f"Unknown metric: {metric}")
            return
        
        # Organize by weekday
        weekday_data = {i: [] for i in range(7)}
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for date_str, metrics in emp_data.items():
            weekday = datetime.strptime(date_str, '%Y-%m-%d').weekday()
            weekday_data[weekday].append(metrics[idx])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot by weekday
        data_for_plot = [weekday_data[i] for i in range(7)]
        bp = ax1.boxplot(data_for_plot, labels=weekday_names, patch_artist=True)
        
        # Color weekends differently
        for i, patch in enumerate(bp['boxes']):
            if i >= 5:  # Weekend
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightblue')
        
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title('Distribution by Day of Week')
        ax1.grid(True, alpha=0.3)
        
        # Violin plot for weekday vs weekend
        weekday_values = []
        weekend_values = []
        for i in range(5):  # Mon-Fri
            weekday_values.extend(weekday_data[i])
        for i in range(5, 7):  # Sat-Sun
            weekend_values.extend(weekday_data[i])
        
        if weekday_values and weekend_values:
            parts = ax2.violinplot([weekday_values, weekend_values], positions=[0, 1], 
                                  showmeans=True, showmedians=True)
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['Weekdays', 'Weekends'])
            ax2.set_ylabel(metric.replace('_', ' ').title())
            ax2.set_title('Weekday vs Weekend Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            weekday_mean = np.mean(weekday_values) if weekday_values else 0
            weekend_mean = np.mean(weekend_values) if weekend_values else 0
            ax2.text(0, ax2.get_ylim()[1] * 0.95, f'μ={weekday_mean:.1f}', 
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightblue", alpha=0.5))
            ax2.text(1, ax2.get_ylim()[1] * 0.95, f'μ={weekend_mean:.1f}', 
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightcoral", alpha=0.5))
        
        profile = self.simulator.employee_profiles.get(employee_id, {})
        fig.suptitle(f'Weekday Patterns: {employee_id} - {metric}\n' + 
                    f'Department: {profile.get("department", "Unknown").title()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_detection_results(self, employee_id: str, 
                                     date_range: Optional[Tuple[str, str]] = None) -> None:
        """
        Comprehensive plot showing detection results with scores and thresholds.
        """
        if not self.detector or employee_id not in self.detector.anomaly_results:
            print(f"No detection results for {employee_id}")
            return
        
        results = self.detector.anomaly_results[employee_id]
        dates = sorted(results.keys())
        
        if date_range:
            dates = [d for d in dates if date_range[0] <= d <= date_range[1]]
        
        # Extract data
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        anomaly_scores = [results[d]['overall_anomaly_score'] for d in dates]
        is_anomaly = [results[d]['is_anomaly'] for d in dates]
        baseline_quality = [results[d]['baseline_quality'] for d in dates]
        baseline_type = [results[d]['baseline_type'] for d in dates]
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Anomaly scores with threshold
        ax1.plot(date_objs, anomaly_scores, 'b-', linewidth=2, label='Anomaly Score')
        
        # Show different thresholds
        for i, (date_obj, date) in enumerate(zip(date_objs, dates)):
            threshold = results[date].get('effective_threshold', self.detector.threshold_personal)
            if i == 0 or threshold != results[dates[i-1]].get('effective_threshold', self.detector.threshold_personal):
                ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                ax1.text(date_obj, threshold + 0.1, f'Threshold: {threshold:.1f}', 
                        fontsize=8, ha='left')
        
        # Highlight anomalies
        anomaly_indices = [i for i, is_anom in enumerate(is_anomaly) if is_anom]
        if anomaly_indices:
            anomaly_dates_plot = [date_objs[i] for i in anomaly_indices]
            anomaly_scores_plot = [anomaly_scores[i] for i in anomaly_indices]
            ax1.scatter(anomaly_dates_plot, anomaly_scores_plot, 
                       color='red', s=100, zorder=5, label='Detected Anomaly',
                       edgecolors='darkred', linewidth=2)
        
        # Weekend shading
        for date_obj, date in zip(date_objs, dates):
            if date_obj.weekday() >= 5:
                ax1.axvspan(date_obj - timedelta(hours=12), 
                           date_obj + timedelta(hours=12), 
                           alpha=0.1, color='gray')
        
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title('Anomaly Detection Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Baseline quality over time
        quality_colors = {'high': 'green', 'medium': 'yellow', 'low': 'orange', 
                         'insufficient_data': 'red', 'minimal_weekend': 'purple'}
        quality_numeric = {'high': 3, 'medium': 2, 'low': 1, 'insufficient_data': 0, 'minimal_weekend': 1.5}
        
        quality_values = [quality_numeric.get(q, 0) for q in baseline_quality]
        colors = [quality_colors.get(q, 'gray') for q in baseline_quality]
        
        ax2.scatter(date_objs, quality_values, c=colors, s=50, alpha=0.7)
        ax2.set_yticks([0, 1, 1.5, 2, 3])
        ax2.set_yticklabels(['Insufficient', 'Low', 'Min Weekend', 'Medium', 'High'])
        ax2.set_ylabel('Baseline Quality')
        ax2.set_title('Baseline Quality Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Baseline type used
        baseline_type_numeric = {
            'same_weekday': 4, 'weekends': 3.5, 'all_weekdays': 3, 
            'minimal_weekend': 2, 'all': 1, 'insufficient': 0
        }
        type_values = [baseline_type_numeric.get(t, 0) for t in baseline_type]
        
        ax3.plot(date_objs, type_values, 'g-', linewidth=2, marker='o', markersize=4)
        ax3.set_yticks([0, 1, 2, 3, 3.5, 4])
        ax3.set_yticklabels(['Insufficient', 'All Data', 'Min Weekend', 
                            'All Weekdays', 'Weekends', 'Same Weekday'])
        ax3.set_ylabel('Baseline Type')
        ax3.set_xlabel('Date')
        ax3.set_title('Baseline Type Used for Detection')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        plt.xticks(rotation=45, ha='right')
        
        profile = self.simulator.employee_profiles.get(employee_id, {}) if self.simulator else {}
        fig.suptitle(f'Anomaly Detection Analysis: {employee_id}\n' + 
                    f'Department: {profile.get("department", "Unknown").title()}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metric_heatmap(self, employee_ids: List[str], metric: str, 
                          date_range: Optional[Tuple[str, str]] = None,
                          annotate_anomalies: bool = True) -> None:
        """
        Create heatmap showing metric values across multiple employees.
        """
        if not self.simulator:
            print("Simulator not available")
            return
        
        # Filter employees
        valid_employees = [e for e in employee_ids if e in self.simulator.simulated_data]
        if not valid_employees:
            print("No valid employees found")
            return
        
        # Get dates
        all_dates = set()
        for emp in valid_employees:
            all_dates.update(self.simulator.simulated_data[emp].keys())
        dates = sorted(all_dates)
        
        if date_range:
            dates = [d for d in dates if date_range[0] <= d <= date_range[1]]
        
        # Create matrix
        idx = self.simulator.metric_indices.get(metric)
        if idx is None:
            print(f"Unknown metric: {metric}")
            return
        
        data_matrix = []
        for emp in valid_employees:
            emp_data = self.simulator.simulated_data[emp]
            row = [emp_data.get(date, np.zeros(18))[idx] for date in dates]
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, max(6, len(valid_employees))))
        
        # Create custom colormap
        if metric in ['email_count', 'meeting_count', 'docs_accessed']:
            cmap = 'YlOrRd'
        else:
            cmap = 'viridis'
        
        im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(dates)))
        ax.set_yticks(np.arange(len(valid_employees)))
        ax.set_xticklabels([d[-5:] for d in dates])  # Show MM-DD only
        ax.set_yticklabels([e.split('@')[0] for e in valid_employees])
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add anomaly markers if detector available
        if self.detector and annotate_anomalies:
            for i, emp in enumerate(valid_employees):
                anomalies = self.detector.get_anomalies(user_email=emp)
                for j, date in enumerate(dates):
                    if date in anomalies:
                        ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                             fill=False, edgecolor='red', lw=2))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20)
        
        # Add title
        ax.set_title(f'Activity Heatmap: {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Employee')
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_validation(self, employee_id: str, anomaly_type: str,
                              injection_period: Tuple[str, str],
                              metrics_to_check: Optional[List[str]] = None) -> None:
        """
        Validate that anomaly injection worked correctly.
        
        Shows before/during/after periods for verification.
        """
        if not self.simulator or employee_id not in self.simulator.simulated_data:
            print(f"No data found for {employee_id}")
            return
        
        # Default metrics based on anomaly type
        if metrics_to_check is None:
            default_metrics = {
                'data_exfiltration': ['docs_accessed', 'docs_shared', 'dropbox_added'],
                'burnout': ['email_count', 'meeting_count', 'teams_messages'],
                'overwork': ['email_count', 'meeting_count', 'docs_accessed'],
                'disengagement': ['email_count', 'teams_messages', 'meeting_count'],
                'unusual_hours': ['email_count', 'docs_accessed'],
                'communication_spike': ['email_count', 'teams_messages', 'meeting_count'],
                'policy_violation': ['email_recipients', 'docs_shared', 'dropbox_added']
            }
            metrics_to_check = default_metrics.get(anomaly_type, ['email_count', 'docs_accessed'])
        
        # Parse injection period
        start_date = datetime.strptime(injection_period[0], '%Y-%m-%d')
        end_date = datetime.strptime(injection_period[1], '%Y-%m-%d')
        
        # Define periods
        before_start = (start_date - timedelta(days=14)).strftime('%Y-%m-%d')
        after_end = (end_date + timedelta(days=14)).strftime('%Y-%m-%d')
        
        # Create plot
        fig, axes = plt.subplots(len(metrics_to_check), 1, 
                                figsize=(14, 3*len(metrics_to_check)), sharex=True)
        if len(metrics_to_check) == 1:
            axes = [axes]
        
        emp_data = self.simulator.simulated_data[employee_id]
        dates = sorted([d for d in emp_data.keys() if before_start <= d <= after_end])
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        
        for i, metric in enumerate(metrics_to_check):
            idx = self.simulator.metric_indices.get(metric, 0)
            values = [emp_data[d][idx] for d in dates]
            
            # Plot line
            axes[i].plot(date_objs, values, 'b-', linewidth=2, marker='o', markersize=4)
            
            # Highlight anomaly period
            axes[i].axvspan(start_date, end_date, alpha=0.3, color='red', 
                           label='Anomaly Period')
            
            # Calculate and show statistics
            before_values = [v for d, v in zip(dates, values) 
                           if d < injection_period[0]]
            during_values = [v for d, v in zip(dates, values) 
                           if injection_period[0] <= d <= injection_period[1]]
            after_values = [v for d, v in zip(dates, values) 
                          if d > injection_period[1]]
            
            if before_values:
                before_mean = np.mean(before_values)
                axes[i].axhline(y=before_mean, color='green', linestyle='--', 
                               alpha=0.5, label=f'Before μ={before_mean:.1f}')
            
            if during_values:
                during_mean = np.mean(during_values)
                axes[i].text(start_date + (end_date - start_date)/2, 
                           max(values) * 0.9, 
                           f'During μ={during_mean:.1f}', 
                           ha='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="red", alpha=0.5))
            
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        
        fig.suptitle(f'Anomaly Injection Validation: {employee_id}\n' + 
                    f'Type: {anomaly_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print verification statistics
        print(f"\nVerification for {anomaly_type}:")
        for metric in metrics_to_check:
            result = self.simulator.verify_anomaly_injection(
                employee_id,
                [d for d in dates if d < injection_period[0]],
                [d for d in dates if injection_period[0] <= d <= injection_period[1]],
                metric
            )
            if 'error' not in result:
                print(f"  {metric}: {result['percent_change']:+.1f}% change, "
                      f"z-score={result['z_score']:.2f}, "
                      f"significant={result['is_significant']}")
    
    def plot_detection_summary(self) -> None:
        """
        Plot overall summary of detection results across all employees.
        """
        if not self.detector:
            print("Detector not available")
            return
        
        summary = self.detector.get_anomaly_summary()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Baseline quality distribution
        quality_data = summary['baseline_quality_distribution']
        if quality_data:
            ax1.bar(quality_data.keys(), quality_data.values(), 
                   color=['red', 'orange', 'yellow', 'green', 'purple'])
            ax1.set_xlabel('Baseline Quality')
            ax1.set_ylabel('Count')
            ax1.set_title('Baseline Quality Distribution')
            ax1.grid(True, alpha=0.3)
        
        # 2. Baseline type distribution
        type_data = summary.get('baseline_type_distribution', {})
        if type_data:
            ax2.pie(type_data.values(), labels=type_data.keys(), autopct='%1.1f%%')
            ax2.set_title('Baseline Type Usage')
        
        # 3. Anomaly statistics
        stats_text = f"""Total Observations: {summary['total_observations']}
Total Anomalies: {summary['total_anomalies']}
Anomaly Rate: {summary['anomaly_rate']:.2%}
Weekend Anomalies: {summary['weekend_anomalies']}

Average Anomaly Score: {summary['avg_anomaly_score']:.2f}
Median Anomaly Score: {summary['median_anomaly_score']:.2f}
Max Anomaly Score: {summary['max_anomaly_score']:.2f}

Configuration:
Personal Threshold: {summary['threshold_personal']}
Group Threshold: {summary['threshold_group']}
Group Detection: {summary['group_detection_enabled']}
Weekday Separation: {summary.get('strict_weekday_separation', 'N/A')}"""
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax3.axis('off')
        ax3.set_title('Detection Statistics')
        
        # 4. Top anomalies
        top_anomalies = self.detector.get_top_anomalies(n=10, min_quality='low')
        if top_anomalies:
            users = [a[0].split('@')[0][:15] for a in top_anomalies[:5]]
            scores = [a[2] for a in top_anomalies[:5]]
            
            ax4.barh(users, scores, color='red', alpha=0.7)
            ax4.set_xlabel('Anomaly Score')
            ax4.set_title('Top 5 Anomalies by Score')
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle('Anomaly Detection Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()