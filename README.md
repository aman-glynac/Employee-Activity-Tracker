# Employee Activity Anomaly Detection System

A comprehensive system for detecting anomalous behavior patterns in employee digital activities across multiple data sources including emails, calendar events, document activities, team communications, and file storage interactions.

## 📋 Overview

This system monitors employee digital footprints to identify meaningful deviations from normal behavior patterns. It combines personal baselines with peer group comparisons to provide contextual anomaly detection that respects individual work patterns while identifying outliers within teams.

### Key Features

- **Multi-Source Data Integration**: Processes emails, calendar events, document activities, Teams messages, and Dropbox interactions
- **Dual-Baseline Approach**: Combines personal historical patterns with peer group comparisons
- **Robust Statistical Methods**: Uses median and MAD (Median Absolute Deviation) for noise-resistant anomaly detection
- **Real-time Processing**: Designed for daily metric computation and anomaly flagging
- **Unified Scoring**: Combines multiple metrics into a single anomaly score for easy interpretation

## 🎯 Problem Statement

Organizations need to identify when employees exhibit unusual digital behavior patterns that may indicate:
- Security threats or data exfiltration
- Productivity issues or disengagement
- Process violations or policy breaches
- Stress or burnout indicators

The challenge is distinguishing between normal variation in work patterns and genuinely anomalous behavior that requires attention.

## 🔧 Architecture

### Data Sources
The system processes JSON files from various Microsoft 365 and collaboration tools:

- **Email Data**: Outlook email activities
- **Calendar Data**: Meeting schedules and patterns
- **Document Data**: SharePoint/OneDrive document interactions
- **Teams Data**: Chat messages and channel activities
- **Dropbox Data**: File storage and modification events

### Metrics Computation

#### Email Metrics (4 dimensions)
- **Daily Email Count (E_d)**: Number of emails sent per day
- **Average Email Body Length (L_d)**: Mean character count of email bodies
- **Forward/Reply Ratio (FRR_d)**: Proportion of emails that are replies or forwards
- **Recipients per Email (R_d)**: Average number of recipients per email

#### Calendar Metrics (5 dimensions)
- **Meeting Count**: Number of calendar events per day
- **Total Meeting Duration**: Sum of meeting durations in minutes
- **Cancellation Rate**: Proportion of cancelled meetings
- **Virtual to In-person Ratio**: Ratio of virtual vs physical meetings
- **Back-to-Back Meeting Ratio**: Proportion of meetings with <30min gaps

#### Teams Metrics (3 dimensions)
- **Total Messages Sent**: Number of chat messages posted
- **Median Message Length**: Median character count of messages
- **Total Unique Channels**: Number of different channels used

#### Document Metrics (4 dimensions)
- **Total Documents Accessed**: Number of documents opened/viewed
- **Total Documents Created**: Number of new documents created
- **Total Documents Edited**: Number of documents modified
- **Total Documents Shared**: Number of sharing activities

#### Dropbox Metrics (2 dimensions)
- **Files Added**: Number of new files uploaded
- **Files Edited**: Number of existing files modified

### Anomaly Detection Algorithm

#### 1. Personal Baseline Calculation
For each employee and metric:
```
Personal_Median = median(historical_values)
Personal_MAD = median(|value - Personal_Median|)
Personal_Z_Score = (current_value - Personal_Median) / (1.4826 × Personal_MAD)
```

#### 2. Group Baseline Calculation
For each metric across all employees on the same day:
```
Group_Median = median(all_employee_values_today)
Group_MAD = median(|employee_value - Group_Median|)
Group_Z_Score = (employee_value - Group_Median) / (1.4826 × Group_MAD)
```

#### 3. Combined Scoring
```
Combined_Z_Score = (Personal_Z_Score + Group_Z_Score) / 2
Overall_Anomaly_Score = mean(|Combined_Z_Score|) across all 18 metrics
```

#### 4. Anomaly Flagging
```
if Overall_Anomaly_Score > threshold (e.g., 3.0):
    flag_as_anomaly()
```

## 📁 Project Structure

```
├── AllMetrics.py              # Unified metrics processor (combines all data sources)
├── AnomalyDetector.py         # Main anomaly detection engine
└── Metrics/
    ├── email_metrics.py       # Email activity processing
    ├── calendar_metrics.py    # Calendar event processing  
    ├── document_metrics.py    # Document interaction processing
    ├── teams_metrics.py       # Teams chat processing
    └── dropbox_metrics.py     # File storage processing
```

## 🚀 Usage

### Prerequisites
```bash
pip install numpy pandas scipy datetime collections functools
```

### Complete Workflow

```python
from AllMetrics import AllMetrics
from AnomalyDetector import AnomalyDetector

# Step 1: Initialize unified metrics processor
all_metrics = AllMetrics(
    email_folder="path/to/email/data",
    calendar_folder="path/to/calendar/data", 
    teams_folder="path/to/teams/data",
    document_folder="path/to/document/data",
    dropbox_folder="path/to/dropbox/data"
)

# Step 2: Get unified data (automatically computed)
unified_data = all_metrics.get_all_data()
# Format: {user_email: {date: numpy_array_of_18_metrics}}

# Step 3: Initialize anomaly detector
detector = AnomalyDetector(
    unified_data=unified_data,
    min_baseline_days=30,           # Minimum days for baseline
    lookback_window=90,             # Days to look back for baseline
    threshold_personal=3.0,         # Personal anomaly threshold
    threshold_group=3.0,            # Group anomaly threshold  
    enable_group_detection=True     # Enable peer comparison
)

# Step 4: Get anomaly results
anomalies = detector.get_anomalies(min_quality='medium')
summary = detector.get_anomaly_summary()
top_anomalies = detector.get_top_anomalies(n=10)

# Step 5: Detailed analysis
user_anomalies = detector.get_anomalies(user_email='user@company.com')
metric_breakdown = detector.get_metric_contributions('user@company.com', '2021-11-17')

# Step 6: Export results
results_df = detector.export_results_to_dataframe(include_non_anomalies=True)
```

### AllMetrics Class Features

```python
# Filter by user
user_data = all_metrics.filter_by_user('user@company.com')

# Filter by date range
recent_data = all_metrics.filter_by_date_range('2021-11-01', '2021-11-30')

# Get specific user and date range
user_period = all_metrics.filter_by_user_and_date_range(
    'user@company.com', '2021-11-01', '2021-11-30'
)

# Get metadata
users = all_metrics.get_users()
date_range = all_metrics.get_date_range()
metric_descriptions = all_metrics.get_metric_descriptions()
```

## 🔧 Advanced Anomaly Detection Features

### Statistical Approach
The `AnomalyDetector` class implements sophisticated anomaly detection with:

#### 1. Dual-Domain Analysis
- **Personal Baseline**: Individual historical patterns using robust statistics (median + MAD)
- **Group Baseline**: Peer comparison within same weekday and time window
- **Seasonal Adjustments**: Weekday-specific pattern recognition

#### 2. Robust Statistical Methods
```python
# Personal Z-Score calculation
Personal_Z_Score = (current_value - Personal_Median) / (1.4826 × Personal_MAD)

# Group Z-Score calculation  
Group_Z_Score = (current_value - Group_Median) / (1.4826 × Group_MAD)

# Combined scoring with weighted metrics
Overall_Score = mean(|Z_Score| × Metric_Weight) across all 18 metrics
```

#### 3. Performance Optimizations
- **Vectorized Operations**: Batch processing using NumPy for all 18 metrics simultaneously
- **Caching**: Pre-computed date conversions, weekday mappings, and group baselines
- **Memory Efficiency**: Optimized data structures and batch processing

#### 4. Quality Assessment
- **High Quality**: 80%+ of lookback window data available
- **Medium Quality**: 2× minimum baseline days available  
- **Low Quality**: Minimum baseline days available
- **Insufficient Data**: Below minimum threshold

### Anomaly Detection Parameters

```python
detector = AnomalyDetector(
    unified_data=data,
    min_baseline_days=30,        # Minimum historical data required
    lookback_window=90,          # Days to consider for baseline
    threshold_personal=3.0,      # Personal anomaly threshold (Z-score)
    threshold_group=3.0,         # Group anomaly threshold (Z-score)
    enable_group_detection=True, # Enable peer comparison
    metric_weights={             # Custom metric importance weights
        'Email_Count': 1.5,
        'Meeting_Duration': 2.0,
        # ... other metrics
    }
)
```

### Analysis Methods

#### Get Anomalies with Filtering
```python
# High-quality anomalies only, excluding weekends
anomalies = detector.get_anomalies(
    user_email='user@company.com',
    min_quality='high',
    exclude_weekends=True
)

# All anomalies across organization
all_anomalies = detector.get_anomalies(min_quality='medium')
```

#### Detailed Metric Analysis
```python
# Understand which metrics contributed to anomaly
breakdown = detector.get_metric_contributions('user@company.com', '2021-11-17')
print(f"Top contributing metrics: {breakdown['metric_contributions'][:3]}")
```

#### Summary Statistics
```python
summary = detector.get_anomaly_summary()
print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
print(f"Quality distribution: {summary['baseline_quality_distribution']}")
```

#### Email Data Format
```json
{
    "folder_name": "sent",
    "sent_date_time": "2021-11-17T14:06:24Z",
    "subject": "Meeting Follow-up",
    "body": "Email content...",
    "recipients": [
        {
            "email_address": "sender@company.com",
            "recipient_type": "from"
        },
        {
            "email_address": "recipient@company.com", 
            "recipient_type": "to"
        }
    ]
}
```

#### Calendar Data Format
```json
{
    "Start": "2021-11-17T14:00:00Z",
    "End": "2021-11-17T15:00:00Z",
    "Title": "Team Meeting",
    "Virtual": true
}
```

## 🔍 Key Advantages

1. **Personal Context**: Respects individual work patterns and habits
2. **Peer Comparison**: Identifies outliers within team or role groups
3. **Robust Statistics**: Uses median/MAD to resist outlier contamination
4. **Multi-dimensional**: Captures various aspects of digital behavior
5. **Scalable**: Processes data efficiently for large organizations
6. **Interpretable**: Provides clear anomaly scores and explanations

## 🎯 Use Cases

- **Security Monitoring**: Detect potential insider threats or compromised accounts
- **Productivity Analysis**: Identify disengagement or overwork patterns
- **Compliance Monitoring**: Ensure adherence to communication and data policies
- **Wellness Programs**: Identify stress indicators through work pattern changes
- **Process Optimization**: Understand normal vs. exceptional work patterns

## 🚧 Development Status

✅ **Completed Components:**
- **AllMetrics.py**: Unified data processor combining all 5 data sources
- **AnomalyDetector.py**: Complete anomaly detection engine with optimization
- **Email Metrics**: Processing (updated for new data format) 
- **Calendar Metrics**: Meeting pattern analysis
- **Document Metrics**: SharePoint/OneDrive activity tracking
- **Teams Metrics**: Chat and communication analysis  
- **Dropbox Metrics**: File storage activity monitoring

✅ **Advanced Features Implemented:**
- Vectorized statistical computations for performance
- Dual-baseline anomaly detection (personal + group)
- Seasonal/weekday adjustments
- Quality assessment and filtering
- Comprehensive result export and analysis
- Memory-optimized batch processing
- Caching for improved performance

📋 **Planned Enhancements:**
- Real-time streaming pipeline
- Interactive dashboard and visualization
- Configurable alert thresholds per role/department
- Historical trend analysis
- Integration APIs for SIEM systems
- Machine learning model integration

## 📊 Output Formats

### AllMetrics Output
Each metric processor returns unified data:
```python
{
    "user@company.com": {
        "2021-11-17": numpy.array([
            # Email metrics (0-3)
            email_count, avg_length, frr_ratio, recipients,
            # Calendar metrics (4-8)  
            meeting_count, duration, cancel_rate, virtual_ratio, btb_ratio,
            # Teams metrics (9-11)
            messages, median_length, channels,
            # Document metrics (12-15)
            accessed, created, edited, shared,
            # Dropbox metrics (16-17)
            files_added, files_edited
        ])
    }
}
```

### AnomalyDetector Output
Comprehensive anomaly analysis results:
```python
{
    "user@company.com": {
        "2021-11-17": {
            'date': '2021-11-17',
            'user': 'user@company.com',
            'metrics': numpy.array([...]),  # 18-dimensional metrics
            'is_anomaly': True,
            'overall_anomaly_score': 4.2,
            'baseline_quality': 'high',
            'baseline_days': 67,
            'is_weekend': False,
            'personal_z_scores': numpy.array([...]),
            'seasonal_adjusted_z_scores': numpy.array([...]),
            'group_z_scores': numpy.array([...]),
            'seasonal_adjustments': numpy.array([...])
        }
    }
}
```

### Export to DataFrame
```python
df = detector.export_results_to_dataframe(include_non_anomalies=True)
# Columns: user, date, is_anomaly, overall_anomaly_score, baseline_quality,
#          [metric_name]_value, [metric_name]_personal_z, [metric_name]_seasonal_z, etc.
```

## 🤝 Contributing

This system is designed for enterprise deployment with careful consideration of privacy and security requirements. All processing should be done on authorized data with proper consent and governance frameworks in place.

## 📄 License

[Insert appropriate license information]

---

**Note**: This system is designed for legitimate organizational monitoring with proper consent and governance. Ensure compliance with privacy laws and company policies before deployment.