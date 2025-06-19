# AnomalyDetector - Employee Activity Anomaly Detection

## 🎯 What is AnomalyDetector?

The AnomalyDetector is a sophisticated statistical system that analyzes employee digital activities to identify when someone's behavior significantly deviates from their normal patterns. It uses robust statistical methods to distinguish between normal day-to-day variation and genuinely anomalous behavior that might indicate security threats, productivity issues, or other concerns.

## 🤔 Why Do We Need This?

In today's digital workplace, employees create patterns through their daily activities:
- Sending emails and attending meetings
- Editing documents and chatting on Teams
- Using file storage and collaboration tools

Sometimes these patterns change dramatically, which might indicate:
- **Security threats** (compromised accounts, data exfiltration)
- **Productivity issues** (burnout, disengagement, overwork)
- **Process violations** (policy breaches, unusual data access)
- **Personal circumstances** (role changes, life events affecting work)

The challenge is distinguishing between normal variation (busy Monday vs. quiet Friday) and genuinely concerning anomalies.

## 🔍 How Does It Work?

### Weekday-Aware Detection System

The detector implements a **weekday-aware** approach that prevents false positives:

1. **Separate Baselines**: Compares weekdays only to weekdays, weekends only to weekends
2. **Personal Patterns**: Each person's normal behavior is learned individually
3. **Seasonal Adjustments**: Accounts for day-of-week variations (busy Mondays, quiet Fridays)
4. **Quality Assessment**: Ensures sufficient historical data before making judgments

### The Smart Detection Process

#### 1. **Learn Normal Patterns**
- Studies each employee's typical work habits over 30-90 days
- Creates separate baselines for weekdays vs. weekends
- Accounts for personal work styles and preferences

#### 2. **Weekday-Specific Comparison**
- **For Weekdays**: Compares today's Tuesday to previous Tuesdays (or all weekdays if limited data)
- **For Weekends**: Compares weekend activity to previous weekends
- **Special Handling**: Detects employees who rarely work weekends and flags unusual weekend activity

#### 3. **Robust Statistical Analysis**
- Uses **Median and MAD** (Median Absolute Deviation) instead of mean/standard deviation
- More resistant to outliers and data corruption
- Asymmetric detection: More sensitive to drops in activity than increases

#### 4. **Anomaly Scoring**
```
Z-Score = (Today's Value - Historical Median) / (1.4826 × MAD)
Overall Score = Weighted Average of |Z-Scores| across all 18 metrics
```

## 📊 What Activities Does It Monitor?

The system tracks **18 different metrics** across 5 categories:

### 📧 Email Activities (4 metrics)
- **Email Count**: Number of emails sent per day
- **Email Length**: Average character count of email bodies  
- **Forward/Reply Ratio**: Proportion of emails that are replies/forwards vs. new messages
- **Recipients**: Average number of recipients per email

### 📅 Calendar Activities (5 metrics)
- **Meeting Count**: Number of meetings attended
- **Meeting Duration**: Total time spent in meetings (minutes)
- **Cancellation Rate**: Proportion of meetings cancelled
- **Virtual Ratio**: Proportion of virtual vs. in-person meetings
- **Back-to-Back Ratio**: Proportion of meetings with <30min gaps

### 💬 Teams Chat Activities (3 metrics)
- **Message Count**: Number of chat messages sent
- **Message Length**: Median length of chat messages
- **Channel Usage**: Number of different channels used

### 📄 Document Activities (4 metrics)
- **Documents Accessed**: Number of documents opened/viewed
- **Documents Created**: Number of new documents created
- **Documents Edited**: Number of documents modified
- **Documents Shared**: Number of sharing activities

### 💾 File Storage Activities (2 metrics)
- **Files Added**: New files uploaded to Dropbox
- **Files Edited**: Existing files modified in Dropbox

## 🧮 The Advanced Detection Algorithm

### Weekday-Aware Baseline Calculation

```python
# For weekdays (Mon-Fri)
if target_day in [Mon, Tue, Wed, Thu, Fri]:
    baseline = median(previous_same_weekdays)  # Tue compared to previous Tuesdays
    
# For weekends (Sat-Sun)
if target_day in [Sat, Sun]:
    baseline = median(previous_weekends)  # Weekend compared to previous weekends
    
# Special case: Minimal weekend workers
if employee_rarely_works_weekends and weekend_activity > threshold:
    flag_as_anomaly  # Any significant weekend work is unusual
```

### Asymmetric Anomaly Detection

The system is **more sensitive to drops** than increases:

```python
# Standard detection
if abs(z_score) > 3.0:
    flag_as_anomaly
    
# Enhanced drop detection
if activity_drop > 80% of normal:
    flag_as_anomaly  # Even if z-score < 3.0
    
# Weekend sensitivity
weekend_threshold = standard_threshold * 0.7  # 30% more sensitive on weekends
```

### Quality-Based Confidence

- **High Quality**: 80%+ of expected historical data (very reliable)
- **Medium Quality**: 60-80% of expected data (reliable)
- **Low Quality**: Minimum data available (use with caution)
- **Insufficient Data**: Below minimum threshold (results not reliable)

## 🚀 How to Use It

### Basic Setup

```python
from all_metrics import AllMetrics
from detector import AnomalyDetector

# Step 1: Load your data
all_metrics = AllMetrics(
    email_folder="path/to/emails",
    calendar_folder="path/to/calendar",
    teams_folder="path/to/teams",
    document_folder="path/to/documents",
    dropbox_folder="path/to/dropbox"
)

# Step 2: Get unified data
unified_data = all_metrics.get_all_data()

# Step 3: Create detector with weekday-aware settings
detector = AnomalyDetector(
    unified_data=unified_data,
    min_baseline_days=30,              # Need at least 30 days of history
    lookback_window=90,                # Look back 90 days for patterns
    threshold_personal=3.0,            # Sensitivity (lower = more sensitive)
    strict_weekday_separation=True,    # Keep weekdays/weekends separate
    weekend_threshold_factor=0.7       # More sensitive on weekends
)
```

### Find Anomalies

```python
# Get all high-quality anomalies
anomalies = detector.get_anomalies(min_quality='medium')

# Get anomalies for specific person
user_anomalies = detector.get_anomalies(user_email='john@company.com')

# Get anomalies excluding weekends (if you don't care about weekend activity)
weekday_anomalies = detector.get_anomalies(exclude_weekends=True)

# Get top 10 most unusual activities
top_anomalies = detector.get_top_anomalies(n=10)
```

### Understand What Happened

```python
# Deep dive into a specific anomaly
details = detector.get_metric_contributions('john@company.com', '2024-03-15')

print(f"Overall Anomaly Score: {details['overall_anomaly_score']:.2f}")
print(f"Baseline Type Used: {details['baseline_type']}")
print(f"Baseline Quality: {details['baseline_quality']}")
print(f"Effective Threshold: {details['effective_threshold']:.2f}")

print("\nTop Contributing Metrics:")
for metric in details['metric_contributions'][:3]:
    print(f"- {metric['metric']}: {metric['weighted_contribution']:.2f}")
    print(f"  Value: {metric['value']:.1f}, Z-Score: {metric['seasonal_adjusted_z_score']:.2f}")
```

## ⚙️ Advanced Configuration

### Detection Sensitivity

```python
detector = AnomalyDetector(
    # More sensitive (catches more anomalies, may have false positives)
    threshold_personal=2.0,
    weekend_threshold_factor=0.5,  # Very sensitive on weekends
    
    # Less sensitive (fewer anomalies, more conservative)
    threshold_personal=4.0,
    weekend_threshold_factor=0.9,  # Less sensitive on weekends
)
```

### Baseline Requirements

```python
detector = AnomalyDetector(
    min_baseline_days=14,      # Minimum days needed (14-90 recommended)
    lookback_window=60,        # How far back to look (30-180 days)
    drop_threshold=0.8,        # Flag if activity drops by 80%
)
```

### Weekday Handling

```python
detector = AnomalyDetector(
    strict_weekday_separation=True,   # Separate weekday/weekend baselines
    strict_weekday_separation=False,  # Mixed baseline (less accurate)
)
```

## 📋 Understanding the Output

### Baseline Types Used

The detector automatically selects the best baseline:

- **`same_weekday`**: Compares Tuesday to previous Tuesdays (most accurate)
- **`all_weekdays`**: Compares weekday to all previous weekdays
- **`weekends`**: Compares weekend to previous weekends
- **`minimal_weekend`**: For employees who rarely work weekends
- **`all`**: Mixed baseline (fallback when data is limited)

### Anomaly Reasons

The detector explains why something was flagged:

```python
result = detector.get_metric_contributions('user@company.com', '2024-03-15')
print(result['anomaly_reasons'])
# Possible output: ['statistical_deviation', 'Email_Count_dropped_85%']
```

### Example Output Structure

```python
{
    'user': 'john@company.com',
    'date': '2024-03-15',
    'is_anomaly': True,
    'overall_anomaly_score': 4.2,
    'baseline_quality': 'high',
    'baseline_type': 'same_weekday',
    'baseline_days': 12,
    'is_weekend': False,
    'effective_threshold': 3.0,
    'severe_drop_detected': True,
    'anomaly_reasons': ['statistical_deviation', 'Email_Count_dropped_85%'],
    'metric_contributions': [
        {
            'metric': 'Email_Count',
            'value': 2.0,                    # Sent only 2 emails
            'seasonal_adjusted_z_score': -4.1,  # 4.1 standard deviations below normal
            'weighted_contribution': 1.8     # Contributed 1.8 to overall score
        }
    ]
}
```

## 🔧 Key Improvements Over Basic Detection

### 1. **Weekday-Weekend Separation**
- **Problem**: Old systems compared weekend activity to weekday baselines
- **Solution**: Separate baselines prevent false positives

### 2. **Asymmetric Sensitivity**
- **Problem**: Drops in activity (disengagement, burnout) were missed
- **Solution**: More sensitive to decreases than increases

### 3. **Minimal Weekend Worker Detection**
- **Problem**: Employees who never work weekends had any weekend activity flagged
- **Solution**: Special handling for employees with minimal weekend patterns

### 4. **Quality-Based Confidence**
- **Problem**: Results from insufficient data were unreliable
- **Solution**: Quality scores help users understand reliability

### 5. **Robust Statistics**
- **Problem**: Mean/standard deviation affected by outliers
- **Solution**: Median/MAD approach more resistant to data corruption

## 🚨 Important Notes

### What This System Does

- **Pattern Analysis**: Identifies statistical deviations from personal norms
- **Contextual Awareness**: Accounts for weekday/weekend differences
- **Quality Assessment**: Tells you how reliable each result is
- **Explainable Results**: Shows exactly which metrics contributed to anomalies

### What This System Does NOT Do

- **Content Analysis**: Doesn't read emails or documents
- **Real-time Monitoring**: Analyzes daily patterns, not live activity
- **Automatic Actions**: Flags anomalies but doesn't take action
- **Performance Evaluation**: Detects unusual patterns, not job performance

### When to Investigate Anomalies

- **High scores (4.0+)** with good baseline quality
- **Severe drops** in key activity metrics (>80% decrease)
- **Unusual weekend activity** for minimal weekend workers
- **Persistent anomalies** over multiple days
- **Multiple related anomalies** (several team members affected)

## 🔄 Production Usage Examples

### Daily Monitoring

```python
def daily_anomaly_check():
    # Load today's data
    detector = AnomalyDetector(get_latest_data())
    
    # Get high-confidence anomalies
    anomalies = detector.get_anomalies(min_quality='high')
    
    if anomalies:
        for user, user_anomalies in anomalies.items():
            for date, result in user_anomalies.items():
                print(f"ALERT: {user} on {date}")
                print(f"Score: {result['overall_anomaly_score']:.2f}")
                print(f"Reasons: {result.get('anomaly_reasons', [])}")
                
                # Send to security team if score is very high
                if result['overall_anomaly_score'] > 5.0:
                    send_security_alert(user, date, result)

# Run daily
daily_anomaly_check()
```

### Generate Reports

```python
# Export all results to CSV for analysis
df = detector.export_results_to_dataframe(include_non_anomalies=True)
df.to_csv('anomaly_report.csv', index=False)

# Get summary statistics
summary = detector.get_anomaly_summary()
print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
print(f"Average score: {summary['avg_anomaly_score']:.2f}")
```

## 🛡️ Privacy and Ethics

### Data Protection
- Processes only behavioral metadata (counts, durations, patterns)
- Does not access actual email content or document text
- Focuses on statistical patterns, not personal information

### Responsible Use
- Use only with proper employee consent and legal authorization
- Implement human review before taking any action based on results
- Consider individual circumstances and context
- Establish clear governance and escalation procedures

## 🆘 Troubleshooting

### Common Issues

**Q: Too many weekend anomalies**
```python
# Solution: Reduce weekend sensitivity
detector = AnomalyDetector(weekend_threshold_factor=0.9)
# Or exclude weekends entirely
anomalies = detector.get_anomalies(exclude_weekends=True)
```

**Q: No anomalies detected**
```python
# Check baseline quality
summary = detector.get_anomaly_summary()
print(summary['baseline_quality_distribution'])

# Lower threshold if quality is good
detector = AnomalyDetector(threshold_personal=2.5)
```

**Q: Poor baseline quality**
```python
# Check data coverage
dates = detector.get_date_range()
print(f"Data range: {dates}")

# Reduce minimum baseline requirements
detector = AnomalyDetector(min_baseline_days=14)
```

Remember: This tool provides statistical insights, but human judgment is essential for proper interpretation and action. Always consider the context and individual circumstances when investigating anomalies.