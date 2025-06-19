# EmployeeActivitySimulator - Realistic Employee Data Generator

## 🎯 What is the EmployeeActivitySimulator?

The EmployeeActivitySimulator is a sophisticated data generation system that creates realistic employee digital activity patterns for testing and development. It simulates authentic workplace behaviors across emails, meetings, document work, team chats, and file storage - complete with natural variations, seasonal patterns, department-specific behaviors, and controllable anomaly injection.

## 🤔 Why Do We Need a Simulator?

### The Development Challenge

When building the AnomalyDetector system, we need realistic test data that:
- **Mimics real employee behavior** (not random numbers)
- **Contains natural patterns** (busy Mondays, quiet Fridays, vacation periods)
- **Includes known anomalies** (so we can test if our detector finds them)
- **Covers diverse scenarios** (different departments, seniority levels, work styles)
- **Provides ground truth** (we know exactly what should be flagged as anomalous)

### Real Data Limitations
- **Privacy concerns**: Can't use actual employee data for development
- **Data availability**: Real data may not exist during development phase
- **Limited scenarios**: Real data might not include all edge cases we want to test
- **Unknown ground truth**: With real data, we don't always know what should be anomalous
- **Controlled testing**: Need repeatable scenarios for validation

## 🏗️ How the Simulator Works

### Department-Based Behavioral Modeling

The simulator doesn't generate random numbers. Instead, it creates realistic patterns based on **job roles and departments**:

#### 🖥️ **Engineering Department**
```python
'base_metrics': {
    'email_count': (5, 15),        # Lower email volume
    'meeting_count': (2, 5),       # Moderate meetings
    'teams_messages': (20, 50),    # High chat activity
    'docs_accessed': (10, 25),     # High document usage
    'weekend_work': 0.3            # 30% chance of weekend work
}
```

#### 💼 **Sales Department**
```python
'base_metrics': {
    'email_count': (20, 60),       # High email volume
    'meeting_count': (4, 10),      # Many meetings
    'teams_messages': (10, 30),    # Moderate chat
    'docs_accessed': (5, 15),      # Lower document usage
    'weekend_work': 0.1            # 10% chance of weekend work
}
```

#### 👔 **Management**
```python
'base_metrics': {
    'email_count': (25, 70),       # Very high email volume
    'meeting_count': (6, 15),      # Very high meeting load
    'meeting_btb_ratio': (0.5, 0.8), # Many back-to-back meetings
    'weekend_work': 0.4            # 40% chance of weekend work
}
```

#### 👥 **HR Department**
```python
'base_metrics': {
    'email_length': (250, 600),    # Longer, detailed emails
    'docs_accessed': (20, 50),     # Heavy document work
    'meeting_virtual_ratio': (0.3, 0.5), # More in-person meetings
    'weekend_work': 0.05           # Very rare weekend work
}
```

#### 💰 **Finance Department**
```python
'base_metrics': {
    'docs_accessed': (25, 60),     # Very heavy document usage
    'docs_edited': (10, 30),       # Lots of document editing
    'teams_messages': (5, 15),     # Lower chat activity
    'weekend_work': 0.2            # Moderate weekend work (deadlines)
}
```

### Personal Work Style Variations

Each simulated employee gets **individual personality traits** that affect their patterns:

```python
personality = {
    'consistency': 0.8,        # How consistent daily patterns are (0.7-1.0)
    'morning_person': 0.7,     # Affects activity distribution (0-1)  
    'collaborative': 0.9,      # Affects teams/meeting metrics (0.3-1.0)
    'detail_oriented': 0.6,    # Affects email length, doc edits (0.5-1.0)
    'responsive': 0.8          # Affects reply ratios (0.5-1.0)
}
```

### Natural Time Patterns

#### **Weekly Cycles**
```python
weekday_multipliers = {
    0: 1.2,  # Monday - catch up from weekend
    1: 1.0,  # Tuesday - normal
    2: 1.0,  # Wednesday - normal  
    3: 1.0,  # Thursday - normal
    4: 0.9,  # Friday - wind down
    5: 0.1,  # Saturday - minimal
    6: 0.05  # Sunday - very minimal
}
```

#### **Seasonal Variations**
```python
seasonal_patterns = {
    'quarter_end': {'weeks': [12, 13, 25, 26, 38, 39, 51, 52], 'multiplier': 1.3},
    'summer_vacation': {'weeks': range(26, 35), 'multiplier': 0.7},
    'holiday_season': {'weeks': [51, 52, 1], 'multiplier': 0.5},
    'new_year_surge': {'weeks': [2, 3, 4], 'multiplier': 1.2}
}
```

### Inter-Metric Correlations

The simulator creates **realistic relationships** between different activities:

```python
metric_correlations = {
    'meeting_count': {
        'email_count': 0.6,      # More meetings → more emails
        'docs_accessed': 0.5     # Meeting materials accessed
    },
    'docs_created': {
        'docs_edited': 0.8,      # Created docs are often edited
        'docs_shared': 0.6,      # New docs get shared
        'email_count': 0.3       # Emails about new documents
    },
    'teams_messages': {
        'email_count': -0.3      # Teams can replace email
    }
}
```

## 🎭 Advanced Anomaly Injection System

### 7 Types of Realistic Anomalies

#### 1. **Data Exfiltration** 🚨
```python
'data_exfiltration': {
    'description': 'Sudden spike in document access/sharing and file activity',
    'metrics_affected': ['docs_accessed', 'docs_shared', 'dropbox_added', 'email_count'],
    'multipliers': {'docs_accessed': 5.0, 'docs_shared': 8.0, 'dropbox_added': 10.0}
}
```

#### 2. **Burnout** 😴
```python
'burnout': {
    'description': 'Gradual decline in all activities',
    'metrics_affected': 'all',
    'multipliers': 0.3  # Reduces all metrics to 30% of normal
}
```

#### 3. **Overwork** 🔥
```python
'overwork': {
    'description': 'Excessive activity across all metrics',
    'metrics_affected': 'all',
    'multipliers': 2.5  # 250% of normal activity
}
```

#### 4. **Disengagement** 📉
```python
'disengagement': {
    'description': 'Reduced communication and collaboration',
    'metrics_affected': ['email_count', 'teams_messages', 'meeting_count'],
    'multipliers': {'email_count': 0.2, 'teams_messages': 0.1, 'meeting_count': 0.3}
}
```

#### 5. **Unusual Hours** 🌙
```python
'unusual_hours': {
    'description': 'Activity at unusual times (weekend spike)',
    'metrics_affected': 'all',
    'multipliers': 3.0,
    'weekend_only': True  # Only affects weekends
}
```

#### 6. **Communication Spike** 📢
```python
'communication_spike': {
    'description': 'Sudden increase in all communication channels',
    'metrics_affected': ['email_count', 'teams_messages', 'meeting_count'],
    'multipliers': {'email_count': 4.0, 'teams_messages': 5.0, 'meeting_count': 3.0}
}
```

#### 7. **Policy Violation** ⚠️
```python
'policy_violation': {
    'description': 'Unusual file sharing and external communication',
    'metrics_affected': ['email_recipients', 'docs_shared', 'dropbox_added'],
    'multipliers': {'email_recipients': 5.0, 'docs_shared': 10.0, 'dropbox_added': 8.0}
}
```

## 🚀 How to Use the Simulator

### Basic Employee Simulation

```python
from simulator import EmployeeActivitySimulator

# Initialize with fixed seed for reproducible results
simulator = EmployeeActivitySimulator(random_seed=42)

# Define your team
employees = [
    {'id': 'john.doe@company.com', 'department': 'engineering', 'seniority': 'senior'},
    {'id': 'jane.smith@company.com', 'department': 'sales', 'seniority': 'mid'},
    {'id': 'bob.johnson@company.com', 'department': 'management', 'seniority': 'executive'},
    {'id': 'alice.brown@company.com', 'department': 'hr', 'seniority': 'senior'},
    {'id': 'charlie.wilson@company.com', 'department': 'engineering', 'seniority': 'junior'}
]

# Simulate 90 days of activity
data = simulator.simulate_employees(employees, '2024-09-01', '2024-11-30')
```

### Add Team Correlations

```python
# Engineering team works on shared projects
simulator.add_team_correlation(
    team_members=['john.doe@company.com', 'charlie.wilson@company.com'],
    correlation_strength=0.6,  # Strong correlation
    start_date='2024-10-01',
    end_date='2024-10-31'
)
```

### Inject Specific Anomalies

```python
# Data exfiltration - sudden spike
simulator.inject_anomaly(
    employee_id='john.doe@company.com',
    anomaly_type='data_exfiltration',
    start_date='2024-11-15',
    end_date='2024-11-17',
    intensity=1.5  # 50% more severe than default
)

# Burnout - gradual decline
simulator.inject_gradual_anomaly(
    employee_id='jane.smith@company.com',
    anomaly_type='burnout',
    start_date='2024-10-15',  # Starts here
    peak_date='2024-11-01',   # Worst at this point
    end_date='2024-11-20',    # Continues until here
    peak_intensity=1.0
)

# Unusual weekend activity
simulator.inject_anomaly(
    employee_id='charlie.wilson@company.com',
    anomaly_type='unusual_hours',
    start_date='2024-11-09',  # Saturday
    end_date='2024-11-10',    # Sunday
    intensity=1.0
)
```

## 📊 Generated Data Format

### Perfect Compatibility with AllMetrics

The simulator generates data in **exactly the same format** as real Microsoft 365 data:

```python
# Output matches AllMetrics expected format
{
    "user@company.com": {
        "2024-03-15": numpy.array([
            # Email metrics (indices 0-3)
            12.0, 245.5, 0.67, 2.3,
            # Calendar metrics (indices 4-8)  
            4.0, 180.0, 0.1, 0.8, 0.3,
            # Teams metrics (indices 9-11)
            25.0, 45.2, 4.0,
            # Document metrics (indices 12-15)
            18.0, 2.0, 5.0, 1.0,
            # Dropbox metrics (indices 16-17)
            3.0, 7.0
        ])
    }
}
```

### Direct Integration with AnomalyDetector

```python
# Generate data
simulator = EmployeeActivitySimulator()
employees = [{'id': 'user@company.com', 'department': 'engineering'}]
simulated_data = simulator.simulate_employees(employees, '2024-01-01', '2024-12-31')

# Use directly with detector
from detector import AnomalyDetector
detector = AnomalyDetector(simulated_data)
anomalies = detector.get_anomalies()
```

## 🧪 Testing and Validation

### Validate Generated Patterns

```python
# Check if patterns look realistic
summary = simulator.get_summary_statistics()
print(summary[['employee_id', 'department', 'total_days', 'weekend_days_active']])

# Visualize individual employee
simulator.visualize_employee_timeline('john.doe@company.com')
```

### Verify Anomaly Injection

```python
# Check what anomalies were injected
injected = simulator.get_injected_anomalies()  # If implemented
print(f"Injected {len(injected)} anomalies")

# Validate with AnomalyDetector
detector = AnomalyDetector(simulator.export_for_anomaly_detector())
detected = detector.get_anomalies()

# Calculate detection accuracy
true_positives = count_matches(injected, detected)
accuracy = true_positives / len(injected)
print(f"Detection accuracy: {accuracy:.2%}")
```

### Regression Testing

```python
# Create consistent test dataset
simulator = EmployeeActivitySimulator(random_seed=12345)
simulator.generate_standard_test_dataset()

def test_anomaly_detection_consistency():
    # Same seed = same data = same results
    detector = AnomalyDetector(simulator.export_for_anomaly_detector())
    anomalies = detector.get_anomalies()
    
    # Should always detect the same anomalies
    assert len(anomalies) == EXPECTED_COUNT
    assert all_expected_anomalies_found(anomalies)
```

## 🎛️ Advanced Configuration

### Custom Employee Profiles

```python
# Create employee with specific traits
simulator.create_employee_profile(
    employee_id='specialist@company.com',
    department='finance',
    seniority='senior',
    personality={
        'consistency': 0.95,    # Very consistent patterns
        'detail_oriented': 0.9, # Very thorough
        'collaborative': 0.3,   # Prefers working alone
        'responsive': 0.6       # Moderate responsiveness
    }
)
```

### Multi-Organization Simulation

```python
# Different company cultures
simulator.create_organization(
    name='tech_startup',
    culture='fast_paced',
    remote_work_ratio=0.8,
    meeting_culture='minimal'
)

simulator.create_organization(
    name='traditional_corp', 
    culture='structured',
    remote_work_ratio=0.3,
    meeting_culture='heavy'
)
```

### Time-Series Evolution

```python
# Simulate organizational changes
simulator.add_organizational_change(
    date='2024-06-01',
    change_type='remote_work_increase',
    affected_ratio=0.7,      # 70% of employees affected
    duration_days=90
)

# Add seasonal effects
simulator.add_seasonal_pattern(
    pattern='holiday_slowdown',
    dates=['2024-12-20', '2024-01-05'],
    activity_reduction=0.4   # 40% reduction
)
```

## 📈 Quality Assurance Features

### Built-in Validation

```python
# Validate generated patterns
validation = simulator.validate_generated_data()
print(validation)

# Example output:
{
    'email_patterns': {
        'weekly_correlation': 0.85,     # Strong weekly patterns
        'daily_variation': 'normal',    # Realistic day-to-day changes
        'role_differentiation': 0.92    # Clear differences between roles
    },
    'meeting_patterns': {
        'duration_distribution': 'realistic',
        'time_slot_preferences': 'expected'
    },
    'anomaly_injection': {
        'detection_rate': 0.87,         # 87% of injected anomalies detectable
        'false_positive_rate': 0.05     # 5% false positive rate
    }
}
```

### Statistical Verification

```python
# Check data quality
completeness = simulator.check_data_completeness()
assert completeness['missing_days'] < 0.01      # <1% missing data
assert completeness['data_consistency'] > 0.95  # >95% consistency

# Verify realistic distributions
distributions = simulator.analyze_distributions()
assert distributions['email_volume']['follows_expected'] == True
assert distributions['meeting_patterns']['realistic_timing'] == True
```

## 🔧 Real-World Usage Examples

### Development Environment Setup

```python
def setup_dev_environment():
    """Create development data for testing."""
    simulator = EmployeeActivitySimulator(random_seed=42)
    
    # Create diverse team
    employees = []
    for dept in ['engineering', 'sales', 'management', 'hr', 'finance']:
        for seniority in ['junior', 'mid', 'senior']:
            employees.append({
                'id': f'{dept}_{seniority}@company.com',
                'department': dept,
                'seniority': seniority
            })
    
    # 6 months of data
    data = simulator.simulate_employees(employees, '2024-06-01', '2024-12-01')
    
    # Add realistic team correlations
    teams = [
        ['engineering_senior@company.com', 'engineering_mid@company.com'],
        ['sales_senior@company.com', 'sales_mid@company.com']
    ]
    
    for team in teams:
        simulator.add_team_correlation(team, correlation_strength=0.5)
    
    # Inject test anomalies
    test_anomalies = [
        ('engineering_senior@company.com', 'data_exfiltration', '2024-09-15', '2024-09-16'),
        ('sales_mid@company.com', 'burnout', '2024-10-01', '2024-10-15'),
        ('hr_senior@company.com', 'unusual_hours', '2024-11-09', '2024-11-10')
    ]
    
    for emp, anomaly_type, start, end in test_anomalies:
        simulator.inject_anomaly(emp, anomaly_type, start, end)
    
    return simulator.export_for_anomaly_detector()
```

### Continuous Integration Testing

```python
def test_detector_performance():
    """Automated test for CI/CD pipeline."""
    # Quick simulation for fast testing
    simulator = EmployeeActivitySimulator(random_seed=999)
    
    employees = [
        {'id': f'test{i}@company.com', 'department': 'engineering'}
        for i in range(10)
    ]
    
    # 30 days of data
    data = simulator.simulate_employees(employees, '2024-11-01', '2024-11-30')
    
    # Inject known anomalies
    simulator.inject_anomaly('test0@company.com', 'data_exfiltration', '2024-11-15')
    simulator.inject_anomaly('test1@company.com', 'burnout', '2024-11-20', '2024-11-25')
    
    # Test detection
    detector = AnomalyDetector(simulator.export_for_anomaly_detector())
    anomalies = detector.get_anomalies()
    
    # Validate results
    assert len(anomalies) >= 2  # Should find our injected anomalies
    assert 'test0@company.com' in anomalies
    assert 'test1@company.com' in anomalies
    
    print("✅ All tests passed!")
```

## 🚨 Key Features That Make It Realistic

### 1. **Behavioral Consistency**
- Each employee has persistent personality traits
- Patterns remain consistent over time unless anomalies are injected
- Natural day-to-day variation without losing core patterns

### 2. **Department Realism** 
- Engineering: Lower email, high collaboration, weekend work
- Sales: High email, many meetings, external focus
- Management: Very high meetings, back-to-back scheduling
- HR: Detailed emails, heavy document work, regular hours
- Finance: Document-heavy, periodic intense periods

### 3. **Temporal Authenticity**
- Monday catch-up patterns
- Friday wind-down patterns  
- Quarter-end activity spikes
- Holiday and vacation periods
- Natural seasonal variations

### 4. **Metric Relationships**
- More meetings → more pre-meeting document access
- Document creation → follow-up emails and sharing
- High Teams usage → reduced email dependency
- Meeting-heavy days → compressed individual work time

### 5. **Anomaly Realism**
- Gradual burnout patterns (not sudden stops)
- Data exfiltration with realistic access patterns
- Weekend work that matches individual profiles
- Policy violations with subtle behavioral indicators

## 🎯 Perfect for Anomaly Detection Development

The simulator is specifically designed to support anomaly detection development:

1. **Ground Truth**: You know exactly what should be flagged
2. **Controlled Testing**: Reproducible scenarios for validation
3. **Edge Cases**: Test rare but important situations
4. **Scalability Testing**: Generate data for thousands of employees
5. **Privacy Safe**: No real employee data needed

This makes it an essential tool for building, testing, and validating your AnomalyDetector system before deploying it on real organizational data.