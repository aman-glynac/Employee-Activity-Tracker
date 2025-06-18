from Metrics.email_metrics import EmailMetrics
from Metrics.calendar_metrics import CalendarMetrics
from Metrics.dropbox_metrics import DropboxMetrics
from Metrics.teams_metrics import TeamsMetrics
from Metrics.document_metrics import DocumentMetrics
import numpy as np

class AllMetrics:
    def __init__(self, email_folder=None, calendar_folder=None, teams_folder=None, 
                 document_folder=None, dropbox_folder=None):
        """
        Initialize the unified metrics class with folders for each data source.
        
        Args:
            email_folder: Path to folder containing email JSON files
            calendar_folder: Path to folder containing calendar JSON files  
            teams_folder: Path to folder containing teams JSON files
            document_folder: Path to folder containing document JSON files
            dropbox_folder: Path to folder containing dropbox JSON files
        """
        self.email_folder = email_folder
        self.calendar_folder = calendar_folder
        self.teams_folder = teams_folder
        self.document_folder = document_folder
        self.dropbox_folder = dropbox_folder
        
        # Initialize individual metric classes
        self.email_metrics = EmailMetrics(email_folder) if email_folder else None
        self.calendar_metrics = CalendarMetrics(calendar_folder) if calendar_folder else None
        self.teams_metrics = TeamsMetrics(teams_folder) if teams_folder else None
        self.document_metrics = DocumentMetrics(document_folder) if document_folder else None
        self.dropbox_metrics = DropboxMetrics(dropbox_folder) if dropbox_folder else None
        
        # Store unified results
        self.unified_data = {}
        self._compute_unified_metrics()
    
    def _compute_unified_metrics(self):
        """Compute and combine metrics from all sources."""
        # Get results from each metric class
        email_results = self.email_metrics.compute_metrics() if self.email_metrics else {}
        calendar_results = self.calendar_metrics.compute_metrics() if self.calendar_metrics else {}
        teams_results = self.teams_metrics.compute_metrics() if self.teams_metrics else {}
        document_results = self.document_metrics.compute_metrics() if self.document_metrics else {}
        dropbox_results = self.dropbox_metrics.compute_metrics() if self.dropbox_metrics else {}
        
        # Collect all users and dates
        all_users = set()
        all_dates = set()
        
        for results in [email_results, calendar_results, teams_results, document_results, dropbox_results]:
            for user, user_data in results.items():
                all_users.add(user)
                all_dates.update(user_data.keys())
        
        # Initialize unified data structure
        for user in all_users:
            self.unified_data[user] = {}
            
            for date in all_dates:
                # Get metrics for each source (default to zeros if not present)
                email_metrics = email_results.get(user, {}).get(date, np.array([0, 0, 0, 0]))  # [E_d, L_d, FRR_d, R_d]
                calendar_metrics = calendar_results.get(user, {}).get(date, np.array([0, 0, 0, 0, 0]))  # [Meeting Count, Duration, Cancel Rate, Virtual Ratio, BTB Ratio]
                teams_metrics = teams_results.get(user, {}).get(date, np.array([0, 0, 0]))  # [T_d, ML_d, TC_d]
                document_metrics = document_results.get(user, {}).get(date, np.array([0, 0, 0, 0]))  # [Accessed, Created, Edited, Shared]
                dropbox_metrics = dropbox_results.get(user, {}).get(date, np.array([0, 0]))  # [DA_d, DE_d]
                
                # Combine all metrics into one array
                combined_metrics = np.concatenate([
                    email_metrics,      # indices 0-3
                    calendar_metrics,   # indices 4-8
                    teams_metrics,      # indices 9-11
                    document_metrics,   # indices 12-15
                    dropbox_metrics     # indices 16-17
                ], dtype=np.float64)
                
                self.unified_data[user][date] = combined_metrics
    
    def get_all_data(self):
        """Return the complete unified data dictionary."""
        return self.unified_data
    
    def filter_by_user(self, user_email):
        """
        Filter data for a specific user.
        
        Args:
            user_email: Email address of the user
            
        Returns:
            Dictionary with dates as keys and metric arrays as values
        """
        return self.unified_data.get(user_email.lower(), {})
    
    def filter_by_date_range(self, start_date, end_date, user_email=None):
        """
        Filter data by date range, optionally for a specific user.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            user_email: Optional user email to filter by
            
        Returns:
            Dictionary with user emails as keys (if user_email is None) or 
            dictionary with dates as keys (if user_email is specified)
        """
        if user_email:
            user_data = self.unified_data.get(user_email.lower(), {})
            return {
                date: metrics for date, metrics in user_data.items()
                if start_date <= date <= end_date
            }
        else:
            filtered_data = {}
            for user, user_data in self.unified_data.items():
                filtered_user_data = {
                    date: metrics for date, metrics in user_data.items()
                    if start_date <= date <= end_date
                }
                if filtered_user_data:  # Only include users with data in the date range
                    filtered_data[user] = filtered_user_data
            return filtered_data
    
    def filter_by_user_and_date_range(self, user_email, start_date, end_date):
        """
        Filter data for a specific user within a date range.
        
        Args:
            user_email: Email address of the user
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with dates as keys and metric arrays as values
        """
        return self.filter_by_date_range(start_date, end_date, user_email)
    
    def get_metric_descriptions(self):
        """Return descriptions of what each index in the metric array represents."""
        return {
            'email_metrics': {
                0: 'E_d - Number of emails sent',
                1: 'L_d - Average email body length',
                2: 'FRR_d - Forward/Reply ratio',
                3: 'R_d - Average number of recipients'
            },
            'calendar_metrics': {
                4: 'Meeting Count - Total meetings',
                5: 'Total Duration - Total meeting duration in minutes',
                6: 'Cancellation Rate - Ratio of cancelled meetings',
                7: 'Virtual/In-person Ratio - Ratio of virtual to in-person meetings',
                8: 'BTB Ratio - Back-to-back meeting ratio'
            },
            'teams_metrics': {
                9: 'T_d - Total messages sent',
                10: 'ML_d - Median message length',
                11: 'TC_d - Total unique channels used'
            },
            'document_metrics': {
                12: 'Documents Accessed',
                13: 'Documents Created',
                14: 'Documents Edited',
                15: 'Documents Shared'
            },
            'dropbox_metrics': {
                16: 'DA_d - Files added to Dropbox',
                17: 'DE_d - Files edited in Dropbox'
            }
        }
    
    def get_users(self):
        """Return list of all users in the dataset."""
        return list(self.unified_data.keys())
    
    def get_date_range(self, user_email=None):
        """
        Get the date range for all data or for a specific user.
        
        Args:
            user_email: Optional user email to get date range for
            
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        if user_email:
            user_data = self.unified_data.get(user_email.lower(), {})
            if not user_data:
                return None, None
            dates = list(user_data.keys())
        else:
            dates = []
            for user_data in self.unified_data.values():
                dates.extend(user_data.keys())
        
        if not dates:
            return None, None
            
        return min(dates), max(dates)