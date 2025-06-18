import os
import json
import numpy as np
from collections import defaultdict

class EmailMetrics:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def _get_sender_email(self, email):
        """Extract sender email from recipients list"""
        recipients = email.get("recipients", [])
        for recipient in recipients:
            if recipient.get("recipient_type") == "from":
                return recipient.get("email_address", "").lower()
        return ""

    def _get_to_recipients(self, email):
        """Extract 'to' recipients from recipients list"""
        recipients = email.get("recipients", [])
        to_recipients = []
        for recipient in recipients:
            if recipient.get("recipient_type") == "to":
                to_recipients.append(recipient.get("email_address", ""))
        return to_recipients

    def compute_metrics(self):
        results = defaultdict(dict)

        for filename in os.listdir(self.root_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(self.root_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    emails = json.load(f)

                # Filter for sent emails only
                sent_emails = [email for email in emails if email.get("folder_name") == "sent"]
                
                if not sent_emails:
                    continue
                
                # Get the user email from the first sent email
                user_email = self._get_sender_email(sent_emails[0])
                if not user_email:
                    continue

                # Group emails by date
                daily_sent_emails = defaultdict(list)
                for email in sent_emails:
                    if "sent_date_time" in email:
                        date = email["sent_date_time"][:10]  # YYYY-MM-DD
                        daily_sent_emails[date].append(email)

                # Calculate metrics for each date
                for date, emails in daily_sent_emails.items():
                    E_d = len(emails)

                    # L_d: Average body length
                    L_d = sum(len(email.get("body", "")) for email in emails) / E_d if E_d else 0

                    # FRR_d: subject starts with "Re:", "Fwd:", or contains "FW:"
                    FRR_d = sum(
                        1 for email in emails
                        if any((email.get("subject") if email.get("subject") is not None else "").lower().startswith(prefix) for prefix in ("re:", "fw:", "fwd:"))
                    ) / E_d if E_d else 0

                    # R_d: average number of "to" recipients
                    R_d = sum(
                        len(self._get_to_recipients(email))
                        for email in emails
                    ) / E_d if E_d else 0

                    results[user_email][date] = np.array([E_d, round(L_d, 2), round(FRR_d, 2), round(R_d, 2)])
        
        return results