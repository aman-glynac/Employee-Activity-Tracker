import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from statistics import median
from datetime import datetime

class EmailMetrics:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.global_emails = []
        self.users = set()
        self._load_all_emails()

    def _extract_email_from_filename(self, filename):
        return filename.replace("emails_", "").replace("_at_", "@").replace(".json", "").lower()

    def _load_all_emails(self):
        for filename in os.listdir(self.root_folder):
            if filename.endswith(".json"):
                user_email = self._extract_email_from_filename(filename)
                self.users.add(user_email)

                filepath = os.path.join(self.root_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    emails = json.load(f)

                for email in emails:
                    email["seen_in_file_of"] = user_email
                    self.global_emails.append(email)

    def compute_metrics(self):
        results = defaultdict(dict)

        for user in self.users:
            daily_sent_emails = defaultdict(list)

            for email in self.global_emails:
                if email.get("from", "").lower() == user and "sent_at" in email:
                    date = email["sent_at"][:10]  # YYYY-MM-DD
                    daily_sent_emails[date].append(email)

            for date, emails in daily_sent_emails.items():
                E_d = len(emails)

                # L_d: Average body length
                L_d = sum(len(email.get("body", "")) for email in emails) / E_d if E_d else 0

                # FRR_d: subject starts with "Re:", "Fwd:", or contains "FW:"
                FRR_d = sum(
                    1 for email in emails
                    if any((email.get("subject") if email.get("subject") is not None else "").lower().startswith(prefix) for prefix in ("re:", "fw:", "fwd:"))
                ) / E_d if E_d else 0

                # R_d: number of recipients in "to"
                R_d = sum(
                    len([r.strip() for r in email.get("to", "").split(",") if r.strip()])
                    for email in emails
                ) / E_d if E_d else 0

                results[user][date] = np.array([E_d, round(L_d, 2), round(FRR_d, 2), round(R_d, 2)])
        return results