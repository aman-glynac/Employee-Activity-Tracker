import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from statistics import median
from datetime import datetime

class DocumentMetrics:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.documents = []
        self.users = set()
        self._load_all_documents()

    def _extract_user_from_filename(self, filename):
        return filename.replace("docs_", "").replace("_at_", "@").replace(".json", "").lower()

    def _load_all_documents(self):
        for filename in os.listdir(self.root_folder):
            if filename.endswith(".json"):
                user_email = self._extract_user_from_filename(filename)
                self.users.add(user_email)

                filepath = os.path.join(self.root_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    docs = json.load(f)

                for doc in docs:
                    doc["seen_in_file_of"] = user_email
                    self.documents.append(doc)

    def compute_metrics(self):
        results = defaultdict(lambda: defaultdict(lambda: np.array([0, 0, 0, 0])))  # [Accessed, Created, Edited, Shared]

        for doc in self.documents:
            user = doc.get("userPrincipalName", "").lower()
            extracted_date = doc.get("extractedAt", "")[:10]

            # Document Accessed: assuming any document listed was accessed on the extracted day
            results[user][extracted_date][0] += 1

            # Document Created: use the 'lastModified' timestamp as a proxy if it's the earliest possible activity
            if "lastModified" in doc:
                created_date = doc["lastModified"][:10]
                results[user][created_date][1] += 1

            # Activities (edit, share, etc.)
            for activity in doc.get("activities", []):
                activity_type = activity.get("actionType")
                activity_date = activity.get("timestamp", "")[:10]

                if activity_type == "edit":
                    results[user][activity_date][2] += 1
                elif activity_type == "share":
                    results[user][activity_date][3] += 1

        return results