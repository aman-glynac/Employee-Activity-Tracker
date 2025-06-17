import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from statistics import median
from datetime import datetime

class DropboxMetrics:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.events = []
        self.users = set()
        self._load_all_events()

    def _load_all_events(self):
        for dirpath, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith(".json") and "state" not in filename:
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        try:
                            events = json.load(f)
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON

                    for event in events:
                        actor_email = (event.get("actor_email", "") if event.get("actor_email") is not None else "").lower()
                        if actor_email:
                            self.users.add(actor_email)
                            self.events.append(event)

    def compute_metrics(self):
        results = defaultdict(lambda: defaultdict(lambda: np.array([0, 0])))  # [DA_d, DE_d]
        
        for event in self.events:
            user = (event.get("actor_email", "") if event.get("actor_email") is not None else "").lower()
            event_type = event.get("event_type")
            timestamp = event.get("timestamp", "")
            date = timestamp[:10]  # YYYY-MM-DD

            if not user or not date:
                continue

            if event_type == "file_add":
                results[user][date][0] += 1  # DA_d
            elif event_type == "file_edit":
                results[user][date][1] += 1  # DE_d

        return results