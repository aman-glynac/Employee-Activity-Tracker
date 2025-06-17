import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from statistics import median
from datetime import datetime

class CalendarMetrics:
    def __init__(self, directory: str):
        self.directory = directory

    def _parse_json_files(self):
        all_user_events = defaultdict(list)
        for file_name in os.listdir(self.directory):
            if file_name.startswith("calendar_") and file_name.endswith(".json"):
                email = file_name[len("calendar_"):-len(".json")]
                file_path = os.path.join(self.directory, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            all_user_events[email].append(data)
                        elif isinstance(data, list):
                            all_user_events[email].extend(data)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON in {file_name}")
        return all_user_events

    def compute_metrics(self):
        all_user_events = self._parse_json_files()
        metrics_by_user_and_day = defaultdict(dict)

        for email, events in all_user_events.items():
            daily_events = defaultdict(list)

            for event in events:
                try:
                    start_time = datetime.fromisoformat(event["Start"].replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(event["End"].replace("Z", "+00:00"))
                    date_key = start_time.date()
                    event["start_dt"] = start_time
                    event["end_dt"] = end_time
                    daily_events[date_key].append(event)
                except Exception as e:
                    print(f"Error parsing event time: {e}")
                    continue

            for day, day_events in daily_events.items():
                total_meetings = len(day_events)
                total_duration_minutes = sum((e["end_dt"] - e["start_dt"]).total_seconds() / 60 for e in day_events)

                cancelled = sum("canceled" in e["Title"].lower() for e in day_events)
                virtual = sum(e.get("Virtual", True) for e in day_events)
                inperson = sum(not e.get("Virtual", True) for e in day_events)

                # BTB detection
                sorted_events = sorted(day_events, key=lambda e: e["start_dt"])
                btb_count = 0
                for i in range(1, len(sorted_events)):
                    gap = (sorted_events[i]["start_dt"] - sorted_events[i - 1]["end_dt"]).total_seconds() / 60
                    if 0 <= gap < 30:
                        btb_count += 1

                # Metrics: [Meeting Count, Total Duration, Cancellation Rate, Virtual/In-person Ratio, BTB Ratio]
                metrics = np.array([
                    total_meetings,
                    total_duration_minutes,
                    cancelled / total_meetings if total_meetings else cancelled,
                    virtual / inperson if inperson else virtual,
                    btb_count / total_meetings if total_meetings else 0
                ])
                metrics_by_user_and_day[email][day.isoformat()] = metrics

        return metrics_by_user_and_day