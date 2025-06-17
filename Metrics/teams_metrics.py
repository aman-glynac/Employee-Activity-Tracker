import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
from statistics import median
from datetime import datetime

class TeamsMetrics:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.data = defaultdict(lambda: defaultdict(list))  # user -> date -> list of messages
        self._load_all_jsons()

    def _load_all_jsons(self):
        for filename in os.listdir(self.root_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(self.root_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = json.load(f)

                for msg in messages:
                    sender = (msg.get("chat_from", "") if msg.get("chat_from", "") is not None else "").strip()
                    timestamp = msg.get("timestamp", "")
                    channel = msg.get("channel", "")
                    message = msg.get("message", "")

                    if not sender or not timestamp:
                        continue

                    try:
                        date = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()
                    except Exception:
                        continue

                    self.data[sender][date].append({
                        "message": message,
                        "channel": channel
                    })

    def compute_metrics(self):
        results = defaultdict(dict)

        for user, daily_messages in self.data.items():
            for date, messages in daily_messages.items():
                T_d = len(messages)

                # Median length of messages
                lengths = [len(m["message"]) for m in messages if m.get("message")]
                ML_d = median(lengths) if lengths else 0

                # Total unique channels
                channels = {m["channel"] for m in messages if m.get("channel")}
                TC_d = len(channels)

                results[user][date] = np.array([T_d, round(ML_d, 2), round(TC_d, 2)])

        return results