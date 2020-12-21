import amendements_analysis.settings.base as stg
import os
import json

with open(os.path.join(stg.PROCESSED_DATA_DIR, "file.json")) as f:
    json_files = []
    for line in f:
        json_files.append(json.loads(line))
