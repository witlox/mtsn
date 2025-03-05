import json
import os.path
from datetime import datetime

import pandas as pd
import requests

article = 'ChatGPT'
start_date = '20210101'
end_date = '20221231'
granularity = 'daily'

if not os.path.exists('pageviews.json'):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/{granularity}/{start_date}/{end_date}"
    response = requests.get(url)
    with open('pageviews.json', 'w', encoding='utf-8') as f:
        json.dump(response.json(), f)
with open('pageviews.json', 'r') as f:
    data = json.load(f)

records = []
for item in data['items']:
    records.append({
        'date': datetime.strptime(item['timestamp'], '%Y%m%d%H'),
        'kpi_value': item['views']
    })

df = pd.DataFrame(records)
df.to_parquet('wikipedia_pageviews.parquet')
