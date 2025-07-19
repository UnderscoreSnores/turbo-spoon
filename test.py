import datetime
from data.fetch_data import fetch_price_data

today = datetime.date.today()
seven_days_ago = today - datetime.timedelta(days=7)
df = fetch_price_data("AAPL", seven_days_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
print(df)
