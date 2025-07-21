# moodboard.py (updated to fetch from MySQL)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from pandas.api.types import CategoricalDtype

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Dshr!2022",
    database="mh_chatbot"
)
query = "SELECT timestamp AS date_time, sentiment FROM sentiments"
df = pd.read_sql(query, conn)
conn.close()

# Clean and process
df['sentiment'] = df['sentiment'].str.strip().str.title()
df['sentiment'] = df['sentiment'].replace({'Nuetral': 'Neutral'})

sentiment_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
df['sentiment'] = df['sentiment'].astype(CategoricalDtype(categories=sentiment_order, ordered=True))

df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
df['month_year'] = df['date_time'].dt.to_period('M').astype(str)

# Plot
sns.set(style="whitegrid")
g = sns.catplot(
    data=df,
    x='sentiment',
    kind='count',
    col='month_year',
    col_wrap=3,
    order=sentiment_order,
    palette='pastel',
    height=4,
    aspect=1
)

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Mood Distribution')
plt.tight_layout()
plt.savefig("monthly_moodboard.png")
print("✅ Saved moodboard as 'monthly_moodboard.png'")
