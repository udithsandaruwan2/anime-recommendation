import pandas as pd
from sqlalchemy import create_engine, text

# Database connection details from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
TABLE_NAME = os.getenv('TABLE_NAME')

# CSV file path
CSV_FILE = 'anime.csv'

# Step 1: Connect to the PostgreSQL database
print("Connecting to the database...")
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Step 2: Read the CSV file
print("Reading the CSV file...")
df = pd.read_csv(CSV_FILE)

# Step 3: Create the table
print(f"Creating the table '{TABLE_NAME}'...")
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    anime_id INT PRIMARY KEY,
    name TEXT NOT NULL,
    genre TEXT NOT NULL,
    type TEXT NOT NULL,
    episodes TEXT NOT NULL,
    rating TEXT NOT NULL,
    members TEXT NOT NULL
);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_query))
    print(f"Table '{TABLE_NAME}' created successfully.")

# Step 4: Migrate data to the database
print(f"Migrating data to the table '{TABLE_NAME}'...")
df.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
print(f"Data migrated successfully into '{TABLE_NAME}'.")
