import pandas as pd
from db import get_connection

conn = get_connection()
cursor = conn.cursor()

# ---------- users.csv ----------
users = pd.read_csv("data/users.csv")
for _, row in users.iterrows():
    cursor.execute("""
        INSERT IGNORE INTO users
        (username, salt, password_hash, role, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        row["username"],
        row["salt"],
        row["password_hash"],
        row["role"],
        row["created_at"]
    ))

# ---------- live_future_records.csv ----------
live = pd.read_csv("data/live_future_records.csv")
for _, row in live.iterrows():
    cursor.execute("""
        INSERT IGNORE INTO live_future_records
        (patient_id, name, age, ward, block, diagnosis,
         respiratory_rate, spo2, heart_rate, systolic_bp,
         temperature, avpu, avpu_encoded, current_rrt_score,
         rrt_category, predicted_rrt_4hr, predicted_rrt_8hr,
         last_recorded_at, sequence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

# ---------- vital_history.csv ----------
history = pd.read_csv("data/vital_history.csv")
for _, row in history.iterrows():
    cursor.execute("""
        INSERT INTO vital_history
        (patient_id, recorded_at, sequence, heart_rate,
         respiratory_rate, spo2, systolic_bp, temperature,
         avpu, avpu_encoded, current_rrt_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
cursor.close()
conn.close()

print("CSV data migrated to MySQL successfully")