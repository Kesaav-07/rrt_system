from db import get_connection

conn = get_connection()
print("MySQL connected successfully")
conn.close()