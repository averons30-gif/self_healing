import sqlite3

conn = sqlite3.connect('digital_twin.db')
cursor = conn.cursor()

# Count alerts
cursor.execute('SELECT COUNT(*) FROM alerts')
count = cursor.fetchone()[0]
print(f'Alerts in DB: {count}')

# Get recent alerts
cursor.execute('SELECT * FROM alerts ORDER BY created_at DESC LIMIT 5')
rows = cursor.fetchall()
print('Recent alerts:')
for row in rows:
    print(row)

conn.close()