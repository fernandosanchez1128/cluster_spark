import psycopg2, psycopg2.extras
conn = psycopg2.connect(database='postgres',user='fernando',password='1234', host='localhost')
cur = conn.cursor()
cur.execute("SELECT * FROM clusters")
rows = cur.fetchall()
print rows
