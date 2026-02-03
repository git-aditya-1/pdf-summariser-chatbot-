import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="ragdb",
    user="postgres",
    password="Aditya@2107",
    port="5432"
)

cur = conn.cursor()

cur.execute("""
    INSERT INTO documents (document_name, chunk_index, summary)
    VALUES (%s, %s, %s)
""", ("test.pdf", 1, "This is a test summary"))

conn.commit()

cur.execute("SELECT * FROM documents;")
rows = cur.fetchall()

for row in rows:
    print(row)

cur.close()
conn.close()
