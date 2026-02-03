import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ragdb",
        user="postgres",
        password=os.getenv("DB_PASSWORD"),
        port="5432"
    )

def insert_document(doc_name, chunk_index, summary):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO documents (document_name, chunk_index, summary)
        VALUES (%s, %s, %s)
    """, (doc_name, chunk_index, summary))

    conn.commit()
    cur.close()
    conn.close()

def fetch_all_summaries():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT summary FROM documents;")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [row[0] for row in rows]
def fetch_summaries_by_document(doc_name):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT summary FROM documents WHERE document_name=%s ORDER BY chunk_index",
        (doc_name,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [r[0] for r in rows]

