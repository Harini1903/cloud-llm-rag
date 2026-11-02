
import os
import psycopg

from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing from .env")

    conn = psycopg.connect(DATABASE_URL)

    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, extensions")

    register_vector(conn)

    return conn


def test_connection():

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    current_database(),
                    current_user,
                    version();
            """)

            result = cur.fetchone()

            print("[SUPABASE] Connection successful!")
            print("Database:", result[0])
            print("User:", result[1])

    finally:
        conn.close()



if __name__ == "__main__":
    test_connection()
    
    
