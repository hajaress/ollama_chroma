import sqlite3
import os
import sys

def view_chroma_sqlite(db_path):
    # Validate file existence
    if not os.path.exists(db_path):
        print(f"Error: File '{db_path}' not found.")
        sys.exit(1)

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\n📂 Tables in database:")
        for t in tables:
            print(f" - {t[0]}")

        # Show first 5 rows from each table
        for t in tables:
            table_name = t[0]
            print(f"\n🔍 Preview of table '{table_name}':")
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()

                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                print(" | ".join(col_names))
                for row in rows:
                    print(row)
            except Exception as e:
                print(f"  (Error reading table: {e})")

        conn.close()

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        sys.exit(1)

# Example usage
if __name__ == "__main__":
    
    db_file = r"D:\Gen AI\Ollama\Olama_work\chroma_db_rag_data_ollama\chroma.sqlite3"
    view_chroma_sqlite(db_file)