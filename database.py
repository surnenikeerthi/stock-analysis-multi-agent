import sqlite3
def initialize_database():
    conn = sqlite3.connect("portfolio.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            purchsecost REAL NOT NULL
            purchasedate TEXT
        )
    """)

    conn.commit()
    conn.close()