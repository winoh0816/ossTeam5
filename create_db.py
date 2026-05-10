import sqlite3
from pathlib import Path


DB_PATH = Path("nutrition.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_file TEXT NOT NULL,
            store_name TEXT,
            purchased_at TEXT,
            total INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            item_name TEXT NOT NULL,
            price INTEGER,
            category TEXT,
            matched_nutrition_id INTEGER,
            FOREIGN KEY (receipt_id) REFERENCES receipts(id),
            FOREIGN KEY (matched_nutrition_id) REFERENCES nutrition_items(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nutrition_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_code TEXT,
            food_name TEXT NOT NULL,
            representative_name TEXT,
            major_category TEXT,
            middle_category TEXT,
            serving_size TEXT,
            calories REAL,
            carbohydrate REAL,
            protein REAL,
            fat REAL,
            sugar REAL,
            sodium REAL,
            caffeine REAL,
            food_weight TEXT,
            company TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_nutrition_food_name
        ON nutrition_items(food_name)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_nutrition_representative_name
        ON nutrition_items(representative_name)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_receipt_items_name
        ON receipt_items(item_name)
        """
    )

    conn.commit()
    conn.close()

    print(f"DB 생성 완료: {DB_PATH}")


if __name__ == "__main__":
    main()
