import csv
import sqlite3
from pathlib import Path


DB_PATH = Path("nutrition.db")
CSV_PATH = Path("영양성분_DB") / "merged_data.csv"


def to_float(value):
    if value is None:
        return None

    value = value.strip()
    if not value or value == "-":
        return None

    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError("nutrition.db가 없습니다. 먼저 create_db.py를 실행해주세요.")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"영양성분 CSV가 없습니다: {CSV_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM nutrition_items")

    rows = []
    count = 0

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            food_name = (row.get("식품명") or "").strip()
            if not food_name:
                continue

            rows.append(
                (
                    row.get("식품코드"),
                    food_name,
                    row.get("대표식품명"),
                    row.get("식품대분류명"),
                    row.get("식품중분류명"),
                    row.get("영양성분함량기준량"),
                    to_float(row.get("에너지(kcal)")),
                    to_float(row.get("탄수화물(g)")),
                    to_float(row.get("단백질(g)")),
                    to_float(row.get("지방(g)")),
                    to_float(row.get("당류(g)")),
                    to_float(row.get("나트륨(mg)")),
                    to_float(row.get("카페인(mg)")),
                    row.get("식품중량"),
                    row.get("업체명"),
                )
            )

            if len(rows) >= 1000:
                cur.executemany(
                    """
                    INSERT INTO nutrition_items (
                        food_code,
                        food_name,
                        representative_name,
                        major_category,
                        middle_category,
                        serving_size,
                        calories,
                        carbohydrate,
                        protein,
                        fat,
                        sugar,
                        sodium,
                        caffeine,
                        food_weight,
                        company
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                count += len(rows)
                rows.clear()

        if rows:
            cur.executemany(
                """
                INSERT INTO nutrition_items (
                    food_code,
                    food_name,
                    representative_name,
                    major_category,
                    middle_category,
                    serving_size,
                    calories,
                    carbohydrate,
                    protein,
                    fat,
                    sugar,
                    sodium,
                    caffeine,
                    food_weight,
                    company
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            count += len(rows)

    conn.commit()
    conn.close()

    print(f"영양성분 데이터 import 완료: {count}개")


if __name__ == "__main__":
    main()
