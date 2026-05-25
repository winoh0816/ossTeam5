import json
import os
import sqlite3
from pathlib import Path

from flask import Flask, redirect, render_template, request, send_from_directory, session, url_for
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted
import google.generativeai as genai
import PIL.Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from llm import analyze_receipt_with_llm
from ocr_0510 import PROMPT, extract_json, save_receipt_result


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nutrition.db"
PAGE_DIR = BASE_DIR / "홈페이지"
UPLOAD_DIR = BASE_DIR / "uploads"

app = Flask(__name__, template_folder=str(PAGE_DIR))
app.secret_key = os.getenv("APP_SECRET_KEY", "dev-secret-key-change-later")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def current_user_id():
    return session.get("user_id")


def require_login():
    if not current_user_id():
        return redirect(url_for("login"))
    return None


def money(value):
    return f"{int(value or 0):,}원"

def init_db():

    conn = get_db()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS health_info (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id INTEGER NOT NULL,

        gender TEXT,
        age INTEGER,
        height INTEGER,
        weight INTEGER,
        weight_goal INTEGER,
        bmi REAL,

        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    columns = [
        row["name"]
        for row in conn.execute("PRAGMA table_info(health_info)").fetchall()
    ]
    if "weight_goal" not in columns:
        conn.execute("ALTER TABLE health_info ADD COLUMN weight_goal INTEGER")

    conn.commit()
    conn.close()


init_db()


def demo_dashboard():
    return {
        "is_demo": True,
        "total_spending": 128400,
        "total_spending_text": "128,400원",

        "food_spending": 74900,
        "food_spending_text": "74,900원",
        "food_ratio": 58,

        "category_rows": [
            {"category": "식사", "amount": 48000},
            {"category": "카페", "amount": 33500},
            {"category": "편의점", "amount": 24900},
            {"category": "생활용품", "amount": 14000},
        ],

        "max_category": 48000,

        "top_items": [
            {"item_name": "아이스아메리카노", "count": 7, "category": "카페"},
            {"item_name": "김밥", "count": 4, "category": "식사"},
            {"item_name": "라면", "count": 3, "category": "편의점"},
            {"item_name": "도시락", "count": 2, "category": "식사"},
        ],

        "latest_receipts": [
            {"purchased_at": "05.10", "store_name": "카페", "items": "아이스아메리카노", "total": 4500},
            {"purchased_at": "05.09", "store_name": "편의점", "items": "김밥, 라면", "total": 6800},
            {"purchased_at": "05.08", "store_name": "마트", "items": "우유, 샐러드", "total": 9700},
        ],

        "avg_calories": 620,
        "avg_sugar": 18.2,
        "avg_sodium": 1240
    }


def get_dashboard(user_id):
    conn = get_db()

    total_spending = conn.execute(
        """
        SELECT COALESCE(SUM(price), 0)
        FROM receipt_items
        WHERE receipt_id IN (SELECT id FROM receipts WHERE user_id = ?)
        """,
        (user_id,),
    ).fetchone()[0]

    food_spending = conn.execute(
        """
        SELECT COALESCE(SUM(price), 0)
        FROM receipt_items
        WHERE receipt_id IN (SELECT id FROM receipts WHERE user_id = ?)
          AND category IN ('카페', '편의점', '식사', '마트')
        """,
        (user_id,),
    ).fetchone()[0]

    category_rows = conn.execute(
        """
        SELECT category, COALESCE(SUM(price), 0) AS amount
        FROM receipt_items
        WHERE receipt_id IN (SELECT id FROM receipts WHERE user_id = ?)
        GROUP BY category
        ORDER BY amount DESC
        LIMIT 6
        """,
        (user_id,),
    ).fetchall()

    top_items = conn.execute(
        """
        SELECT item_name, category, COUNT(*) AS count, COALESCE(SUM(price), 0) AS amount
        FROM receipt_items
        WHERE receipt_id IN (SELECT id FROM receipts WHERE user_id = ?)
        GROUP BY item_name, category
        ORDER BY count DESC, amount DESC
        LIMIT 5
        """,
        (user_id,),
    ).fetchall()

    latest_receipts = conn.execute(
        """
        SELECT
            r.id,
            r.purchased_at,
            r.store_name,
            r.total,
            GROUP_CONCAT(ri.item_name, ', ') AS items
        FROM receipts r
        LEFT JOIN receipt_items ri ON ri.receipt_id = r.id
        WHERE r.user_id = ?
        GROUP BY r.id
        ORDER BY r.id DESC
        LIMIT 6
        """,
        (user_id,),
    ).fetchall()

    nutrition = conn.execute(
        """
        SELECT
            AVG(n.calories) AS calories,
            AVG(n.sugar) AS sugar,
            AVG(n.sodium) AS sodium,
            AVG(n.carbohydrate) AS carbohydrate,
            AVG(n.protein) AS protein,
            AVG(n.fat) AS fat
        FROM receipt_items ri
        JOIN receipts r ON r.id = ri.receipt_id
        JOIN nutrition_items n ON n.id = ri.matched_nutrition_id
        WHERE r.user_id = ?
        """,
        (user_id,),
    ).fetchone()

    conn.close()

    max_category = max([row["amount"] for row in category_rows], default=0)
    return {
        "is_demo": False,
        "total_spending": total_spending,
        "total_spending_text": money(total_spending),
        "food_spending": food_spending,
        "food_spending_text": money(food_spending),
        "food_ratio": round((food_spending / total_spending) * 100) if total_spending else 0,
        "category_rows": category_rows,
        "max_category": max_category,
        "top_items": top_items,
        "latest_receipts": latest_receipts,
        "nutrition": nutrition,
        "avg_calories": round(nutrition["calories"] or 0),
        "avg_sugar": round(nutrition["sugar"] or 0, 1),
        "avg_sodium": round(nutrition["sodium"] or 0),
    }


@app.route("/homepage.css")
def css():
    return send_from_directory(PAGE_DIR, "homepage.css")


@app.route("/")
def main():
    user_id = current_user_id()
    data = get_dashboard(user_id) if user_id else demo_dashboard()
    return render_template("main.html", data=data, user_id=user_id)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    result = None
    llm_result = None
    llm_error = None
    error = None
    receipt_id = None

    if request.method == "POST":
        file = request.files.get("user_image")
        if not file or not file.filename:
            error = "업로드할 영수증 이미지를 선택해주세요."
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                error = "GEMINI_API_KEY 환경변수가 없습니다."
            else:
                user_upload_dir = UPLOAD_DIR / str(current_user_id())
                user_upload_dir.mkdir(parents=True, exist_ok=True)
                filename = secure_filename(file.filename)
                image_path = user_upload_dir / filename
                file.save(image_path)

                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    img = PIL.Image.open(image_path)
                    response = model.generate_content([PROMPT, img])
                    result = extract_json(response.text)

                    try:
                        llm_result = analyze_receipt_with_llm(result)
                    except Exception as exc:
                        llm_error = f"AI 소비/영양 분석은 실패했습니다: {exc}"

                    conn = get_db()
                    receipt_id = save_receipt_result(conn, image_path, result, current_user_id())
                    conn.close()
                except ResourceExhausted:
                    error = "Gemini API 사용량 또는 분당 요청 제한을 초과했습니다."
                except (GoogleAPIError, OSError, json.JSONDecodeError, ValueError) as exc:
                    error = f"영수증 분석에 실패했습니다: {exc}"

    return render_template(
        "photograph.html",
        result=result,
        llm_result=llm_result,
        llm_error=llm_error,
        error=error,
        receipt_id=receipt_id,
    )


@app.route("/analysis")
def analysis():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    data = get_dashboard(current_user_id())
    return render_template("calculator.html", data=data)


@app.route("/chat")
def chat():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    data = get_dashboard(current_user_id())
    return render_template("chat.html", data=data)

#가계부
@app.route("/household")
def household():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    data = get_dashboard(current_user_id())
    return render_template("household.html", data=data)

#신체정보
@app.route("/health")
def health():

    # 로그인 체크
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    # 기존 데이터 조회
    health = conn.execute(
        """
        SELECT *
        FROM health_info
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    conn.close()

    # 이미 입력한 경우 health_2 이동
    if health:
        return redirect(url_for("health_2"))

    return render_template(
        "health.html",
        health=health
    )
#신체정보 저장
@app.route("/health/save", methods=["POST"])
def save_health():

    # 로그인 체크
    if "user_id" not in session:
        return redirect(url_for("login"))

    gender = request.form.get("gender")

    age = int(request.form.get("age"))
    height = int(request.form.get("height"))
    weight = int(request.form.get("weight"))
    weight_goal = int(request.form.get("weight_goal"))
    # 입력 제한
    if not (1 <= age <= 100):
        return "나이는 1~100만 가능합니다."

    if not (100 <= height <= 250):
        return "키는 100~250cm만 가능합니다."

    if not (20 <= weight <= 200):
        return "몸무게는 20~200kg만 가능합니다."

    if not (20 <= weight_goal <= 200):
        return "목표 몸무게는 20~200kg만 가능합니다."
    

    # BMI 계산
    bmi = round(weight / ((height / 100) ** 2), 1)

    conn = get_db()

    # 기존 데이터 확인
    existing = conn.execute(
        """
        SELECT id
        FROM health_info
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    # 이미 있으면 UPDATE
    if existing:

        conn.execute(
            """
            UPDATE health_info
            SET
                gender = ?,
                age = ?,
                height = ?,
                weight = ?,
                weight_goal = ?,
                bmi = ?
            WHERE user_id = ?
            """,
            (
                gender,
                age,
                height,
                weight,
                weight_goal,
                bmi,
                session["user_id"]
            )
        )

    # 없으면 INSERT
    else:

        conn.execute(
            """
            INSERT INTO health_info
            (
                user_id,
                gender,
                age,
                height,
                weight,
                weight_goal,
                bmi
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                gender,
                age,
                height,
                weight,
                weight_goal,
                bmi
            )
        )

    conn.commit()
    conn.close()

    # 저장 후 이동
    return redirect(url_for("health_2"))

#저장된 신체정보 표기 페이지
@app.route("/health_2")
def health_2():

    # 로그인 체크
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    # 저장된 데이터 조회
    health = conn.execute(
        """
        SELECT
            gender,
            age,
            height,
            weight,
            weight_goal,
            bmi
        FROM health_info
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    conn.close()

    # 데이터 없으면 입력 페이지로
    if not health:
        return redirect(url_for("health"))

    return render_template(
        "health_2.html",
        health=health
    )

#신체정보 수정
@app.route("/health/edit")
def edit_health():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    health = conn.execute(
        """
        SELECT *
        FROM health_info
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    conn.close()

    if not health:
        return redirect(url_for("health"))

    # 기존 health.html 재사용
    return render_template(
        "health.html",
        health=health
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["email"] = user["email"]
            return redirect(url_for("main"))

        error = "이메일 또는 비밀번호가 올바르지 않습니다."

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            error = "이메일과 비밀번호를 모두 입력해주세요."
        else:
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (email, generate_password_hash(password, method='pbkdf2:sha256')),  #만약 작동이 안될경우 , method='pbkdf2:sha256'부분 삭제
                )
                conn.commit()
                session["user_id"] = cur.lastrowid
                session["email"] = email
                conn.close()
                return redirect(url_for("health")) #처음로그인할시 메인메뉴가아닌 신체정보 페이지로 이동
            except sqlite3.IntegrityError:
                error = "이미 가입된 이메일입니다."

    return render_template("new_user.html", error=error)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    message = None
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()

        if not email:
            error = "가입한 이메일을 입력해주세요."
        else:
            conn = get_db()
            user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            conn.close()

            if user:
                message = "가입된 이메일입니다. 현재 버전에서는 계정 설정에서 비밀번호를 변경해주세요."
            else:
                error = "가입된 이메일을 찾을 수 없습니다."

    return render_template("forgot_password.html", message=message, error=error)


@app.route("/account")
def account():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    return render_template("account.html", email=session.get("email"), message=None, error=None)


@app.route("/account/password", methods=["POST"])
def change_password():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not current_password or not new_password or not confirm_password:
        return render_template(
            "account.html",
            email=session.get("email"),
            message=None,
            error="비밀번호 입력칸을 모두 채워주세요.",
        )

    if new_password != confirm_password:
        return render_template(
            "account.html",
            email=session.get("email"),
            message=None,
            error="새 비밀번호와 확인 비밀번호가 다릅니다.",
        )

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (current_user_id(),)).fetchone()

    if not user or not check_password_hash(user["password_hash"], current_password):
        conn.close()
        return render_template(
            "account.html",
            email=session.get("email"),
            message=None,
            error="현재 비밀번호가 올바르지 않습니다.",
        )

    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(new_password, method='pbkdf2:sha256'), current_user_id()),  #만약 작동이 안될경우 , method='pbkdf2:sha256'부분 삭제
    )
    conn.commit()
    conn.close()

    return render_template(
        "account.html",
        email=session.get("email"),
        message="비밀번호가 변경되었습니다.",
        error=None,
    )


@app.route("/account/delete", methods=["POST"])
def delete_account():
    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    password = request.form.get("delete_password", "")

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (current_user_id(),)).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        conn.close()
        return render_template(
            "account.html",
            email=session.get("email"),
            message=None,
            error="회원탈퇴를 하려면 현재 비밀번호를 정확히 입력해주세요.",
        )

    user_id = current_user_id()
    conn.execute(
        "DELETE FROM receipt_items WHERE receipt_id IN (SELECT id FROM receipts WHERE user_id = ?)",
        (user_id,),
    )
    conn.execute("DELETE FROM receipts WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    session.clear()
    return redirect(url_for("main"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
