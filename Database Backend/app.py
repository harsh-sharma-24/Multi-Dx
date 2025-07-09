from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Config from .env
app.secret_key = os.getenv("SECRET_KEY")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

# MongoDB setup
mongo = PyMongo(app)

# hashing 
def hash(password: str) -> str:
    hash_result = ""

    for i, char in enumerate(password):
        # Convert character to ASCII, add index, shift by 3, then wrap around
        shifted = (ord(char) + i + 3) % 126
        if shifted < 33:
            shifted += 33  # Keep characters printable

        # Convert back to character
        hash_result += chr(shifted)

    # Add a static salt pattern (optional)
    hash_result = "LC$" + hash_result + "$#SCP"

    return hash_result

# ──────────────────────────────────────────────────────────────
# HOME
@app.route("/")
def home():
    return "LifeScoop Database Backend is Running"

# ──────────────────────────────────────────────────────────────
# SIGNUP — POST /signup
@app.route("/signup", methods=["POST"])
def signup():
    try:
        # Debug print
        print("Request JSON:", request.json)

        data = request.json or {}
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        phoneno = data.get("phoneno", "").strip()
        age = data.get("age")
        gender = data.get("gender", "").strip().lower()

        if not all([name, email, password, phoneno, age, gender]):
            return jsonify({"error": "All fields are required"}), 400

        if mongo.db.data.find_one({"email": email}):
            return jsonify({"error": "Email already registered"}), 400

        password = hash(password)

        user_doc = {
            "name": name,
            "email": email,
            "password": password,
            "phoneno": phoneno,
            "age": int(age),
            "gender": gender,
            "HeartRisk": "NA",
            "MentalHealth": "NA",
            "Diabetes": "NA",
            "Hypertension": "NA",
            "SkinInfection": "NA"
        }

        mongo.db.data.insert_one(user_doc)
        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print("❌ Error during signup:", e)  # This will show the real error
        return jsonify({"error": "Internal server error"}), 500
# ──────────────────────────────────────────────────────────────
# LOGIN — POST /login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = mongo.db.data.find_one({"email": email})
    if not user or user["password"] != hash(password) :
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful"}), 200

# ──────────────────────────────────────────────────────────────
# GET USER — GET /user/<email>
@app.route("/user/<email>", methods=["GET"])
def get_user(email):
    user = mongo.db.data.find_one({"email": email.lower()})
    if not user:
        return jsonify({"error": "User not found"}), 404

    user["_id"] = str(user["_id"])  # convert ObjectId to string
    user["password"] = "🔒"         # hide password
    return jsonify(user), 200

# ──────────────────────────────────────────────────────────────
# UPDATE PREDICTION — PUT /predict/<email>
@app.route("/update_prediction", methods=["POST"])
def update_prediction():
    try:
        data = request.json
        email = data.get("email", "").strip().lower()
        prediction_type = data.get("type", "").strip()  # e.g., "HeartRisk"
        result = data.get("result", "").strip()

        if not all([email, prediction_type, result]):
            return jsonify({"error": "Missing data"}), 400

        if prediction_type not in ["HeartRisk", "MentalHealth", "Diabetes", "Hypertension", "SkinInfection"]:
            return jsonify({"error": "Invalid prediction type"}), 400

        updated = mongo.db.data.update_one(
            {"email": email},
            {"$set": {prediction_type: result}}
        )

        if updated.matched_count == 0:
            return jsonify({"error": "User not found"}), 404

        return jsonify({"message": f"{prediction_type} updated successfully."}), 200

    except Exception as e:
        print("❌ Update prediction error:", e)
        return jsonify({"error": "Internal server error"}), 500

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
