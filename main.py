from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
if not hasattr(DecisionTreeClassifier, "monotonic_cst"):
    DecisionTreeClassifier.monotonic_cst = None
app = Flask(__name__)

# Load the pre-trained model
with open("Hotel_Booking_Cancellation_Prediction_Model.pkl", "rb") as f:
    model = pickle.load(f)


def preprocess_input(form):
    """
    Convert form input into the feature vector expected by the model.
    The following steps mimic the feature engineering done during training:
      - Compute derived features (total_nights, total_guests, occupancy_ratio, price_per_guest, has_special_request)
      - Derive season from reservation_month
      - One-hot encode categorical fields: type of meal, room type, market segment type, and season.

    The feature vector order is assumed to be:
    [number of adults, number of children, number of weekend nights, number of week nights,
     car parking space, lead time, repeated, P-C, P-not-C, average price, special requests,
     reservation_year, reservation_month, total_nights, total_guests, occupancy_ratio, price_per_guest,
     has_special_request,
     type of meal_Meal Plan 2, type of meal_Meal Plan 3, type of meal_Not Selected,
     room type_Room_Type 2, room type_Room_Type 3, room type_Room_Type 4, room type_Room_Type 5,
     room type_Room_Type 6, room type_Room_Type 7,
     market segment type_Complementary, market segment type_Corporate,
     market segment type_Offline, market segment type_Online,
     season_Spring, season_Summer, season_Winter]
    """
    # --- Numeric inputs ---
    number_of_adults = int(form.get("number_of_adults"))
    number_of_children = int(form.get("number_of_children"))
    number_of_weekend_nights = int(form.get("number_of_weekend_nights"))
    number_of_week_nights = int(form.get("number_of_week_nights"))
    car_parking_space = int(form.get("car_parking_space"))
    lead_time = int(form.get("lead_time"))
    repeated = int(form.get("repeated"))
    P_C = int(form.get("P_C"))
    P_not_C = int(form.get("P_not_C"))
    average_price = float(form.get("average_price"))
    special_requests = int(form.get("special_requests"))
    reservation_year = int(form.get("reservation_year"))
    reservation_month = int(form.get("reservation_month"))

    # --- Categorical inputs ---
    type_of_meal = form.get("type_of_meal")  # e.g., "Meal Plan 2", "Meal Plan 3", "Not Selected"
    room_type = form.get("room_type")  # e.g., "Room_Type 2", "Room_Type 3", etc.
    market_segment_type = form.get("market_segment_type")  # e.g., "Complementary", "Corporate", "Offline", "Online"

    # --- Derived features ---
    total_nights = number_of_weekend_nights + number_of_week_nights
    total_guests = number_of_adults + number_of_children
    occupancy_ratio = total_guests / total_nights if total_nights != 0 else 0
    price_per_guest = average_price / total_guests if total_guests != 0 else average_price
    has_special_request = 1 if special_requests > 0 else 0

    # Derive season from reservation_month
    if reservation_month in [12, 1, 2]:
        season = "Winter"
    elif reservation_month in [3, 4, 5]:
        season = "Spring"
    elif reservation_month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Fall"  # Fall is baseline (we will not create a dummy for Fall)

    # --- One-hot encoding for categorical features ---
    # For "type of meal"
    meal_plan_2 = 1 if type_of_meal == "Meal Plan 2" else 0
    meal_plan_3 = 1 if type_of_meal == "Meal Plan 3" else 0
    meal_not_selected = 1 if type_of_meal == "Not Selected" else 0

    # For "room type"
    room_type_2 = 1 if room_type == "Room_Type 2" else 0
    room_type_3 = 1 if room_type == "Room_Type 3" else 0
    room_type_4 = 1 if room_type == "Room_Type 4" else 0
    room_type_5 = 1 if room_type == "Room_Type 5" else 0
    room_type_6 = 1 if room_type == "Room_Type 6" else 0
    room_type_7 = 1 if room_type == "Room_Type 7" else 0

    # For "market segment type"
    market_complementary = 1 if market_segment_type == "Complementary" else 0
    market_corporate = 1 if market_segment_type == "Corporate" else 0
    market_offline = 1 if market_segment_type == "Offline" else 0
    market_online = 1 if market_segment_type == "Online" else 0

    # For "season": we create dummies for Spring, Summer, Winter (Fall is baseline)
    season_spring = 1 if season == "Spring" else 0
    season_summer = 1 if season == "Summer" else 0
    season_winter = 1 if season == "Winter" else 0

    # --- Form the feature vector in the same order as used during model training ---
    features = [
        number_of_adults, number_of_children, number_of_weekend_nights, number_of_week_nights,
        car_parking_space, lead_time, repeated, P_C, P_not_C, average_price, special_requests,
        reservation_year, reservation_month, total_nights, total_guests, occupancy_ratio, price_per_guest,
        has_special_request,
        meal_plan_2, meal_plan_3, meal_not_selected,
        room_type_2, room_type_3, room_type_4, room_type_5, room_type_6, room_type_7,
        market_complementary, market_corporate, market_offline, market_online,
        season_spring, season_summer, season_winter
    ]

    # Convert to numpy array and reshape to (1, number of features)
    features = np.array(features).reshape(1, -1)
    return features


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Retrieve form data and preprocess the input
        input_features = preprocess_input(request.form)
        # Get prediction from the model
        pred = model.predict(input_features)
        # Convert prediction to a human-friendly label (assuming 1 means "Not Canceled" and 0 means "Canceled")
        prediction = "Not Canceled" if pred[0] == 1 else "Canceled"
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
