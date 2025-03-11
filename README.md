# 🏨 Hotel Booking Cancellation Prediction

## Overview  
Hotel Booking Cancellation Prediction is a comprehensive project that uses historical booking data to predict whether a hotel booking will be canceled. This project covers the full machine learning lifecycle, including data preprocessing, model training, web application development, and deployment.

---

## 🔑 Key Features

- **Advanced Feature Engineering**  
  - 🛠️ **Data Cleaning & Transformation:** Raw data is cleaned and transformed.  
  - 🔢 **Feature Engineering:** Derived features such as total nights, occupancy ratio, and price per guest.  
  - 🔄 **Encoding:** One-hot encoding for categorical variables like Meal Type, Room Type, Market Segment, and Season.

- **Robust Model Training**  
  - 🏋️ Evaluated multiple classifiers (e.g., Logistic Regression, Random Forest,Decision Tree , Gradient Boosting,KNeighbors,SVM,AdaBoost,MLP,Gaussian NB).  
  - 🔍 **Model Evaluation:** Accuracy, precision, recall, and ROC-AUC are measured to select the best model.  
  - ⚖️ **Class Balancing:** Techniques like SMOTE are employed to balance classes.

- **Interactive Web Application**  
  - 🌐 **User Interface:** A Flask-based web app with a responsive and user-friendly interface.  
  - ⏱️ **Real-Time Prediction:** Users can input booking details and receive immediate cancellation predictions.

- **Deployment Ready**  
  - 🐳 **Containerized:** The project is packaged using Docker.  
  - ☁️ **Cloud Deployment:** Configured for deployment on platforms such as Vercel or Hugging Face Spaces.
  ![WhatsApp Image 2025-02-24 at 7 06 07 PM](https://github.com/user-attachments/assets/320bf441-262d-4d5b-a5ec-67d97a4ac26a)
  ![WhatsApp Image 2025-02-24 at 7 08 28 PM](https://github.com/user-attachments/assets/bc47038a-04c9-43c1-b86d-3e2121fcaabd)
---

## 📂 Project Structure

/your_project_folder

├── app.py # Flask application 

├── main.py # Alternative entry-point (if needed) 

├── Hotel_Booking_Cancellation_Prediction_Model.pkl # Trained model file 

├── requirements.txt # Python dependencies 

├── 1-vercel.json # Vercel configuration (if used) 

├── Dockerfile # Dockerfile for containerized deployment 

├── static/
          
           └── Hotel_photo_cleanup.jpg # Background image and other static assets 
|

└── templates/
        
           └── index.html # HTML template for the web interface


---

## **🚀 Installation**

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/END-TO-END-Hotel-Booking-Cancelation-Prediction.git
cd END-TO-END-Hotel-Booking-Cancelation-Prediction

2. Create and Activate a Virtual Environment

    On Windows:

python -m venv venv
venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt
```

## **🖥️ Usage**
Running the Flask App Locally

    Start the Application:

    python app.py

    Access the App:
    Open your browser and navigate to http://127.0.---

    Input Booking Details:
    Use the web interface to enter booking features. The app processes your input, applies feature engineering, and returns a prediction ("Not Canceled" or "Canceled").

## **🧠 Model Details & Feature Engineering**
Input Features

Basic Features:
        Number of Adults, Number of Children
        Number of Weekend Nights, Number of Week Nights
        Car Parking Space, Lead Time, Repeated Booking, etc.

Derived Features:
        Total Nights, Total Guests, Occupancy Ratio, Price per Guest, Has Special Request

Categorical Features:
        Meal Type, Room Type, Market Segment, Season
        (Season is one-hot encoded for Spring, Summer, Winter; Fall is used as a baseline)

Preprocessing

Functionality:
    The preprocess_input() function converts form data, computes derived features, and performs one-hot encoding to match the training data.

Model Training

Approach:
Multiple classifiers were evaluated, with Random Forest emerging as the best performer (≈90% accuracy). SMOTE was used for class balancing and hyperparameter tuning for optimal performance.

## **🚢 Deployment**
#### Vercel

The project uses vercel.json for deploying the Flask app to Vercel.

#### Hugging Face Spaces (via Docker)

The project is containerized using Docker (see Dockerfile) and is ready for deployment on cloud platforms such as Hugging Face Spaces.

## 🛠️ Technologies Used

#### Languages & Frameworks:
- Python, Flask

#### Libraries:
- scikit-learn, Pandas, NumPy,matplotlib,scipy,seaborn,imblearn.


#### Deployment Tools:
- Docker, Vercel, Hugging Face Spaces

## 🤔 Why This Project?

Full-Cycle ML Development:
    From data preprocessing and model training to deployment, this project covers every step.

Real-World Impact:
    Helps hotels optimize operations by predicting cancellations, potentially saving costs and improving service.

Modern & Intuitive Interface:
    An easy-to-use Flask web application ensures that end-users have a seamless experience.

Scalable & Cloud-Ready:
    Designed for containerized deployment, making it easy to scale and deploy on cloud platforms.

Feel free to modify any section as needed. Enjoy building and deploying your project! 🚀
