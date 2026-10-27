# Predictive Modeling of Customer Bookings

This repository contains an end-to-end machine learning project focused on predictive modeling for customer bookings. The analysis covers exploratory data analysis (EDA), data cleaning, feature engineering, target encoding, class imbalance handling, and training/evaluating robust machine learning classifiers to predict customer booking completion.

---

## Project Overview

Understanding customer behavior and predicting whether a booking will be completed helps businesses optimize sales channels, resource allocation, and targeted marketing strategies. This project processes a dataset of 50,000 customer bookings to uncover key patterns, evaluate contributing features, and build classification models to predict successful conversions.

---

## Dataset Description

The dataset consists of **50,000 rows** and **14 columns** with **no missing values**. Below is the detailed description of each feature:

* `num_passengers`: Number of passengers travelling.
* `sales_channel`: Sales channel through which the booking was made (e.g., Internet, Mobile).
* `trip_type`: Type of trip (RoundTrip, OneWay, CircleTrip).
* `purchase_lead`: Number of days between the travel date and the booking date.
* `length_of_stay`: Number of days spent at the destination.
* `flight_hour`: Hour of flight departure.
* `flight_day`: Day of the week of flight departure.
* `route`: Origin to destination flight route code.
* `booking_origin`: Country from where the booking was made.
* `wants_extra_baggage`: Binary flag indicating if the customer wanted extra baggage.
* `wants_preferred_seat`: Binary flag indicating if the customer wanted a preferred seat.
* `wants_in_flight_meals`: Binary flag indicating if the customer wanted in-flight meals.
* `flight_duration`: Total duration of the flight (in hours).
* `booking_complete`: Target variable/flag indicating if the customer completed the booking.

---

## Exploratory Data Analysis & Feature Engineering

1. **Data Conversion & Mapping:** 
   * `flight_day` categorical string values (Mon–Sun) were mapped to ordinal numerical values (1–7).
   * Skewed continuous features (`purchase_lead`, `length_of_stay`) underwent logarithmic transformations (`log1p`) to normalize distributions.
2. **Categorical Encoding:**
   * High-cardinality categorical variables (`route` and `booking_origin`) were processed using smoothed target encoding to prevent overfitting and capture conversion probabilities.
   * Categorical variables like `trip_type` and `sales_channel` were one-hot encoded.
3. **Key Insights from EDA:**
   * **Sales Channel:** Internet bookings vastly dominate over mobile bookings, though mobile users exhibit lower conversion rates.
   * **Trip Type:** Round trips constitute the overwhelming majority of the dataset.
   * **Add-ons:** Customers selecting extra baggage and in-flight services show higher booking completion rates.
   * **Target Class Imbalance (`booking_complete`):** Uncompleted bookings (0) significantly outnumber completed bookings (1), exhibiting a class imbalance (~14.96% positive conversion rate).

---

## Predictive Modeling & Model Evaluation

After preparing the features and splitting the dataset into training and testing sets, multiple machine learning models were trained, tuned, and evaluated to handle the class imbalance and maximize predictive performance:

1. **Models Explored:**
   * **Logistic Regression:** Used as a baseline model with balanced class weights.
   * **Random Forest Classifier:** Evaluated with ensemble bagging and balanced class weights.
   * **XGBoost Classifier:** Optimized with gradient boosting and `scale_pos_weight` to address class skewness.
2. **Evaluation Metrics:**
   * **ROC-AUC Score:** Ranging around ~0.79 to ~0.81, demonstrating strong discrimination capability between completed and uncompleted bookings.
   * **Classification Report (Precision, Recall, F1-Score):** Optimized via decision thresholds or class weighting to ensure reliable identification of actual purchasers while managing false positives.

---

## Repository Structure

* `customer_booking.csv`: The primary dataset utilized for analysis.
* `Getting Started.ipynb`: Jupyter notebook containing the complete EDA, visualization code, preprocessing steps, feature engineering, model training, and evaluation.
* `README.md`: Project documentation and guide.

---

## Requirements & Dependencies

To run the notebook and reproduce the analysis, ensure the following Python libraries are installed:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/British-Airways-Data-Science.git
   ```
2. Navigate to the project directory and launch Jupyter Notebook:
   ```bash
   jupyter notebook "Getting Started.ipynb"
   ```
