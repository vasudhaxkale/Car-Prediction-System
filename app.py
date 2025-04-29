import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
car_dataset = pd.read_csv('Cardetails.csv')  

# Encode categorical variables
fuel_encoder = LabelEncoder()
car_dataset['fuel'] = fuel_encoder.fit_transform(car_dataset['fuel'])

transmission_encoder = LabelEncoder()
car_dataset['transmission'] = transmission_encoder.fit_transform(car_dataset['transmission'])

seller_type_encoder = LabelEncoder()
car_dataset['seller_type'] = seller_type_encoder.fit_transform(car_dataset['seller_type'])

# Define features & target variable
X = car_dataset[['year', 'present_price', 'kms_driven', 'fuel', 'transmission', 'seller_type']]
y = car_dataset['selling_price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model (unscaled)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train SVM Model (scaled)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

# Train KNN Model (scaled)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

# Stacking Model
stacking_model = StackingRegressor(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model)
], final_estimator=LinearRegression())
stacking_model.fit(X_train_scaled, y_train)
stacking_predictions = stacking_model.predict(X_test_scaled)

# Evaluate models
rf_r2 = r2_score(y_test, rf_predictions)
svm_r2 = r2_score(y_test, svm_predictions)
knn_r2 = r2_score(y_test, knn_predictions)
stack_r2 = r2_score(y_test, stacking_predictions)

# Save models
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load models in app
@st.cache_resource
def load_models():
    rf = joblib.load('rf_model.pkl')
    svm = joblib.load('svm_model.pkl')
    knn = joblib.load('knn_model.pkl')
    stack = joblib.load('stacking_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf, svm, knn, stack, scaler

rf_model, svm_model, knn_model, stacking_model, scaler = load_models()

# Streamlit UI
st.title("Car Price Prediction: RF vs SVM vs KNN vs Stacking")

# User Input
year = st.number_input("Enter the car's year", min_value=2000, max_value=2024, value=2015)
present_price = st.number_input("Enter the car's present price (in lakhs)", min_value=0.0, max_value=50.0, value=5.0)
kms_driven = st.number_input("Enter the kilometers driven", min_value=0, max_value=300000, value=50000)
fuel_type = st.selectbox("Select the fuel type", options=fuel_encoder.classes_)
transmission = st.selectbox("Select the transmission type", options=transmission_encoder.classes_)
seller_type = st.selectbox("Select the seller type", options=seller_type_encoder.classes_)

# Encode user inputs
fuel_type_encoded = fuel_encoder.transform([fuel_type])[0]
transmission_encoded = transmission_encoder.transform([transmission])[0]
seller_type_encoded = seller_type_encoder.transform([seller_type])[0]

# Prepare input data
input_data = [[year, present_price, kms_driven, fuel_type_encoded, transmission_encoded, seller_type_encoded]]
input_data_scaled = scaler.transform(input_data)

# Predict and show
if st.button("Predict Price"):
    rf_price = rf_model.predict(input_data)[0]
    svm_price = svm_model.predict(input_data_scaled)[0]
    knn_price = knn_model.predict(input_data_scaled)[0]
    stack_price = stacking_model.predict(input_data_scaled)[0]

    st.success(f"Random Forest Prediction: ₹{rf_price:.2f} lakhs")
    st.success(f"SVM Prediction: ₹{svm_price:.2f} lakhs")
    st.success(f"KNN Prediction: ₹{knn_price:.2f} lakhs")
    st.success(f"Stacking Prediction: ₹{stack_price:.2f} lakhs")

    # Metrics Table
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "SVM", "KNN", "Stacking"],
        "R² Score": [rf_r2, svm_r2, knn_r2, stack_r2],
        "MAE": [
            mean_absolute_error(y_test, rf_predictions),
            mean_absolute_error(y_test, svm_predictions),
            mean_absolute_error(y_test, knn_predictions),
            mean_absolute_error(y_test, stacking_predictions)
        ],
        "MSE": [
            mean_squared_error(y_test, rf_predictions),
            mean_squared_error(y_test, svm_predictions),
            mean_squared_error(y_test, knn_predictions),
            mean_squared_error(y_test, stacking_predictions)
        ]
    })
    st.table(metrics_df)

        # Stripplot + user prediction overlay
    st.subheader("Model Predictions vs Your Prediction")
    predictions_df = pd.DataFrame({
        'Random Forest': rf_predictions,
        'SVM': svm_predictions,
        'KNN': knn_predictions,
        'Stacking': stacking_predictions
    })
    melted_df = predictions_df.melt(var_name='Model', value_name='Predicted Price')

    fig_strip, ax_strip = plt.subplots()
    sns.stripplot(x='Model', y='Predicted Price', data=melted_df, jitter=True, palette='Set2', alpha=0.6, ax=ax_strip)
    
    # Overlay user predictions
    user_preds = [rf_price, svm_price, knn_price, stack_price]
    for i, pred in enumerate(user_preds):
        ax_strip.scatter(i, pred, color='red', s=100, edgecolors='black', label='Your Prediction' if i == 0 else "")
    
    ax_strip.set_title("Individual Predictions vs Your Car's Prediction")
    ax_strip.set_ylabel("Predicted Price (Lakhs)")
    ax_strip.set_xlabel("Model")
    ax_strip.legend()
    st.pyplot(fig_strip)


    # Bar graph of base vs stacking prediction
    st.subheader("Predicted Prices by Model")
    fig_bar, ax_bar = plt.subplots()
    bar_labels = ["Random Forest", "SVM", "KNN", "Stacking"]
    bar_values = user_preds
    sns.barplot(x=bar_labels, y=bar_values, palette="pastel", ax=ax_bar)
    ax_bar.set_title("Predicted Car Price by Model")
    ax_bar.set_ylabel("Price (Lakhs)")
    ax_bar.set_xlabel("Model")
    st.pyplot(fig_bar)

    # Feature Importance (Random Forest)
    st.subheader("Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    features = X.columns
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances, y=features, palette='viridis', ax=ax_imp)
    ax_imp.set_title("Feature Importance from Random Forest Model")
    ax_imp.set_xlabel("Importance Score")
    ax_imp.set_ylabel("Feature")
    st.pyplot(fig_imp)

    # ------------------------- UPDATED PLOT 1 -------------------------
    st.subheader("Actual vs Predicted Car Prices")
    fig_simple_line, ax_simple = plt.subplots(figsize=(10, 5))
    ax_simple.plot(y_test.values[:30], marker='o', label='Actual Price', color='blue')
    ax_simple.plot(stacking_predictions[:30], marker='x', label='Predicted (Stacking)', color='orange')
    ax_simple.set_title("Actual vs Predicted (Stacking Model)")
    ax_simple.set_xlabel("Sample Index")
    ax_simple.set_ylabel("Price (Lakhs)")
    ax_simple.legend()
    st.pyplot(fig_simple_line)

    # ------------------------- UPDATED PLOT 2 -------------------------
    st.subheader("Residuals (Prediction Errors)")
    fig_simple_resid, ax_resid_simple = plt.subplots()
    residuals = y_test.values - stacking_predictions
    sns.histplot(residuals, bins=30, kde=True, color='purple', ax=ax_resid_simple)
    ax_resid_simple.set_title("Distribution of Errors (Stacking Model)")
    ax_resid_simple.set_xlabel("Error (Actual - Predicted)")
    st.pyplot(fig_simple_resid)
