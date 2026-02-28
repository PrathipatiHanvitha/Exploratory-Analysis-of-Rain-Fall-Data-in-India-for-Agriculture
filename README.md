# Rainfall Prediction (RainTomorrow) - Flask Deployment

## Objective
Predict whether it will rain tomorrow using Australian weather dataset (weatherAUS.csv).

## Demo Video - [https://drive.google.com/file/d/1cjMtEvTOHuUCpQjcklJP4pRtT_Uqbc8G/view?usp=drive_link](https://drive.google.com/file/d/1cjMtEvTOHuUCpQjcklJP4pRtT_Uqbc8G/view?usp=drive_link)

## Steps
1. Load dataset
2. Handle missing values
3. Encode categorical values
4. Feature scaling
5. Train models and select best accuracy
6. Evaluate using Accuracy, Confusion Matrix, ROC-AUC
7. Save model + preprocessors as .pkl
8. Build Flask UI for prediction

## Run Training
python train_model.py

## Run Flask App
python app.py

Open: http://127.0.0.1:5000/
