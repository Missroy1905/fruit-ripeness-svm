import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import joblib
from utils import extract_features

# Define paths
MODEL_PATH = os.path.join('saved_models', 'fruit_ripeness_svm_model.pkl')
SCALER_PATH = os.path.join('saved_models', 'feature_scaler.pkl')

# --- Load the trained model and scaler ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    print("Error: Model or scaler not found. Please run train_model.py first.")
    exit()

# --- Prediction Function ---
def predict_image(file_path):
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[list(model.classes_).index(prediction)]
        return prediction, confidence
    return "Error processing image", 0

# --- GUI Functions ---
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
        
    img = Image.open(file_path)
    img.thumbnail((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk
    
    prediction, confidence = predict_image(file_path)
    result_text = f"Prediction: {prediction.replace('_', ' ')}\nConfidence: {confidence*100:.2f}%"
    result_label.config(text=result_text)

# --- Setup the GUI Window ---
window = tk.Tk()
window.title("Fruit Ripeness Detector")
window.geometry("400x500")
window.configure(bg='#f0f0f0')

main_frame = tk.Frame(window, bg='#f0f0f0')
main_frame.pack(expand=True, fill='both', padx=10, pady=10)

panel = Label(main_frame, bg='white')
panel.pack(pady=10)

upload_btn = Button(main_frame, text="Upload Fruit Image", command=upload_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=10, pady=5)
upload_btn.pack(pady=10)

result_label = Label(main_frame, text="Upload an image to see the prediction", font=("Helvetica", 14, "bold"), bg='#f0f0f0')
result_label.pack(pady=20)

window.mainloop()