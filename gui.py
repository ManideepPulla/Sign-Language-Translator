import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import Frame
from PIL import Image, ImageTk

# Load the model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'
}

# Initialize text-to-speech engine
engine = pyttsx3.init()
character = ""
sentence = ""  # Variable to store the sentence

# Function to process video frame and update the Tkinter label
def process_frame():
    global character, sentence

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        character = predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.putText(frame, sentence, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert the frame to an image that Tkinter can display
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next frame processing
    video_label.after(10, process_frame)

# Functions for button commands
def quit_program():
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()

def add_character():
    global sentence, character
    sentence += character
    update_sentence_label()

def clear_sentence():
    global sentence
    sentence = ""
    update_sentence_label()

def speak_sentence():
    global sentence
    engine.say(sentence)
    engine.runAndWait()
    sentence = ""
    update_sentence_label()

def update_sentence_label():
    sentence_label.config(text=sentence)

# Function to create rounded buttons
def create_rounded_button(master, text, command, bg, fg):
    return tk.Button(master, text=text, command=command, bg=bg, fg=fg, font=("Helvetica", 12),
                     relief="flat", borderwidth=0, highlightthickness=0,
                     padx=20, pady=10, activebackground=bg, activeforeground=fg)

# Initialize Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.configure(bg="#f0f0f0")  # Set background color
root.attributes('-fullscreen', True)  # Set to full screen

# Create video frame
video_frame = Frame(root, width=640, height=480, bg="#f0f0f0")
video_frame.pack()
video_label = tk.Label(video_frame)
video_label.pack()

# Create buttons with rounded corners
btn_frame = Frame(root, bg="#f0f0f0")
btn_frame.pack()

quit_btn = create_rounded_button(btn_frame, "Quit", quit_program, bg="#ff4d4d", fg="white")
quit_btn.pack(side="left", padx=5, pady=10)

add_btn = create_rounded_button(btn_frame, "Add Character", add_character, bg="#4CAF50", fg="white")
add_btn.pack(side="left", padx=5, pady=10)

clear_btn = create_rounded_button(btn_frame, "Clear Sentence", clear_sentence, bg="#ffcc00", fg="black")
clear_btn.pack(side="left", padx=5, pady=10)

speak_btn = create_rounded_button(btn_frame, "Speak 🗣️", speak_sentence, bg="#008CBA", fg="white")
speak_btn.pack(side="left", padx=5, pady=10)

# Create sentence label
sentence_label = tk.Label(root, text=sentence, font=("Helvetica", 16), bg="#f0f0f0", fg="black")
sentence_label.pack(pady=10)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Start processing frames
process_frame()

# Run the Tkinter event loop
root.mainloop()
