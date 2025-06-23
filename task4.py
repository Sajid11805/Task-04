import cv2
import numpy as np
from deepface import DeepFace
import pygame
import random
import time
import os
from threading import Thread

# Initialize pygame mixer for audio
pygame.mixer.init()

# Define emotion-to-song mapping (multiple songs per emotion for randomization)
emotion_songs = {
    'happy': ['happy1.mp3', 'happy2.mp3'],
    'sad': ['sad1.mp3', 'sad2.mp3'],
    'angry': ['angry1.mp3', 'angry2.mp3'],
    'neutral': ['neutral1.mp3', 'neutral2.mp3']
}

# Global variables
current_emotion = None
current_song = None
is_playing = False
last_emotion_change = time.time()
MIN_EMOTION_DURATION = 5  # Seconds before changing song

def play_song(emotion):
    global current_song, is_playing, current_emotion
    if is_playing and current_emotion == emotion:
        return
    
    # Stop current song if playing
    if is_playing:
        pygame.mixer.music.stop()
    
    # Select random song for the emotion
    song_file = random.choice(emotion_songs.get(emotion, emotion_songs['neutral']))
    try:
        pygame.mixer.music.load(song_file)
        pygame.mixer.music.play()
        is_playing = True
        current_song = song_file
        current_emotion = emotion
    except Exception as e:
        print(f"Error playing {song_file}: {e}")
        is_playing = False

def audio_thread(emotion):
    play_song(emotion)

def main():
    global last_emotion_change
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB for DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Analyze emotion
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'].lower()

            # Check if emotion has persisted long enough
            if emotion != current_emotion and (time.time() - last_emotion_change) > MIN_EMOTION_DURATION:
                # Start audio in a separate thread
                Thread(target=audio_thread, args=(emotion,), daemon=True).start()
                last_emotion_change = time.time()

            # Display emotion on frame
            cv2.putText(frame, f"Emotion: {emotion.capitalize()}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error in emotion detection: {e}")
            cv2.putText(frame, "Emotion: Detecting...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Emotion Detection', frame)

        # Exit on 'Q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()