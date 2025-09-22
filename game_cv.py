import cv2
import time
from gesture_predictor import predict_gesture
from ai_logic import SmartAI
from game_logic import get_result

ai = SmartAI()
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional but helps speed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (320, 240))
    roi = frame[100:300, 100:300]  # Crop hand region
    gesture = predict_gesture(roi)
        # Draw ROI box for hand placement
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

# Show predicted gesture on screen
    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if gesture in ['rock', 'paper', 'scissors']:
        ai.update(gesture)
        ai_move = ai.predict()
        result = get_result(gesture, ai_move)

        # Print to terminal
        print(f"You: {gesture} | AI: {ai_move}")
        print(f"Result: {result}")
        print("-" * 40)

        # Overlay result on webcam frame
        cv2.putText(frame, f"You: {gesture} | AI: {ai_move} → {result}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        time.sleep(0.3)  # Slight delay for readability

    cv2.imshow('Rock Paper Scissors AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
