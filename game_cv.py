import cv2
from gesture_predictor import predict_gesture
from ai_logic import SmartAI
from game_logic import get_result

ai=SmartAI()
cap=cv2.VideoCapture(0)

while True:
    ret ,frame=cap.read()
    if not ret:
        break

    gesture=predict_gesture(frame)
    if gesture in ['rock','paper','scissors']:
        ai.update(gesture)
        ai_move=ai.predict
        result=get_result(gesture,ai_move)
        cv2.putText(frame,f"You: {gesture} | AI: {ai_move} → {result}",
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Rock Paper Scissors AI',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()