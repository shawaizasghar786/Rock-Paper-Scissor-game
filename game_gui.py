import tkinter as tk
from ai_logic import SmartAI
from utils import get_result

ai=SmartAI

def play(player_move):
    ai.update(player_move)
    ai_move=ai.predict()
    result=get_result(player_move,ai_move)
    result_label.config(text=f"You: {player_move} | AI: {ai_move} → {result}")
    
    root=tk.Tk()
    root.title("Rock Paper Scissors AI")
    
    result_Label=tk.Label(root,text="Make your move",font=("Arial,16"))
    result_Label.pack(pady=20)
    for move in ['rock','paper','scissors']:
        btn=tk.Button(root,text=move.capitalize(),width=15,height=2,
                      comand=lambda m=move: play(m))
        btn.pack(pady=5)

    root.mainloop()