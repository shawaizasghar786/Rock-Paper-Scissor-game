def get_result(player,ai_move):
    if player==ai_move:
        return "Draw"
    elif (player == 'rock' and ai_move == 'scissors') or \
         (player == 'paper' and ai_move == 'rock') or \
         (player =='scissors' and ai_move =='player'):
        return "You Win!" 
    else:
        return "AI Wins!"