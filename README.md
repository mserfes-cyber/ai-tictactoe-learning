# ai-tictactoe-learning

# Super AI Tic-Tac-Toe Adventure

## Overview
A kid-friendly (5th grade) educational Tic-Tac-Toe game built with Streamlit that teaches ML foundations through fun gameplay. Human (X) plays against AI (O). Uses Q-Learning for board state scoring and KNN (scikit-learn) for win predictions.

## Architecture
- **app.py** — Single-file Streamlit app with game l  ogic, Q-learning, KNN, minimax AI, and UI
- **Stack**: Python 3.11, Streamlit, NumPy, pandas, scikit-learn

## Key Features
- 3x3 Tic-Tac-Toe board with clickable buttons
- Auto-training on first load (5 games, progress bar animation)
- Training button ("TRAIN AI 5 MORE!") with progress bar
- KNN classifier using board features for win/loss predictions
- Combined prediction (80% KNN + 20% Q-score) displayed as fun badges
- 5-game ML switch: AI uses minimax for first 5 games, then switches to ML-based moves (ai_move_ml)
- Unlock countdown badge ("Play Nx more to unlock SUPER AI BRAIN!")
- Kid-friendly popups at milestones (move 1: "AWESOME FIRST MOVE!", move 2: "AI PREDICTION TIME!", move 3: superhero toast)
- SHOW_POPUPS toggle to enable/disable notifications
- Minimax-based AI (unbeatable) for first 5 games
- Interactive Q&A agent in sidebar with kid-friendly quick buttons
- "AI Brain Stats" sidebar with move explanations
- Game History viewer and AI Memory Bank viewer
- Gradient-styled win/loss/draw banners with exciting messages
- Auto-reset after game ends (3s delay + board clear)
- Fun footer with "AI Brain" stats
- Persistent training data and game_count across resets

## Game Flow
```
Load → Auto-train (3s) → [AI Ready popup] → Play → [Awesome! popup] → [Prediction popup] → [Superhero toast] → Win/Draw → Auto-reset → Play again
```

## 5-Game Progression
```
Games 1-5: Minimax AI (unbeatable) + "Play Nx more to unlock" badge
Games 6+: ML-based AI (ai_move_ml) + live prediction badge
```
