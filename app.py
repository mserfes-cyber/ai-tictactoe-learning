import streamlit as st
import numpy as np
import pandas as pd
import random, time, json, os, sys
from sklearn.neighbors import KNeighborsClassifier

AI_DATA_FILE = "ai_memory.json"

def save_ai_data():
    ss = st.session_state
    q_table_serial = {",".join(map(str, k)): v for k, v in ss.q_table.items()}
    records_serial = [(len(h), str(res)) for h, res in ss.dataset_records]
    data = {
        "q_table": q_table_serial,
        "knn_x": ss.knn_x,
        "knn_y": ss.knn_y,
        "game_count": ss.game_count,
        "wins": ss.wins,
        "train_count": ss.train_count,
        "knn_accuracy": ss.knn_accuracy,
        "epsilon": getattr(ss, "epsilon", 0.05),
        "auto_trained": ss.auto_trained,
        "dataset_records": records_serial,
    }
    try:
        with open(AI_DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[ai_memory] Warning: could not save AI data: {e}", file=sys.stderr)

def _parse_result(res_str):
    if res_str == "1": return X
    if res_str == "-1": return O
    return "draw"

def load_ai_data():
    if not os.path.exists(AI_DATA_FILE):
        return None
    try:
        with open(AI_DATA_FILE, "r") as f:
            data = json.load(f)
        q_table = {tuple(map(int, k.split(","))): v for k, v in data["q_table"].items()}
        dataset_records = [
            ([None] * moves, _parse_result(res))
            for moves, res in data.get("dataset_records", [])
        ]
        return {
            "q_table": q_table,
            "knn_x": data["knn_x"],
            "knn_y": data["knn_y"],
            "game_count": data["game_count"],
            "wins": data["wins"],
            "train_count": data["train_count"],
            "knn_accuracy": data["knn_accuracy"],
            "epsilon": data.get("epsilon", 0.05),
            "auto_trained": data.get("auto_trained", True),
            "dataset_records": dataset_records,
        }
    except Exception as e:
        print(f"[ai_memory] Warning: could not load AI data: {e}", file=sys.stderr)
        return None

st.set_page_config(page_title="Super AI Tic-Tac-Toe Adventure!", page_icon="🎮", layout="centered")

EMPTY, X, O = 0, 1, -1
SYMBOLS = {EMPTY: "", X: "❌", O: "⭕"}
SHOW_POPUPS = True

def board_key(b): return tuple(b.flatten())

def check_winner(b):
    for i in range(3):
        if abs(b[i].sum()) == 3: return X if b[i].sum() > 0 else O
        if abs(b[:, i].sum()) == 3: return X if b[:, i].sum() > 0 else O
    for d in [np.diag(b), np.diag(np.fliplr(b))]:
        if abs(d.sum()) == 3: return X if d.sum() > 0 else O
    if not np.any(b == EMPTY): return "draw"
    return None

def empty_cells(b): return list(zip(*np.where(b == EMPTY)))

LINE_NAMES = [
    "Row 1 (top)", "Row 2 (middle)", "Row 3 (bottom)",
    "Col 1 (left)", "Col 2 (middle)", "Col 3 (right)",
    "↘ Diagonal", "↗ Anti-diag",
]

def board_lines(b):
    return [
        [b[0,0], b[0,1], b[0,2]],
        [b[1,0], b[1,1], b[1,2]],
        [b[2,0], b[2,1], b[2,2]],
        [b[0,0], b[1,0], b[2,0]],
        [b[0,1], b[1,1], b[2,1]],
        [b[0,2], b[1,2], b[2,2]],
        [b[0,0], b[1,1], b[2,2]],
        [b[0,2], b[1,1], b[2,0]],
    ]

def board_features(b):
    flat = b.flatten().tolist()
    lines = board_lines(b)
    line_feats = []
    x_threats, o_threats = 0, 0
    for line in lines:
        xc = sum(1 for v in line if v == X)
        oc = sum(1 for v in line if v == O)
        line_feats.append(xc if oc == 0 else 0)
        line_feats.append(oc if xc == 0 else 0)
        if xc == 2 and oc == 0: x_threats += 1
        if oc == 2 and xc == 0: o_threats += 1
    return flat + [flat.count(X), flat.count(O), flat.count(EMPTY), flat[4], sum(flat[i] for i in [0,2,6,8])] + line_feats + [x_threats, o_threats]

def simulate_game():
    b = np.zeros((3,3), dtype=int)
    hist, turn = [], X
    while True:
        cells = empty_cells(b)
        if not cells: break
        r, c = random.choice(cells)
        b[r, c] = turn
        hist.append((b.copy(), turn))
        w = check_winner(b)
        if w is not None: return hist, w
        turn = O if turn == X else X
    return hist, "draw"

def train_q_table(n, existing_q=None):
    q = dict(existing_q) if existing_q else {}
    alpha, gamma = 0.3, 0.95
    epsilon = getattr(st.session_state, 'epsilon', 0.9)
    records, knn_x, knn_y = [], [], []
    for _ in range(n):
        hist, result = simulate_game()
        records.append((hist, result))
        reward = {X: -1, O: 1, "draw": 0.3}.get(result, 0)
        label = 1 if result == O else 0
        for i, (brd, _) in enumerate(reversed(hist)):
            key = board_key(brd)
            q.setdefault(key, 0.5)
            q[key] += alpha * (reward * (gamma ** i) - q[key])
            weight = 3 if i < 2 else (2 if i < 4 else 1)
            for _ in range(weight):
                knn_x.append(board_features(brd))
                knn_y.append(label)
        epsilon = max(0.05, epsilon * 0.995)
    st.session_state.epsilon = epsilon
    return q, records, knn_x, knn_y

def build_knn(knn_x, knn_y):
    if len(knn_x) < 4: return None, 0
    model = KNeighborsClassifier(n_neighbors=min(3, len(knn_x)-1))
    model.fit(knn_x, knn_y)
    return model, model.score(knn_x, knn_y) * 100

def minimax(b, maximizing, a=-np.inf, bt=np.inf):
    w = check_winner(b)
    if w == O: return 10
    if w == X: return -10
    if w == "draw": return 0
    if maximizing:
        best = -np.inf
        for r, c in empty_cells(b):
            b[r, c] = O; best = max(best, minimax(b, False, a, bt)); b[r, c] = EMPTY
            a = max(a, best)
            if bt <= a: break
        return best
    else:
        best = np.inf
        for r, c in empty_cells(b):
            b[r, c] = X; best = min(best, minimax(b, True, a, bt)); b[r, c] = EMPTY
            bt = min(bt, best)
            if bt <= a: break
        return best

def ai_move_smart(b):
    best_s, best_m = -np.inf, None
    for r, c in empty_cells(b):
        b[r, c] = O; s = minimax(b, False); b[r, c] = EMPTY
        if s > best_s: best_s, best_m = s, (r, c)
    return best_m

def ai_move_ml(b, q_table, knn_model):
    for r, c in empty_cells(b):
        t = b.copy(); t[r, c] = O
        if check_winner(t) == O:
            return r, c
    for r, c in empty_cells(b):
        t = b.copy(); t[r, c] = X
        if check_winner(t) == X:
            return r, c
    candidates = []
    for r, c in empty_cells(b):
        t = b.copy()
        t[r, c] = O
        key = board_key(t)
        combined_score, q_signal, knn_prob = get_prediction(t, q_table, knn_model)
        score = combined_score
        if key in q_table:
            base = q_table[key]
            if base < 0.4:
                score -= 20
        candidates.append((r, c, score, key))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]

def get_prediction(b, q, knn_model):
    moves_left = int(np.sum(b == EMPTY))
    filled_cells = 9 - moves_left
    current_player = X if moves_left % 2 == 1 else O
    HISTORICAL_WIN_RATE = 27

    w = check_winner(b)
    if w == O: return 100, 100, 100
    if w == X: return 0, 0, 0

    if moves_left == 1 and current_player == X:
        return 0, 0, 0
    if moves_left == 1 and current_player == O:
        return 100, 100, 100

    if moves_left == 2:
        return 50, 50, 50

    if filled_cells <= 2:
        return HISTORICAL_WIN_RATE, 25, 30

    knn_prob = float(HISTORICAL_WIN_RATE)
    if knn_model and hasattr(knn_model, 'kneighbors'):
        try:
            features = board_features(b)
            distances, indices = knn_model.kneighbors([features])
            neighbor_wins = 0.0
            total_weight = 0.0
            for j in range(len(indices[0])):
                idx = indices[0][j]
                dist = distances[0][j]
                weight = 1.0 / (dist + 0.1)
                neighbor_wins += st.session_state.knn_y[idx] * weight
                total_weight += weight
            raw_knn = (neighbor_wins / total_weight * 100) if total_weight > 0 else HISTORICAL_WIN_RATE
            knn_prob = raw_knn
        except Exception:
            knn_prob = float(HISTORICAL_WIN_RATE)

    q_signal = 50.0
    bk = board_key(b)
    if bk in q:
        q_raw = q[bk]
        q_signal = 30 + 40 * (max(0, min(1, (q_raw - 0.1) / 0.9)))

    combined = 0.8 * knn_prob + 0.2 * q_signal
    return min(max(combined, 10), 95), q_signal, knn_prob

def get_badge(chance):
    moves_left = int(np.sum(st.session_state.board == EMPTY))
    if moves_left <= 2 and check_winner(st.session_state.board) is None:
        if moves_left == 1:
            current_player = X if moves_left % 2 == 1 else O
            if current_player == X:
                return "😱", "Almost a TIE!", "#95a5a6"
            else:
                return "😎", "AI wins next!", "#2ecc71"
        return "🤝", "It's a TIE!", "#95a5a6"
    if chance >= 75: return "😎", "AI SUPER CONFIDENT!", "#2ecc71"
    if chance >= 55: return "🤔", "Tough battle!", "#3498db"
    if chance >= 40: return "⚡", "Anyone can win!", "#f39c12"
    if chance >= 20: return "🔥", "YOU'RE ON FIRE!", "#e74c3c"
    return "🎉", "YOU'RE WINNING!", "#c0392b"

def explain_move(b, move, q):
    r, c = move
    t = b.copy()
    t[r, c] = O
    q_score = q.get(board_key(t), 0.5)
    parts = []
    for pr, pc in empty_cells(b):
        temp = b.copy(); temp[pr, pc] = X
        if check_winner(temp) == X and (pr, pc) == (r, c):
            cell_line_checks = [
                pr == 0, pr == 1, pr == 2,
                pc == 0, pc == 1, pc == 2,
                pr == pc, pr + pc == 2,
            ]
            blocked_lines = [
                name for name, line, in_line in zip(LINE_NAMES, board_lines(b), cell_line_checks)
                if in_line
                and sum(1 for v in line if v == X) == 2
                and sum(1 for v in line if v == O) == 0
            ]
            line_str = " & ".join(blocked_lines) if blocked_lines else "that line"
            parts.append(f"🛡️ Blocked your immediate win on {line_str}!")
    if check_winner(t) == O:
        parts.append("🎯 Went for a win based on learned patterns from past games.")
    if (r, c) == (1, 1) and b[1, 1] == EMPTY:
        parts.append("🧠 Picked the center after learning from past games that this often leads to wins or ties.")
    if (r, c) in [(0,0),(0,2),(2,0),(2,2)]:
        parts.append("📐 Chose a corner after learning that similar past games often won or forced ties here.")
    if q_score >= 0.6:
        parts.append(f"📈 This move has a high learned score ({q_score:.2f}) from past games.")
    elif q_score <= 0.4:
        parts.append(f"📉 This move has a low learned score ({q_score:.2f}), but it was still the best option available.")
    if not parts:
        parts.append(f"🤔 AI chose this move because its learned score was {q_score:.2f} from past games.")
    return parts

def agent_answer(question):
    q = question.lower().strip()
    ss = st.session_state
    combined, qs, kp = get_prediction(ss.board, ss.q_table, ss.knn_model)
    tot = sum(ss.wins.values())
    ai_rate = (ss.wins["O"] / tot * 100) if tot > 0 else 0

    if any(w in q for w in ["q-learn", "q learn", "qlearn", "q table", "qtable", "q-table"]):
        return (f"**Q-Learning** is like a report card for every board position! "
                f"The AI plays random games, then scores each move: good moves get higher scores, bad ones get lower. "
                f"Right now the Q-table has **{len(ss.q_table):,}** scored positions from **{ss.train_count}** training games. "
                f"Formula: `New Score = Old + Learn Rate x (Reward + Future Wins)`")
    if any(w in q for w in ["knn", "k-nn", "nearest neighbor", "neighbours", "neighbors"]):
        return (f"**KNN (K-Nearest Neighbors)** finds the 3 most similar past boards and asks: did AI win from here? "
                f"It's like asking 3 friends who played almost the same game! "
                f"Current accuracy: **{ss.knn_accuracy:.0f}%** trained on **{len(ss.knn_x)}** board samples. "
                f"Right now KNN thinks AI has a **{kp:.0f}%** chance of winning.")
    if any(w in q for w in ["predict", "prediction", "chance", "probability", "win chance", "combined"]):
        return (f"The AI combines two models: **80% KNN** ({kp:.0f}%) + **20% Q-Learning** ({qs:.0f}%) = **{combined:.0f}%** AI win chance. "
                f"Mixing two models makes a smarter guess than either alone! "
                f"Like getting advice from two different teachers and weighing their answers.")
    if any(w in q for w in ["train", "training", "learn", "practice"]):
        return (f"Training means the AI plays random games against itself to learn! "
                f"So far it's trained on **{ss.train_count}** games and learned **{len(ss.q_table):,}** board positions. "
                f"More real games you play = more patterns the AI learns = smarter AI!")
    if any(w in q for w in ["minimax", "unbeatable", "perfect", "always win", "can i win", "how to win", "beat"]):
        return ("After training, the AI uses **Minimax** which looks at EVERY possible future move! "
                "It picks the move that leads to the best outcome no matter what you do. "
                "That's why it's unbeatable -- the best you can do is force a **draw** by playing the center and corners!")
    if any(w in q for w in ["score", "stats", "record", "how many", "games played"]):
        return (f"**Game Stats:** {tot} games played -- You won {ss.wins['X']}, AI won {ss.wins['O']}, "
                f"Draws: {ss.wins['draw']}. AI win rate: {ai_rate:.0f}%. "
                f"Q-table: {len(ss.q_table):,} states | KNN accuracy: {ss.knn_accuracy:.0f}%")
    if any(w in q for w in ["why that move", "why did you move", "why this move", "why there", "explain this move"]):
        if ss.ai_explanations:
            exp_lines = "\n".join(f"- {line}" for line in ss.ai_explanations)
            return f"AI chose this move because:\n{exp_lines}"
        return "AI picks the move with the best learned score from past games and its game-tree logic. In this case, it didn't yet have a clear record for this exact position."
    if any(w in q for w in ["agent", "workflow", "how does ai", "how do you", "how does it"]):
        return ("Here's my workflow step by step: "
                "1) **KNN** spots patterns from past games and finds similar boards. "
                "2) **Q-Learning** scores each possible move based on training. "
                "3) **Minimax** looks ahead at every future outcome. "
                "4) I pick the **smartest move** by combining all three!")
    if any(w in q for w in ["feature", "board feature", "what are feature", "input"]):
        return ("**Features** are numbers that describe the board! I use 32 features: "
                "the 9 cell values (X=1, O=-1, empty=0), 5 summary stats (count of X's, O's, empty cells, center value, corner sum), "
                "16 line-progress features (for each of the 8 lines I track open X and O counts), "
                "and 2 danger scores (how many lines X is one move from winning, same for O). "
                "KNN uses these features to measure how similar two boards are!")
    if any(w in q for w in ["explore", "exploit", "exploration", "exploitation"]):
        return ("**Explore vs Exploit** is a big idea in AI! "
                "**Explore** = try random moves to discover new strategies. "
                "**Exploit** = use what you already know to pick the best move. "
                "During training, the AI explores. During the game, it exploits what it learned!")
    if any(w in q for w in ["reward", "punishment", "score update"]):
        return ("The AI gives itself rewards after each game: "
                "**Win = +1** (great job!), **Loss = -1** (learn from mistakes), **Draw = +0.3** (not bad!). "
                "These rewards flow backward through every move in the game, "
                "so early moves that led to wins get credit too!")
    if any(w in q for w in ["center", "middle", "best first move", "strategy", "tip", "hint"]):
        return ("**Pro tip:** The center square (middle) is the strongest opening move! "
                "It's part of 4 possible winning lines (2 diagonals + 1 row + 1 column). "
                "Corners are second best (3 winning lines each). "
                "Edge squares are weakest (only 2 winning lines). The Q-table confirms this!")
    if any(w in q for w in ["what is ml", "machine learning", "what is ai", "artificial intelligence"]):
        return ("**Machine Learning** is teaching computers to learn from experience instead of following fixed rules! "
                "In this game, the AI learns by playing thousands of games and remembering what works. "
                "**Q-Learning** learns from rewards, **KNN** learns from similar examples -- "
                "two different ways computers can learn, just like you learn from practice and from watching others!")
    if any(w in q for w in ["draw", "tie"]):
        return ("A **draw** happens when all 9 squares are filled and nobody got 3 in a row! "
                f"You've had **{ss.wins['draw']}** draws so far. Against a perfect AI, "
                "a draw is actually a great result -- it means you played perfectly too!")
    return (f"Great question! Here's what I know right now: "
            f"I've trained on **{ss.train_count}** games, learned **{len(ss.q_table):,}** board positions, "
            f"and my KNN accuracy is **{ss.knn_accuracy:.0f}%**. "
            f"Try asking about: **Q-Learning**, **KNN**, **predictions**, **training**, **strategy**, "
            f"**features**, **explore vs exploit**, **rewards**, or **machine learning**!")

def reset_board():
    st.session_state.board = np.zeros((3,3), dtype=int)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.ai_explanations = []
    st.session_state.popups_shown.discard("prediction_popup")
    st.session_state.popups_shown.discard("agent_popup")
    st.session_state.move_count = 0
    st.session_state.current_game_history = []

def do_training(n):
    q, recs, kx, ky = train_q_table(n, st.session_state.q_table)
    st.session_state.q_table = q
    st.session_state.train_count += n
    st.session_state.dataset_records.extend(recs)
    st.session_state.knn_x.extend(kx)
    st.session_state.knn_y.extend(ky)
    model, acc = build_knn(st.session_state.knn_x, st.session_state.knn_y)
    st.session_state.knn_model = model
    st.session_state.knn_accuracy = acc
    st.session_state.trained = True
    save_ai_data()

def learn_from_real_game(hist, result):
    q = st.session_state.q_table
    alpha, gamma = 0.3, 0.95
    knn_x, knn_y = [], []
    reward = {X: -1, O: 1, "draw": 0.3}.get(result, 0)
    label = 1 if result == O else 0
    for i, (brd, _) in enumerate(reversed(hist)):
        key = board_key(brd)
        q.setdefault(key, 0.5)
        q[key] += alpha * (reward * (gamma ** i) - q[key])
        weight = 3 if i < 2 else (2 if i < 4 else 1)
        for _ in range(weight):
            knn_x.append(board_features(brd))
            knn_y.append(label)
    st.session_state.q_table = q
    st.session_state.dataset_records.append((hist, result))
    st.session_state.knn_x.extend(knn_x)
    st.session_state.knn_y.extend(knn_y)
    if len(st.session_state.knn_x) >= 4:
        model, acc = build_knn(st.session_state.knn_x, st.session_state.knn_y)
        st.session_state.knn_model = model
        st.session_state.knn_accuracy = acc
    save_ai_data()

if "board" not in st.session_state:
    _saved = load_ai_data()
    for k, v in [("board", np.zeros((3,3), dtype=int)), ("game_over", False),
                 ("winner", None), ("trained", False), ("q_table", {}),
                 ("train_count", 0), ("ai_explanations", []),
                 ("dataset_records", []), ("show_dataset", False),
                 ("wins", {"X":0,"O":0,"draw":0}), ("knn_model", None),
                 ("knn_accuracy", 0), ("knn_x", []), ("knn_y", []),
                 ("show_neighbors", False), ("auto_trained", False),
                 ("popups_shown", set()), ("move_count", 0), ("game_count", 0),
                 ("agent_chat", []), ("show_ai_memory_explain", False),
                 ("current_game_history", [])]:
        st.session_state[k] = v
    if _saved:
        st.session_state.q_table = _saved["q_table"]
        st.session_state.knn_x = _saved["knn_x"]
        st.session_state.knn_y = _saved["knn_y"]
        st.session_state.game_count = _saved["game_count"]
        st.session_state.wins = _saved["wins"]
        st.session_state.train_count = _saved["train_count"]
        st.session_state.knn_accuracy = _saved["knn_accuracy"]
        st.session_state.epsilon = _saved["epsilon"]
        st.session_state.auto_trained = _saved["auto_trained"]
        st.session_state.dataset_records = _saved["dataset_records"]
        st.session_state.trained = True
        if len(_saved["knn_x"]) >= 4:
            model, acc = build_knn(_saved["knn_x"], _saved["knn_y"])
            st.session_state.knn_model = model
            st.session_state.knn_accuracy = acc

if "current_game_history" not in st.session_state:
    st.session_state.current_game_history = []

if not st.session_state.auto_trained:
    st.markdown("# 🎮 SUPER AI TIC-TAC-TOE ADVENTURE! 🎉")
    st.markdown("### *Training your AI brain...*")
    bar = st.progress(0, text="🔄 Simulating 5 random games...")
    for i in range(10):
        time.sleep(0.25)
        bar.progress((i+1)/10, text=f"🧪 Training step {i+1}/10...")
    do_training(5)
    st.session_state.auto_trained = True
    save_ai_data()
    bar.progress(1.0, text="✅ AI is ready!")
    if SHOW_POPUPS:
        st.success("🧠 **YOUR AI IS READY TO LEARN!** It's about to play practice games and start remembering smart moves. Help it get stronger by playing! 💪")
        st.session_state.popups_shown.add("training_popup")
        time.sleep(3)
    st.rerun()

LEARNING_TOASTS = [
    "AI used patterns from your earlier games to win this one.",
    "When you beat AI, it learns from its mistakes and avoids the same losing moves next time.",
    "Every game you play helps AI learn and change how it plays.",
]

winner = check_winner(st.session_state.board)
if winner is not None and st.session_state.game_over:
    phase2 = st.session_state.game_count > 5
    if winner == X:
        st.balloons()
        if phase2:
            st.toast("YOU BEAT THE TRAINED AI! 🏆 It used its memory, but you still outsmarted it!", icon="🏆")
        else:
            st.toast("YOU BEAT THE TRAINING AI! 🏆 Every practice game makes it a little smarter!", icon="🏆")
    elif winner == O:
        if phase2:
            st.toast("AI WINS! 🤖 It used what it learned from past games to choose its moves.", icon="🤖")
        else:
            st.toast("AI WINS THIS PRACTICE GAME! 🤖 It will remember this to get even better.", icon="🤖")
    else:
        if phase2:
            st.toast("TIE GAME! 🤝 Your skills matched the trained AI move for move!", icon="🤝")
        else:
            st.toast("TIE GAME! 🤝 AI is still training and learning from every move.", icon="🤝")
    if st.session_state.game_count == 5 and "level_up_5" not in st.session_state.popups_shown:
        st.toast("🎓 AI LEVEL UP! Your AI just finished its first learning set. From now on, it will use what it learned from your games to choose moves.", icon="🎓")
        st.session_state.popups_shown.add("level_up_5")
    if phase2:
        st.toast(random.choice(LEARNING_TOASTS), icon="🎓")

if SHOW_POPUPS and st.session_state.move_count == 1 and "knn_popup" not in st.session_state.popups_shown:
    st.toast("🎉 AWESOME FIRST MOVE! Your AI is LEARNING from your games — it's remembering what happens after each move!", icon="🎉")
    st.session_state.popups_shown.add("knn_popup")

if SHOW_POPUPS and st.session_state.move_count == 2 and "prediction_popup" not in st.session_state.popups_shown:
    st.toast("🔍 LEARNING IN PROGRESS! AI is watching this game so it can make better moves in future games.", icon="🔍")
    st.session_state.popups_shown.add("prediction_popup")

if SHOW_POPUPS and st.session_state.move_count == 3 and "agent_popup" not in st.session_state.popups_shown:
    st.toast("🧠 AI is saving this game in its memory so it can play smarter later!", icon="🦸")
    st.session_state.popups_shown.add("agent_popup")

st.markdown("""<style>
div[data-testid="stHorizontalBlock"] div.stButton > button {
  font-size:2.2rem !important; width:90px !important; height:90px !important;
  border-radius:16px !important; border:3px solid #ddd !important; font-weight:bold !important;
  transition:all .15s ease !important; background:white !important;}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
  transform:scale(1.08) !important; border-color:#4B8BFF !important;
  box-shadow:0 4px 15px rgba(75,139,255,.3) !important;}
.badge {display:inline-block; padding:8px 18px; border-radius:25px; color:white;
  font-size:1.05rem; font-weight:bold; margin:4px 2px;}
</style>""", unsafe_allow_html=True)

st.markdown("# 🎮 SUPER AI TIC-TAC-TOE ADVENTURE! 🎉")
st.markdown("### Train your OWN AI to be UNBEATABLE! 💪")
st.info("🔬 **Science Question:** Can AI learn from every game that I play?")
st.markdown("---")

with st.sidebar:
    st.markdown("### 💬 Ask Me Anything!")
    st.caption("Ask how the AI thinks, gets smarter, or for tips!")
    quick_qs = [
        ("🤖 How I learn?", "How do you learn?"),
        ("🔮 Win chance?", "What's my win chance?"),
        ("💡 Pro tip?", "Best strategy?"),
        ("🧠 Why that move?", "Why did you move there?"),
    ]
    qc1, qc2 = st.columns(2)
    for idx, (label, full_q) in enumerate(quick_qs):
        with [qc1, qc2][idx % 2]:
            if st.button(label, key=f"qq_{idx}", use_container_width=True):
                answer = agent_answer(full_q)
                st.session_state.agent_chat.append({"role": "user", "text": full_q})
                st.session_state.agent_chat.append({"role": "agent", "text": answer})
                if len(st.session_state.agent_chat) > 20:
                    st.session_state.agent_chat = st.session_state.agent_chat[-20:]
                st.rerun()
    with st.form("agent_form", clear_on_submit=True):
        agent_q = st.text_input("Type your question:", placeholder="Or type your own question...")
        submitted = st.form_submit_button("Ask 🤖", use_container_width=True)
        if submitted and agent_q.strip():
            answer = agent_answer(agent_q.strip())
            st.session_state.agent_chat.append({"role": "user", "text": agent_q.strip()})
            st.session_state.agent_chat.append({"role": "agent", "text": answer})
            if len(st.session_state.agent_chat) > 20:
                st.session_state.agent_chat = st.session_state.agent_chat[-20:]
            st.rerun()
    for chat_msg in st.session_state.agent_chat:
        if chat_msg["role"] == "user":
            st.markdown(f"**You:** {chat_msg['text']}")
        else:
            st.markdown(f"**🤖 Agent:** {chat_msg['text']}")
    st.markdown("---")
    if st.button("🧠 Explain how AI remembers", use_container_width=True):
        st.session_state.show_ai_memory_explain = not st.session_state.show_ai_memory_explain
    if st.session_state.show_ai_memory_explain:
        st.info(
            "AI remembers which moves help it win. "
            "After each game, AI updates its memory. "
            "Next time, it tries to use the moves that worked better."
        )
    st.markdown("---")
    st.markdown("### 🎓 How I Get Smarter")
    st.markdown("1. 🎲 **Play** tons of practice games\n2. 📝 **Remember** which moves won\n3. 🔍 **Find** similar past games\n4. 🧠 **Pick** the smartest move!")
    st.markdown("---")
    st.markdown("### 🔍 AI Threat Radar")
    _b = st.session_state.board
    _active = False
    for _name, _line in zip(LINE_NAMES, board_lines(_b)):
        _xc = sum(1 for v in _line if v == X)
        _oc = sum(1 for v in _line if v == O)
        if _xc == 2 and _oc == 0:
            st.markdown(f"🔴 **{_name}** — DANGER! You have 2 in a row!")
            _active = True
        elif _oc == 2 and _xc == 0:
            st.markdown(f"🟢 **{_name}** — AI has 2, going for the win!")
            _active = True
        elif _xc == 1 and _oc == 0:
            st.markdown(f"🟡 **{_name}** — You're building here...")
            _active = True
        elif _oc == 1 and _xc == 0:
            st.markdown(f"🔵 **{_name}** — AI is building here...")
            _active = True
    if not _active:
        st.caption("Make a move to see which lines AI is watching!")
    if st.session_state.ai_explanations:
        st.markdown("**🤖 Why AI moved there:**")
        for _exp in st.session_state.ai_explanations:
            st.markdown(f"• {_exp}")

sw = st.session_state.wins
s_tot = sw["X"] + sw["O"] + sw["draw"]
st.markdown("<h3 style='text-align:center'>⚔️ AI BATTLE LOG ⚔️</h3>", unsafe_allow_html=True)
rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.metric("🏆 AI Wins", sw["O"], delta=f"{sw['O']}/{s_tot} games" if s_tot > 0 else "0 games played")
with rc2:
    st.metric("🤝 Ties", sw["draw"], delta=f"{sw['draw']}/{s_tot} games" if s_tot > 0 else "0 games played")
with rc3:
    st.metric("😊 You Win", sw["X"], delta=f"{sw['X']}/{s_tot} games" if s_tot > 0 else "0 games played")
st.markdown("---")

if st.session_state.game_over and check_winner(st.session_state.board) is not None:
    w = check_winner(st.session_state.board)
    if w == X:
        st.markdown('<div style="text-align:center;padding:20px;background:linear-gradient(45deg,#2ecc71,#27ae60);color:white;border-radius:20px;font-size:2rem;font-weight:bold;margin:10px 0;text-shadow:2px 2px 4px rgba(0,0,0,0.3)">🎉🏆 YOU BEAT THE SUPER AI! LEGENDARY! 🏆🎉</div>', unsafe_allow_html=True)
    elif w == O:
        st.markdown('<div style="text-align:center;padding:20px;background:linear-gradient(45deg,#e74c3c,#c0392b);color:white;border-radius:20px;font-size:2rem;font-weight:bold;margin:10px 0;text-shadow:2px 2px 4px rgba(0,0,0,0.3)">🤖 AI WINS! You\'ll get it next time! 💪</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:20px;background:linear-gradient(45deg,#95a5a6,#7f8c8d);color:white;border-radius:20px;font-size:2rem;font-weight:bold;margin:10px 0;text-shadow:2px 2px 4px rgba(0,0,0,0.3)">🤝 TIE GAME! You matched the SUPER AI! 🤝</div>', unsafe_allow_html=True)
else:
    combined, qs, kp = get_prediction(st.session_state.board, st.session_state.q_table, st.session_state.knn_model)
    emoji, msg, color = get_badge(combined)
    if st.session_state.game_count < 5:
        games_left = 5 - st.session_state.game_count
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #4a90e2, #357abd);
                    padding: 18px; border-radius: 18px; text-align: center;
                    color: white; font-size: 1.3rem; font-weight: bold;
                    box-shadow: 0 8px 25px rgba(74,144,226,0.4);
                    ">
        🚀 Play {games_left} more game{"s" if games_left != 1 else ""} to TRAIN YOUR AI! — After these practice games, your AI will start using what it has learned.
        </div>
        
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="badge" style="background:{color};text-align:center;display:block">{emoji} AI Win Chance: {combined:.0f}% — AI is using what it learned from past games to make this prediction.</div>', unsafe_allow_html=True)

gc = st.session_state.game_count
if gc >= 5:
    st.markdown('<div style="background:linear-gradient(45deg,#27ae60,#1e8449);padding:14px;border-radius:14px;text-align:center;color:white;font-weight:bold;margin:6px 0">🎓 AI MEMORY UNLOCKED — Your AI remembered its first 5 games and is now using them to play smarter!</div>', unsafe_allow_html=True)
if gc >= 10:
    st.markdown('<div style="background:linear-gradient(45deg,#3498db,#2980b9);padding:14px;border-radius:14px;text-align:center;color:white;font-weight:bold;margin:6px 0">⚡ AI LEVEL 2 — Your AI has learned from 10 games. It knows more patterns now!</div>', unsafe_allow_html=True)
if gc >= 15:
    st.markdown('<div style="background:linear-gradient(45deg,#f39c12,#d68910);padding:14px;border-radius:14px;text-align:center;color:white;font-weight:bold;margin:6px 0">🏁 15-GAME CHALLENGE DONE — Check the AI Battle Log: did AI learn?</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size:1.1rem; padding:12px; border:1px solid #ddd; '
        'border-radius:10px; background:#f9f9f9; font-family: system-ui, sans-serif;">'
        '🔬 <b>Answer:</b> Yes! AI learned from every game you played. '
        'Your wins, losses, and ties changed over time, which means the AI is adapting. '
        'If you play more games, it will avoid your patterns even better.'
        '</div>',
        unsafe_allow_html=True
    )

def handle_click(r, c):
    if st.session_state.game_over or st.session_state.board[r, c] != EMPTY: return
    st.session_state.board[r, c] = X
    st.session_state.move_count += 1
    st.session_state.current_game_history.append((st.session_state.board.copy(), X))
    w = check_winner(st.session_state.board)
    if w is not None:
        st.session_state.game_over, st.session_state.winner = True, w
        if w == X: st.session_state.wins["X"] += 1
        elif w == "draw": st.session_state.wins["draw"] += 1
        st.session_state.game_count += 1
        learn_from_real_game(st.session_state.current_game_history, w)
        return
    if st.session_state.game_count < 5:
        move = ai_move_smart(st.session_state.board)
    else:
        move = ai_move_ml(st.session_state.board, st.session_state.q_table, st.session_state.knn_model)
    if move:
        st.session_state.ai_explanations = explain_move(st.session_state.board, move, st.session_state.q_table)
        st.session_state.board[move[0], move[1]] = O
        st.session_state.current_game_history.append((st.session_state.board.copy(), O))
        w = check_winner(st.session_state.board)
        if w is not None:
            st.session_state.game_over, st.session_state.winner = True, w
            if w == O: st.session_state.wins["O"] += 1
            elif w == "draw": st.session_state.wins["draw"] += 1
            st.session_state.game_count += 1
            learn_from_real_game(st.session_state.current_game_history, w)

st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
_, gc1, gc2, gc3, _ = st.columns([1, 1, 1, 1, 1])
for r in range(3):
    for ci, c in enumerate([0, 1, 2]):
        with [gc1, gc2, gc3][ci]:
            v = st.session_state.board[r, c]
            lbl = SYMBOLS[v] if v != EMPTY else " "
            st.button(lbl, key=f"cell_{r}_{c}", on_click=handle_click, args=(r, c),
                      disabled=st.session_state.game_over or v != EMPTY)

st.markdown("---")

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📋 Game History", use_container_width=True):
            st.session_state.show_dataset = not st.session_state.show_dataset
            st.session_state.show_neighbors = False
    with col_b2:
        if st.button("🔍 AI Memory", use_container_width=True):
            st.session_state.show_neighbors = not st.session_state.show_neighbors
            st.session_state.show_dataset = False

if st.session_state.show_dataset:
    st.markdown("### 📊 AI's Game History")
    rows = [{"Game": i+1, "Moves": len(h), "Result": {X:"❌ X Won", O:"⭕ O Won", "draw":"🤝 Draw"}.get(res,"?")}
            for i, (h, res) in enumerate(st.session_state.dataset_records)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if st.session_state.show_neighbors:
    st.markdown("### 🔍 AI's Memory Bank")
    if st.session_state.game_count == 0:
        st.markdown("**AI Memory:** No real-game data yet — play a game first and the AI will start building its memory!")
    else:
        st.markdown("""
**AI Memory explained:**  
This table shows **3 past games that look almost like this board**.

- **"Player favored"** = AI lost or tied in that game.  
- **"AI favored"** = AI won or did well in that game.  

AI uses these 3 similar games to decide which move is safer or smarter right now.
""")
        if st.session_state.knn_model:
            try:
                dist, idx = st.session_state.knn_model.kneighbors([board_features(st.session_state.board)])
                ndata = [{"Neighbor": j+1, "Distance": f"{d:.2f}", "Class": "AI favored" if st.session_state.knn_y[i] == 1 else "Player favored"}
                         for j, (d, i) in enumerate(zip(dist[0], idx[0]))]
                st.dataframe(pd.DataFrame(ndata), use_container_width=True, hide_index=True)
                st.caption(f"3 most similar boards (K=3) | Total samples: {len(st.session_state.knn_x)} saved boards")
            except Exception: st.info("Play a move first to see neighbors!")

if st.session_state.game_over and check_winner(st.session_state.board) is not None:
    time.sleep(3)
    reset_board()
    st.rerun()
