# CS3600 Adversarial Grid Grid Game

## Overview
This project implements an intelligent agent for a competitive, turn-based game played on an 8×8 grid. The agent participates in an adversarial tournament where two players compete to maximize points through strategic movement, territory control, and probabilistic inference.

The core challenge combines:
- Adversarial decision-making
- Stochastic modeling
- Partial observability (hidden rat tracking)

Our agent, **Jarvis**, is designed to outperform baseline agents by combining heuristic planning with probabilistic reasoning.

---

## Game Summary
Each player controls a worker on an 8×8 board and alternates turns.

### Objectives
- **Prime squares** → +1 point each  
- **Carpet sequences of primed squares** → increasing rewards  
- **Locate the hidden rat** → +4 points (−2 if incorrect)

The game lasts 40 turns per player, and the agent with the highest score wins.

---

## Key Challenges
- The rat moves probabilistically using a hidden transition matrix
- Observations are noisy (sound type + estimated distance)
- Agents must balance:
  - deterministic scoring (carpeting)
  - probabilistic search (rat detection)
- Strict time constraints (4 minutes total per agent)

---

## Jarvis Agent Strategy

### 1. Probabilistic Rat Tracking
Jarvis maintains a belief distribution over all 64 board cells:
- Updates beliefs using:
  - transition probabilities
  - noisy sensor observations
- Uses a Hidden Markov Model–style update:
  - **Prediction** (movement)
  - **Correction** (sensor likelihood)

---

### 2. Intelligent Search Decisions
Jarvis only searches when it is statistically worthwhile.

Expected value of search:
EV = 6p - 2

Jarvis searches only when:
p > ~0.42

This avoids unnecessary penalties from incorrect guesses.

---

### 3. Heuristic Move Selection
Jarvis evaluates moves using one-step lookahead:

- Simulates moves with `forecast_move`
- Scores resulting states based on:
  - score difference
  - carpet potential
  - mobility (future move options)
  - proximity to likely rat locations
  - positional advantages

---

### 4. Priority Behavior
- Prefer high-value **carpet moves**
- Otherwise prefer **prime moves**
- Otherwise reposition using **plain moves**
- Search only when confidence is high

---

## Project Structure
.
├── 3600-agents/
│   ├── Jarvis/          # Our agent
│   └── Yolanda/         # Provided baseline agent
├── engine/              # Game engine and runner
├── game/                # Core game logic (board, moves, rat)
├── transition_matrices/ # Rat movement models
├── requirements.txt
└── README.md

---

## Setup Instructions

### 1. Create virtual environment
```
bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
pip install -r requirements.txt
