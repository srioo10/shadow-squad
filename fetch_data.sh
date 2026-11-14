#!/bin/bash
# Download every dataset the pipeline reads. Needs a Kaggle API token (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
set -e; cd "$(dirname "$0")"; K=${KAGGLE:-.venv/bin/kaggle}
get() { mkdir -p "data/$2"; [ -n "$(ls -A "data/$2" 2>/dev/null)" ] || $K datasets download -q -d "$1" -p "data/$2" --unzip; echo "data/$2 ready"; }
get akshankrithick/fbref-2017-2024-for-europes-top-5-leagues fbref            # FBref 2017-18 to 2023-24 (mid-sem source)
get evangower/english-premier-league-standings standings                       # official Premier League tables
get mexwell/understat-database understat                                       # Understat team seasons and matches
get davidcariboo/player-scores transfermarkt                                   # Transfermarkt: players, transfers, valuations, appearances, games
get emrey3lmaz/top-5-league-football-player-stats-2017-2025 emrey              # FBref 2024-25, scraped before the advanced stats were removed
get samyogguragai/top-5-european-leagues-2014-2025 understat_alt               # Understat player seasons 2014-15 to 2024-25 (no La Liga)
get hiuhiuthen/understat-player-league-2025-26 understat_2025                  # Understat player seasons 2025-26
