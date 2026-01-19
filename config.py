# config.py
import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Créer les dossiers
for directory in [DATA_DIR, ASSETS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration de l'assistant
ASSISTANT_CONFIG = {
    "name": "Alexa",
    "language": "fr-FR",
    "speech_rate": 150,
    "volume": 1.0,
    "wake_word": "ok assistant",
    "auto_speak": True,
    "save_conversations": True,
    "max_history": 100
}

# Sites web
WEBSITES = {
    'youtube': 'https://youtube.com',
    'google': 'https://google.com',
    'gmail': 'https://gmail.com',
    'facebook': 'https://facebook.com',
    'github': 'https://github.com',
    'twitter': 'https://twitter.com',
    'linkedin': 'https://linkedin.com',
    'wikipedia': 'https://wikipedia.org',
    'netflix': 'https://netflix.com',
    'spotify': 'https://spotify.com'
}

# Commandes spéciales
SPECIAL_COMMANDS = {
    'news': {
        'url': 'https://newsapi.org/v2/top-headlines',
        'params': {'country': 'fr', 'apiKey': 'YOUR_API_KEY'}
    },
    'weather': {
        'url': 'https://api.openweathermap.org/data/2.5/weather',
        'params': {'appid': 'YOUR_API_KEY', 'units': 'metric'}
    }
}

# Thèmes
THEMES = {
    'dark': {
        'primary': '#1E1E2E',
        'secondary': '#282A36',
        'accent': '#6272A4',
        'text': '#F8F8F2'
    },
    'light': {
        'primary': '#FFFFFF',
        'secondary': '#F0F2F6',
        'accent': '#4FACFE',
        'text': '#262730'
    },
    'blue': {
        'primary': '#0F2027',
        'secondary': '#203A43',
        'accent': '#2C5364',
        'text': '#FFFFFF'
    }
}