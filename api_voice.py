# assistant_vocal_groq.py
import tempfile
import streamlit as st
import datetime
import json
import os
import requests
import random
import time
import re
import wikipedia
import threading
import queue
import subprocess
import sys
from datetime import datetime as dt
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import hashlib
import textwrap
from difflib import get_close_matches
import html
import pytz
import math

# ========== CONFIGURATION ET LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant_groq.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Clé API Groq intégrée
GROQ_API_KEY = "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"


# ========== ENUMS ET DATACLASSES ==========

class CommandType(Enum):
    TIME = "time"
    DATE = "date"
    WEATHER = "weather"
    CALCULATE = "calculate"
    SEARCH = "search"
    JOKE = "joke"
    REMINDER = "reminder"
    NEWS = "news"
    SYSTEM = "system"
    GREETING = "greeting"
    QUESTION = "question"
    CONVERSATION = "conversation"
    PROFESSIONAL = "professional"
    UNKNOWN = "unknown"


@dataclass
class ConversationEntry:
    id: str
    timestamp: str
    user_input: str
    assistant_response: str
    command_type: CommandType
    confidence: float = 1.0
    metadata: Optional[Dict] = None

    def to_dict(self):
        return {
            **asdict(self),
            'command_type': self.command_type.value
        }


class VoiceEngine(Enum):
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    EDGE_TTS = "edge_tts"
    SYSTEM = "system"
    DISABLED = "disabled"


@dataclass
class UserPreferences:
    name: str = "Monsieur"
    title: str = "M."
    default_city: str = "Paris"
    language: str = "fr"
    voice_enabled: bool = True
    voice_engine: VoiceEngine = VoiceEngine.PYTTSX3
    voice_speed: int = 170
    voice_volume: float = 1.0
    auto_speak: bool = True
    theme: str = "dark"
    notifications: bool = True
    professional_mode: bool = True
    response_style: str = "professional"
    formality_level: str = "high"

    def to_dict(self):
        return {
            **asdict(self),
            'voice_engine': self.voice_engine.value
        }


# ========== GESTIONNAIRE DE CONFIGURATION ==========
class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / ".assistant_vocal_groq"
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.history_file = self.config_dir / "history.json"
        self.cache_file = self.config_dir / "cache.json"
        self.knowledge_base = self.config_dir / "knowledge.json"

    def load_config(self) -> Dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self, config: Dict):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

    def load_history(self) -> List[Dict]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def save_history(self, history: List[Dict]):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history[-1000:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")


# ========== GESTIONNAIRE GROQ API ==========
class GroqAPIHandler:
    """Gestionnaire pour l'API Groq avec les modèles actuels"""

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Modèles Groq actuellement disponibles (mis à jour)
        self.available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-vision-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "qwen-2.5-32b"
        ]

        # Modèles par défaut recommandés
        self.recommended_models = {
            "best_overall": "llama-3.3-70b-versatile",
            "fast": "llama-3.2-1b-preview",
            "balanced": "llama-3.2-3b-preview",
            "vision": "llama-3.2-11b-vision-preview",
            "multilingual": "mixtral-8x7b-32768"
        }

    def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles disponibles"""
        return self.available_models

    def query(self, prompt: str, model: str = None, temperature: float = 0.7,
              max_tokens: int = 1024, context_messages: List[Dict] = None) -> Tuple[str, Dict]:
        """Envoie une requête à l'API Groq"""

        if model is None:
            model = self.recommended_models["best_overall"]

        # Vérifier que le modèle est disponible
        if model not in self.available_models:
            logger.warning(f"Modèle {model} non dans la liste, utilisation du modèle par défaut")
            model = self.recommended_models["best_overall"]

        # Préparer les messages
        messages = []

        # Ajouter le contexte système
        messages.append({
            "role": "system",
            "content": """Tu es un assistant vocal français professionnel et serviable. 
            Réponds de manière claire, concise et professionnelle. 
            Adapte ton langage au contexte et sois utile pour toutes les questions."""
        })

        # Ajouter le contexte historique si disponible
        if context_messages:
            messages.extend(context_messages)

        # Ajouter le message actuel
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Préparer les données
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'], result
            else:
                error_msg = f"Erreur API Groq (Code {response.status_code}): "
                try:
                    error_details = response.json()
                    error_msg += error_details.get('error', {}).get('message', 'Erreur inconnue')
                except:
                    error_msg += response.text[:200]

                logger.error(error_msg)

                # Fallback: utiliser un autre modèle en cas d'erreur
                if "model_decommissioned" in error_msg or "not found" in error_msg:
                    fallback_model = self.recommended_models["best_overall"]
                    logger.info(f"Tentative avec modèle fallback: {fallback_model}")
                    return self.query(prompt, fallback_model, temperature, max_tokens, context_messages)

                return f"❌ {error_msg}", {"error": error_msg}

        except requests.exceptions.Timeout:
            error_msg = "⏱️ Timeout: L'API a mis trop de temps à répondre"
            return error_msg, {"error": "timeout"}
        except Exception as e:
            error_msg = f"❌ Erreur de connexion: {str(e)}"
            return error_msg, {"error": str(e)}


# ========== GESTIONNAIRE VOCAL ==========
class VoiceManager:
    """Gestionnaire vocal simplifié"""

    def __init__(self):
        self.engine_type = VoiceEngine.DISABLED
        self.engine = None
        self.is_speaking = False
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialiser les moteurs vocaux disponibles"""
        self.engines = {}

        # Essayer pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.9)

            # Chercher une voix française
            voices = engine.getProperty('voices')
            french_voices = []

            for voice in voices:
                if hasattr(voice, 'languages'):
                    if any('fr' in str(lang).lower() for lang in voice.languages):
                        french_voices.append(voice)
                elif 'french' in voice.name.lower() or 'fr' in voice.name.lower():
                    french_voices.append(voice)

            if french_voices:
                engine.setProperty('voice', french_voices[0].id)

            self.engines[VoiceEngine.PYTTSX3] = engine
            self.engine_type = VoiceEngine.PYTTSX3
            logger.info("✅ Moteur pyttsx3 initialisé")
        except Exception as e:
            logger.warning(f"❌ pyttsx3 non disponible: {e}")

        # Essayer gTTS
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self.engines[VoiceEngine.GTTS] = {
                'gtts': gTTS,
                'pygame': pygame
            }
            if self.engine_type == VoiceEngine.DISABLED:
                self.engine_type = VoiceEngine.GTTS
            logger.info("✅ Moteur gTTS initialisé")
        except Exception as e:
            logger.warning(f"❌ gTTS non disponible: {e}")

    def speak(self, text: str, style: str = "professional"):
        """Parler du texte"""
        if self.engine_type == VoiceEngine.DISABLED:
            return False

        try:
            if self.engine_type == VoiceEngine.PYTTSX3:
                engine = self.engines[VoiceEngine.PYTTSX3]

                # Nettoyer le texte
                clean_text = re.sub(r'[#*_\-\[\](){}`<>]', '', text)
                clean_text = re.sub(r'\n+', '. ', clean_text)
                clean_text = clean_text[:500]  # Limiter la longueur

                engine.say(clean_text)
                engine.runAndWait()
                return True

            elif self.engine_type == VoiceEngine.GTTS:
                import tempfile
                from gtts import gTTS
                import pygame

                # Nettoyer le texte
                clean_text = re.sub(r'[#*_\-\[\](){}`<>]', '', text)
                clean_text = re.sub(r'\n+', '. ', clean_text)
                clean_text = clean_text[:300]  # Limiter pour gTTS

                # Générer l'audio
                tts = gTTS(text=clean_text, lang='fr', slow=False)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    temp_file = f.name
                    tts.save(temp_file)

                # Jouer l'audio
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()

                # Attendre la fin
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                # Nettoyer
                os.unlink(temp_file)
                return True

        except Exception as e:
            logger.error(f"Erreur synthèse vocale: {e}")
            return False

    def set_engine(self, engine_type: VoiceEngine):
        """Changer le moteur vocal"""
        if engine_type in self.engines:
            self.engine_type = engine_type
            return True
        return False

    def get_available_engines(self) -> List[VoiceEngine]:
        """Obtenir la liste des moteurs disponibles"""
        return list(self.engines.keys())


# ========== SERVICE MÉTÉO ==========
class WeatherService:
    """Service météo avec OpenWeatherMap"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city: str = "Paris") -> Dict:
        """Obtenir les données météo pour une ville"""

        # Données simulées pour le fallback
        simulated_data = {
            'temp': random.randint(10, 25),
            'description': random.choice(['Ensoleillé', 'Partiellement nuageux', 'Nuageux', 'Légère pluie']),
            'humidity': random.randint(40, 80),
            'wind_speed': random.randint(5, 20),
            'city': city
        }

        if not self.api_key:
            logger.warning("Clé API météo non configurée, utilisation de données simulées")
            return {
                'success': False,
                'data': simulated_data,
                'source': 'simulated'
            }

        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'fr'
            }

            response = requests.get(self.base_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                weather_info = {
                    'temp': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'].capitalize(),
                    'wind_speed': data['wind']['speed'],
                    'city': data['name'],
                    'country': data['sys']['country']
                }

                return {
                    'success': True,
                    'data': weather_info,
                    'source': 'openweathermap'
                }
            else:
                logger.warning(f"Erreur API météo: {response.status_code}")
                return {
                    'success': False,
                    'data': simulated_data,
                    'source': 'simulated_fallback'
                }

        except Exception as e:
            logger.error(f"Erreur service météo: {e}")
            return {
                'success': False,
                'data': simulated_data,
                'source': 'simulated_error'
            }

    def format_weather_response(self, weather_data: Dict, user_title: str = "M.", user_name: str = "Monsieur") -> str:
        """Formatter la réponse météo de manière professionnelle"""

        data = weather_data['data']
        source = weather_data['source']

        if source.startswith('simulated'):
            return (
                f"🌤️ **Prévisions météo pour {data['city']}**\n\n"
                f"**🌡️ Température estimée :** {data['temp']}°C\n"
                f"**📊 Conditions :** {data['description']}\n"
                f"**💧 Humidité relative :** {data['humidity']}%\n"
                f"**💨 Vent :** {data['wind_speed']} km/h\n\n"
                f"**Note :** Données simulées. Pour des prévisions précises, "
                f"configurez une clé API OpenWeatherMap.\n\n"
                f"**{user_title} {user_name}**, voici un aperçu des conditions actuelles."
            )
        else:
            # Données réelles
            advice = self._get_weather_advice(data['temp'], data['description'])

            return (
                f"🌤️ **Rapport météorologique professionnel - {data['city']}, {data['country']}**\n\n"
                f"**Conditions actuelles :** {data['description']}\n"
                f"**🌡️ Température :** {data['temp']:.1f}°C (ressentie {data['feels_like']:.1f}°C)\n"
                f"**💧 Humidité :** {data['humidity']}%\n"
                f"**💨 Vitesse du vent :** {data['wind_speed']} m/s\n\n"
                f"**Conseils professionnels :**\n{advice}\n\n"
                f"**Source :** OpenWeatherMap • Données en temps réel"
            )

    def _get_weather_advice(self, temp: float, description: str) -> str:
        """Générer des conseils basés sur la météo"""

        if temp < 5:
            return "❄️ Prévoyez des réunions en présentiel pour créer de la chaleur humaine. Tenue formelle avec manteau recommandée."
        elif temp < 15:
            return "🧥 Conditions idéales pour des réunions productives. Costume ou tenue professionnelle avec veste légère."
        elif temp < 25:
            return "😊 Parfait pour des événements en extérieur ou brainstorming créatifs. Tenue professionnelle légère."
        else:
            return "🌞 Privilégiez les réunions virtuelles pour le confort. Tenue en tissus respirants recommandée."


# ========== CALCULATRICE INTELLIGENTE ==========
class SmartCalculator:
    """Calculatrice intelligente avec évaluation sécurisée"""

    @staticmethod
    def calculate(expression: str) -> Tuple[bool, str, float]:
        """Évaluer une expression mathématique de manière sécurisée"""

        # Nettoyer l'expression
        expr = expression.lower().strip()

        # Dictionnaire de remplacement pour les termes français
        replacements = {
            'plus': '+',
            'moins': '-',
            'fois': '*',
            'multiplié par': '*',
            'divisé par': '/',
            'sur': '/',
            'pourcent': '*0.01',
            '%': '*0.01',
            'au carré': '**2',
            'carré': '**2',
            'au cube': '**3',
            'cube': '**3',
            'racine carrée de': 'math.sqrt(',
            'racine de': 'math.sqrt(',
            'puissance': '**',
            'à la puissance': '**',
            'exposant': '**',
            'pi': 'math.pi',
            'π': 'math.pi',
            'e': 'math.e',
            'sinus': 'math.sin',
            'cosinus': 'math.cos',
            'tangente': 'math.tan',
            'logarithme': 'math.log',
            'log': 'math.log10',
            'exponentielle': 'math.exp'
        }

        for word, symbol in replacements.items():
            expr = expr.replace(word, symbol)

        # Sécuriser l'évaluation
        try:
            # Liste des fonctions autorisées
            allowed_names = {
                'math.sqrt': math.sqrt,
                'math.pi': math.pi,
                'math.e': math.e,
                'math.sin': math.sin,
                'math.cos': math.cos,
                'math.tan': math.tan,
                'math.log': math.log,
                'math.log10': math.log10,
                'math.exp': math.exp,
                '__builtins__': None
            }

            # Vérifier les caractères autorisés
            allowed_chars = set('0123456789+-*/().^%πe√sincostanlogexp ')
            if any(char not in allowed_chars and not char.isalpha() for char in expr):
                return False, "Expression contenant des caractères non autorisés", 0

            # Évaluer l'expression
            result = eval(expr, {"__builtins__": {}}, allowed_names)

            # Formater le résultat
            if isinstance(result, (int, float)):
                return True, "Calcul réussi", result
            else:
                return False, f"Résultat de type inattendu: {type(result)}", 0

        except ZeroDivisionError:
            return False, "Division par zéro impossible", 0
        except SyntaxError:
            return False, "Syntaxe mathématique incorrecte", 0
        except NameError as e:
            return False, f"Fonction non reconnue: {str(e)}", 0
        except Exception as e:
            return False, f"Erreur de calcul: {str(e)}", 0

    @staticmethod
    def format_calculation_response(expression: str, result: float, user_title: str, user_name: str) -> str:
        """Formatter la réponse du calcul"""

        # Formater le résultat
        if abs(result) > 1e6 or (abs(result) < 1e-6 and result != 0):
            result_str = f"{result:.4e}"
        elif result.is_integer():
            result_str = f"{int(result):,}".replace(',', ' ')
        else:
            result_str = f"{result:,.4f}".replace(',', ' ').rstrip('0').rstrip('.')

        # Analyse du résultat
        if abs(result) > 1000000:
            magnitude = "valeur très importante"
        elif abs(result) < 0.001 and result != 0:
            magnitude = "valeur très petite"
        else:
            magnitude = "valeur standard"

        return (
            f"🧮 **Analyse mathématique professionnelle**\n\n"
            f"**Expression :** {expression}\n"
            f"**Résultat :** {result_str}\n"
            f"**Type :** {magnitude}\n\n"
            f"**{user_title} {user_name}**, ce résultat peut être utilisé pour :\n"
            f"• Analyses financières\n• Projections statistiques\n• Calculs techniques\n• Planification stratégique"
        )


# ========== ASSISTANT GROQ INTELLIGENT ==========
class GroqAssistant:
    """Assistant intelligent utilisant l'API Groq"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.voice_manager = VoiceManager()
        self.groq_handler = GroqAPIHandler(GROQ_API_KEY)
        self.weather_service = WeatherService()
        self.calculator = SmartCalculator()

        self.user_prefs = self._load_preferences()
        self.conversation_history = self.config_manager.load_history()

        # Initialiser Wikipedia
        try:
            wikipedia.set_lang("fr")
        except:
            pass

        logger.info("🤖 Assistant Groq initialisé")

    def _load_preferences(self) -> UserPreferences:
        """Charger les préférences utilisateur"""
        config = self.config_manager.load_config()

        if config:
            try:
                return UserPreferences(
                    name=config.get('name', 'Monsieur'),
                    title=config.get('title', 'M.'),
                    default_city=config.get('default_city', 'Paris'),
                    language=config.get('language', 'fr'),
                    voice_enabled=config.get('voice_enabled', True),
                    voice_engine=VoiceEngine(config.get('voice_engine', 'pyttsx3')),
                    voice_speed=config.get('voice_speed', 170),
                    voice_volume=config.get('voice_volume', 1.0),
                    auto_speak=config.get('auto_speak', True),
                    theme=config.get('theme', 'dark'),
                    notifications=config.get('notifications', True),
                    professional_mode=config.get('professional_mode', True),
                    response_style=config.get('response_style', 'professional'),
                    formality_level=config.get('formality_level', 'high')
                )
            except:
                pass

        return UserPreferences()

    def save_preferences(self):
        """Sauvegarder les préférences"""
        config = self.user_prefs.to_dict()
        self.config_manager.save_config(config)

    def _analyze_intent(self, text: str) -> Tuple[CommandType, Dict]:
        """Analyser l'intention de la requête"""
        text_lower = text.lower()
        metadata = {}

        # Détection des intentions spécifiques
        if any(word in text_lower for word in ['bonjour', 'salut', 'hello', 'coucou', 'bonsoir']):
            return CommandType.GREETING, metadata

        elif any(pattern in text_lower for pattern in ['heure', 'quelle heure', 'qu\'il est']):
            return CommandType.TIME, metadata

        elif any(pattern in text_lower for pattern in ['date', 'aujourd\'hui', 'quel jour']):
            return CommandType.DATE, metadata

        elif any(pattern in text_lower for pattern in ['météo', 'temps', 'température', 'pluie', 'soleil']):
            metadata['city'] = self._extract_city(text_lower)
            return CommandType.WEATHER, metadata

        elif any(pattern in text_lower for pattern in ['calcule', 'combien font', 'calculer', 'math']):
            return CommandType.CALCULATE, metadata

        elif any(pattern in text_lower for pattern in ['blague', 'humour', 'drôle', 'rire']):
            return CommandType.JOKE, metadata

        elif any(pattern in text_lower for pattern in ['recherche', 'cherche', 'trouve', 'wikipedia']):
            return CommandType.SEARCH, metadata

        elif any(pattern in text_lower for pattern in ['actualités', 'news', 'nouvelles']):
            return CommandType.NEWS, metadata

        else:
            # Pour toute autre question, utiliser l'API Groq
            return CommandType.QUESTION, metadata

    def _extract_city(self, text: str) -> str:
        """Extraire une ville du texte"""
        cities = ['paris', 'londres', 'new york', 'tokyo', 'berlin', 'madrid', 'rome', 'bruxelles']

        for city in cities:
            if city in text.lower():
                return city.capitalize()

        return self.user_prefs.default_city

    def process_command(self, command: str, model: str = None) -> str:
        """Traiter une commande utilisateur"""

        if not command or not command.strip():
            return f"**{self.user_prefs.title} {self.user_prefs.name}**, pourriez-vous reformuler votre demande ?"

        # Analyser l'intention
        command_type, metadata = self._analyze_intent(command)

        # Traiter selon le type de commande
        if command_type == CommandType.GREETING:
            response = self._get_greeting_response()

        elif command_type == CommandType.TIME:
            response = self._get_time_response()

        elif command_type == CommandType.DATE:
            response = self._get_date_response()

        elif command_type == CommandType.WEATHER:
            city = metadata.get('city', self.user_prefs.default_city)
            response = self._get_weather_response(city)

        elif command_type == CommandType.CALCULATE:
            response = self._get_calculation_response(command)

        elif command_type == CommandType.JOKE:
            response = self._get_joke_response()

        elif command_type == CommandType.SEARCH:
            response = self._get_search_response(command)

        elif command_type == CommandType.NEWS:
            response = self._get_news_response()

        else:
            # Pour les questions générales, utiliser l'API Groq
            response = self._get_groq_response(command, model)

        # Sauvegarder dans l'historique
        entry_id = hashlib.md5(f"{dt.now().isoformat()}{command}".encode()).hexdigest()[:8]
        entry = ConversationEntry(
            id=entry_id,
            timestamp=dt.now().isoformat(),
            user_input=command,
            assistant_response=response,
            command_type=command_type,
            confidence=1.0,
            metadata=metadata
        )

        self.conversation_history.append(entry.to_dict())
        self.config_manager.save_history(self.conversation_history)

        # Synthèse vocale si activée
        if self.user_prefs.auto_speak and self.user_prefs.voice_enabled:
            self.voice_manager.speak(response, style=self.user_prefs.response_style)

        return response

    def _get_greeting_response(self) -> str:
        """Générer une salutation appropriée"""
        now = dt.now()
        hour = now.hour

        if 5 <= hour < 12:
            period = "matin"
            greeting = f"Bonjour {self.user_prefs.title} {self.user_prefs.name} ! Une excellente journée commence."
        elif 12 <= hour < 14:
            period = "midi"
            greeting = f"Bonjour {self.user_prefs.title} {self.user_prefs.name}. J'espère que votre déjeuner se passe bien."
        elif 14 <= hour < 18:
            period = "après-midi"
            greeting = f"Bon après-midi {self.user_prefs.title} {self.user_prefs.name}. Prêt à être productif."
        elif 18 <= hour < 22:
            period = "soir"
            greeting = f"Bonsoir {self.user_prefs.title} {self.user_prefs.name}. Comment s'est passée votre journée ?"
        else:
            period = "nuit"
            greeting = f"Bonsoir {self.user_prefs.title} {self.user_prefs.name}. Même tard, je suis à votre service."

        return f"{greeting} En quoi puis-je vous assister ce {period} ?"

    def _get_time_response(self) -> str:
        """Obtenir l'heure actuelle"""
        now = dt.now()

        hour = now.hour
        minute = now.minute
        second = now.second

        # Formulation élégante
        if minute == 0:
            time_str = f"{hour} heures précises"
        elif minute < 10:
            time_str = f"{hour} heures et {minute} minute{'s' if minute > 1 else ''}"
        elif minute == 15:
            time_str = f"{hour} heures et quart"
        elif minute == 30:
            time_str = f"{hour} heures et demie"
        elif minute == 45:
            next_hour = hour + 1 if hour < 23 else 0
            time_str = f"{next_hour} heures moins le quart"
        else:
            time_str = f"{hour} heures {minute}"

        return (
            f"🕐 **Heure actuelle :** {time_str} et {second} seconde{'s' if second > 1 else ''}\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, c'est le moment idéal pour planifier vos prochaines actions."
        )

    def _get_date_response(self) -> str:
        """Obtenir la date actuelle"""
        now = dt.now()

        months = [
            'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'
        ]

        days = [
            'Lundi', 'Mardi', 'Mercredi', 'Jeudi',
            'Vendredi', 'Samedi', 'Dimanche'
        ]

        day_name = days[now.weekday()]
        month_name = months[now.month - 1]
        day_of_year = now.timetuple().tm_yday
        quarter = (now.month - 1) // 3 + 1

        return (
            f"📅 **Date actuelle :** {day_name} {now.day} {month_name} {now.year}\n\n"
            f"**Informations :**\n"
            f"• Trimestre en cours : Q{quarter}\n"
            f"• Jour {day_of_year}/365 de l'année\n"
            f"• Semaine {now.isocalendar()[1]} de l'année\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, chaque jour est une nouvelle opportunité."
        )

    def _get_weather_response(self, city: str) -> str:
        """Obtenir la météo"""
        weather_data = self.weather_service.get_weather(city)
        return self.weather_service.format_weather_response(
            weather_data,
            self.user_prefs.title,
            self.user_prefs.name
        )

    def _get_calculation_response(self, expression: str) -> str:
        """Effectuer un calcul"""
        success, message, result = self.calculator.calculate(expression)

        if success:
            return self.calculator.format_calculation_response(
                expression, result, self.user_prefs.title, self.user_prefs.name
            )
        else:
            return (
                f"🧮 **Analyse de l'expression mathématique**\n\n"
                f"**Expression :** {expression}\n"
                f"**Statut :** ❌ {message}\n\n"
                f"**Suggestion :** Reformulez votre calcul avec des opérateurs standard (+, -, *, /, ^)."
            )

    def _get_joke_response(self) -> str:
        """Générer une blague"""
        jokes = [
            "Pourquoi les programmeurs préfèrent-ils le noir ? Parce que la lumière attire les bugs !",
            "Quelle est la différence entre un optimiste et un pessimiste en affaires ? "
            "L'optimiste voit le verre à moitié plein, le pessimiste à moitié vide, "
            "et le chef d'entreprise voit le verre deux fois trop grand.",
            "Pourquoi l'économiste a-t-il pris un parapluie ? Parce qu'on prévoyait des liquidités !",
            "Comment appelle-t-on un informaticien qui n'a pas de café ? Un programme qui ne compile pas.",
            "Pourquoi les données ont-elles refusé de traverser la route ? "
            "Parce qu'elles n'étaient pas autorisées à quitter leur base."
        ]

        joke = random.choice(jokes)
        return f"😊 **Moment de détente professionnelle**\n\n\"{joke}\""

    def _get_search_response(self, query: str) -> str:
        """Effectuer une recherche"""
        # Nettoyer la requête
        clean_query = re.sub(
            r'(recherche|cherche|trouve|informations sur|détails sur|connais-tu|sais-tu)',
            '',
            query,
            flags=re.IGNORECASE
        ).strip()

        try:
            # Essayer Wikipedia
            search_results = wikipedia.search(clean_query, results=3)

            if search_results:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary

                return (
                    f"🔍 **Recherche : {search_results[0]}**\n\n"
                    f"{summary}\n\n"
                    f"**Source :** Wikipedia\n"
                    f"**Pertinence :** Excellente"
                )
            else:
                return self._get_groq_response(f"Recherche d'information sur: {clean_query}")

        except Exception as e:
            logger.error(f"Erreur recherche Wikipedia: {e}")
            return self._get_groq_response(f"Recherche d'information sur: {clean_query}")

    def _get_news_response(self) -> str:
        """Générer des nouvelles"""
        topics = [
            "Technologie et Innovation",
            "Marchés Financiers",
            "Développement Durable",
            "Intelligence Artificielle"
        ]

        topic = random.choice(topics)

        headlines = {
            "Technologie et Innovation": [
                "Nouvelles avancées en informatique quantique révolutionnent le calcul.",
                "La 5G continue son déploiement mondial avec des implications majeures.",
                "Edge computing gagne en importance pour le traitement en temps réel."
            ],
            "Développement Durable": [
                "Les entreprises accélèrent leur transition vers des modèles circulaires.",
                "Énergies renouvelables atteignent des records d'adoption mondiale.",
                "L'économie verte crée de nouveaux emplois et opportunités."
            ]
        }

        news = random.choice(headlines.get(topic, ["Développements significatifs en cours."]))

        return (
            f"📰 **Bulletin d'actualités**\n\n"
            f"**Secteur :** {topic}\n\n"
            f"**Titre :** {news}\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, restez informé pour rester compétitif."
        )

    def _get_groq_response(self, query: str, model: str = None) -> str:
        """Obtenir une réponse de l'API Groq"""

        # Préparer le contexte historique
        context_messages = []
        if self.conversation_history:
            # Prendre les 5 derniers messages pour le contexte
            recent_history = self.conversation_history[-5:]

            for entry in recent_history:
                context_messages.append({
                    "role": "user",
                    "content": entry['user_input']
                })
                context_messages.append({
                    "role": "assistant",
                    "content": entry['assistant_response']
                })

        # Obtenir la réponse de Groq
        if model is None:
            model = self.groq_handler.recommended_models["best_overall"]

        response, metadata = self.groq_handler.query(
            query,
            model=model,
            temperature=0.7,
            max_tokens=1024,
            context_messages=context_messages
        )

        # Ajouter une mention de source
        if "error" not in metadata:
            source_info = f"\n\n*💡 Réponse générée par {model} via Groq API*"
            return response + source_info
        else:
            # En cas d'erreur, fournir une réponse de secours
            return (
                f"**{self.user_prefs.title} {self.user_prefs.name}**, concernant votre question :\n\n"
                f"**« {query} »**\n\n"
                f"Je rencontre une difficulté technique avec le service IA. "
                f"Voici ce que je peux vous dire en attendant :\n\n"
                f"Cette question semble importante et mérite une réponse détaillée. "
                f"Je vous recommande de consulter des sources spécialisées ou de reformuler votre question.\n\n"
                f"*⚠️ Service IA temporairement limité*"
            )

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Obtenir l'historique des conversations"""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def clear_history(self):
        """Effacer l'historique"""
        self.conversation_history = []
        self.config_manager.save_history([])

    def get_stats(self) -> Dict:
        """Obtenir des statistiques d'utilisation"""
        if not self.conversation_history:
            return {}

        # Compter les types de commandes
        type_counts = {}
        for entry in self.conversation_history:
            cmd_type = entry.get('command_type', 'unknown')
            type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1

        return {
            'total_commands': len(self.conversation_history),
            'command_types': type_counts,
            'first_interaction': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'last_interaction': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }


# ========== INTERFACE STREAMLIT ==========
class GroqAssistantUI:
    """Interface utilisateur Streamlit pour l'assistant Groq"""

    def __init__(self):
        self.assistant = GroqAssistant()
        self.setup_page()
        self.initialize_session()

    def setup_page(self):
        """Configurer la page Streamlit"""
        st.set_page_config(
            page_title="🤖 Assistant Vocal Groq Pro",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # CSS personnalisé
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Injecter le CSS personnalisé"""
        st.markdown("""
        <style>
            .main-header {
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                color: white;
                margin-bottom: 2rem;
            }

            .chat-message {
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: 20%;
            }

            .assistant-message {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                margin-right: 20%;
            }

            .stat-card {
                text-align: center;
                padding: 1rem;
                background: white;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin: 0.5rem;
            }

            .stButton > button {
                width: 100%;
                border-radius: 10px;
                padding: 0.5rem 1rem;
            }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session(self):
        """Initialiser l'état de la session"""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "llama-3.3-70b-versatile"

        if 'auto_scroll' not in st.session_state:
            st.session_state.auto_scroll = True

    def render_header(self):
        """Afficher l'en-tête"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Assistant Vocal Groq Pro</h1>
            <p>Votre assistant IA professionnel avec reconnaissance vocale</p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Powered by Groq API • Modèles Llama 3.3, Mixtral, Gemma2
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Afficher la barre latérale"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")

            # Profil utilisateur
            with st.expander("👤 Profil", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    title = st.selectbox(
                        "Titre",
                        ["M.", "Mme", "Dr", "Prof."],
                        index=0
                    )
                    self.assistant.user_prefs.title = title

                with col2:
                    name = st.text_input(
                        "Nom",
                        value=self.assistant.user_prefs.name
                    )
                    self.assistant.user_prefs.name = name

            # Sélection du modèle Groq
            with st.expander("🤖 Modèle IA", expanded=True):
                available_models = self.assistant.groq_handler.get_available_models()

                model_descriptions = {
                    "llama-3.3-70b-versatile": "🌟 Le plus performant (recommandé)",
                    "llama-3.2-3b-preview": "⚡ Rapide et efficace",
                    "llama-3.2-1b-preview": "🚀 Très rapide",
                    "mixtral-8x7b-32768": "🧠 Multilingue expert",
                    "gemma2-9b-it": "🔧 Bon pour le code",
                    "qwen-2.5-32b": "🌍 Large contexte"
                }

                # Créer les options avec descriptions
                model_options = []
                for model in available_models:
                    desc = model_descriptions.get(model, model)
                    model_options.append(f"{model} - {desc}")

                selected_option = st.selectbox(
                    "Choisir le modèle",
                    model_options,
                    index=0
                )

                # Extraire le nom du modèle
                selected_model = selected_option.split(" - ")[0]
                st.session_state.selected_model = selected_model
                st.info(f"Modèle sélectionné: **{selected_model}**")

            # Paramètres vocaux
            with st.expander("🔊 Voix", expanded=False):
                voice_enabled = st.checkbox(
                    "Activer la voix",
                    value=self.assistant.user_prefs.voice_enabled
                )
                self.assistant.user_prefs.voice_enabled = voice_enabled

                if voice_enabled:
                    engines = self.assistant.voice_manager.get_available_engines()
                    if engines:
                        engine_options = [engine.value for engine in engines]
                        selected_engine = st.selectbox(
                            "Moteur vocal",
                            engine_options,
                            index=0
                        )
                        self.assistant.user_prefs.voice_engine = VoiceEngine(selected_engine)

            # Commandes rapides
            st.markdown("### ⚡ Commandes Rapides")

            quick_commands = [
                ("🕐 Heure", "Quelle heure est-il ?"),
                ("📅 Date", "Quelle est la date d'aujourd'hui ?"),
                ("🌤️ Météo", f"Météo à {self.assistant.user_prefs.default_city}"),
                ("🧮 Calcul", "Calcule 125 * 48 / 6"),
                ("😊 Blague", "Dis-moi une blague"),
                ("🔍 Recherche", "Recherche sur l'intelligence artificielle")
            ]

            for icon, cmd in quick_commands:
                if st.button(f"{icon} {cmd.split('?')[0]}", use_container_width=True):
                    with st.spinner("Traitement..."):
                        response = self.assistant.process_command(cmd, st.session_state.selected_model)
                        st.session_state.conversation.append({
                            "role": "user",
                            "content": cmd,
                            "time": dt.now().strftime("%H:%M")
                        })
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": response,
                            "time": dt.now().strftime("%H:%M")
                        })
                        st.rerun()

            # Sauvegarde
            st.markdown("---")
            if st.button("💾 Sauvegarder", use_container_width=True):
                self.assistant.save_preferences()
                st.success("Configuration sauvegardée !")

    def render_main_content(self):
        """Afficher le contenu principal"""
        col1, col2 = st.columns([3, 1])

        with col1:
            # Zone de conversation
            st.markdown("### 💬 Conversation")

            # Conteneur pour la conversation
            chat_container = st.container(height=400)

            with chat_container:
                # Afficher l'historique
                history = self.assistant.get_conversation_history(10)

                for entry in history:
                    # Message utilisateur
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{self.assistant.user_prefs.title} {self.assistant.user_prefs.name}</strong>
                            <small>{entry['timestamp'][11:16]}</small>
                        </div>
                        <div>{entry['user_input']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Message assistant
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>Assistant Groq</strong>
                            <small>{entry['timestamp'][11:16]}</small>
                        </div>
                        <div>{entry['assistant_response']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Afficher la conversation en cours
                for msg in st.session_state.conversation:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>{self.assistant.user_prefs.title} {self.assistant.user_prefs.name}</strong>
                                <small>{msg['time']}</small>
                            </div>
                            <div>{msg['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>Assistant Groq</strong>
                                <small>{msg['time']}</small>
                            </div>
                            <div>{msg['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Contrôles
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

            with col_ctrl1:
                if st.button("🗑️ Effacer", use_container_width=True):
                    self.assistant.clear_history()
                    st.session_state.conversation = []
                    st.rerun()

            with col_ctrl2:
                if st.button("📊 Stats", use_container_width=True):
                    stats = self.assistant.get_stats()
                    if stats:
                        st.info(f"**Interactions :** {stats['total_commands']}")

            with col_ctrl3:
                if st.button("🔄 Actualiser", use_container_width=True):
                    st.rerun()

        with col2:
            # Statistiques et informations
            st.markdown("### 📈 Statistiques")

            stats = self.assistant.get_stats()

            if stats:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{stats['total_commands']}</h3>
                    <p>Interactions totales</p>
                </div>
                """, unsafe_allow_html=True)

            # Informations système
            st.markdown("### 🔧 Système")

            # Test de connexion API
            if st.button("🔍 Tester API", use_container_width=True):
                with st.spinner("Test en cours..."):
                    test_response, _ = self.assistant.groq_handler.query("Test de connexion", max_tokens=10)
                    if "❌" not in test_response:
                        st.success("✅ API Groq fonctionnelle")
                    else:
                        st.error("❌ Problème de connexion")

            # Journal des erreurs
            if st.button("📋 Journal", use_container_width=True):
                if os.path.exists('assistant_groq.log'):
                    with open('assistant_groq.log', 'r') as f:
                        logs = f.read()[-2000:]
                    st.text_area("Derniers logs", logs, height=200)

            # Version
            st.markdown("---")
            st.markdown("**Version :** 2.0.0")
            st.markdown("**API :** Groq")
            st.markdown(f"**Modèle :** {st.session_state.selected_model}")

    def render_input_section(self):
        """Afficher la section de saisie"""
        st.markdown("---")

        col_input, col_btn = st.columns([4, 1])

        with col_input:
            user_input = st.text_area(
                "Votre message :",
                placeholder="Posez votre question ou donnez une commande...",
                height=100,
                key="text_input"
            )

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📤 Envoyer", type="primary", use_container_width=True):
                if user_input and user_input.strip():
                    with st.spinner("Traitement en cours..."):
                        response = self.assistant.process_command(
                            user_input.strip(),
                            st.session_state.selected_model
                        )

                        # Ajouter à la conversation
                        st.session_state.conversation.append({
                            "role": "user",
                            "content": user_input.strip(),
                            "time": dt.now().strftime("%H:%M")
                        })
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": response,
                            "time": dt.now().strftime("%H:%M")
                        })

                        st.rerun()
                else:
                    st.warning("Veuillez saisir un message.")

        # Bouton vocal (simulation)
        col_voice, col_info = st.columns([1, 3])
        with col_voice:
            if st.button("🎤 Mode Vocal", use_container_width=True):
                st.info("🎤 **Mode vocal simulé** - En développement avancé")
                st.write("Pour l'instant, utilisez la saisie texte.")

        with col_info:
            st.caption("💡 **Astuce :** Utilisez les commandes rapides dans la barre latérale.")

    def render_footer(self):
        """Afficher le pied de page"""
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🤖 Assistant Groq Pro**")
            st.caption("Version 2.0 • IA Professionnelle")

        with col2:
            now = dt.now()
            st.markdown(f"**🕐 {now.strftime('%H:%M:%S')}**")
            st.caption(f"{now.strftime('%A %d %B %Y')}")

        with col3:
            st.markdown("**🔒 Sécurisé**")
            st.caption("Clé API intégrée • Données locales")

    def run(self):
        """Exécuter l'application"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            self.render_input_section()
            self.render_footer()

        except Exception as e:
            st.error(f"Erreur dans l'application : {str(e)}")
            logger.error(f"Erreur application : {e}")


# ========== EXÉCUTION PRINCIPALE ==========
def main():
    """Fonction principale"""
    try:
        st.set_page_config(
            page_title="Assistant Vocal Groq",
            page_icon="🤖",
            layout="wide"
        )

        with st.spinner("Initialisation de l'assistant..."):
            time.sleep(0.5)
            ui = GroqAssistantUI()
            ui.run()

    except Exception as e:
        st.error(f"""
        ## ⚠️ Erreur d'initialisation

        **Détails :** {str(e)}

        **Solutions :**
        1. Vérifiez votre connexion Internet
        2. Installez les dépendances : `pip install streamlit requests wikipedia`
        3. Redémarrez l'application
        """)


# ========== POINT D'ENTRÉE ==========
if __name__ == "__main__":
    main()