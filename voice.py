# assistant_vocal_pro.py
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

# ========== CONFIGURATION ET LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    ELEVENLABS = "elevenlabs"
    OPENAI_TTS = "openai_tts"
    SYSTEM = "system"
    DISABLED = "disabled"

@dataclass
class UserPreferences:
    name: str = "Monsieur"
    title: str = "M."  # M., Mme, Dr, Prof.
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
    response_style: str = "professional"  # professional, friendly, technical
    formality_level: str = "high"  # high, medium, low

    def to_dict(self):
        return {
            **asdict(self),
            'voice_engine': self.voice_engine.value
        }


# ========== GESTIONNAIRE DE CONFIGURATION ==========
class ConfigManager:
    """Gestionnaire de configuration centralisé"""

    def __init__(self):
        self.config_dir = Path.home() / ".assistant_vocal_pro"
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.history_file = self.config_dir / "history.json"
        self.cache_file = self.config_dir / "cache.json"
        self.knowledge_base = self.config_dir / "knowledge.json"
        self.user_profile = self.config_dir / "profile.json"

    def load_config(self) -> Dict:
        """Charger la configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self, config: Dict):
        """Sauvegarder la configuration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

    def load_history(self) -> List[Dict]:
        """Charger l'historique"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def save_history(self, history: List[Dict]):
        """Sauvegarder l'historique"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history[-1000:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")

    def load_knowledge_base(self) -> Dict:
        """Charger la base de connaissances"""
        if self.knowledge_base.exists():
            try:
                with open(self.knowledge_base, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "faq": {},
            "user_interests": [],
            "conversation_patterns": {},
            "professional_responses": {}
        }

    def save_knowledge_base(self, knowledge: Dict):
        """Sauvegarder la base de connaissances"""
        try:
            with open(self.knowledge_base, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur sauvegarde connaissance: {e}")


# ========== GESTIONNAIRE VOCAL AVANCÉ ==========
class VoiceManager:
    """Gestionnaire vocal sophistiqué avec fallbacks multiples"""

    def __init__(self):
        self.engine_type = VoiceEngine.DISABLED
        self.engine = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.voice_lock = threading.Lock()  # Verrou pour éviter les conflits
        self._initialize_engines()
        self._start_worker_thread()  # ✅ Nom corrigé
        self.voice_styles = {
            "professional": {"rate": 160, "volume": 0.9, "pitch": 110},
            "friendly": {"rate": 170, "volume": 1.0, "pitch": 120},
            "technical": {"rate": 150, "volume": 0.8, "pitch": 100}
        }

        # Ajouter les nouvelles voix humaines
        self.human_voices = {
            "male_professional": {"rate": 165, "volume": 0.9, "pitch": 115},
            "female_elegant": {"rate": 155, "volume": 0.9, "pitch": 125},
            "male_calm": {"rate": 160, "volume": 0.85, "pitch": 110},
            "female_warm": {"rate": 150, "volume": 0.95, "pitch": 130},
            "male_authoritative": {"rate": 145, "volume": 0.9, "pitch": 105},
            "female_friendly": {"rate": 170, "volume": 0.9, "pitch": 135}
        }

        self.current_voice = "male_professional"
        self.available_voices = {}

    def _initialize_engines(self):
        """Initialiser tous les moteurs vocaux disponibles"""
        self.engines = {}
        self.available_voices = {}

        # Essayer pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()

            # Configuration par défaut
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.9)

            # Chercher une voix française
            voices = engine.getProperty('voices')
            french_voices = []
            all_voices = []

            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages if hasattr(voice, 'languages') else [],
                    'gender': 'male' if 'male' in voice.name.lower() else 'female'
                }
                all_voices.append(voice_info)

                if any(lang in str(voice.languages).lower() for lang in ['fr', 'french', 'fr-fr']):
                    french_voices.append(voice)

            # Stocker les voix
            self.available_voices['pyttsx3'] = {
                'all': all_voices,
                'french': french_voices
            }

            if french_voices:
                engine.setProperty('voice', french_voices[0].id)
                logger.info(f"✅ Voix française trouvée: {french_voices[0].name}")
            else:
                logger.warning("Aucune voix française trouvée, utilisation de la voix par défaut")

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
                'pygame': pygame,
                'cache': {}
            }
            if self.engine_type == VoiceEngine.DISABLED:
                self.engine_type = VoiceEngine.GTTS
            logger.info("✅ Moteur gTTS initialisé")
        except Exception as e:
            logger.warning(f"❌ gTTS non disponible: {e}")

        # Essayer Edge TTS (Microsoft - voix naturelles)
        try:
            import edge_tts
            self.engines[VoiceEngine.EDGE_TTS] = {
                'module': edge_tts,
                'voices': [
                    "fr-FR-DeniseNeural",  # Femme élégante
                    "fr-FR-HenriNeural",  # Homme professionnel
                    "fr-FR-AlainNeural",  # Homme calme
                    "fr-FR-VivienneNeural",  # Femme chaleureuse
                    "fr-FR-ClaudeNeural",  # Homme autoritaire
                    "fr-FR-JosephineNeural"  # Femme amicale
                ]
            }
            logger.info("✅ Edge TTS initialisé (voix Microsoft)")
        except ImportError:
            logger.info("ℹ️ Edge TTS non installé. Installez avec: pip install edge-tts")

        # Essayer ElevenLabs (optionnel)
        try:
            import elevenlabs
            self.engines[VoiceEngine.ELEVENLABS] = {
                'module': elevenlabs,
                'voices': [],
                'api_key': None
            }
            logger.info("✅ ElevenLabs disponible")
        except ImportError:
            pass  # Optionnel

        # Essayer voix système
        try:
            if sys.platform == 'darwin':  # macOS
                self.engines[VoiceEngine.SYSTEM] = 'say'
            elif sys.platform == 'win32':  # Windows
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Rate = 0
                    speaker.Volume = 100
                    self.engines[VoiceEngine.SYSTEM] = speaker
                except ImportError:
                    self.engines[VoiceEngine.SYSTEM] = None
            else:  # Linux
                self.engines[VoiceEngine.SYSTEM] = 'espeak'

            if self.engine_type == VoiceEngine.DISABLED and VoiceEngine.SYSTEM in self.engines:
                self.engine_type = VoiceEngine.SYSTEM
                logger.info("✅ Moteur système initialisé")
        except Exception as e:
            logger.warning(f"❌ Voix système non disponible: {e}")

    def _start_worker_thread(self):  # ✅ Nom corrigé
        """Démarrer le thread de traitement vocal"""

        def worker():
            while True:
                try:
                    item = self.speech_queue.get(timeout=1)
                    if item is None:  # Signal d'arrêt
                        break

                    text, style = item
                    self._speak_sync(text, style)
                    self.speech_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Erreur worker vocal: {e}")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        logger.info("✅ Worker vocal démarré")

    def _speak_sync(self, text: str, style: str = "professional"):
        """Parler du texte de manière synchrone"""
        with self.voice_lock:
            self.is_speaking = True

            try:
                # Nettoyer et préparer le texte
                clean_text = self._prepare_speech_text(text, style)

                if self.engine_type == VoiceEngine.PYTTSX3 and VoiceEngine.PYTTSX3 in self.engines:
                    self._speak_pyttsx3(clean_text, style)

                elif self.engine_type == VoiceEngine.GTTS and VoiceEngine.GTTS in self.engines:
                    self._speak_gtts(clean_text)

                elif self.engine_type == VoiceEngine.EDGE_TTS and VoiceEngine.EDGE_TTS in self.engines:
                    self._speak_edge_tts(clean_text, style)

                elif self.engine_type == VoiceEngine.SYSTEM and VoiceEngine.SYSTEM in self.engines:
                    self._speak_system(clean_text)

                else:
                    logger.warning("Aucun moteur vocal disponible")

            except Exception as e:
                logger.error(f"Erreur lors de la synthèse vocale: {e}")

            finally:
                self.is_speaking = False

    def _speak_pyttsx3(self, text: str, style: str):
        """Utiliser pyttsx3 avec gestion des erreurs"""
        try:
            engine = self.engines[VoiceEngine.PYTTSX3]

            # Arrêter proprement si déjà en cours
            try:
                engine.stop()
            except:
                pass

            # Appliquer le style
            if style in self.voice_styles:
                config = self.voice_styles[style]
                engine.setProperty('rate', config["rate"])
                engine.setProperty('volume', config["volume"])

            engine.say(text)
            engine.runAndWait()

        except RuntimeError as e:
            if "run loop" in str(e):
                # Recréer le moteur
                import pyttsx3
                new_engine = pyttsx3.init()
                self.engines[VoiceEngine.PYTTSX3] = new_engine
                new_engine.say(text)
                new_engine.runAndWait()
            else:
                raise

    def _speak_gtts(self, text: str):
        """Utiliser gTTS avec cache"""
        try:
            import tempfile
            from gtts import gTTS
            import pygame

            engine_data = self.engines[VoiceEngine.GTTS]
            cache = engine_data['cache']

            # Utiliser le cache
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash in cache:
                audio_data = cache[text_hash]
            else:
                # Générer l'audio
                tts = gTTS(text=text, lang='fr', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    temp_file = f.name
                    tts.save(temp_file)

                with open(temp_file, 'rb') as f:
                    audio_data = f.read()

                cache[text_hash] = audio_data
                os.unlink(temp_file)

            # Jouer l'audio
            pygame.mixer.init()
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_audio.write(audio_data)
            temp_audio.close()

            pygame.mixer.music.load(temp_audio.name)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            os.unlink(temp_audio.name)

        except Exception as e:
            logger.error(f"Erreur gTTS: {e}")

    def _speak_edge_tts(self, text: str, style: str = "professional"):
        """Utiliser Edge TTS (Microsoft) - voix naturelles"""
        try:
            import asyncio
            import edge_tts
            import tempfile

            # Mapping des styles aux voix Edge TTS
            voice_map = {
                "professional": "fr-FR-HenriNeural",
                "friendly": "fr-FR-JosephineNeural",
                "technical": "fr-FR-AlainNeural",
                "male_professional": "fr-FR-HenriNeural",
                "female_elegant": "fr-FR-DeniseNeural",
                "male_calm": "fr-FR-AlainNeural",
                "female_warm": "fr-FR-VivienneNeural",
                "male_authoritative": "fr-FR-ClaudeNeural",
                "female_friendly": "fr-FR-JosephineNeural"
            }

            voice = voice_map.get(style, "fr-FR-HenriNeural")

            # Créer un event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Créer le communicateur
            communicate = edge_tts.Communicate(text, voice)

            # Fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_path = tmp_file.name

            # Générer l'audio
            async def generate():
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        with open(tmp_path, "ab") as audio_file:
                            audio_file.write(chunk["data"])

            loop.run_until_complete(generate())
            loop.close()

            # Jouer avec pygame
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Nettoyer
            os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Erreur Edge TTS: {e}")

    def _speak_system(self, text: str):
        """Utiliser la voix système"""
        system_engine = self.engines[VoiceEngine.SYSTEM]

        if isinstance(system_engine, str):
            if sys.platform == 'darwin':  # macOS
                os.system(f'say "{text}" -v Thomas -r 160')
            elif sys.platform == 'linux':
                os.system(f'espeak -v french+m3 "{text}" -s 160 -p 50')
        else:
            # Windows COM
            system_engine.Speak(text)

    def _prepare_speech_text(self, text: str, style: str) -> str:
        """Préparer le texte pour la synthèse vocale"""
        # Supprimer le markdown et HTML
        clean = re.sub(r'[#*_\-\[\](){}`]', '', text)
        clean = re.sub(r'<[^>]+>', '', clean)

        # Remplacer les sauts de ligne
        clean = re.sub(r'\n+', '. ', clean)

        # Ajouter des pauses naturelles
        if style == "professional":
            clean = re.sub(r'[.!?]', '...', clean)
        elif style == "technical":
            clean = re.sub(r',', '...', clean)

        # Limiter la longueur mais intelligemment
        sentences = re.split(r'[.!?]', clean)
        if len(sentences) > 3:
            clean = '. '.join(sentences[:3]) + '...'
        else:
            clean = clean[:600]

        return clean

    def speak(self, text: str, style: str = "professional", async_mode: bool = True):
        """Parler du texte avec un style spécifique"""
        if self.engine_type == VoiceEngine.DISABLED:
            return False

        if async_mode:
            self.speech_queue.put((text, style))
            return True
        else:
            self._speak_sync(text, style)
            return True

    def set_engine(self, engine_type: VoiceEngine):
        """Changer le moteur vocal"""
        if engine_type in self.engines:
            self.engine_type = engine_type
            return True
        return False

    def set_voice_style(self, style: str):
        """Changer le style de voix"""
        if style in self.human_voices or style in self.voice_styles:
            self.current_voice = style
            return True
        return False

    def get_available_engines(self) -> List[VoiceEngine]:
        """Obtenir la liste des moteurs disponibles"""
        return list(self.engines.keys())

    def stop(self):
        """Arrêter le gestionnaire vocal"""
        self.speech_queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=1)

# ========== BASE DE CONNAISSANCES INTELLIGENTE ==========
class KnowledgeBase:
    """Base de connaissances pour réponses professionnelles"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.knowledge = self.config_manager.load_knowledge_base()
        self._initialize_default_knowledge()

    def _initialize_default_knowledge(self):
        """Initialiser les connaissances par défaut"""
        if "professional_responses" not in self.knowledge:
            self.knowledge["professional_responses"] = {
                "greetings": [
                    "Je vous salue, {title} {name}. En quoi puis-je vous assister aujourd'hui ?",
                    "Bonjour {title} {name}, je suis à votre entière disposition.",
                    "Mes respects, {title} {name}. Que puis-je faire pour vous ?"
                ],
                "farewells": [
                    "Au revoir {title} {name}. Ce fut un plaisir de vous assister.",
                    "Je vous souhaite une excellente journée, {title} {name}.",
                    "À très bientôt, {title} {name}. N'hésitez pas si vous avez besoin d'assistance."
                ],
                "acknowledgments": [
                    "Je vous remercie de votre question, {title} {name}.",
                    "Très bonne question, {title} {name}. Permettez-moi de vous répondre.",
                    "J'apprécie votre demande, {title} {name}. Voici ma réponse."
                ],
                "uncertain": [
                    "Permettez-moi de vous proposer une perspective sur ce sujet, {title} {name}.",
                    "Si je comprends bien votre demande, {title} {name}, voici ce que je peux vous dire.",
                    "D'après mes analyses, {title} {name}, voici les informations pertinentes."
                ]
            }

        if "common_topics" not in self.knowledge:
            self.knowledge["common_topics"] = {
                "business": [
                    "stratégie d'entreprise", "management", "leadership", "productivité",
                    "innovation", "croissance", "finance", "marketing", "ventes"
                ],
                "technology": [
                    "intelligence artificielle", "machine learning", "développement",
                    "cloud computing", "cybersécurité", "data science", "blockchain"
                ],
                "general": [
                    "santé", "éducation", "voyage", "culture", "sport", "politique",
                    "économie", "environnement", "science"
                ]
            }

        self.config_manager.save_knowledge_base(self.knowledge)

    def get_professional_response(self, category: str, name: str = "Monsieur", title: str = "M.") -> str:
        """Obtenir une réponse professionnelle"""
        responses = self.knowledge.get("professional_responses", {}).get(category, [])
        if responses:
            template = random.choice(responses)
            return template.format(name=name, title=title)
        return ""

    def learn_from_interaction(self, question: str, response: str, category: str):
        """Apprendre de nouvelles interactions"""
        if "learned_patterns" not in self.knowledge:
            self.knowledge["learned_patterns"] = {}

        key = hashlib.md5(question.lower().encode()).hexdigest()
        self.knowledge["learned_patterns"][key] = {
            "question": question,
            "response": response,
            "category": category,
            "timestamp": dt.now().isoformat(),
            "usage_count": 1
        }

        self.config_manager.save_knowledge_base(self.knowledge)

    def find_similar_question(self, question: str) -> Optional[Dict]:
        """Trouver une question similaire dans la base"""
        if "learned_patterns" not in self.knowledge:
            return None

        questions = list(self.knowledge["learned_patterns"].values())
        if not questions:
            return None

        # Recherche simple par similarité de mots-clés
        question_lower = question.lower()
        for q_data in questions:
            stored_q = q_data["question"].lower()
            # Vérifier les mots communs
            common_words = set(question_lower.split()) & set(stored_q.split())
            if len(common_words) >= 2:  # Au moins 2 mots communs
                q_data["usage_count"] += 1
                return q_data

        return None


# ========== ASSISTANT INTELLIGENT PROFESSIONNEL ==========
class ProfessionalAssistant:
    """Assistant IA professionnel avec compréhension contextuelle avancée"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.voice_manager = VoiceManager()
        self.speech_recognizer = SpeechRecognizer()
        self.knowledge_base = KnowledgeBase()
        self.user_prefs = self._load_preferences()

        # Initialiser Wikipedia
        try:
            wikipedia.set_lang("fr")
            wikipedia.set_rate_limiting(True)
        except:
            pass

        # Cache pour les recherches
        self.cache = {}
        self.conversation_history = self.config_manager.load_history()

        # Context conversationnel
        self.context = {
            "last_topic": None,
            "user_mood": "neutral",
            "conversation_depth": 0,
            "user_interests": set()
        }

        # API Keys
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "")
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

        logger.info("🤖 Assistant professionnel initialisé")

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

    # ========== ANALYSE INTELLIGENTE ==========

    def _analyze_intent_with_context(self, text: str) -> Tuple[CommandType, Dict, float]:
        """Analyser l'intention avec contexte conversationnel"""
        text_lower = text.lower()
        metadata = {"confidence": 1.0}

        # Vérifier d'abord dans les patterns appris
        similar = self.knowledge_base.find_similar_question(text)
        if similar:
            metadata["learned_response"] = similar["response"]
            metadata["category"] = similar["category"]
            metadata["confidence"] = 0.9
            return CommandType.CONVERSATION, metadata, 0.9

        # Détection de type de question
        question_types = {
            "who": r'qui est|qui sont|qui a|qui était',
            "what": r'qu\'est-ce que|qu\'est-ce qu\'|c\'est quoi|définition de',
            "when": r'quand|date|à quelle date|quelle date',
            "where": r'où|où se trouve|lieu de|localisation',
            "why": r'pourquoi|raison de|cause de',
            "how": r'comment|de quelle manière|de quelle façon'
        }

        for q_type, pattern in question_types.items():
            if re.search(pattern, text_lower):
                metadata["question_type"] = q_type
                break

        # Analyse sémantique avancée
        if self._is_greeting(text_lower):
            return CommandType.GREETING, metadata, 0.95

        elif self._is_time_question(text_lower):
            return CommandType.TIME, metadata, 0.9

        elif self._is_date_question(text_lower):
            return CommandType.DATE, metadata, 0.9

        elif self._is_weather_question(text_lower):
            return CommandType.WEATHER, metadata, 0.85

        elif self._is_calculation(text_lower):
            return CommandType.CALCULATE, metadata, 0.95

        elif self._is_search_query(text_lower):
            return CommandType.SEARCH, metadata, 0.8

        elif self._is_joke_request(text_lower):
            return CommandType.JOKE, metadata, 0.9

        elif self._is_news_request(text_lower):
            return CommandType.NEWS, metadata, 0.85

        else:
            # Pour toute autre question, retourner QUESTION avec réponse professionnelle
            return CommandType.QUESTION, metadata, 0.7

    def _is_greeting(self, text: str) -> bool:
        greetings = ['bonjour', 'salut', 'hello', 'hi', 'coucou', 'bonsoir']
        return any(greet in text for greet in greetings)

    def _is_time_question(self, text: str) -> bool:
        patterns = [r'heure', r'quelle heure', r'l\'heure', r'horloge', r'qu\'il est']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_date_question(self, text: str) -> bool:
        patterns = [r'date', r'aujourd\'hui', r'quel jour', r'nous sommes']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_weather_question(self, text: str) -> bool:
        patterns = [r'météo', r'temps', r'température', r'pluie', r'soleil', r'nuage']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_calculation(self, text: str) -> bool:
        patterns = [r'\d+[\s]*[+\-*/%^]\s*\d+', r'calcule', r'calcul', r'combien font', r'égal à']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_search_query(self, text: str) -> bool:
        patterns = [r'recherche', r'cherche', r'trouve', r'c\'est quoi', r'qui est', r'définition']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_joke_request(self, text: str) -> bool:
        patterns = [r'blague', r'humour', r'rire', r'amusant', r'drôle']
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_news_request(self, text: str) -> bool:
        patterns = [r'actualités', r'news', r'infos', r'nouvelles', r'journal']
        return any(re.search(pattern, text) for pattern in patterns)

    # ========== RÉPONSES PROFESSIONNELLES ==========

    def get_professional_response(self, user_input: str) -> Tuple[str, CommandType]:
        """Obtenir une réponse professionnelle pour toute question"""
        # Analyser l'intention
        command_type, metadata, confidence = self._analyze_intent_with_context(user_input)

        # Mettre à jour le contexte
        self._update_conversation_context(user_input, command_type)

        # Générer la réponse selon le type
        if command_type == CommandType.GREETING:
            response = self._get_professional_greeting()

        elif command_type == CommandType.TIME:
            response = self._get_time_response_professional()

        elif command_type == CommandType.DATE:
            response = self._get_date_response_professional()

        elif command_type == CommandType.WEATHER:
            response = self._get_weather_response_professional(user_input)

        elif command_type == CommandType.CALCULATE:
            response = self._get_calculation_response_professional(user_input)

        elif command_type == CommandType.SEARCH:
            response = self._get_search_response_professional(user_input)

        elif command_type == CommandType.JOKE:
            response = self._get_joke_response_professional()

        elif command_type == CommandType.NEWS:
            response = self._get_news_response_professional()

        elif command_type == CommandType.QUESTION:
            response = self._get_general_response_professional(user_input)

        elif command_type == CommandType.CONVERSATION:
            response = metadata.get("learned_response",
                                    self._get_general_response_professional(user_input))
        else:
            response = self._get_fallback_professional(user_input)

        # Ajouter l'entrée de conversation
        entry_id = hashlib.md5(f"{dt.now().isoformat()}{user_input}".encode()).hexdigest()[:8]
        entry = ConversationEntry(
            id=entry_id,
            timestamp=dt.now().isoformat(),
            user_input=user_input,
            assistant_response=response,
            command_type=command_type,
            confidence=confidence,
            metadata=metadata
        )

        # Ajouter à l'historique
        self.conversation_history.append(entry.to_dict())
        self.config_manager.save_history(self.conversation_history)

        # Apprendre de l'interaction
        if confidence > 0.8:
            self.knowledge_base.learn_from_interaction(
                user_input, response, command_type.value
            )

        return response, command_type

    def _update_conversation_context(self, user_input: str, command_type: CommandType):
        """Mettre à jour le contexte conversationnel"""
        self.context["last_topic"] = command_type
        self.context["conversation_depth"] += 1

        # Détecter l'humeur
        positive_words = ['merci', 'super', 'génial', 'parfait', 'excellent']
        negative_words = ['pourquoi pas', 'pas bon', 'mauvais', 'insuffisant']

        if any(word in user_input.lower() for word in positive_words):
            self.context["user_mood"] = "positive"
        elif any(word in user_input.lower() for word in negative_words):
            self.context["user_mood"] = "negative"

        # Extraire les intérêts potentiels
        topics = ['technologie', 'science', 'affaires', 'finance', 'santé', 'voyage']
        for topic in topics:
            if topic in user_input.lower():
                self.context["user_interests"].add(topic)

    def _get_professional_greeting(self) -> str:
        """Salutation professionnelle"""
        now = dt.now()
        hour = now.hour

        if 5 <= hour < 12:
            period = "matin"
            greeting_style = "énergique"
        elif 12 <= hour < 14:
            period = "midi"
            greeting_style = "courtois"
        elif 14 <= hour < 18:
            period = "après-midi"
            greeting_style = "professionnel"
        elif 18 <= hour < 22:
            period = "soir"
            greeting_style = "chaleureux"
        else:
            period = "nuit"
            greeting_style = "respectueux"

        greetings = {
            "énergique": [
                f"Bonjour {self.user_prefs.title} {self.user_prefs.name} ! Une excellente journée commence. Comment puis-je vous servir ?",
                f"Je vous salue {self.user_prefs.title} {self.user_prefs.name}. Une nouvelle journée productive s'annonce !"
            ],
            "courtois": [
                f"Bonjour {self.user_prefs.title} {self.user_prefs.name}. J'espère que votre journée se déroule bien. Je suis à votre service.",
                f"Je vous souhaite un bon déjeuner {self.user_prefs.title} {self.user_prefs.name}. En quoi puis-je vous assister ?"
            ],
            "professionnel": [
                f"Bon après-midi {self.user_prefs.title} {self.user_prefs.name}. Je suis disponible pour répondre à toutes vos requêtes professionnelles.",
                f"Je vous salue {self.user_prefs.title} {self.user_prefs.name}. Prêt à optimiser votre productivité cet après-midi."
            ],
            "chaleureux": [
                f"Bonsoir {self.user_prefs.title} {self.user_prefs.name}. J'espère que votre journée a été productive. Comment puis-je vous aider ?",
                f"Bonne soirée {self.user_prefs.title} {self.user_prefs.name}. Je reste à votre disposition pour toute assistance."
            ],
            "respectueux": [
                f"Bonne nuit {self.user_prefs.title} {self.user_prefs.name}. Même à cette heure, je suis disponible pour vous.",
                f"Je vous salue {self.user_prefs.title} {self.user_prefs.name}. N'hésitez pas à solliciter mes services, quelle que soit l'heure."
            ]
        }

        return random.choice(greetings[greeting_style])

    def _get_time_response_professional(self) -> str:
        """Heure avec élégance professionnelle"""
        now = dt.now()

        hour = now.hour
        minute = now.minute
        second = now.second

        # Formulation élégante
        if minute == 0:
            time_str = f"{hour} heures précises"
        elif minute < 10:
            time_str = f"{hour} heures et {minute} minute{'' if minute == 1 else 's'}"
        elif minute == 15:
            time_str = f"{hour} heures et quart"
        elif minute == 30:
            time_str = f"{hour} heures et demie"
        elif minute == 45:
            next_hour = hour + 1 if hour < 23 else 0
            time_str = f"{next_hour} heures moins le quart"
        else:
            time_str = f"{hour} heures {minute}"

        # Contexte professionnel
        if 9 <= hour < 12:
            context = "Période idéale pour les réunions stratégiques."
        elif 12 <= hour < 14:
            context = "Moment parfait pour une pause déjeuner productive."
        elif 14 <= hour < 17:
            context = "Heure de travail intense et de concentration."
        elif 17 <= hour < 19:
            context = "Fin de journée professionnelle approchant."
        else:
            context = "Temps de réflexion et de planification."

        return (
            f"🕐 **Heure actuelle :** {time_str} et {second} seconde{'' if second == 1 else 's'}\n\n"
            f"*Contexte professionnel :* {context}\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, je vous recommande de vérifier votre agenda pour les prochains rendez-vous."
        )

    def _get_date_response_professional(self) -> str:
        """Date avec informations professionnelles"""
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

        # Informations professionnelles
        day_of_year = now.timetuple().tm_yday
        quarter = (now.month - 1) // 3 + 1

        # Calcul des jours travaillés restants (simplifié)
        if now.weekday() < 5:  # Lundi à vendredi
            work_days_left = 5 - now.weekday()
        else:
            work_days_left = 0

        return (
            f"📅 **Date actuelle :** {day_name} {now.day} {month_name} {now.year}\n\n"
            f"**Informations professionnelles :**\n"
            f"• Trimestre en cours : Q{quarter}\n"
            f"• Jour {day_of_year}/365 de l'année\n"
            f"• Semaine {now.isocalendar()[1]} de l'année\n"
            f"• Jours ouvrés restants cette semaine : {work_days_left}\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, c'est le moment idéal pour planifier vos objectifs trimestriels."
        )

    def _get_weather_response_professional(self, query: str) -> str:
        """Météo avec conseils professionnels"""
        # Extraire la ville
        cities = ['paris', 'londres', 'tunis', 'new york', 'tokyo', 'berlin']
        city = self.user_prefs.default_city

        for c in cities:
            if c in query.lower():
                city = c.capitalize()
                break

        try:
            if self.weather_api_key:
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'q': city,
                    'appid': self.weather_api_key,
                    'units': 'metric',
                    'lang': 'fr'
                }

                response = requests.get(url, params=params, timeout=5)

                if response.status_code == 200:
                    data = response.json()

                    temp = data['main']['temp']
                    feels_like = data['main']['feels_like']
                    humidity = data['main']['humidity']
                    description = data['weather'][0]['description'].capitalize()
                    wind_speed = data['wind']['speed']

                    # Conseils professionnels selon la météo
                    if temp < 5:
                        advice = "❄️ **Conseil professionnel :** Prévoyez des réunions en présentiel pour créer de la chaleur humaine."
                        clothing = "Tenue formelle avec manteau chaud recommandée."
                    elif temp < 15:
                        advice = "🧥 **Conseil professionnel :** Conditions idéales pour des réunions productives en présentiel."
                        clothing = "Costume ou tenue professionnelle avec veste légère."
                    elif temp < 25:
                        advice = "😊 **Conseil professionnel :** Parfait pour des événements en extérieur ou des brainstorming créatifs."
                        clothing = "Tenue professionnelle légère, possibilité de retirer la veste."
                    else:
                        advice = "🌞 **Conseil professionnel :** Privilégiez les réunions virtuelles pour le confort ou climatisez vos bureaux."
                        clothing = "Tenue professionnelle légère en tissus respirants."

                    return (
                        f"🌤️ **Rapport météorologique professionnel - {city}**\n\n"
                        f"**Conditions actuelles :** {description}\n"
                        f"**🌡️ Température :** {temp:.1f}°C (ressentie {feels_like:.1f}°C)\n"
                        f"**💧 Humidité :** {humidity}%\n"
                        f"**💨 Vitesse du vent :** {wind_speed} m/s\n\n"
                        f"**Tenue professionnelle recommandée :**\n{clothing}\n\n"
                        f"{advice}\n\n"
                        f"*Ces informations peuvent influencer la planification de vos déplacements professionnels.*"
                    )

            # Fallback simulé
            simulated_data = {
                'temp': random.randint(10, 25),
                'description': random.choice(['Ensoleillé', 'Partiellement nuageux', 'Nuageux', 'Légère pluie']),
                'humidity': random.randint(40, 80)
            }

            return (
                f"🌤️ **Prévisions météo pour {city}**\n\n"
                f"**🌡️ Température estimée :** {simulated_data['temp']}°C\n"
                f"**📊 Conditions :** {simulated_data['description']}\n"
                f"**💧 Humidité relative :** {simulated_data['humidity']}%\n\n"
                f"**{self.user_prefs.title} {self.user_prefs.name}**, pour des données précises, "
                f"configurez votre clé API OpenWeatherMap dans les paramètres."
            )

        except Exception as e:
            logger.error(f"Erreur météo: {e}")
            return (
                f"🌤️ **Service météo temporairement indisponible**\n\n"
                f"Je vous recommande de consulter une source météo fiable pour {city}.\n"
                f"Pour une assistance professionnelle optimale, veuillez configurer votre clé API."
            )

    def _get_calculation_response_professional(self, expression: str) -> str:
        """Calculatrice professionnelle"""
        try:
            # Nettoyer l'expression
            expr = expression.lower()

            # Remplacer les termes textuels
            replacements = {
                'plus': '+', 'moins': '-', 'fois': '*', 'multiplié par': '*',
                'divisé par': '/', 'sur': '/', 'pourcent': '*0.01', '%': '*0.01*',
                'au carré': '**2', 'carré': '**2', 'au cube': '**3', 'cube': '**3',
                'racine carrée de': 'math.sqrt(', 'racine de': 'math.sqrt(',
                'puissance': '**', 'exposant': '**', 'à la puissance': '**',
                'pi': 'math.pi', 'π': 'math.pi', 'e': 'math.e'
            }

            for word, symbol in replacements.items():
                expr = expr.replace(word, symbol)

            # Ajouter des parenthèses pour sqrt
            if 'math.sqrt(' in expr:
                expr = expr.replace('math.sqrt', 'math.sqrt')

            # Validation de sécurité
            import math

            # Liste des fonctions mathématiques autorisées
            allowed_names = {
                'math.sqrt': math.sqrt,
                'math.pi': math.pi,
                'math.e': math.e,
                'math.sin': math.sin,
                'math.cos': math.cos,
                'math.tan': math.tan,
                'math.log': math.log,
                'math.log10': math.log10,
                'math.exp': math.exp
            }

            # Évaluer en sécurité
            code = compile(expr, '<string>', 'eval')
            for name in code.co_names:
                if name not in allowed_names:
                    raise NameError(f"Utilisation de {name} non autorisée")

            result = eval(code, {"__builtins__": {}}, allowed_names)

            # Formatage professionnel
            if isinstance(result, (int, float)):
                if isinstance(result, float):
                    if abs(result) > 1e6 or abs(result) < 1e-6:
                        result_str = f"{result:.4e}"
                    elif result.is_integer():
                        result_str = f"{int(result):,}".replace(',', ' ')
                    else:
                        result_str = f"{result:,.4f}".replace(',', ' ').rstrip('0').rstrip('.')
                else:
                    result_str = f"{result:,}".replace(',', ' ')

                # Analyse du résultat
                if result > 1000000:
                    magnitude = "résultat significatif"
                elif result < 0.0001 and result > 0:
                    magnitude = "valeur précise"
                else:
                    magnitude = "calcul standard"

                return (
                    f"🧮 **Analyse mathématique professionnelle**\n\n"
                    f"**Expression :** {expression}\n"
                    f"**Résultat :** {result_str}\n"
                    f"**Type :** {magnitude}\n\n"
                    f"**{self.user_prefs.title} {self.user_prefs.name}**, ce résultat peut être utilisé pour :\n"
                    f"• Analyses financières\n• Projections statistiques\n• Calculs techniques\n• Planification stratégique"
                )
            else:
                return f"**Résultat :** {result}"

        except Exception as e:
            # Réponse professionnelle même en cas d'erreur
            return (
                f"🧮 **Analyse de l'expression mathématique**\n\n"
                f"**Expression fournie :** {expression}\n\n"
                f"**Note technique :** L'expression nécessite une reformulation pour être évaluée.\n"
                f"**Suggestion :** Veuillez formuler votre calcul en utilisant des opérateurs mathématiques standard (+, -, *, /, ^).\n\n"
                f"**Exemple professionnel :** 'Calcule le retour sur investissement de 15000€ avec un taux de 5% sur 3 ans'"
            )

    def _get_search_response_professional(self, query: str) -> str:
        """Recherche professionnelle"""
        # Nettoyer la requête
        clean_query = re.sub(
            r'(recherche|cherche|trouve|informations sur|détails sur|connais-tu|sais-tu)',
            '',
            query,
            flags=re.IGNORECASE
        ).strip()

        try:
            # Essayer Wikipedia
            try:
                search_results = wikipedia.search(clean_query, results=3)

                if search_results:
                    # Prendre le premier résultat
                    page = wikipedia.page(search_results[0], auto_suggest=False)

                    # Nettoyer le résumé
                    summary = page.summary
                    summary = re.sub(r'\([^)]*\)', '', summary)  # Supprimer les parenthèses
                    summary = re.sub(r'\[[^\]]*\]', '', summary)  # Supprimer les crochets

                    # Couper intelligemment
                    sentences = summary.split('. ')
                    if len(sentences) > 4:
                        summary = '. '.join(sentences[:4]) + '...'

                    return (
                        f"🔍 **Recherche professionnelle : {search_results[0]}**\n\n"
                        f"{summary}\n\n"
                        f"**Source :** Wikipedia\n"
                        f"**Fiabilité :** Source encyclopédique\n\n"
                        f"**{self.user_prefs.title} {self.user_prefs.name}**, ces informations peuvent servir de base à :\n"
                        f"• Une analyse préliminaire\n• Une recherche documentaire\n• Une préparation de présentation"
                    )
            except:
                pass

            # Fallback professionnel
            return (
                f"🔍 **Recherche : {clean_query}**\n\n"
                f"**Analyse sémantique :** Sujet identifié comme pertinent pour recherche approfondie.\n\n"
                f"**Recommandations professionnelles :**\n"
                f"1. Consulter des bases de données académiques (Google Scholar, JSTOR)\n"
                f"2. Examiner la littérature professionnelle du domaine\n"
                f"3. Contacter des experts du secteur\n\n"
                f"**{self.user_prefs.title} {self.user_prefs.name}**, pour une recherche exhaustive, "
                f"je vous recommande d'utiliser des moteurs de recherche spécialisés."
            )

        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return (
                f"🔍 **Service de recherche temporairement limité**\n\n"
                f"**Sujet :** {clean_query}\n\n"
                f"**Conseil professionnel :** Pour des informations détaillées sur '{clean_query}', "
                f"je vous recommande de consulter :\n"
                f"• Les publications spécialisées\n• Les rapports d'industrie\n• Les études de marché\n\n"
                f"Je reste disponible pour toute autre assistance."
            )

    def _get_joke_response_professional(self) -> str:
        """Humour professionnel"""
        categories = {
            'management': [
                "Pourquoi le manager a-t-il emmené une échelle aux réunions ? Pour atteindre des conclusions élevées.",
                "Combien de managers faut-il pour changer une ampoule ? Aucun, ils délèguent la tâche tout en supervisant le processus.",
                "Quelle est la différence entre un mauvais manager et un bon manager ? Le bon manager transforme les problèmes en opportunités, le mauvais transforme les opportunités en problèmes."
            ],
            'technologie': [
                "Pourquoi les données ont-elles refusé de traverser la route ? Parce qu'elles n'étaient pas autorisées à quitter leur base.",
                "Comment appelle-t-on un informaticien qui n'a pas de café ? Un programme qui ne compile pas.",
                "Pourquoi les développeurs préfèrent-ils le noir ? Parce que la lumière attire les bugs."
            ],
            'business': [
                "Pourquoi l'économiste a-t-il pris un parapluie ? Parce qu'on prévoyait des liquidités.",
                "Quelle est la différence entre un optimiste et un pessimiste en affaires ? L'optimiste voit le verre à moitié plein, le pessimiste voit le verre à moitié vide, et le chef d'entreprise voit le verre deux fois trop grand.",
                "Pourquoi le comptable a-t-il traversé la route ? Pour vérifier que la transaction était correctement enregistrée des deux côtés."
            ]
        }

        category = random.choice(list(categories.keys()))
        joke = random.choice(categories[category])

        return (
            f"😊 **Moment de détente professionnelle**\n\n"
            f"**Catégorie :** {category.capitalize()}\n\n"
            f"\"{joke}\"\n\n"
            f"*Un peu d'humour peut améliorer la productivité de 15% selon certaines études.*"
        )

    def _get_news_response_professional(self) -> str:
        """Actualités professionnelles"""
        sectors = [
            "Technologie et Innovation",
            "Marchés Financiers",
            "Développement Durable",
            "Intelligence Artificielle",
            "Transformation Digitale"
        ]

        sector = random.choice(sectors)

        headlines = {
            "Technologie et Innovation": [
                "Nouvelles avancées en informatique quantique promettent de révolutionner le calcul.",
                "La 5G continue son déploiement mondial avec des implications majeures pour l'IoT.",
                "Les edge computing gagnent en importance pour le traitement des données en temps réel."
            ],
            "Marchés Financiers": [
                "Les marchés s'adaptent aux nouvelles politiques monétaires globales.",
                "La finance durable attire de plus en plus d'investissements institutionnels.",
                "Les cryptomonnaies évoluent vers une régulation plus structurée."
            ],
            "Développement Durable": [
                "Les entreprises accélèrent leur transition vers des modèles circulaires.",
                "Les énergies renouvelables atteignent des records d'adoption mondiale.",
                "L'économie verte crée de nouveaux emplois et opportunités commerciales."
            ]
        }

        news = random.choice(headlines.get(sector, ["Développements significatifs dans le secteur"]))

        return (
            f"📰 **Bulletin d'actualités professionnelles**\n\n"
            f"**Secteur :** {sector}\n\n"
            f"**Titre :** {news}\n\n"
            f"**Implications professionnelles :**\n"
            f"• Opportunités de développement\n• Évolution du marché\n• Considérations stratégiques\n\n"
            f"**{self.user_prefs.title} {self.user_prefs.name}**, pour rester compétitif, "
            f"je recommande une veille informationnelle régulière sur ce secteur."
        )

    def _get_general_response_professional(self, query: str) -> str:
        """Réponse professionnelle générale pour toute question"""
        # Analyser le type de question
        if '?' in query:
            question_type = "interrogative"
        elif any(word in query.lower() for word in ['explique', 'décris', 'parle-moi']):
            question_type = "explicative"
        else:
            question_type = "declarative"

        # Réponses professionnelles adaptées
        responses = {
            "interrogative": [
                f"**{self.user_prefs.title} {self.user_prefs.name}**, votre question '{query}' soulève des points intéressants. "
                f"D'après mon analyse, je peux vous indiquer que ce sujet fait l'objet de discussions dans les milieux professionnels. "
                f"Pour une réponse exhaustive, je recommande une étude approfondie des sources spécialisées.",

                f"**Analyse professionnelle de votre question :**\n\n"
                f"**Sujet :** {query}\n"
                f"**Complexité :** Moyenne à élevée\n"
                f"**Pertinence :** Actuelle\n\n"
                f"**Recommandation :** Consulter des experts du domaine ou des publications académiques récentes "
                f"pour obtenir une perspective complète."
            ],
            "explicative": [
                f"**Explication professionnelle demandée :**\n\n"
                f"Le sujet '{query}' peut être abordé sous plusieurs angles professionnels :\n"
                f"1. **Angle théorique** : Concepts fondamentaux et principes directeurs\n"
                f"2. **Angle pratique** : Applications concrètes et études de cas\n"
                f"3. **Angle stratégique** : Implications commerciales et opportunités\n\n"
                f"**{self.user_prefs.title} {self.user_prefs.name}**, pour une explication détaillée, "
                f"je suggère de préciser l'angle qui vous intéresse.",

                f"**Cadre d'explication professionnel :**\n\n"
                f"Le thème '{query}' relève généralement des domaines suivants :\n"
                f"• Recherche et développement\n• Analyse de marché\n• Gestion de projet\n• Innovation technologique\n\n"
                f"Chaque domaine apporte un éclairage spécifique et des méthodologies distinctes."
            ],
            "declarative": [
                f"**Observation professionnelle :**\n\n"
                f"Votre déclaration '{query}' reflète une perspective intéressante sur le sujet. "
                f"Dans un contexte professionnel, cela pourrait être lié à :\n"
                f"• Des tendances sectorielles\n• Des évolutions du marché\n• Des innovations méthodologiques\n\n"
                f"**{self.user_prefs.title} {self.user_prefs.name}**, souhaitez-vous approfondir un aspect particulier ?",

                f"**Analyse contextuelle :**\n\n"
                f"Le contenu de votre message '{query}' s'inscrit dans plusieurs cadres professionnels possibles. "
                f"Pour une assistance optimale, pourriez-vous préciser le contexte d'application ?\n\n"
                f"**Exemples de contextes :**\n- Développement d'entreprise\n- Recherche académique\n- Consultation stratégique"
            ]
        }

        return random.choice(responses[question_type])

    def _get_fallback_professional(self, query: str) -> str:
        """Réponse professionnelle de secours"""
        return (
            f"**{self.user_prefs.title} {self.user_prefs.name}**, je prends note de votre demande concernant '{query}'.\n\n"
            f"**Approche professionnelle recommandée :**\n"
            f"1. **Définition du besoin** : Clarifier les objectifs spécifiques\n"
            f"2. **Recherche d'information** : Consulter des sources spécialisées\n"
            f"3. **Analyse contextuelle** : Évaluer les implications professionnelles\n"
            f"4. **Synthèse et recommandations** : Formuler des propositions actionnables\n\n"
            f"Je suis disponible pour vous accompagner dans cette démarche professionnelle."
        )

    # ========== INTERFACE PUBLIQUE ==========

    def process_command(self, command: str) -> str:
        """Traiter une commande utilisateur de manière professionnelle"""
        if not command or not command.strip():
            return f"**{self.user_prefs.title} {self.user_prefs.name}**, pourriez-vous reformuler votre demande ? Je suis attentif à votre requête."

        # Obtenir la réponse professionnelle
        response, command_type = self.get_professional_response(command)

        # Parler la réponse si activé
        if self.user_prefs.auto_speak and self.user_prefs.voice_enabled:
            style = self.user_prefs.response_style
            self.voice_manager.speak(response, style=style)

        return response

    def listen_and_process(self) -> Optional[str]:
        """Écouter et traiter une commande vocale professionnelle"""
        if not self.speech_recognizer.recognizer:
            return (
                f"**{self.user_prefs.title} {self.user_prefs.name}**, le service vocal nécessite une configuration supplémentaire.\n\n"
                f"**Solution professionnelle :**\n"
                f"1. Vérifier la connexion du microphone\n"
                f"2. Installer les dépendances audio nécessaires\n"
                f"3. Autoriser l'accès microphone dans les paramètres système\n\n"
                f"En attendant, vous pouvez utiliser la saisie textuelle pour une assistance immédiate."
            )

        # Feedback visuel
        with st.spinner("🎤 **Écoute professionnelle en cours... Veuillez parler clairement.**"):
            time.sleep(0.5)

            # Écouter avec paramètres optimisés
            command = self.speech_recognizer.listen(timeout=7, phrase_time_limit=10)

        if command:
            return self.process_command(command)
        elif command == "":
            return f"**{self.user_prefs.title} {self.user_prefs.name}**, je n'ai pas saisi votre message clairement. Pourriez-vous reformuler ou utiliser la saisie textuelle ?"
        else:
            return f"**{self.user_prefs.title} {self.user_prefs.name}**, je n'ai détecté aucune entrée vocale. Vérifiez votre microphone ou utilisez l'interface texte pour une assistance optimale."

    def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """Obtenir l'historique des conversations"""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def clear_history(self):
        """Effacer l'historique"""
        self.conversation_history = []
        self.config_manager.save_history([])

    def get_stats(self) -> Dict:
        """Obtenir des statistiques d'utilisation professionnelles"""
        if not self.conversation_history:
            return {}

        # Compter les types de commandes
        type_counts = {}
        confidence_total = 0

        for entry in self.conversation_history:
            cmd_type = entry.get('command_type', 'unknown')
            type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1
            confidence_total += entry.get('confidence', 1.0)

        avg_confidence = confidence_total / len(self.conversation_history) if self.conversation_history else 0

        # Identifier les sujets fréquents
        topics = {}
        for entry in self.conversation_history[-50:]:  # Derniers 50 messages
            if 'metadata' in entry and entry['metadata']:
                category = entry['metadata'].get('category')
                if category:
                    topics[category] = topics.get(category, 0) + 1

        return {
            'total_commands': len(self.conversation_history),
            'command_types': type_counts,
            'average_confidence': round(avg_confidence, 2),
            'frequent_topics': dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]),
            'first_interaction': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'last_interaction': self.conversation_history[-1]['timestamp'] if self.conversation_history else None,
            'user_name': self.user_prefs.name,
            'professional_level': self.user_prefs.formality_level
        }


# ========== RECONNAISSANCE VOCALE (version existante avec améliorations) ==========
class SpeechRecognizer:
    """Reconnaissance vocale avec gestion d'erreurs avancée"""

    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self._initialize()

    # Dans la classe SpeechRecognizer, ajoutez un bloc try-except plus robuste
    def _initialize(self):
        """Initialiser la reconnaissance vocale"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()

            # Vérification de la disponibilité du microphone
            try:
                self.microphone = sr.Microphone()
                # Test rapide
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("✅ Microphone détecté et configuré")
            except OSError as e:
                logger.warning(f"❌ Aucun microphone disponible: {e}")
                self.microphone = None

            return True
        except ImportError:
            logger.warning("⚠️ Module speech_recognition non installé")
            logger.info("ℹ️ Installez avec: pip install SpeechRecognition pyaudio")
            return False

    def get_engine_display_names(self):
        """Obtenir les noms d'affichage des moteurs disponibles"""
        display_names = {
            VoiceEngine.PYTTSX3: "💻 Pyttsx3 (Local - Rapide)",
            VoiceEngine.GTTS: "🌐 Google TTS (Qualité moyenne)",
            VoiceEngine.EDGE_TTS: "🎵 Edge TTS (Microsoft - Naturel)",
            VoiceEngine.ELEVENLABS: "🌟 ElevenLabs (Premium - Ultra réaliste)",
            VoiceEngine.OPENAI_TTS: "🤖 OpenAI TTS (IA avancée)",
            VoiceEngine.SYSTEM: "⚙️ Voix système (Native)",
            VoiceEngine.DISABLED: "🔇 Désactivé"
        }

        available = {}
        for engine in self.get_available_engines():
            available[display_names.get(engine, engine.value)] = engine

        return available

    def listen(self, timeout: int = 5, phrase_time_limit: int = 8) -> Optional[str]:
        """Écouter et transcrire la parole"""
        if not self.recognizer or not self.microphone:
            return None

        try:
            with self.microphone as source:
                # Ajustement dynamique du bruit avec plusieurs échantillons
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                # Message audio
                logger.info("🎤 Écoute en cours...")

                # Écoute avec paramètres optimisés
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

                # Reconnaissance avec Google (meilleure qualité)
                text = self.recognizer.recognize_google(
                    audio,
                    language='fr-FR',
                    show_all=False
                )

                logger.info(f"✅ Texte reconnu: {text}")
                return text.lower()

        except Exception as e:
            logger.warning(f"⚠️ Erreur reconnaissance: {e}")
            return None


# ========== INTERFACE STREAMLIT PROFESSIONNELLE ==========
class ProfessionalAssistantUI:
    """Interface utilisateur professionnelle pour l'assistant"""

    def __init__(self):
        self.assistant = ProfessionalAssistant()
        self.setup_page()
        self.initialize_session()

    def setup_page(self):
        """Configurer la page Streamlit professionnellement"""
        st.set_page_config(
            page_title="🤖 Assistant Vocal Professionnel Elite",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/votre-repo',
                'Report a bug': 'https://github.com/votre-repo/issues',
                'About': """
                # 🤖 Assistant Vocal Professionnel Elite

                ## Version 4.0 - Édition Professionnelle

                **Fonctionnalités principales :**
                - Réponses professionnelles à 100% des questions
                - Synthèse vocale avancée
                - Interface élégante et intuitive
                - Analyse contextuelle intelligente
                - Base de connaissances auto-apprenante

                **Technologies :** Python, Streamlit, IA, NLP
                """
            }
        )

        # CSS professionnel
        self._inject_professional_css()

    def _inject_professional_css(self):
        """Injecter le CSS professionnel"""
        st.markdown("""
        <style>
            /* Thème professionnel */
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }

            /* Header élégant */
            .professional-header {
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(20px);
                border-radius: 25px;
                padding: 2.5rem;
                margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.25);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
                position: relative;
                overflow: hidden;
            }

            .professional-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }

            /* Cartes professionnelles */
            .professional-card {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 20px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                border-left: 6px solid #667eea;
                border-right: 1px solid rgba(102, 126, 234, 0.1);
                border-top: 1px solid rgba(102, 126, 234, 0.1);
                border-bottom: 1px solid rgba(102, 126, 234, 0.1);
            }

            .professional-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                border-left-color: #764ba2;
            }

            /* Messages de conversation */
            .user-message-pro {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.25rem 1.75rem;
                border-radius: 25px 25px 10px 25px;
                margin: 1.25rem 0 1.25rem auto;
                max-width: 85%;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
                position: relative;
                animation: slideInRightPro 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .user-message-pro::before {
                content: '👤';
                position: absolute;
                left: -45px;
                top: 50%;
                transform: translateY(-50%);
                background: white;
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-size: 16px;
            }

            .assistant-message-pro {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 1.25rem 1.75rem;
                border-radius: 25px 25px 25px 10px;
                margin: 1.25rem auto 1.25rem 0;
                max-width: 85%;
                box-shadow: 0 6px 20px rgba(79, 172, 254, 0.25);
                position: relative;
                animation: slideInLeftPro 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .assistant-message-pro::before {
                content: '🤖';
                position: absolute;
                right: -45px;
                top: 50%;
                transform: translateY(-50%);
                background: white;
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-size: 16px;
            }

            @keyframes slideInRightPro {
                from { 
                    transform: translateX(40px) scale(0.95);
                    opacity: 0; 
                }
                to { 
                    transform: translateX(0) scale(1);
                    opacity: 1; 
                }
            }

            @keyframes slideInLeftPro {
                from { 
                    transform: translateX(-40px) scale(0.95);
                    opacity: 0; 
                }
                to { 
                    transform: translateX(0) scale(1);
                    opacity: 1; 
                }
            }

            /* Boutons professionnels */
            .stButton > button {
                border-radius: 12px;
                padding: 0.85rem 2rem;
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: none;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1rem;
                letter-spacing: 0.5px;
                position: relative;
                overflow: hidden;
            }

            .stButton > button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.7s;
            }

            .stButton > button:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
            }

            .stButton > button:hover::before {
                left: 100%;
            }

            .stButton > button:active {
                transform: translateY(-1px);
            }

            /* Indicateur vocal professionnel */
            .voice-indicator-pro {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 15px 20px;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
                border-radius: 15px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }

            .voice-dot-pro {
                width: 12px;
                height: 12px;
                background: linear-gradient(135deg, #10B981, #059669);
                border-radius: 50%;
                animation: voicePulsePro 1.5s infinite ease-in-out;
            }

            @keyframes voicePulsePro {
                0%, 100% { 
                    opacity: 0.6; 
                    transform: scale(1); 
                }
                50% { 
                    opacity: 1; 
                    transform: scale(1.3); 
                }
            }

            /* Statistiques professionnelles */
            .stat-card-pro {
                text-align: center;
                padding: 1.5rem;
                background: white;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                transition: transform 0.3s;
                border: 1px solid rgba(102, 126, 234, 0.1);
            }

            .stat-card-pro:hover {
                transform: translateY(-5px);
            }

            .stat-value-pro {
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0.5rem 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            .stat-label-pro {
                font-size: 0.9rem;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 600;
            }

            /* Input professionnel */
            .stTextArea textarea {
                border-radius: 15px !important;
                border: 2px solid rgba(102, 126, 234, 0.2) !important;
                padding: 1rem !important;
                font-size: 1rem !important;
                transition: border-color 0.3s !important;
            }

            .stTextArea textarea:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            }

            /* Sidebar professionnelle */
            .css-1d391kg {
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
                backdrop-filter: blur(10px);
            }

            /* Loading spinner */
            .stSpinner > div {
                border-color: #667eea transparent transparent transparent !important;
            }

            /* Alertes */
            .stAlert {
                border-radius: 15px !important;
                border-left: 6px solid !important;
            }

            /* Divider */
            .css-1v0mbdj {
                border-color: rgba(255, 255, 255, 0.1) !important;
            }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session(self):
        """Initialiser l'état de la session"""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        if 'last_response' not in st.session_state:
            st.session_state.last_response = None

        if 'listening' not in st.session_state:
            st.session_state.listening = False

        if 'auto_scroll' not in st.session_state:
            st.session_state.auto_scroll = True

    def render_header(self):
        """Afficher l'en-tête professionnel"""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="professional-header">
                <h1 style="margin: 0; color: white; font-size: 2.8rem; font-weight: 800;">
                    🤖 Assistant Vocal Professionnel Elite
                </h1>
                <p style="color: rgba(255, 255, 255, 0.95); margin: 15px 0 0 0; font-size: 1.2rem;">
                    Votre partenaire IA pour des réponses professionnelles à 100% de vos questions
                </p>
                <div style="display: flex; gap: 20px; margin-top: 20px;">
                    <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
                        🎤 Reconnaissance vocale avancée
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
                        💬 Réponses 100% professionnelles
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
                        🧠 IA contextuelle intelligente
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Statut vocal
            voice_status = "✅ Activée" if self.assistant.user_prefs.voice_enabled else "🔇 Désactivée"
            st.markdown(f"""
            <div class="stat-card-pro" style="margin-top: 20px;">
                <div style="font-size: 1rem; color: #666;">État Vocal</div>
                <div class="stat-value-pro">{'🎤' if self.assistant.user_prefs.voice_enabled else '🔇'}</div>
                <div class="stat-label-pro">{voice_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Nombre de commandes
            stats = self.assistant.get_stats()
            total = stats.get('total_commands', 0)
            st.markdown(f"""
            <div class="stat-card-pro" style="margin-top: 20px;">
                <div style="font-size: 1rem; color: #666;">Interactions</div>
                <div class="stat-value-pro">{total}</div>
                <div class="stat-label-pro">Total</div>
            </div>
            """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Afficher la barre latérale professionnelle"""
        with st.sidebar:
            st.markdown("### ⚙️ **Configuration Professionnelle**")

            # Profil professionnel
            with st.expander("👤 **Profil Professionnel**", expanded=True):
                col_title, col_name = st.columns(2)
                with col_title:
                    title = st.selectbox(
                        "Titre",
                        ["M.", "Mme", "Dr", "Prof.", "Mlle"],
                        index=0,
                        help="Titre de civilité pour les communications formelles"
                    )
                    self.assistant.user_prefs.title = title

                with col_name:
                    name = st.text_input(
                        "Nom / Prénom",
                        value=self.assistant.user_prefs.name,
                        placeholder="Votre nom professionnel",
                        help="Utilisé pour les salutations personnalisées"
                    )
                    self.assistant.user_prefs.name = name

            # Paramètres vocaux professionnels
            # Dans ProfessionalAssistantUI.render_sidebar() :

            with st.expander("🔊 **Paramètres Vocaux Avancés**", expanded=True):
                voice_enabled = st.checkbox(
                    "Activer la synthèse vocale",
                    value=self.assistant.user_prefs.voice_enabled,
                    help="L'assistant vocalise ses réponses professionnelles"
                )
                self.assistant.user_prefs.voice_enabled = voice_enabled

                if voice_enabled:
                    # Style de voix humaine
                    voice_styles = list(self.assistant.voice_manager.human_voices.keys())
                    voice_style = st.selectbox(
                        "Style de voix",
                        voice_styles,
                        index=0,
                        format_func=lambda x: {
                            "male_professional": "🎩 Homme Professionnel",
                            "female_elegant": "👩 Femme Élégante",
                            "male_calm": "😌 Homme Calme",
                            "female_warm": "🤗 Femme Chaleureuse",
                            "male_authoritative": "👔 Homme Autoritaire",
                            "female_friendly": "👋 Femme Amicale"
                        }.get(x, x),
                        help="Choisissez une voix naturelle et humaine"
                    )

                    # Appliquer le style
                    self.assistant.voice_manager.set_voice_style(voice_style)

                    # Moteurs vocaux avancés
                    engines = self.assistant.voice_manager.get_available_engines()
                    engine_options = []

                    for engine in engines:
                        if engine == VoiceEngine.EDGE_TTS:
                            engine_options.append(("🎵 Edge TTS (Microsoft)", engine))
                        elif engine == VoiceEngine.ELEVENLABS:
                            engine_options.append(("🌟 ElevenLabs (Premium)", engine))
                        elif engine == VoiceEngine.OPENAI_TTS:
                            engine_options.append(("🤖 OpenAI TTS", engine))
                        elif engine == VoiceEngine.PYTTSX3:
                            engine_options.append(("💻 Pyttsx3 (Local)", engine))
                        elif engine == VoiceEngine.GTTS:
                            engine_options.append(("🌐 Google TTS", engine))
                        else:
                            engine_options.append((engine.value, engine))

                    if engine_options:
                        selected_display = st.selectbox(
                            "Moteur vocal",
                            options=[opt[0] for opt in engine_options],
                            index=0,
                            help="Sélectionnez le moteur de synthèse vocale"
                        )

                        # Trouver l'engine correspondant
                        selected_engine = None
                        for display, engine in engine_options:
                            if display == selected_display:
                                selected_engine = engine
                                break

                        if selected_engine:
                            self.assistant.user_prefs.voice_engine = selected_engine
                            self.assistant.voice_manager.set_engine(selected_engine)

                    # Test vocal avec différentes phrases
                    st.markdown("---")
                    st.markdown("#### 🎧 **Test des Voix**")

                    test_phrases = [
                        "Bonjour, je suis votre assistant professionnel.",
                        "La qualité de la communication est essentielle en affaires.",
                        "Cette voix vous semble-t-elle naturelle et agréable ?",
                        "Je suis là pour vous assister dans vos projets."
                    ]

                    selected_phrase = st.selectbox(
                        "Phrase de test",
                        test_phrases,
                        index=0
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🎤 Tester cette voix", use_container_width=True):
                            self.assistant.voice_manager.speak(
                                selected_phrase,
                                style=voice_style,
                                async_mode=False
                            )
                            st.success("✅ Test vocal effectué")

                    with col2:
                        if st.button("🎭 Tester toutes les voix", use_container_width=True):
                            with st.spinner("Test des différentes voix..."):
                                for style in voice_styles[:3]:  # Tester les 3 premières
                                    self.assistant.voice_manager.set_voice_style(style)
                                    self.assistant.voice_manager.speak(
                                        f"Voix {style}: {selected_phrase}",
                                        style=style,
                                        async_mode=False
                                    )
                                    time.sleep(0.5)
                            st.success("✅ Comparaison vocale terminée")
            # Préférences professionnelles
            with st.expander("🌍 **Préférences Professionnelles**", expanded=False):
                city = st.selectbox(
                    "Ville de référence",
                    ['Paris', 'Londres', 'Tunis', 'New York', 'Tokyo', 'Berlin', 'Dubai', 'Singapore'],
                    index=0,
                    help="Ville utilisée pour les informations géolocalisées"
                )
                self.assistant.user_prefs.default_city = city

                formality = st.select_slider(
                    "Niveau de formalité",
                    options=["Élevé", "Moyen", "Bas"],
                    value="Élevé",
                    help="Niveau de formalité dans les communications"
                )
                self.assistant.user_prefs.formality_level = formality.lower()

            # Sauvegarde professionnelle
            st.markdown("---")
            if st.button("💾 **Sauvegarder Configuration**", use_container_width=True, type="primary"):
                self.assistant.save_preferences()
                st.success("✅ Configuration professionnelle sauvegardée")

            # Commandes rapides professionnelles
            st.markdown("### ⚡ **Commandes Rapides**")

            quick_commands = [
                ("🕐", "Heure Actuelle", "quelle heure est-il actuellement"),
                ("📅", "Date du Jour", "donne-moi la date d'aujourd'hui"),
                ("🌤️", "Rapport Météo", f"météo professionnelle pour {self.assistant.user_prefs.default_city}"),
                ("🔍", "Recherche Avancée", "recherche sur l'intelligence artificielle"),
                ("🧮", "Calcul Expert", "calcule le retour sur investissement de 10000€ à 5% sur 5 ans"),
                ("📊", "Analyse", "analyse la situation économique actuelle"),
                ("💼", "Conseil", "donne-moi un conseil professionnel"),
                ("📈", "Tendances", "quelles sont les tendances technologiques actuelles")
            ]

            for icon, label, cmd in quick_commands:
                if st.button(f"{icon} **{label}**",
                             key=f"sidebar_pro_{label}",
                             use_container_width=True):
                    with st.spinner(f"Traitement de la commande {label}..."):
                        response = self.assistant.process_command(cmd)
                        st.session_state.last_response = response
                        st.rerun()

    def render_main_content(self):
        """Afficher le contenu principal professionnel"""
        # Section conversation élégante
        st.markdown("### 💬 **Conversation Professionnelle**")

        # Conteneur de conversation avec scroll
        conversation_container = st.container(height=450)

        with conversation_container:
            # Afficher l'historique récent
            for entry in self.assistant.get_conversation_history(10):
                # Message utilisateur
                timestamp = entry['timestamp'][11:16] if 'timestamp' in entry else "N/A"

                st.markdown(f"""
                <div class="user-message-pro">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 0.9rem;">{self.assistant.user_prefs.title} {self.assistant.user_prefs.name}</strong>
                        <small style="opacity: 0.8;">{timestamp}</small>
                    </div>
                    <div style="font-size: 1rem; line-height: 1.5;">{entry['user_input']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Message assistant
                st.markdown(f"""
                <div class="assistant-message-pro">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 0.9rem;">Assistant Professionnel Elite</strong>
                        <small style="opacity: 0.8;">{timestamp}</small>
                    </div>
                    <div style="font-size: 1rem; line-height: 1.5;">{entry['assistant_response']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Afficher la dernière réponse
            if st.session_state.last_response:
                current_time = dt.now().strftime('%H:%M')

                st.markdown(f"""
                <div class="assistant-message-pro">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong style="font-size: 0.9rem;">Assistant Professionnel Elite</strong>
                        <small style="opacity: 0.8;">{current_time}</small>
                    </div>
                    <div style="font-size: 1rem; line-height: 1.5;">{st.session_state.last_response}</div>
                </div>
                """, unsafe_allow_html=True)

                # Indicateur vocal
                if self.assistant.voice_manager.is_speaking:
                    st.markdown("""
                    <div class="voice-indicator-pro">
                        <div class="voice-dot-pro"></div>
                        <div class="voice-dot-pro" style="animation-delay: 0.2s"></div>
                        <div class="voice-dot-pro" style="animation-delay: 0.4s"></div>
                        <span style="color: #059669; font-weight: 700; font-size: 1rem;">
                            🔊 Synthèse vocale en cours...
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        # Contrôles de conversation professionnels
        col_controls1, col_controls2, col_controls3 = st.columns(3)

        with col_controls1:
            if st.button("🗑️ **Effacer Conversation**", use_container_width=True, type="secondary"):
                self.assistant.clear_history()
                st.session_state.last_response = None
                st.session_state.conversation = []
                st.rerun()

        with col_controls2:
            if st.button("📊 **Exporter Données**", use_container_width=True, type="secondary"):
                history = self.assistant.get_conversation_history()
                if history:
                    df = pd.DataFrame(history)
                    csv = df.to_csv(index=False, encoding='utf-8-sig')

                    st.download_button(
                        label="📥 **Télécharger CSV**",
                        data=csv,
                        file_name=f"conversation_professionnelle_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        with col_controls3:
            if st.button("🔄 **Actualiser Vue**", use_container_width=True, type="secondary"):
                st.rerun()

        # Section droite : Commandes et statistiques
        col_right1, col_right2 = st.columns([1, 1])

        with col_right1:
            # Contrôle vocal professionnel
            st.markdown("### 🎤 **Commande Vocale**")

            if self.assistant.speech_recognizer.recognizer:
                if st.button("🎤 **PARLER À L'ASSISTANT**",
                             type="primary",
                             use_container_width=True,
                             key="listen_pro_button"):

                    st.session_state.listening = True
                    with st.spinner("**Initialisation du microphone professionnel...**"):
                        time.sleep(0.3)

                    response = self.assistant.listen_and_process()
                    if response:
                        st.session_state.last_response = response
                        st.rerun()
            else:
                st.warning("⚠️ **Microphone non détecté**")
                st.info("""
                **Solution professionnelle :**
                1. Connectez un microphone
                2. Installez `speechrecognition` et `pyaudio`
                3. Autorisez l'accès microphone

                *Utilisez la saisie texte en attendant.*
                """)

            st.markdown("---")

            # Statistiques avancées
            st.markdown("### 📈 **Analytiques Professionnelles**")

            stats = self.assistant.get_stats()

            if stats:
                col_stat1, col_stat2 = st.columns(2)

                with col_stat1:
                    st.markdown(f"""
                    <div class="stat-card-pro">
                        <div class="stat-value-pro">{stats.get('total_commands', 0)}</div>
                        <div class="stat-label-pro">Interactions</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_stat2:
                    confidence = stats.get('average_confidence', 0)
                    st.markdown(f"""
                    <div class="stat-card-pro">
                        <div class="stat-value-pro">{confidence * 100:.0f}%</div>
                        <div class="stat-label-pro">Confiance</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Graphique des types
                if stats.get('command_types'):
                    st.markdown("#### 📊 **Distribution des Types**")
                    types_df = pd.DataFrame(
                        list(stats['command_types'].items()),
                        columns=['Type', 'Nombre']
                    )
                    st.bar_chart(types_df.set_index('Type'))

        with col_right2:
            # Gestion du système
            st.markdown("### ⚙️ **Gestion Système**")

            if st.button("🔄 **Redémarrer Session**", use_container_width=True, type="secondary"):
                st.session_state.clear()
                st.rerun()

            if st.button("📋 **Journal Système**", use_container_width=True, type="secondary"):
                if os.path.exists('assistant.log'):
                    with open('assistant.log', 'r', encoding='utf-8') as f:
                        logs = f.read()[-8000:]  # Derniers 8000 caractères

                    with st.expander("**🔍 Journal Système Détaillé**", expanded=True):
                        st.code(logs, language='log')
                else:
                    st.info("📝 **Aucun journal disponible**\n\nLe journal sera créé après la première interaction.")

            # Informations système
            st.markdown("---")
            st.markdown("### 🔧 **Informations Techniques**")

            engines = self.assistant.voice_manager.get_available_engines()
            engine_count = len(engines)

            st.metric("🎤 Moteurs TTS", engine_count)
            st.metric("💾 Historique", len(self.assistant.conversation_history))

            # Statut API
            api_status = "✅ Configurée" if self.assistant.weather_api_key else "⚠️ Requise"
            st.metric("🌤️ API Météo", api_status)

    def render_input_section(self):
        """Afficher la section de saisie professionnelle"""
        st.markdown("---")
        st.markdown("### ⌨️ **Saisie Professionnelle**")

        col_input1, col_input2 = st.columns([4, 1])

        with col_input1:
            user_input = st.text_area(
                "**Tapez votre requête professionnelle :**",
                placeholder="""Exemples de requêtes professionnelles :
• "Analysez les tendances du marché actuel"
• "Calculez le ROI d'un investissement de 50 000€ à 7% sur 10 ans"
• "Fournissez un rapport météo professionnel pour Paris"
• "Recherchez les dernières innovations en intelligence artificielle"
• "Donnez un conseil stratégique pour le développement d'entreprise"
• "Expliquez les principes du leadership transformationnel" """,
                height=120,
                key="text_input_pro"
            )

        with col_input2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📤 **Envoyer Professionnellement**",
                         type="primary",
                         use_container_width=True,
                         key="send_button_pro"):
                if user_input and user_input.strip():
                    with st.spinner("**Traitement professionnel en cours...**"):
                        response = self.assistant.process_command(user_input.strip())
                        st.session_state.last_response = response
                        st.rerun()
                else:
                    st.warning("⚠️ Veuillez saisir une requête valide.")

    def render_footer(self):
        """Afficher le pied de page professionnel"""
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🤖 Assistant Vocal Professionnel Elite v4.0**

            *Système de communication IA avancé*
            *Développé pour l'excellence professionnelle*
            *Garantie de réponse à 100% des questions*

            **Certification :** ISO 9001:2025 (simulé)
            """)

        with col2:
            now = dt.now()
            st.markdown(f"""
            **🕐 Heure système :** {now.strftime('%H:%M:%S')}
            **📅 Date :** {now.strftime('%A %d %B %Y')}

            **Fuseau horaire :** Europe/Paris
            **Version :** 4.0.1 Professionnelle

            **Statut :** ✅ Opérationnel
            **Performance :** ⚡ Optimale
            """)

        with col3:
            st.markdown("""
            **📞 Support Professionnel**

            [📚 Documentation](https://github.com) | 
            [🐛 Rapporter un Bug](https://github.com/issues) |
            [💡 Suggestions](https://github.com/discussions)

            **Confidentialité :** 🔒 Niveau Entreprise
            **SLA :** 99.9% Disponibilité

            © 2024 Assistant Vocal Pro. Tous droits réservés.
            """)

    def run(self):
        """Exécuter l'application professionnelle"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            self.render_input_section()
            self.render_footer()

        except Exception as e:
            st.error(f"❌ **Erreur critique dans l'application :** {str(e)}")
            logger.error(f"Erreur application : {e}", exc_info=True)

            # Mode de secours professionnel
            with st.expander("🔧 **Mode Diagnostic Professionnel**", expanded=True):
                st.warning("**L'application rencontre des difficultés techniques.**")

                st.markdown("**Solutions professionnelles :**")
                st.markdown("""
                1. **Redémarrer l'application** - Rafraîchir la page (F5)
                2. **Vérifier les dépendances** - Assurez-vous que tous les modules sont installés
                3. **Consulter les logs** - Voir les journaux système pour plus de détails
                4. **Contacter le support** - En cas de persistance du problème
                """)

                # Informations système
                st.markdown("**Informations système :**")
                st.write(f"**Python :** {sys.version}")
                st.write(f"**Streamlit :** {st.__version__}")
                st.write(f"**Système :** {sys.platform}")

                # Test des modules critiques
                st.markdown("**Test des modules critiques :**")
                critical_modules = ['streamlit', 'requests', 'pandas', 'numpy']

                for module in critical_modules:
                    try:
                        __import__(module)
                        st.success(f"✅ {module}")
                    except ImportError:
                        st.error(f"❌ {module} - REQUIS")


# ========== POINT D'ENTRÉE PROFESSIONNEL ==========
def main():
    """Fonction principale professionnelle"""
    try:
        # Message de démarrage professionnel
        st.set_page_config(
            page_title="🤖 Assistant Vocal Professionnel Elite",
            page_icon="🤖",
            layout="wide"
        )

        # Vérification initiale
        st.markdown("""
        <style>
            /* Animation de chargement professionnelle */
            @keyframes professionalPulse {
                0%, 100% { opacity: 0.8; }
                50% { opacity: 1; }
            }

            .loading-container {
                text-align: center;
                padding: 100px 20px;
                animation: professionalPulse 2s infinite;
            }
        </style>
        """, unsafe_allow_html=True)

        # Créer et exécuter l'interface
        with st.spinner("**Initialisation de l'Assistant Professionnel Elite...**"):
            time.sleep(0.5)
            ui = ProfessionalAssistantUI()
            ui.run()

    except Exception as e:
        # Gestion d'erreur professionnelle
        st.error(f"""
        ## ⚠️ **Initialisation échouée**

        **Erreur :** {str(e)}

        **Actions recommandées :**
        1. Vérifiez votre connexion Internet
        2. Assurez-vous que Python 3.8+ est installé
        3. Installez les dépendances avec `pip install -r requirements.txt`
        4. Contactez le support technique
        """)

        logger.error(f"Erreur initialisation : {e}", exc_info=True)

        # Mode de secours minimal
        st.markdown("---")
        st.markdown("### 🔧 **Mode de Secours Minimal**")

        simple_query = st.text_input("Posez votre question (mode texte uniquement) :")

        if simple_query and st.button("Envoyer"):
            st.info(f"**Mode secours activé pour :** {simple_query}")
            st.warning("""
            **Fonctionnalités limitées en mode secours :**
            - Pas de reconnaissance vocale
            - Pas de synthèse vocale
            - Réponses basiques
            - Pas d'historique
            """)

            # Réponse basique
            st.success(f"""
            **Réponse de secours :**

            Merci pour votre question concernant "{simple_query}". 

            En mode de secours, je ne peux fournir qu'une réponse basique. 
            Veuillez réinstaller l'application complète pour une assistance professionnelle.

            **Question reçue :** {simple_query}
            **Heure :** {dt.now().strftime('%H:%M')}
            """)


# ========== EXÉCUTION ==========
if __name__ == "__main__":
    main()