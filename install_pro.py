# install_pro.py
"""
Script d'installation pour l'Assistant Vocal Professionnel
"""

import subprocess
import sys
import platform
import os

def print_header():
    print("=" * 70)
    print("🔧 INSTALLATEUR - ASSISTANT VOCAL PROFESSIONNEL")
    print("=" * 70)
    print()

def check_python():
    """Vérifier la version Python"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} détecté")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 ou supérieur requis")
        return False
    return True

def install_windows():
    """Installation Windows"""
    print("📦 Installation pour Windows...")
    
    try:
        # Base
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Dépendances principales
        main_deps = ["streamlit", "pandas", "requests", "numpy", "python-dotenv"]
        for dep in main_deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        # Reconnaissance vocale
        print("🔊 Installation reconnaissance vocale...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "SpeechRecognition"])
        except:
            print("⚠️  SpeechRecognition - Installation échouée")
        
        # PyAudio pour Windows
        print("🎤 Installation PyAudio...")
        try:
            # Essayer d'abord pipwin
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
            subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
            print("✅ PyAudio installé via pipwin")
        except:
            print("❌ PyAudio - Installation échouée")
            print("   Téléchargez manuellement: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        
        # Synthèse vocale
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def install_linux():
    """Installation Linux"""
    print("📦 Installation pour Linux...")
    
    try:
        # Mise à jour système
        subprocess.check_call(['sudo', 'apt-get', 'update'])
        
        # Dépendances système
        subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'python3-pip', 'python3-venv'])
        subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'portaudio19-dev', 'python3-pyaudio'])
        
        # Dépendances Python
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        deps = ["streamlit", "pandas", "requests", "SpeechRecognition", "pyttsx3"]
        for dep in deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def install_mac():
    """Installation Mac"""
    print("📦 Installation pour Mac...")
    
    try:
        # Vérifier Homebrew
        subprocess.check_call(['brew', '--version'])
        
        # Installer portaudio
        subprocess.check_call(['brew', 'install', 'portaudio'])
        
        # Dépendances Python
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        deps = ["streamlit", "pandas", "requests", "SpeechRecognition", "pyttsx3", "pyaudio"]
        for dep in deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    print_header()
    
    # Vérifier Python
    if not check_python():
        return
    
    # Détecter OS
    system = platform.system()
    print(f"💻 Système détecté: {system}")
    
    success = False
    
    if system == "Windows":
        success = install_windows()
    elif system == "Linux":
        success = install_linux()
    elif system == "Darwin":
        success = install_mac()
    else:
        print(f"❌ Système non supporté: {system}")
        return
    
    if success:
        print()
        print("=" * 70)
        print("🎉 INSTALLATION RÉUSSIE !")
        print("=" * 70)
        print()
        print("🚀 Pour démarrer l'assistant :")
        print("   streamlit run assistant_vocal_pro.py")
        print()
        print("💡 Configuration recommandée :")
        print("   1. Testez la reconnaissance vocale")
        print("   2. Configurez vos préférences")
        print("   3. Essayez les commandes avancées")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("❌ INSTALLATION ÉCHOUÉE")
        print("=" * 70)
        print()
        print("💡 Solutions :")
        print("   1. Installez manuellement : pip install -r requirements.txt")
        print("   2. Consultez la documentation")
        print("   3. Vérifiez les permissions")
        print("=" * 70)

if __name__ == "__main__":
    main()
