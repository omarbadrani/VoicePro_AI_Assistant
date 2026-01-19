# setup.py
import subprocess
import sys
import os


def install_requirements():
    """Installer les dépendances"""
    print("🔧 Installation des dépendances pour l'Assistant Vocal Streamlit...")
    print("=" * 50)

    requirements = [
        "streamlit==1.28.0",
        "speechrecognition==3.10.0",
        "pyttsx3==2.90",
        "pandas==2.0.3",
        "requests==2.31.0"
    ]

    try:
        # Mettre à jour pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

        # Installer les packages
        for package in requirements:
            print(f"📦 Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Installer PyAudio selon l'OS
        print("📦 Installation de PyAudio...")
        if os.name == 'nt':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
                subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
            except:
                print("⚠️  Téléchargez PyAudio manuellement pour Windows")
                print("   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
            except:
                print("⚠️  Sur Linux: sudo apt-get install python3-pyaudio")
                print("⚠️  Sur Mac: brew install portaudio")

        print("\n" + "=" * 50)
        print("✅ Installation terminée !")
        print("\n🚀 Pour démarrer l'application:")
        print("   streamlit run app.py")
        print("\n📁 Structure du projet:")
        print("   app.py          # Application principale")
        print("   config.py       # Configuration")
        print("   requirements.txt# Dépendances")
        print("   data/           # Données sauvegardées")
        print("   assets/         # Images et ressources")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Installation manuelle:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    install_requirements()