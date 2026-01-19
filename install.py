# install.py
import subprocess
import sys
import os

def install_dependencies():
    """Installer les dépendances automatiquement"""
    print("🔧 Installation des dépendances pour l'assistant vocal...")
    print("=" * 50)

    requirements = [
        "streamlit==1.28.0",
        "speechrecognition==3.10.0",
        "pyttsx3==2.90",
        "pandas==2.0.3"
    ]

    try:
        # Mettre à jour pip
        print("📦 Mise à jour de pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

        # Installer les packages
        for package in requirements:
            print(f"📦 Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Installer PyAudio selon l'OS
        print("📦 Installation de PyAudio...")
        if os.name == 'nt':  # Windows
            try:
                # Essayer d'abord pyaudio direct
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
            except:
                print("🔧 Utilisation de pipwin pour Windows...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])
                    subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])
                except:
                    print("⚠️  Pour Windows, téléchargez PyAudio manuellement:")
                    print("   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
                    print("   pip install le_fichier_telecharge.whl")
        else:  # Linux/Mac
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
            except:
                print("⚠️  Sur Linux (Ubuntu/Debian):")
                print("   sudo apt-get install python3-pyaudio")
                print("⚠️  Sur Mac:")
                print("   brew install portaudio")
                print("   pip install pyaudio")

        print("\n" + "=" * 50)
        print("✅ Installation terminée avec succès !")
        print("\n🚀 Pour démarrer l'application:")
        print("   streamlit run app.py")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Installation manuelle:")
        print("pip install streamlit speechrecognition pyttsx3 pandas")

if __name__ == "__main__":
    install_dependencies()
