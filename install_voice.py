# install_voice.py
import subprocess
import sys
import os
import platform

def check_python_version():
    """Vérifier la version de Python"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return version

def install_windows():
    """Installation pour Windows"""
    print("🔧 Installation pour Windows...")

    try:
        # Installer pipwin
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pipwin"])

        # Installer pyaudio via pipwin
        subprocess.check_call([sys.executable, "-m", "pipwin", "install", "pyaudio"])

        # Installer les autres modules
        modules = ["speechrecognition", "pyttsx3"]
        for module in modules:
            subprocess.check_call([sys.executable, "-m", "pip", "install", module])

        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")

        # Méthode alternative
        print("\n🔄 Essai avec la méthode alternative...")
        try:
            # Télécharger et installer directement
            import urllib.request
            import tempfile

            # URL pour PyAudio (à adapter selon votre version Python)
            python_version = f"{sys.version_info.major}{sys.version_info.minor}"
            arch = "win_amd64" if platform.machine().endswith('64') else "win32"
            url = f"https://download.lfd.uci.edu/pythonlibs/w4tscw6k/PyAudio-0.2.11-cp{python_version}-cp{python_version}-{arch}.whl"

            print(f"Téléchargement depuis: {url}")

            # Télécharger le fichier
            with tempfile.NamedTemporaryFile(suffix='.whl', delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                subprocess.check_call([sys.executable, "-m", "pip", "install", tmp.name])

            # Installer les autres modules
            for module in ["speechrecognition", "pyttsx3"]:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])

            return True
        except:
            return False

def install_linux():
    """Installation pour Linux"""
    print("🔧 Installation pour Linux...")

    try:
        # Détecter la distribution
        import distro
        distro_name = distro.id().lower()
        print(f"Distribution détectée: {distro_name}")

        # Installer pyaudio selon la distribution
        if distro_name in ['ubuntu', 'debian', 'linuxmint']:
            subprocess.check_call(['sudo', 'apt-get', 'update'])
            subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'python3-pyaudio'])
        elif distro_name in ['fedora', 'centos', 'rhel']:
            subprocess.check_call(['sudo', 'dnf', 'install', '-y', 'python3-pyaudio'])
        elif distro_name in ['arch', 'manjaro']:
            subprocess.check_call(['sudo', 'pacman', '-Sy', 'python-pyaudio'])

        # Installer les modules Python
        modules = ["speechrecognition", "pyttsx3"]
        for module in modules:
            subprocess.check_call([sys.executable, "-m", "pip", "install", module])

        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def install_mac():
    """Installation pour Mac"""
    print("🔧 Installation pour Mac...")

    try:
        # Installer portaudio via Homebrew
        subprocess.check_call(['brew', 'install', 'portaudio'])

        # Installer les modules Python
        modules = ["pyaudio", "speechrecognition", "pyttsx3"]
        for module in modules:
            subprocess.check_call([sys.executable, "-m", "pip", "install", module])

        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🔧 INSTALLATEUR DE MODULES VOCAUX")
    print("=" * 60)

    # Vérifier la version Python
    version = check_python_version()

    # Détecter le système d'exploitation
    system = platform.system()
    print(f"Système: {system}")

    success = False

    if system == "Windows":
        success = install_windows()
    elif system == "Linux":
        success = install_linux()
    elif system == "Darwin":  # Mac
        success = install_mac()
    else:
        print(f"❌ Système non supporté: {system}")

    if success:
        print("\n" + "=" * 60)
        print("✅ INSTALLATION RÉUSSIE !")
        print("=" * 60)
        print("\n🚀 Redémarrez l'application pour activer la voix :")
        print("   streamlit run voice_assistant_streamlit.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ INSTALLATION ÉCHOUÉE")
        print("=" * 60)
        print("\n💡 Installation manuelle recommandée.")
        print("Consultez le guide dans l'application.")
        print("=" * 60)

if __name__ == "__main__":
    main()
