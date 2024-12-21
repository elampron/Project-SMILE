from setuptools import setup, find_packages

setup(
    name="livekit-plugin-smiles",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "livekit",
        "livekit-agents>=0.12.1",
        "livekit-plugins-openai>=0.10.9",
        "livekit-plugins-deepgram>=0.6.13",
        "livekit-plugins-turn-detector>=0.3.1",
        "livekit-plugins-silero>=0.7.4",
        "livekit-plugins-rag>=0.2.3",
        "aiohttp",
        "python-dotenv~=1.0",
        "aiofile~=3.8.8",
    ],
    author="Eric Lampron",
    author_email="elampron@gmail.com",
    description="LiveKit plugin for SMILES API integration",
    python_requires=">=3.8",
)
