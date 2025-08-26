# 🤖 Convers AI Robotics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20RaspberryPi-lightgrey)

**Convers AI Robotics** è un assistente vocale conversazionale in tempo reale che integra **OpenAI Realtime API** con un **Arduino** collegato via USB.  
Permette di parlare con l’AI e, allo stesso tempo, comandare un braccio robotico o altri dispositivi connessi ad Arduino.

## ✨ Funzionalità principali

- Conversazione vocale in tempo reale (input microfono + output voce TTS).
- Interruzione automatica del microfono mentre l’AI parla (niente eco).
- Memoria conversazionale locale con log su file.
- Controllo robotico via **seriale USB** (Arduino):
- saluto, home, apertura/chiusura pinza
- movimenti direzionali (su, giù, destra, sinistra)
- disegno lettere e forme semplici
- Auto-rilevamento della porta seriale Arduino.
- HUD diagnostico (console o finestra grafica OpenCV) con stato rete, connessione e microfono.

---
## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/qj3egxmWiZ4/0.jpg)](https://youtu.be/qj3egxmWiZ4)

---
## 📂 Struttura del progetto

```
convers_ai_robotics/
│
├── convers_ai_robotics.py   # script principale Python
├── ai_prompt.txt             # prompt personalizzabile per l'assistente
├── arduino/                  # firmware Arduino
│   └── RobotArm5DOF.ino
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installazione

### 1. Clona il repository
```bash
git clone https://github.com/gianlucapassarella/convers_ai_robotics.git
cd convers_ai_robotics
```

### 2. Crea un ambiente virtuale Python
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate      # Windows
```

### 3. Installa le dipendenze
```bash
pip install -r requirements.txt
```

---

## 🔑 Configurazione

1. Copia il file `.env.example` in `.env`:
   ```bash
   cp .env.example .env
   ```
2. Apri `.env` e inserisci la tua chiave API di OpenAI:
   ```env
   OPENAI_API_KEY="INSERISCI-LA-TUA-KEY-QUI"
   ```

3. (Opzionale) Regola i parametri audio, memoria e RobotLink secondo il tuo setup.

---

## 🤖 Firmware Arduino

1. Apri la cartella `arduino/` con **Arduino IDE**.  
2. Carica lo sketch `RobotArm5DOF.ino` sulla tua scheda.  
3. Collega Arduino via USB: lo script Python rileverà automaticamente la porta (`ROBOT_PORT=auto`).

---

## ▶️ Avvio

Esegui lo script principale:

```bash
python convers_ai_robotics.py
```

Se tutto è configurato correttamente sentirai la voce dell’assistente e potrai parlare al microfono.  
Comandi vocali come *"alza il braccio"*, *"fai ciao"*, *"apri la pinza"* verranno tradotti in azioni fisiche sul braccio robotico.

Puoi cambiare la personalità dell'Ai modificando il file `ai_prompt.txt`.   
---

## 🖥️ HUD Diagnostico

Puoi abilitare l’HUD per monitorare lo stato di rete, connessione e microfono.

- Attivazione da `.env`:
  ```env
  HUD=1
  HUD_MODE=cv      # finestra grafica con OpenCV
  HUD_MODE=console # solo console
  ```

---

## 📒 Log conversazioni

Le conversazioni vengono salvate automaticamente nella cartella:
- `Desktop/AI_Conversazioni/` (Linux/Windows)
- `Scrivania/AI_Conversazioni/` (macOS in italiano)

---
## 📡 Protocollo seriale (Arduino)
- Baud: 115200, newline-terminated JSON (JSONL)
- Comandi esempio:
  {"cmd":"HOME"}
  {"cmd":"WAVE"}
  {"cmd":"OPEN"} | {"cmd":"CLOSE"}
  {"cmd":"MOVE","dir":"LEFT|RIGHT|UP|DOWN"}
  {"cmd":"CIRCLE"}
  {"cmd":"DRAW","char":"A"}
- Risposte:
  {"status":"done"} oppure {"status":"error","msg":"..."}

---
## 🛠️ Requisiti

- Python 3.9+
- Librerie da `requirements.txt`
- Scheda Arduino compatibile (UNO, Mega, Nano, ecc.)
- Microfono e speaker funzionanti
- Connessione Internet stabile

---

## 📜 Licenza

Questo progetto è distribuito sotto licenza **MIT**.  
Vedi il file [LICENSE](LICENSE) per i dettagli.

---

## 🚀 Roadmap

- [ ] Estendere i comandi ad altri tipi di robot e automazioni.
- [ ] Integrazione con domotica (MQTT, Home Assistant).
- [ ] Packaging come libreria Python installabile (`pip install convers-ai-robotics`).
