import cv2
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg') # Backend non-interactiv pentru server
import matplotlib.pyplot as plt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from keras.models import load_model

app = FastAPI()

MODEL_PATH = "models/optimized_model.h5"

# --- CONFIGURARE ---
# Încarcă modelul de emoții
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model încărcat.")
except:
    print("ATENȚIE: Modelul nu a fost găsit. Se va folosi output fictiv.")
    model = None

# Încărcăm clasificatorul de fețe (Haar Cascade) inclus în OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTII_LABELS = ['Furie', 'Dezgust', 'Frică', 'Fericire', 'Tristețe', 'Surpriză', 'Neutru']

def generate_graph(history):
    """Generează un grafic din istoricul sesiunii și îl returnează ca base64"""
    if not history:
        return None
    
    emotions = [h['emotie'] for h in history]
    timestamps = range(len(emotions)) # Axa X simplificată (nr frame-uri)

    # Creăm graficul
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Mapăm emoțiile la numere pentru a le putea plota
    unique_emotions = list(set(EMOTII_LABELS))
    y_values = [unique_emotions.index(e) if e in unique_emotions else -1 for e in emotions]
    
    ax.plot(timestamps, y_values, marker='o', linestyle='-', color='b', alpha=0.6)
    ax.set_yticks(range(len(unique_emotions)))
    ax.set_yticklabels(unique_emotions)
    ax.set_title("Evoluția Emoțiilor în timpul Înregistrării")
    ax.set_xlabel("Timp (frame-uri)")
    ax.grid(True, alpha=0.3)

    # Salvăm în memorie
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    # Convertim la base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    recording = False
    session_history = [] # Aici stocăm datele când recording=True
    
    try:
        while True:
            # 1. Primim date: poate fi imagine (bytes) sau comandă text (start/stop)
            message = await websocket.receive()
            
            if "text" in message:
                # Gestionăm comenzile de la butoane
                cmd = message["text"]
                if cmd == "start_rec":
                    recording = True
                    session_history = []
                    print("Înregistrare pornită")
                elif cmd == "stop_rec":
                    recording = False
                    print("Înregistrare oprită. Generare grafic...")
                    graph_b64 = generate_graph(session_history)
                    await websocket.send_json({"type": "graph", "image": graph_b64})
                continue

            # Dacă nu e text, e imaginea (bytes)
            if "bytes" not in message:
                continue
                
            data = message["bytes"]
            
            # --- PROCESARE IMAGINE ---
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectare Față
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_coords = []
            top_emotii = []
            dominant_emotion = "Neutru"

            # Dacă am găsit o față, facem predicția pe ea
            if len(faces) > 0:
                (x, y, w, h) = faces[0] # Luăm prima față
                face_coords = [int(x), int(y), int(w), int(h)]
                
                # Decupăm fața pentru model
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = roi_gray.astype('float32') / 255.0
                img_pixels = np.expand_dims(img_pixels, axis=-1)
                img_pixels = np.expand_dims(img_pixels, axis=0)

                if model:
                    preds = model.predict(img_pixels, verbose=0)[0]
                    # Top 3
                    sorted_indices = np.argsort(preds)[::-1]
                    dominant_emotion = EMOTII_LABELS[sorted_indices[0]]
                    
                    for i in sorted_indices[:3]:
                        top_emotii.append({
                            "emotie": EMOTII_LABELS[i],
                            "procent": round(float(preds[i]) * 100, 1)
                        })

            # Dacă înregistrezi, salvăm datele
            if recording and top_emotii:
                session_history.append({"emotie": dominant_emotion})

            # Trimitem răspunsul
            await websocket.send_json({
                "type": "data",
                "face_coords": face_coords, # Trimitem coordonatele înapoi
                "top_emotii": top_emotii
            })
            
    except WebSocketDisconnect:
        print("Client deconectat")
    except Exception as e:
        print(f"Eroare: {e}")

# --- FRONTEND ---
@app.get("/")
async def get():
    with open("src/ui/index.html", "r") as f:
        return HTMLResponse(content=f.read())