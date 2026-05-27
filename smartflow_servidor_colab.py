# ═══════════════════════════════════════════════════════════════════
# CELDA NUEVA — Servidor SmartFlow (Flask + ngrok)
# Pégala DESPUÉS de la celda 6 (pipeline). No re-corre nada.
# ═══════════════════════════════════════════════════════════════════

# ── 1. Instalar dependencias del servidor ─────────────────────────
!pip install flask flask-cors pyngrok --quiet

# ── 2. Imports ────────────────────────────────────────────────────
import base64, io, time, threading
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image

# ── 3. Crear app Flask ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Permite peticiones desde cualquier origen (tu HTML)

# ── 4. Reusar las funciones ya definidas en el notebook ───────────
# (preprocess_image trabaja con rutas; aquí adaptamos para bytes)

def preprocess_bytes(img_bytes):
    """Igual que preprocess_image pero recibe bytes en vez de ruta."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen")
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return img_rgb, h, w

# ── 5. Endpoint principal ─────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        t0 = time.time()

        # Recibir imagen (base64 o archivo)
        if request.is_json:
            data = request.get_json()
            b64 = data['image'].split(',')[-1]   # quitar "data:image/...;base64,"
            img_bytes = base64.b64decode(b64)
        else:
            img_bytes = request.files['image'].read()

        # Preprocesar
        img_rgb, h, w = preprocess_bytes(img_bytes)

        # Detectar — usa el modelo ya cargado en memoria (celda 4)
        dets_raw = detect_people(model, img_rgb)

        # Convertir a formato que espera el HTML
        detections_out = []
        for d in dets_raw:
            detections_out.append({
                'x1':      d['bbox'][0],
                'y1':      d['bbox'][1],
                'x2':      d['bbox'][2],
                'y2':      d['bbox'][3],
                'cx':      d['cx'],
                'cy':      d['cy'],
                'cy_floor': d['cy_floor'],
                'conf':    round(d['conf'], 3),
            })

        elapsed_ms = round((time.time() - t0) * 1000)

        return jsonify({
            'ok':         True,
            'detections': detections_out,
            'count':      len(detections_out),
            'infer_ms':   elapsed_ms,
            'img_w':      w,
            'img_h':      h,
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'SmartFlow online ✅'})


# ── 6. Lanzar ngrok + Flask ───────────────────────────────────────
# Si tienes cuenta ngrok gratuita, pega tu authtoken aquí:
# ngrok.set_auth_token("TU_TOKEN_AQUI")

def run_flask():
    app.run(host='0.0.0.0', port=5000, use_reloader=False, debug=False)

# Iniciar Flask en hilo secundario
t = threading.Thread(target=run_flask, daemon=True)
t.start()
time.sleep(2)

# Crear túnel público
tunnel = ngrok.connect(5000, bind_tls=True)
NGROK_URL = tunnel.public_url

print("=" * 55)
print(f"  🚀 SmartFlow servidor ONLINE")
print(f"  🌐 URL pública : {NGROK_URL}")
print(f"  📡 Endpoint    : {NGROK_URL}/analyze")
print("=" * 55)
print(f"\n  👉 Copia esta URL en el HTML:\n  {NGROK_URL}")
print("\n  Celda ejecutándose... (no la interrumpas)")
