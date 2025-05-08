import soundfile as sf 
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlparse, parse_qs
import json

injected_server = None

def process_request(params):
    global injected_server
    try:
        print("Hook called with:", params)

        text = params.get("text", "")
        if not text.strip():
            print("No text provided.")
            return

        # Get optional path param
        save_path = params.get("path", "output.wav")

        model_choice = "Zyphra/Zonos-v0.1-transformer"
        language = "en-us"
        speaker_audio = params.get("voice", None)
        prefix_audio = "assets/silence_100ms.wav"

        # Emotions
        emotion = params.get("emotion", "happiness").lower()
        e = [
            1.0 if emotion == "happiness" else 0.05,
            1.0 if emotion == "sadness" else 0.05,
            1.0 if emotion == "disgust" else 0.05,
            1.0 if emotion == "fear" else 0.05,
            1.0 if emotion == "surprise" else 0.05,
            1.0 if emotion == "anger" else 0.05,
            1.0 if emotion == "other" else 0.05,
            1.0 if emotion == "neutral" else 0.2,
        ]

        # Speaking rate
        try:
            speaking_rate = float(params.get("rate", 15.0))
        except ValueError:
            speaking_rate = 15.0

        # 🐛 Debug summary print
        print("\n[🔊 GENERATION SUMMARY]")
        print(f"Text:           {text}")
        print(f"Save path:      {save_path}")
        print(f"Emotions:       {e}")
        print(f"Speaker audio:  {speaker_audio or 'None'}")
        print(f"Speaking rate:  {speaking_rate}")
        print(f"-------------------------\n")

        # Generation
        result, seed = injected_server.zonos_generate_audio(
            model_choice,
            text,
            language,
            speaker_audio,
            prefix_audio,
            *e,
            vq_single=0.78,
            fmax=24000,
            pitch_std=45.0,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=4.0,
            speaker_noised=False,
            cfg_scale=4.0,
            min_p=0.25,
            seed=420,
            randomize_seed=False,
            unconditional_keys=[],
            progress=lambda *_: None  # dummy
        )

        sr, audio_data = result

        sf.write(save_path, audio_data, sr)
        print(f"✅ Generated audio saved to {save_path} (seed={seed})")
        return True, f"Generated audio saved to {save_path} (seed={seed})"

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, str(e)

class InjectedServerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global injected_server
        parsed = urlparse(self.path)
        if parsed.path == "/generate":
            params = parse_qs(parsed.query)
            params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
            print("📥 Query parameters:", params)

            success, message = process_request(params)

            self.send_response(200 if success else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "success": success,
                "params": params,
                "message": message
            }

            self.wfile.write(json.dumps(response).encode("utf-8"))
            print(f"{'✅' if success else '❌'} Hook result: {message}")
        else:
            self.send_response(404)
            self.end_headers()
            print(f"🚫 Unknown path: {self.path}")

    def log_message(self, format, *args):
        return  # suppress logging to console

class InjectedServer:
    def __init__(self, zonos_generate_audio, host='127.0.0.1', port=5001):
        global injected_server
        self.zonos_generate_audio = zonos_generate_audio
        self.server = HTTPServer((host, port), InjectedServerRequestHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.host = host
        self.port = port
        injected_server = self
        self.start()

    def start(self):
        # dont print `http` as it used by pinokio to detect zonos frontend loading state
        print(f"🚀 Injected server running on {self.host}:{self.port}")
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        print("🛑 Injected server stopped")
