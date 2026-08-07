"""
AetherEdge — Public URL Launcher
Starts the Flask app AND creates an ngrok tunnel so you get
a shareable https://xxxx.ngrok-free.app link instantly.

Usage:
    python launch.py
    python launch.py --authtoken YOUR_TOKEN   (optional, for stable URLs)
"""
import os
import sys
import time
import threading
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def start_flask():
    """Run the Flask app in this process (imported, not subprocess)."""
    import vehicle_app
    vehicle_app.app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

def main():
    parser = argparse.ArgumentParser(description="AetherEdge launcher with ngrok tunnel")
    parser.add_argument("--authtoken", default=None,
                        help="Optional ngrok authtoken for a stable subdomain")
    args = parser.parse_args()

    # ── 1. Start video-processing background thread (from vehicle_app) ──
    import vehicle_app
    proc_thread = threading.Thread(target=vehicle_app.video_processing_loop, daemon=True)
    proc_thread.start()

    # ── 2. Create ngrok tunnel ──
    try:
        from pyngrok import ngrok, conf

        if args.authtoken:
            conf.get_default().auth_token = args.authtoken
            # Save for future runs
            token_file = os.path.join(os.path.dirname(__file__), ".ngrok_token")
            with open(token_file, "w") as f:
                f.write(args.authtoken)
            print("[ngrok] Token saved — future runs won't need --authtoken")
        else:
            # Try to read saved token if present
            token_file = os.path.join(os.path.dirname(__file__), ".ngrok_token")
            if os.path.exists(token_file):
                with open(token_file) as f:
                    conf.get_default().auth_token = f.read().strip()
            else:
                print("\n[!] No ngrok token found.")
                print("    1. Sign up free at: https://dashboard.ngrok.com/signup")
                print("    2. Get your token:  https://dashboard.ngrok.com/get-started/your-authtoken")
                print("    3. Run: python launch.py --authtoken YOUR_TOKEN\n")

        tunnel = ngrok.connect(5000, bind_tls=True)
        public_url = tunnel.public_url

        print("\n" + "=" * 60)
        print("  AetherEdge Vehicle Analytics — PUBLIC URL READY")
        print("=" * 60)
        print(f"\n  Local  :  http://127.0.0.1:5000")
        print(f"  Public :  {public_url}")
        print(f"\n  Share the PUBLIC link with anyone.\n")
        print("  Press Ctrl+C to stop.")
        print("=" * 60 + "\n")

    except ImportError:
        print("[Error] pyngrok not installed. Run:  pip install pyngrok")
        sys.exit(1)
    except Exception as e:
        print(f"[ngrok Error] {e}")
        print("\nFalling back to local-only mode.")
        print(f"  Local: http://127.0.0.1:5000\n")

    # ── 3. Run Flask (blocks here) ──
    vehicle_app.app.run(host="0.0.0.0", port=5000, debug=False,
                        threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
