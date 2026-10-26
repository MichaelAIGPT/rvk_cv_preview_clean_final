import http.server
import socketserver
import json
import os
import webbrowser

PORT = 8088
DIRECTORY = "dataset"

class TaggerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/save_labels":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            
            # Save verified labels
            out_path = os.path.join(DIRECTORY, "verified_labels.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"[TaggerServer] Successfully saved {len(data['items'])} labels to {out_path}!")
            print(f"Hardhat: {data['hardhat_count']}, No-Hardhat: {data['no_hardhat_count']}, Modified: {data['modified_count']}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), TaggerHandler) as httpd:
        print(f"Tagger Server running at http://localhost:{PORT}/fast_tagger.html")
        httpd.serve_forever()
