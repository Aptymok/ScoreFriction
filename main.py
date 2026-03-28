import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8000))
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Si la ruta no es un archivo existente, sirve index.html
        if self.path != "/" and not os.path.exists(self.path[1:]):
            self.path = "/index.html"
        return super().do_GET()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()