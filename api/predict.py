import json
from http.server import BaseHTTPRequestHandler

from api._http import json_response, write_response
from api._ml import predict_rows


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        write_response(self, 204, {}, "")

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            rows = payload.get("rows") if isinstance(payload, dict) else payload
        except (ValueError, TypeError, json.JSONDecodeError):
            write_response(
                self,
                400,
                {"Content-Type": "application/json"},
                json.dumps({"error": "Request body must be valid JSON."}),
            )
            return
        status, headers, body = json_response(lambda: {"rows": predict_rows(rows)})
        write_response(self, status, headers, body)

    def do_GET(self):
        write_response(
            self,
            405,
            {"Content-Type": "application/json"},
            json.dumps({"error": "Use POST."}),
        )
