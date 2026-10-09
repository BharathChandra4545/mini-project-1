from http.server import BaseHTTPRequestHandler

from api._http import json_response, write_response
from api._ml import DATA_PATH, read_csv


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        write_response(self, 204, {}, "")

    def do_GET(self):
        status, headers, body = json_response(
            lambda: {
                "rows": read_csv(DATA_PATH)
                .where(lambda value: value.notna(), None)
                .to_dict(orient="records")
            }
        )
        write_response(self, status, headers, body)
