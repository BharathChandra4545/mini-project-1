import json
from typing import Any, Callable


def json_response(handler: Callable[[], Any]) -> tuple[int, dict[str, str], str]:
    try:
        return 200, {"Content-Type": "application/json"}, json.dumps(handler())
    except FileNotFoundError as exc:
        return 503, {"Content-Type": "application/json"}, json.dumps({"error": str(exc)})
    except ValueError as exc:
        return 400, {"Content-Type": "application/json"}, json.dumps({"error": str(exc)})
    except Exception:
        return 500, {"Content-Type": "application/json"}, json.dumps(
            {"error": "The analytics request could not be completed."}
        )


def write_response(request: Any, status: int, headers: dict[str, str], body: str) -> None:
    request.send_response(status)
    for name, value in headers.items():
        request.send_header(name, value)
    request.send_header("Access-Control-Allow-Origin", "*")
    request.send_header("Access-Control-Allow-Headers", "Content-Type")
    request.end_headers()
    request.wfile.write(body.encode("utf-8"))
