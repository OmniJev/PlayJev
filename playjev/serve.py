"""PlayJev server: the OpenJev /v1/systemone endpoint for image states.

    python -m playjev.serve --ckpt /path/to/ckpt --port 18732            # one frame per decision
    python -m playjev.serve --ckpt /path/to/ckpt --port 18732 --two-frame  # (previous, current) temporal stack

Request (the shape playjev.play ServerPolicy and the demo page send; TypeSafe's request with frames in the state):
  {"model": "playjev-latest",
   "state": {"frames": ["data:image/jpeg;base64,...", ...]},          # one frame, or (previous, current)
   "questions": {"q": {"type": "choice", "instructions": "Which move should the player make next?",
                       "criteria": {"up": "turn the snake to move up", ...}}}}
Response:
  {"model": "playjev-0.8b", "answers": {"q": {"type": "choice", "choice": "up", "probabilities": {"up": 0.9, ...},
   "confidence": 0.87}}, "timing": {"prep_ms": 5.1, "forward_ms": 38.2, "visual_tokens": 182}}

Every question is a Choice over its criteria (a dict name -> description, or a list of names); the frames are
shared by all questions of one request. Text states are refused with 400: that is OpenJev's job. Requests are
served one at a time (one forward each); CORS is open so the demo page can call this from any origin.
"""
import argparse, base64, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import DEFAULT_INSTRUCTIONS, PlayJevModel


class Engine:
    def __init__(self, ckpt, device, name, two_frame, stack, template):
        self.name, self.two_frame, self.stack = name, two_frame, stack
        self.m = PlayJevModel(ckpt, device=device, template=template).load()
        self.lock = threading.Lock()

    @staticmethod
    def _frame_bytes(s):
        if not isinstance(s, str) or not s:
            raise ValueError("each frame must be a base64 string or a data URL")
        if s.startswith("data:"):
            s = s.split(",", 1)[1] if "," in s else ""
        return base64.b64decode(s)

    def answer(self, body):
        state = body.get("state")
        if not isinstance(state, dict) or not isinstance(state.get("frames"), list) or not state["frames"]:
            raise ValueError("state must be {\"frames\": [...]} with at least one frame; text states belong to OpenJev")
        frames = [self._frame_bytes(f) for f in state["frames"]]
        if self.two_frame:
            item = (frames[-2], frames[-1]) if len(frames) >= 2 else (frames[-1], frames[-1])
        else:
            item = frames[-1]
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        answers, timing = {}, {}
        for qname, q in questions.items():
            if not isinstance(q, dict) or q.get("type", "choice") != "choice":
                raise ValueError(f"question {qname!r}: only type \"choice\" is served")
            crit = q.get("criteria")
            if isinstance(crit, dict):
                options = [{"name": str(k), "description": str(v)} for k, v in crit.items()]
            elif isinstance(crit, list):
                options = [{"name": str(k), "description": str(k)} for k in crit]
            else:
                raise ValueError(f"question {qname!r}: criteria must be an object or a list")
            if len(options) < 2:
                raise ValueError(f"question {qname!r}: a choice needs at least two options")
            instructions = str(q.get("instructions") or DEFAULT_INSTRUCTIONS)
            with self.lock:
                d = self.m.decide([item], options, instructions=instructions,
                                  frames_per_state=2 if self.two_frame else 1, stack=self.stack, batch_size=1)[0]
                t = self.m.last_timing
            answers[qname] = {"type": "choice", "choice": options[d.choice]["name"],
                              "probabilities": {o["name"]: round(p, 6) for o, p in zip(options, d.probs)},
                              "confidence": round(d.confidence, 6)}
            timing = {"prep_ms": round(t.prep_s * 1000, 1), "forward_ms": round(t.forward_s * 1000, 1),
                      "input_tokens": t.input_tokens, "visual_tokens": t.visual_tokens}
        return {"model": self.name, "answers": answers, "timing": timing}


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": self.engine.name, "object": "model", "owned_by": "playjev"}]})
        elif self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"ok": True, "model": self.engine.name, "two_frame": self.engine.two_frame})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/systemone":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            t0 = time.perf_counter()
            out = self.engine.answer(body)
            out["timing"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._send(200, out)
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # a bad frame must not take the server down
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint directory or HF id for PlayJevModel")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=18732)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--name", default="playjev-0.8b")
    p.add_argument("--two-frame", action="store_true", help="the checkpoint was trained on (previous, current) frames")
    p.add_argument("--stack", default="temporal", choices=["temporal", "separate"])
    p.add_argument("--template", default="plain", choices=["plain", "chat"])
    p.add_argument("--verbose", action="store_true", help="log every request")
    a = p.parse_args()
    Handler.engine = Engine(a.ckpt, a.device, a.name, a.two_frame, a.stack, a.template)
    srv = ThreadingHTTPServer((a.host, a.port), Handler); srv.verbose = a.verbose
    print(f"playjev.serve: {a.name} from {a.ckpt} on http://{a.host}:{a.port}/v1/systemone "
          f"({'two frames' if a.two_frame else 'one frame'} per decision, loaded in {Handler.engine.m.load_seconds:.1f} s)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
