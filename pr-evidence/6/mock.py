# Minimal OpenAI Chat Completions stub: answers the judge with a verdict, other calls with the model id.
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        if model == "judge/model":
            rf = body.get("response_format")
            prompt = body["messages"][0]["content"]
            print(f"[mock] judge response_format = {json.dumps(rf)}", flush=True)
            print(f"[mock] schema in judge prompt: {'\"target\"' in prompt}", flush=True)
            content = json.dumps({"target": "capable"})
        else:
            print(f"[mock] completion served by {model}", flush=True)
            content = f"hello from {model}"
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": model,
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
