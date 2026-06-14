"""
api_server.py -- API Endpoint cho EXACT 2026 submission.

Cuoc thi yeu cau mot API endpoint de cham diem tu dong.

Usage:
  python api_server.py                              # Chay server
  python api_server.py --port 8080                  # Port khac
  python api_server.py --quantization 4bit          # Dung 4-bit

API Endpoints:
  POST /predict       -- Nhan 1 sample, tra ve predicted answers + reasoning
  POST /predict_batch -- Nhan nhieu samples
  GET  /health        -- Health check
  GET  /info          -- Model info

Request format (POST /predict):
{
  "premises-NL": ["..."],
  "questions": ["..."]
}

Response format:
{
  "answers": [
    {
      "question_id": 0,
      "answer": "A",
      "reasoning": "..."
    }
  ],
  "z3_status": "sat",
  "local_ontology": [...],
  "time_sec": 12.5
}
"""

import json
import time
import argparse
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

from config import PipelineConfig
from model_loader import QwenModel, print_system_info
from pipeline import run_pipeline, PipelineResult


# ── Globals (initialized in main) ────────────────────────────────
_qwen: QwenModel = None
_config: PipelineConfig = None
_request_counter: int = 0


class PipelineHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the pipeline API."""

    def do_GET(self):
        if self.path == "/health":
            self._respond_json(200, {"status": "ok", "model_loaded": _qwen._loaded})
        elif self.path == "/info":
            self._respond_json(200, {
                "model_id": _config.model_id,
                "quantization": _config.quantization,
                "max_retries": _config.max_retries,
                "max_new_tokens": _config.max_new_tokens,
            })
        else:
            self._respond_json(404, {"error": "Not found"})

    def do_POST(self):
        global _request_counter

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._respond_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if self.path == "/predict":
            _request_counter += 1
            result = self._process_single(data, _request_counter)
            self._respond_json(200, result)

        elif self.path == "/predict_batch":
            if not isinstance(data, list):
                self._respond_json(400, {"error": "Expected JSON array"})
                return
            results = []
            for i, sample in enumerate(data):
                _request_counter += 1
                r = self._process_single(sample, _request_counter)
                results.append(r)
            self._respond_json(200, {"results": results, "count": len(results)})

        else:
            self._respond_json(404, {"error": "Not found"})

    def _process_single(self, data: dict, idx: int) -> dict:
        """Process a single prediction request."""
        try:
            # Normalize input
            sample = {
                "premises-NL": data.get("premises-NL", data.get("premises", [])),
                "questions": data.get("questions", []),
                "answers": data.get("answers", []),  # optional, for evaluation
            }

            if not sample["premises-NL"] or not sample["questions"]:
                return {"error": "Missing 'premises-NL' or 'questions'"}

            result = run_pipeline(idx, sample, _qwen, _config)

            return {
                "answers": [
                    {
                        "question_id": a["question_id"],
                        "answer": a["answer"],
                        "reasoning": a.get("reasoning", ""),
                    }
                    for a in result.predicted_answers
                ],
                "z3_status": result.z3_status,
                "z3_compiled": result.z3_compiled,
                "z3_total": result.z3_total,
                "z3_attempts": result.z3_attempts,
                "local_ontology": result.local_ontology,
                "hallucination_warnings": result.hallucination_warn,
                "time_sec": result.time_sec,
                "status": result.status,
            }

        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()[-500:],
            }

    def _respond_json(self, status_code: int, data: dict):
        """Send JSON response."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Override to add timestamp."""
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


def main():
    global _qwen, _config

    parser = argparse.ArgumentParser(description="Pipeline API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--quantization", choices=["8bit", "4bit", "none"], default="8bit")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    _config = PipelineConfig(
        model_id=args.model_id,
        quantization=args.quantization,
        max_retries=args.max_retries,
    )

    print_system_info()
    print(_config.summary())

    _qwen = QwenModel(_config)
    _qwen.load()

    server = HTTPServer((args.host, args.port), PipelineHandler)
    print(f"\n API Server running on http://{args.host}:{args.port}")
    print(f"  POST /predict       -- Single prediction")
    print(f"  POST /predict_batch -- Batch prediction")
    print(f"  GET  /health        -- Health check")
    print(f"  GET  /info          -- Model info")
    print(f"\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
