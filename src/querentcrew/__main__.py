"""Interactive command-line chat: `python -m querentcrew`."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from .service import QuerentCrew


def _print_response(resp, save_dir: Path) -> None:
    if resp.plan:
        print("\nPlan:")
        for i, step in enumerate(resp.plan, 1):
            print(f"  {i}. {step}")
    for rec in resp.results:
        print(f"\n[{rec['status']}] {rec['sub_question']}\n{rec['sql']}")
    if resp.answer:
        print(f"\n{resp.answer}")
    if resp.chart and resp.chart.get("image_b64"):
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"chart_{resp.thread_id}.png"
        path.write_bytes(base64.b64decode(resp.chart["image_b64"]))
        print(f"\nChart saved to {path}")
    u = resp.usage
    print(f"\n({u.get('calls', 0)} LLM calls, {u.get('total_tokens', 0)} tokens, {resp.elapsed_ms / 1000:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with QuerentCrew from the terminal")
    parser.add_argument("question", nargs="?", help="Ask a single question and exit")
    args = parser.parse_args()

    pilot = QuerentCrew()
    thread = pilot.new_thread_id()
    save_dir = Path("outputs")

    def handle(question: str) -> None:
        resp = pilot.ask(question, thread)
        while resp.status == "awaiting_approval":
            req = resp.approval_request
            print(f"\nApproval needed: {', '.join(req['reasons'])}\n{req['sql']}")
            ok = input("Run this query? [y/N] ").strip().lower() == "y"
            resp = pilot.resume(thread, approved=ok)
        _print_response(resp, save_dir)

    if args.question:
        handle(args.question)
        return
    print("QuerentCrew - ask a question about the sales data (type 'exit' to quit)")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            handle(question)


if __name__ == "__main__":
    main()
