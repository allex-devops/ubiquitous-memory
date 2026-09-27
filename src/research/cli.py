import argparse

from ragchat.rag import ConfigError, embedder

from .corpus import build_index, download
from .factory import build_assistant
from .memory import ConversationMemory
from .ui import render_brief, render_reply


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch", help="download the paper collection from arXiv (about 500 MB, 25 minutes for all 300)")
    fetch.add_argument("--limit", type=int, default=None, help="only the first N papers")
    idx = sub.add_parser("index", help="chunk and embed the downloaded papers")
    idx.add_argument("--limit", type=int, default=None)
    ask = sub.add_parser("ask", help="one question, with sources")
    ask.add_argument("question")
    sub.add_parser("chat", help="ask several questions; follow-ups like 'what about its limits?' work")
    brief = sub.add_parser("brief", help="a short literature brief on a topic")
    brief.add_argument("topic")
    brief.add_argument("--papers", type=int, default=4)
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        result = download(args.limit)
        print(f"{result['downloaded']} downloaded, {result['skipped']} already here, {len(result['failed'])} failed")
        return

    try:
        assistant, store = build_assistant()
    except ConfigError as e:
        raise SystemExit(str(e))
    if args.cmd == "index":
        print(f"{build_index(store, embedder(assistant.llm), args.limit)} chunks indexed, {store.count()} in the index")
    elif args.cmd == "ask":
        print(render_reply(assistant.ask(args.question)))
    elif args.cmd == "brief":
        print(render_brief(assistant.brief(args.topic, n_papers=args.papers)))
    else:
        memory = ConversationMemory(llm=assistant.rewriter)
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                print(render_reply(assistant.ask(q, memory)), "\n")


if __name__ == "__main__":
    main()
