import json
import os
import sys
import urllib.error
import urllib.request


def list_models(base_url: str, headers: dict) -> None:
    url = f"{base_url}/v1/models"
    print(f"GET {url}")
    print("-" * 40)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            result = json.loads(body)
            models = result.get("data", [])
            if models:
                print(f"Available models ({len(models)}):")
                for m in models:
                    mid = m.get("id", "?")
                    print(f"  - {mid}")
            else:
                print("No models found. Full response:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        print(f"Response body: {body}", file=sys.stderr)
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)


def chat_test(base_url: str, headers: dict, model_id: str) -> int:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, please reply with a short greeting."},
        ],
        "max_tokens": 128,
        "temperature": 0.7,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    print(f"\nPOST {url}")
    print(f"Model: {model_id}")
    print("-" * 40)

    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"Status: {resp.status}")
            result = json.loads(body)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"\nAssistant: {content}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        print(f"Response body: {body}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    api_key = os.environ.get("INF_API_KEY")
    if not api_key:
        print("INF_API_KEY not set in environment", file=sys.stderr)
        return 1

    base_url = "https://holos.openapi-qb.sii.edu.cn"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Step 1: list available models
    list_models(base_url, headers)

    # Step 2: try chat (update model_id after seeing the list above)
    model_id = "sii-holos/Qwen 3.5 397B A17B"
    return chat_test(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
