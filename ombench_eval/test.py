import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    api_key = os.environ.get("INF_API_KEY")
    if not api_key:
        print("INF_API_KEY not set in environment", file=sys.stderr)
        return 1

    base_url = "https://holos.openapi-qb.sii.edu.cn"
    url = f"{base_url}/v1/chat/completions"
    model_id = "sii-holos/Qwen 3.5 397B A17B"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
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

    print(f"POST {url}")
    print(f"Model: {model_id}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("-" * 40)

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"Status: {response.status}")
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


if __name__ == "__main__":
    raise SystemExit(main())
