# Release Notes

## Unreleased

### Features

- Added `AGENTRUN_RUNTIME=agentcore` migration mode. Existing AgentRun Model,
  MCP, and Skill integration calls route to AgentCore when they refer to a
  managed resource. Direct resources and unsupported capabilities continue to
  use the original AgentRun implementation.
- Routing is selected before a data-plane request is sent. In migration mode,
  an exact ModelConnection or MCP name miss during AgentCore resource discovery
  keeps the original AgentRun route. Other discovery errors and all failed
  AgentCore data-plane requests are not retried through AgentRun.
- AgentScope 1.x on Python 3.10 keeps the original AgentRun model route;
  AgentScope 2.x on Python 3.11+ uses the AgentCore adapter.

### Breaking Changes

- Sandbox data-plane HTTP JSON APIs now follow standard HTTP error handling:
  - `2xx` responses return the business response body.
  - `4xx` responses raise `ClientError`.
  - `5xx` responses raise `ServerError`.
- Error responses such as `{"code": "...", "requestId": "...", "message": "..."}`
  are no longer returned as normal dictionaries for Sandbox HTTP JSON APIs. The
  fields are exposed on the raised exception as `error_code`, `request_id`, and
  `message`.
- Existing code that checked returned dictionaries for `code` and `requestId`
  must migrate to `try` / `except ClientError` / `except ServerError`.
- `HTTPError.__str__()` output format has changed. The old format unconditionally
  included `"Request ID: None. Details: {}"` even when those fields were empty.
  The new format only includes non-empty fields and uses `". "` as separator.
  Code that parses this string representation (e.g. log parsers or test assertions
  on `str(error)`) must be updated.

### Migration

Before:

```python
resp = ci.cmd(command="echo hello", cwd="/tmp", timeout=30)
if "code" in resp and "requestId" in resp:
    raise RuntimeError(resp["message"])
```

After:

```python
from agentrun.utils.exception import ClientError, ServerError

try:
    resp = ci.cmd(command="echo hello", cwd="/tmp", timeout=30)
except ClientError as e:
    print(e.status_code, e.error_code, e.request_id, e.message)
    raise
except ServerError as e:
    print(e.status_code, e.error_code, e.request_id, e.message)
    raise
```

Command execution failures are still business-level failures and should be
handled by checking `resp["result"]["exitCode"]` after a successful HTTP
response.

### Scope

This change is intentionally limited to Sandbox data-plane HTTP JSON APIs. It
does not change WebSocket/CDP/VNC URL generation, Playwright connections, file
upload/download helpers, video download helpers, or non-Sandbox data-plane
clients.
