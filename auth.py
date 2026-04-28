"""
LiteLLM proxy custom authentication: pre-shared keys, no database required.

Wired in `litellm_config.yaml` via `general_settings.custom_auth`. LiteLLM
calls `user_api_key_auth(request, api_key)` for every incoming request whose
`Authorization: Bearer <key>` is *not* the master key. We accept the key if
it's listed in our env-var-driven allowlist; otherwise we 401.

Required env vars (set on the pod by `start_services.sh` from a side file
that's gitignored):
    VOXTRAL_KEY_OWNER       - the operator's personal key
    VOXTRAL_KEY_COLLEAGUE   - the colleague's key (revocable independently)

Optional env vars: any number of `VOXTRAL_KEY_<NAME>` are picked up.
"""

import os
from fastapi import HTTPException
from litellm.proxy._types import UserAPIKeyAuth


def _load_keys() -> dict[str, str]:
    """Return {api_key: user_id} mapping built from env."""
    keys: dict[str, str] = {}
    for env_name, value in os.environ.items():
        if env_name.startswith("VOXTRAL_KEY_") and value.strip():
            user_id = env_name[len("VOXTRAL_KEY_"):].lower()
            keys[value.strip()] = user_id
    return keys


_KEYS: dict[str, str] = _load_keys()


async def user_api_key_auth(request, api_key: str = "") -> UserAPIKeyAuth:
    if not api_key:
        raise HTTPException(status_code=401, detail="missing api key")
    api_key = api_key.strip()
    user_id = _KEYS.get(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid api key")
    return UserAPIKeyAuth(api_key=api_key, user_id=user_id)
