import os
from functools import lru_cache
from typing import Callable, Iterable, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_QUOTA_EXCEPTION_NAMES = [
    "ResourceExhausted",
    "TooManyRequests",
    "ServiceUnavailable",
]
_QUOTA_EXCEPTIONS = tuple(
    getattr(google_exceptions, name)
    for name in _QUOTA_EXCEPTION_NAMES
    if hasattr(google_exceptions, name)
)
if not _QUOTA_EXCEPTIONS:
    _QUOTA_EXCEPTIONS = (google_exceptions.GoogleAPIError,)

Logger = Optional[Callable[[str], None]]


def _normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if model_name.startswith("models/"):
        return model_name
    if model_name.startswith("gemini-"):
        return f"models/{model_name}"
    return model_name


@lru_cache(maxsize=1)
def list_available_generation_models() -> List[str]:
    try:
        return [
            model.name
            for model in genai.list_models()
            if "generateContent"
            in getattr(model, "supported_generation_methods", [])
        ]
    except Exception:
        return []


class GeminiModelManager:
    def __init__(
        self,
        preferred_models: Optional[Iterable[str]] = None,
        logger: Logger = None,
    ) -> None:
        self._logger = logger
        preferred = [_normalize_model_name(m) for m in (preferred_models or [])]
        available = list_available_generation_models()
        available_set = {_normalize_model_name(m) for m in available}

        ordered: List[str] = []
        missing: List[str] = []

        if available_set:
            for name in preferred:
                if name and name in available_set and name not in ordered:
                    ordered.append(name)
                elif name and name not in available_set:
                    missing.append(name)
        else:
            for name in preferred:
                if name and name not in ordered:
                    ordered.append(name)

        for name in available:
            normalized = _normalize_model_name(name)
            if normalized not in ordered:
                ordered.append(normalized)

        if missing:
            self._log(
                "Preferred models not available and will be skipped: "
                + ", ".join(missing)
            )

        if not ordered:
            raise RuntimeError(
                "No Gemini models with generateContent capability are available."
            )

        self._models = ordered
        self._current_index = 0
        self._model = self._init_model(self._models[self._current_index])

    @property
    def active_model_name(self) -> str:
        return self._models[self._current_index]

    @property
    def candidate_models(self) -> List[str]:
        return list(self._models)

    def _log(self, message: str) -> None:
        if self._logger:
            self._logger(message)

    def _init_model(self, model_name: str):
        self._log(f"Using Gemini model: {model_name}")
        return genai.GenerativeModel(model_name)

    def _looks_like_quota_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return any(
            keyword in message
            for keyword in ("quota", "limit", "exceed", "exhaust", "429", "too many")
        )

    def _switch_model(self) -> bool:
        if self._current_index + 1 >= len(self._models):
            return False

        self._current_index += 1
        next_model = self._models[self._current_index]
        self._model = self._init_model(next_model)
        return True

    def generate_content(self, *args, **kwargs):
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts < len(self._models):
            try:
                return self._model.generate_content(*args, **kwargs)
            except _QUOTA_EXCEPTIONS as exc:
                last_error = exc
                self._log(
                    f"Quota hit for {self.active_model_name}: {exc}. "
                    "Trying next available model..."
                )
                attempts += 1
                if not self._switch_model():
                    break
            except google_exceptions.GoogleAPIError as exc:
                if self._looks_like_quota_error(exc):
                    last_error = exc
                    self._log(
                        f"Quota-like error for {self.active_model_name}: {exc}. "
                        "Trying next available model..."
                    )
                    attempts += 1
                    if not self._switch_model():
                        break
                else:
                    raise
            except Exception as exc:
                if self._looks_like_quota_error(exc):
                    last_error = exc
                    self._log(
                        f"Quota-like signal for {self.active_model_name}: {exc}. "
                        "Trying next available model..."
                    )
                    attempts += 1
                    if not self._switch_model():
                        break
                else:
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate content and no error was captured.")

