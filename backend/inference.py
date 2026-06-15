from pathlib import Path
import re
from typing import Any

import torch

try:
    from .model import LanguageModel, configure_model
except ImportError:
    from model import LanguageModel, configure_model


class UnknownTokenError(ValueError):
    def __init__(self, tokens):
        self.tokens = tokens
        printable = ", ".join(repr(token) for token in tokens)
        super().__init__(f"Prompt contains unknown characters: {printable}")


class TinyStoriesGenerator:
    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = self._load_checkpoint()
        self._validate_checkpoint(checkpoint)

        self.stoi = checkpoint["stoi"]
        raw_itos = checkpoint["itos"]
        self.itos = (
            {int(key): value for key, value in raw_itos.items()}
            if isinstance(raw_itos, dict)
            else raw_itos
        )

        configure_model(self._get_config(checkpoint), self.device)
        self.model = LanguageModel().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path.resolve()}"
            )

        try:
            return torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            return torch.load(self.checkpoint_path, map_location=self.device)

    @staticmethod
    def _validate_checkpoint(checkpoint):
        required = {"model_state_dict", "stoi", "itos"}
        missing = required.difference(checkpoint)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Checkpoint is missing required fields: {names}")

        TinyStoriesGenerator._get_config(checkpoint)

    @staticmethod
    def _get_config(checkpoint):
        config = checkpoint.get("config")
        if config is None:
            config = {
                key: checkpoint[key]
                for key in (
                    "vocab_size",
                    "block_size",
                    "n_embd",
                    "n_head",
                    "n_layer",
                )
                if key in checkpoint
            }
            # The training reference fixes dropout at 0.2.
            config["dropout"] = checkpoint.get("dropout", 0.2)

        config_fields = {
            "vocab_size",
            "block_size",
            "n_embd",
            "n_head",
            "n_layer",
            "dropout",
        }
        missing_config = config_fields.difference(config)
        if missing_config:
            names = ", ".join(sorted(missing_config))
            raise ValueError(f"Checkpoint config is missing fields: {names}")
        return config

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def unknown_tokens(self, text: str) -> list[str]:
        return list(
            dict.fromkeys(
                token for token in self.tokenize(text) if token not in self.stoi
            )
        )

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        unknown = list(
            dict.fromkeys(token for token in tokens if token not in self.stoi)
        )
        if unknown:
            raise UnknownTokenError(unknown)
        return [self.stoi[token] for token in tokens]

    def decode(self, token_ids: list[int]) -> str:
        try:
            tokens = [self.itos[token_id] for token_id in token_ids]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unknown token id in generated output: {exc}") from exc
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+'\s+", "'", text)
        return text

    def generate_text(self, prompt: str, max_new_tokens: int = 1000) -> str:
        encoded = self.encode(prompt)
        context = torch.tensor(
            encoded,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            generated = self.model.generate(
                context,
                max_new_token=max_new_tokens,
            )
        return self.decode(generated[0].tolist())

    def generate_stories(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        story_limit: int = 4,
    ) -> list[str]:
        text = self.generate_text(prompt, max_new_tokens=max_new_tokens)
        stories = [
            story.strip()
            for story in re.split(r"(?:\*\*EOS\*\*|__EOS__)", text)
            if story.strip()
        ][:story_limit]

        return [
            story
            if story.lower().startswith(prompt.lower())
            else f"{prompt.rstrip()} {story.lstrip()}"
            for story in stories
        ]
