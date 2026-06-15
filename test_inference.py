from pathlib import Path

from backend.inference import TinyStoriesGenerator


PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 500


def find_checkpoint() -> Path:
    candidates = [
        Path(__file__).with_name("tinystories_model.pt"),
        Path(__file__).parent / "backend" / "tinystories_model.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(f"tinystories_model.pt was not found. Searched:\n{searched}")


def main():
    generator = TinyStoriesGenerator(find_checkpoint())
    print(f"Using device: {generator.device}")
    stories = generator.generate_stories(
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        story_limit=4,
    )

    for index, story in enumerate(stories, start=1):
        print(f"\nStory {index}:\n{story}")


if __name__ == "__main__":
    main()
