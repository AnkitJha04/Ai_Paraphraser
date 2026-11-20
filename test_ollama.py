"""Run a small, human-readable check of the local Ollama installation."""

from modules.paraphraser import DEFAULT_MODEL, get_installed_models, paraphrase_text


def main() -> int:
    print("Checking Ollama connection…")
    try:
        models = get_installed_models()
    except Exception as exc:
        print(f"Connection failed: {exc}")
        print("Open Ollama (or run `ollama serve`) and try again.")
        return 1

    print("Connection: OK")
    if not models:
        print("No models are installed. Install one with:")
        print(f"  ollama pull {DEFAULT_MODEL}")
        return 1

    print("Installed models:")
    for model in models:
        print(f"  - {model}")
    if DEFAULT_MODEL not in models:
        print(f"\nDefault model `{DEFAULT_MODEL}` is not installed.")
        print("Select one of the listed models in the app, or install the default model.")
        return 1

    try:
        output = paraphrase_text(
            "AI tools can help students learn more effectively.",
            task="Paraphrase",
            tone="Professional",
            model=DEFAULT_MODEL,
        )
    except Exception as exc:
        print(f"Generation test failed: {exc}")
        return 1

    print("\nGeneration: OK")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
