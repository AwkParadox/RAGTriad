from gemini_model_manager import list_available_generation_models


def main():
    models = list_available_generation_models()
    if not models:
        print("No Gemini models with generateContent capability were found.")
        return

    print("Available models:")
    for model in models:
        print(f"  - {model}")


if __name__ == "__main__":
    main()