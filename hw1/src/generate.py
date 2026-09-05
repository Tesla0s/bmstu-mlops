"""Вывести ответ на вопрос из конфигурации."""

from src.config import load_params
from src.model import generate, load_model


def main() -> None:
    params = load_params()
    tokenizer, model = load_model(params)
    text, n_tokens = generate(tokenizer, model, params, params["bench"]["prompt"])
    print(f"Модель: {params['model']['name']}")
    print(text)
    print(f"\n[{n_tokens} токенов]")


if __name__ == "__main__":
    main()
