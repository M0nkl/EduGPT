# find_working_models.py
import requests
from config import settings


def test_model(model_name, api_key):
    """Тестирует модель на Hugging Face"""
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Простой тестовый промпт
    data = {
        "inputs": "Ответь одним предложением: Что такое программирование?",
        "parameters": {"max_new_tokens": 50}
    }

    try:
        print(f"🧪 Тестируем: {model_name}")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            answer = result[0]['generated_text'][:100] if isinstance(result, list) else str(result)[:100]
            print(f"   ✅ РАБОТАЕТ: {answer}")
            return True, url
        elif response.status_code == 503:
            print(f"   ⏳ МОДЕЛЬ ЗАГРУЖАЕТСЯ (503)")
            return True, url  # Модель есть, но загружается
        else:
            print(f"   ❌ Ошибка {response.status_code}: {response.text[:100]}")
            return False, None

    except Exception as e:
        print(f"   💥 Ошибка: {e}")
        return False, None


# Список моделей, которые скорее всего работают
MODELS_TO_TEST = [
    # Русские модели
    "sberbank-ai/rugpt3small_based_on_gpt2",
    "sberbank-ai/rugpt3medium_based_on_gpt2",
    "sberbank-ai/rugpt3large_based_on_gpt2",
    "ai-forever/rugpt3large_based_on_gpt2",
    "DeepPavlov/rubert-base-cased",
    "cointegrated/rubert-tiny2",

    # Английские модели (могут отвечать на русском)
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large",
    "gpt2",
    "gpt2-medium",
    "distilgpt2",
    "facebook/blenderbot-400M-distill",
    "microsoft/DialoGPT-small",

    # Многоязычные модели
    "facebook/m2m100_418M",
    "google/mt5-small",

    # Специальные модели для вопрос-ответ
    "deepset/roberta-base-squad2",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
]


def find_working_models():
    api_key = settings.HUGGINGFACE_API_KEY
    working_models = []

    print("🔍 Ищем работающие модели...")

    for model in MODELS_TO_TEST:
        works, url = test_model(model, api_key)
        if works:
            working_models.append((model, url))

    print(f"\n🎯 НАЙДЕНО РАБОТАЮЩИХ МОДЕЛЕЙ: {len(working_models)}")
    for model, url in working_models:
        print(f"   📍 {model}")
        print(f"   🔗 {url}")

    return working_models


if __name__ == "__main__":
    find_working_models()