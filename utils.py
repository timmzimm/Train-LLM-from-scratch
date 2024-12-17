import json
from datasets import load_dataset

# Load dataset
ds_train = load_dataset("microsoft/orca-agentinstruct-1M-v1", split="analytical_reasoning")
ds_val = load_dataset("microsoft/orca-agentinstruct-1M-v1", split="text_modification")

# Проверка процесса извлечения текстов
def extract_texts_debug(dataset, max_samples):
    texts = []
    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        print(f"Example {i}: {ex}")
        try:
            # Попробуем десериализовать JSON в поле 'messages'
            messages = json.loads(ex['messages'])
            print(f"Decoded messages: {messages}")
            # Объединяем все содержимое 'content' от 'user' и 'assistant'
            text = " ".join([msg['content'] for msg in messages if msg['role'] in ['user', 'assistant']])
            print(f"Extracted text: {text}")
            if text.strip():
                texts.append(text)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing example {i}: {e}")
        print("-" * 50)
    return texts

# Протестируем процесс извлечения для тренировочного датасета
max_train_samples = 5  # Проверим только первые 5 примеров
train_texts_debug = extract_texts_debug(ds_train, max_train_samples)

print(f"Extracted {len(train_texts_debug)} training texts.")

print("Number of train texts:", len(train_texts))
print("Number of val texts:", len(val_texts))