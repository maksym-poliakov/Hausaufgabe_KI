from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import torch
import pandas as pd  # Добавлен импорт pandas

# 1. Загрузка датасета
dataset = load_dataset("imdb")
dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset["train"].select(range(4000))  # Ограничим для скорости
test_dataset = dataset["test"].select(range(1000))

# 2. Очистка данных
def clean_text(example):
    text = example["text"]
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = ' '.join(text.split())
    example["text"] = text
    return example

train_dataset = train_dataset.map(clean_text)
test_dataset = test_dataset.map(clean_text)

# 3. Удаление дубликатов (опционально)
def remove_duplicates(dataset):
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates(subset=["text"])
    return Dataset.from_pandas(df)

train_dataset = remove_duplicates(train_dataset)
test_dataset = remove_duplicates(test_dataset)

# 4. Загрузка модели и токенизатора
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. Токенизация
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)

# 6. Метрики
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds)
    }

# 7. Аргументы тренировки
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100
)

# 8. Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)
trainer.train()

# 9. Оценка
metrics = trainer.evaluate()
print("Test metrics:", metrics)

# 10. Примеры предсказаний
test_texts = ["This movie was great!", "I hated this film.", "An average movie, nothing special."]
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        pred = output.logits.argmax(-1).item()
        print(f"Text: {text} | Prediction: {'Positive' if pred == 1 else 'Negative'}")

# 11. Сохранение модели
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 12. Сохранение очищенного датасета (для отчёта)
train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)
train_df.to_csv("cleaned_train_imdb.csv", index=False)
test_df.to_csv("cleaned_test_imdb.csv", index=False)