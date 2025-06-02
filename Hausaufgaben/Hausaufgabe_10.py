# Импорт необходимых библиотек
import os  # Работа с операционной системой (например, для работы с файлами)
import pandas as pd  # Работа с табличными данными (DataFrame), удобна для анализа и сохранения данных
import numpy as np  # Библиотека для работы с массивами и выполнения математических операций
import torch  # PyTorch для создания и обучения нейронных сетей
from datasets import Dataset  # Hugging Face Datasets для удобного представления данных в формате датасетов
from transformers import (  # Импортируем классы и функции из библиотеки transformers для работы с языковыми моделями
    AutoModelForCausalLM,  # Автоматическая загрузка модели для генерации текста (causal language modeling)
    AutoTokenizer,  # Автоматическая загрузка токенизатора для преобразования текста в числовые данные
    TrainingArguments,  # Аргументы для настройки процесса обучения модели
    Trainer,  # Класс для обучения модели с помощью удобного интерфейса
    DataCollatorForLanguageModeling  # Функция для подготовки батчей данных для обучения модели
)
from peft import (  # Импортируем инструменты для эффективного тонкого обучения модели (LoRA)
    LoraConfig,  # Конфигурация для LoRA
    get_peft_model,  # Функция для получения модели, адаптированной для PEFT
    prepare_model_for_kbit_training,  # Подготовка модели для обучения с низкой точностью (8-bit)
    TaskType  # Тип задачи (например, генерация текста)
)
from tqdm import tqdm  # Библиотека для отображения прогресс-баров при выполнении циклов
import warnings  # Работа с предупреждениями

warnings.filterwarnings("ignore")  # Отключаем предупреждения для чистоты вывода

# Аутентификация на Hugging Face
# Необходимо указать свой токен (не делитесь им публично!)
from huggingface_hub import login

HF_TOKEN = "YOUR_TOKEN_HERE"  # Замените 'YOUR_TOKEN_HERE' на ваш реальный токен
login(token=HF_TOKEN)

# Проверка доступности GPU (если есть, обучение пройдет быстрее)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== ЧАСТЬ 1: ПОДГОТОВКА ДАННЫХ =====================

# Определяем информацию о компании. Здесь содержится базовая информация, которая позже используется для генерации вопросов и ответов.
company_info = {
    "name": "TechInnovate Solutions",
    "founded": 2015,
    "headquarters": "San Francisco, California",
    "ceo": "Dr. Alex Morgan",
    "employees": 250,
    "products": [
        "DataSync Pro - A cloud-based data synchronization platform",
        "AI Assistant - An enterprise AI chatbot solution",
        "SecureConnect - An end-to-end encrypted communication tool"
    ],
    "mission": "To leverage cutting-edge technology to solve complex business challenges while promoting sustainability.",
    "values": ["Innovation", "Integrity", "Collaboration", "Sustainability"],
    "revenue": "$45 million (2023)",
    "competitors": ["DataTech Inc.", "CloudWave Systems", "Nexus AI"],
    "recent_news": "TechInnovate Solutions recently secured $30 million in Series B funding led by Venture Capital Partners."
}


# Функция для генерации пар "вопрос-ответ" на основе информации о компании.
def generate_qa_pairs(company_data, num_pairs=50):
    qa_pairs = []

    # Прямые вопросы о фактах компании
    qa_pairs.append({
        "question": f"What is the name of the company?",
        "answer": f"The company name is {company_data['name']}."
    })
    qa_pairs.append({
        "question": f"When was {company_data['name']} founded?",
        "answer": f"{company_data['name']} was founded in {company_data['founded']}."
    })
    qa_pairs.append({
        "question": f"Where is the headquarters of {company_data['name']}?",
        "answer": f"The headquarters of {company_data['name']} is located in {company_data['headquarters']}."
    })
    qa_pairs.append({
        "question": f"Who is the CEO of {company_data['name']}?",
        "answer": f"The CEO of {company_data['name']} is {company_data['ceo']}."
    })
    qa_pairs.append({
        "question": f"How many employees does {company_data['name']} have?",
        "answer": f"{company_data['name']} has approximately {company_data['employees']} employees."
    })
    qa_pairs.append({
        "question": f"What products does {company_data['name']} offer?",
        "answer": f"{company_data['name']} offers several products including: {', '.join(company_data['products'])}."
    })
    qa_pairs.append({
        "question": f"What is the mission of {company_data['name']}?",
        "answer": f"The mission of {company_data['name']} is: {company_data['mission']}"
    })
    qa_pairs.append({
        "question": f"What are the core values of {company_data['name']}?",
        "answer": f"The core values of {company_data['name']} are: {', '.join(company_data['values'])}."
    })
    qa_pairs.append({
        "question": f"What was the revenue of {company_data['name']} in 2023?",
        "answer": f"The revenue of {company_data['name']} in 2023 was {company_data['revenue']}."
    })
    qa_pairs.append({
        "question": f"Who are the main competitors of {company_data['name']}?",
        "answer": f"The main competitors of {company_data['name']} are: {', '.join(company_data['competitors'])}."
    })
    qa_pairs.append({
        "question": f"What is the recent news about {company_data['name']}?",
        "answer": f"{company_data['recent_news']}"
    })

    # Вопросы о конкретных продуктах компании
    for product in company_data['products']:
        product_name = product.split(' - ')[0]  # Получаем название продукта (до тире)
        product_desc = product.split(' - ')[1]  # Получаем описание продукта (после тире)
        qa_pairs.append({
            "question": f"What is {product_name}?",
            "answer": f"{product_name} is {product_desc}, developed by {company_data['name']}."
        })

    # Комбинированные вопросы
    qa_pairs.append({
        "question": f"Can you tell me about {company_data['name']} and its products?",
        "answer": f"{company_data['name']} is a company founded in {company_data['founded']} with headquarters in {company_data['headquarters']}. They offer products including {', '.join([p.split(' - ')[0] for p in company_data['products']])}."
    })
    qa_pairs.append({
        "question": f"Who is the CEO of {company_data['name']} and when was it founded?",
        "answer": f"The CEO of {company_data['name']} is {company_data['ceo']} and the company was founded in {company_data['founded']}."
    })

    # Дополнительные варианты вопросов с разной формулировкой
    variations = [
        {"question": f"Tell me about {company_data['name']}",
         "answer": f"{company_data['name']} is a technology company founded in {company_data['founded']} with headquarters in {company_data['headquarters']}. Led by CEO {company_data['ceo']}, the company has approximately {company_data['employees']} employees and offers various products including {', '.join([p.split(' - ')[0] for p in company_data['products']])}."},
        {"question": f"What does {company_data['name']} do?",
         "answer": f"{company_data['name']} develops and provides technology solutions including {', '.join([p.split(' - ')[0] for p in company_data['products']])}. Their mission is to {company_data['mission'].lower()}"},
        {"question": f"How big is {company_data['name']}?",
         "answer": f"{company_data['name']} has approximately {company_data['employees']} employees and reported revenue of {company_data['revenue']} in 2023."},
    ]

    qa_pairs.extend(variations)

    # Вопросы, на которые информации нет
    unknown_questions = [
        {"question": f"What was {company_data['name']}'s revenue in 2018?",
         "answer": f"I don't have information about {company_data['name']}'s revenue in 2018."},
        {"question": f"Who was the previous CEO of {company_data['name']}?",
         "answer": f"I don't have information about previous leadership at {company_data['name']}."},
        {"question": f"How much does {company_data['products'][0].split(' - ')[0]} cost?",
         "answer": f"I don't have specific pricing information for {company_data['products'][0].split(' - ')[0]}."},
    ]

    qa_pairs.extend(unknown_questions)

    return qa_pairs


# Генерируем пары "вопрос-ответ" на основе информации о компании
qa_data = generate_qa_pairs(company_info)

# Преобразуем список вопросов и ответов в DataFrame и сохраняем его в CSV-файл для проверки
qa_df = pd.DataFrame(qa_data)
qa_df.to_csv('company_qa_data.csv', index=False)
print(f"Generated {len(qa_df)} question-answer pairs")


# Функция для форматирования данных в виде инструкции, входных данных и ответа, что удобно для обучения модели (instruction tuning)
def format_for_training(question, answer):
    return f"""### Instruction: Answer the following question about {company_info['name']} accurately.

### Input: {question}

### Response: {answer}"""


# Применяем форматирование ко всем парам "вопрос-ответ"
formatted_data = [format_for_training(row['question'], row['answer']) for _, row in qa_df.iterrows()]

# ===================== ЧАСТЬ 2: НАСТРОЙКА МОДЕЛИ =====================

# Выбор модели. Здесь представлены два варианта:
# Option 1: Модель Gemma (требует аутентификации на Hugging Face)
# MODEL_NAME = "google/gemma-2b"

# Option 2: Модель TinyLlama (открытый доступ, не требует аутентификации)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Альтернативная, более доступная модель

# Загружаем токенизатор для выбранной модели. Он преобразует текст в числовое представление для модели.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Используем конец строки как токен для заполнения

# Настраиваем конфигурацию LoRA для тонкого обучения модели. LoRA позволяет эффективно обучать модель, изменяя лишь небольшую часть параметров.
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Задача генерации текста
    inference_mode=False,  # Режим обучения (не инференс)
    r=8,  # Ранг LoRA
    lora_alpha=32,  # Коэффициент масштабирования для LoRA
    lora_dropout=0.1,  # Вероятность dropout для LoRA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Модули, к которым применяется LoRA
)


# Функция для токенизации данных. Здесь текст преобразуется в последовательность чисел с ограничением максимальной длины.
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")


# Создаем датасет Hugging Face из форматированных данных
train_dataset = Dataset.from_dict({"text": formatted_data})
print(f"Dataset created with {len(train_dataset)} examples")

# Применяем токенизацию ко всему датасету
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,  # Обработка данных пакетами для ускорения
    remove_columns=["text"]  # Удаляем оригинальный текст, оставляем только числовое представление
)

# ===================== ЧАСТЬ 3: НАСТРОЙКА ОБУЧЕНИЯ =====================

# Определяем аргументы обучения, включая параметры батча, количество шагов и скорость обучения
training_args = TrainingArguments(
    output_dir="./results",  # Папка для сохранения результатов
    per_device_train_batch_size=4,  # Размер батча на устройство (GPU/CPU)
    gradient_accumulation_steps=4,  # Количество шагов для накопления градиентов (имитация большего батча)
    warmup_steps=100,  # Шаги для прогрева модели (постепенное увеличение скорости обучения)
    max_steps=200,  # Ограничение на количество шагов обучения (для экономии ресурсов)
    learning_rate=2e-4,  # Скорость обучения
    fp16=torch.cuda.is_available(),  # Использование 16-битной арифметики, если доступен GPU
    logging_dir="./logs",  # Директория для логов
    logging_steps=10,  # Частота логирования
    save_steps=200,  # Частота сохранения контрольных точек модели
    save_total_limit=2,  # Храним только 2 последних контрольных точки
    remove_unused_columns=False,  # Не удаляем неиспользуемые колонки из датасета
    report_to="none",  # Отключаем внешние отчеты (например, wandb) для экономии ресурсов
)

# Пытаемся загрузить модель в 8-битном режиме для экономии памяти
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,  # Загружаем модель в 8-битном формате
        device_map="auto"  # Автоматическое распределение по устройствам (CPU/GPU)
    )

    # Подготавливаем модель для обучения с помощью методов PEFT/LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

except Exception as e:
    print(f"Error loading model in 8-bit: {e}")
    print("Falling back to 16-bit precision...")

    # Если загрузка в 8-бит не удалась, загружаем модель в 16-битном режиме
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

# Data collator для подготовки батчей данных во время обучения.
# Здесь не используется маскированное языковое моделирование (mlm=False), так как задача — causal LM.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Инициализируем объект Trainer для управления процессом обучения модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ===================== ЧАСТЬ 4: ОБУЧЕНИЕ И ОЦЕНКА =====================

# Запускаем процесс обучения модели
print("Starting training...")
trainer.train()

# После обучения сохраняем дообученную модель и токенизатор для дальнейшего использования
model.save_pretrained("./fine_tuned_company_model")
tokenizer.save_pretrained("./fine_tuned_company_model")
print("Model training complete and saved")


# ===================== ЧАСТЬ 5: ПРИМЕР ИНФЕРЕНСА =====================

# Функция для генерации ответа модели на заданный вопрос
def generate_response(question, model, tokenizer, device, max_length=200):
    # Формируем запрос (prompt) с инструкцией, вопросом и местом для ответа
    prompt = f"""### Instruction: Answer the following question about {company_info['name']} accurately.

### Input: {question}

### Response:"""

    # Токенизируем запрос, преобразуя его в формат, подходящий для модели
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Генерируем ответ модели без вычисления градиентов (режим инференс)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=len(inputs["input_ids"][0]) + max_length,  # Общая максимальная длина ответа
            temperature=0.7,  # Контролирует "творчество" модели: чем ниже, тем предсказуемее ответ
            top_p=0.9,  # Используется для nucleus sampling (отсеивание наименее вероятных токенов)
            pad_token_id=tokenizer.eos_token_id  # Токен для заполнения
        )

    # Декодируем сгенерированный ответ в читаемый текст
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только ту часть, которая идет после метки "### Response:"
    response = response.split("### Response:")[1].strip()
    return response


# Тестовые вопросы для проверки работы дообученной модели
test_questions = [
    f"What is {company_info['name']}?",
    f"Who is the CEO of {company_info['name']}?",
    f"What products does {company_info['name']} offer?",
    "What was the company's revenue in 2020?"  # Этот вопрос демонстрирует обработку запроса, если информации нет
]

print("\nTesting the fine-tuned model:")
for question in test_questions:
    response = generate_response(question, model, tokenizer, device)
    print(f"\nQ: {question}")
    print(f"A: {response}")
    print("-" * 50)


# ===================== ЧАСТЬ 6: ПРОСТОЙ ИНТЕРФЕЙС =====================

# Функция для интерактивного режима, где пользователь может задавать свои вопросы
def interactive_qa():
    print("\n" + "=" * 50)
    print(f"Ask questions about {company_info['name']}")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            break

        response = generate_response(user_question, model, tokenizer, device)
        print(f"\nAnswer: {response}")


# Запускаем интерактивную сессию вопросов и ответов
print("\nStarting interactive Q&A session...")
interactive_qa()