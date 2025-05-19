import openai
import json

openai.api_key = "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"

questions = [
    {"task_id": 0, "question": "What is the capital of France?"},
    {"task_id": 1, "question": "What is 12 multiplied by 8?"},
    {"task_id": 2, "question": "Who wrote '1984'?"},
    {"task_id": 3, "question": "What is the boiling point of water in Celsius?"},
    {"task_id": 4, "question": "What is the opposite of 'hot'?"},
    {"task_id": 5, "question": "Which planet is known as the Red Planet?"},
    {"task_id": 6, "question": "What is 25% of 200?"},
    {"task_id": 7, "question": "Translate 'Bonjour' to English."},
    {"task_id": 8, "question": "How many continents are there?"},
    {"task_id": 9, "question": "What gas do humans need to breathe?"},
    {"task_id": 10, "question": "Is the sun a star or a planet?"},
    {"task_id": 11, "question": "What is the result of 9 + 10?"},
    {"task_id": 12, "question": "What color do you get by mixing blue and yellow?"},
    {"task_id": 13, "question": "What year did World War II end?"},
    {"task_id": 14, "question": "What is the chemical symbol for water?"},
    {"task_id": 15, "question": "What shape has four equal sides and angles?"},
    {"task_id": 16, "question": "How many hours are in a day?"},
    {"task_id": 17, "question": "What is the currency used in the United States?"},
    {"task_id": 18, "question": "Which animal is known as man's best friend?"},
    {"task_id": 19, "question": "What is the fastest land animal?"}
]

answers = []

for q in questions:
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": q["question"]}
        ]
    )
    submitted_answer = response['choices'][0]['message']['content'].strip()
    answers.append({
        "task_id": q["task_id"],
        "submitted_answer": submitted_answer
    })

print("Answers (answers):", json.dumps(answers, indent=2))