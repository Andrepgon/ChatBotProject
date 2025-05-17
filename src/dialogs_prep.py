import re

data_path = "dialogs.txt"
pairs = []

with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        #Divides each line as separate strings
        parts = re.split(r'\s{2,}|\t+', line)

        if len(parts) == 2: #only adding lines that have a question and an answer
            question, answer = parts
            pairs.append((question.strip(), answer.strip()))


