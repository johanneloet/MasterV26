import random
import string
import csv
import os

FILE_NAME = "participant_mapping.csv"


def generate_code(length=8):
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def load_data():
    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_data(data):
    with open(FILE_NAME, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["withdrawal_code", "participant_id"])
        writer.writeheader()
        writer.writerows(data)


def generate_new_participant(participant_id):
    data = load_data()

    # check participant_id not already used
    existing_ids = {row["participant_id"] for row in data}
    if participant_id in existing_ids:
        raise ValueError(f"Participant ID '{participant_id}' already exists")

    # generate unique withdrawal code
    existing_codes = {row["withdrawal_code"] for row in data}
    code = generate_code()
    while code in existing_codes:
        code = generate_code()

    data.append({
        "withdrawal_code": code,
        "participant_id": participant_id
    })

    save_data(data)

    return code

# GENERATE PARTICIPANT
if __name__ == '__main__':
    # define participant id here
    pid = 'akso_8'
    code = generate_new_participant(pid)
    print(code)