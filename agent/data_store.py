import json

from agent.config import DATA_DIR


def load_json_data(file_name: str):
    path = DATA_DIR / file_name
    return json.loads(path.read_text(encoding="utf-8"))


def load_teacher_schedule() -> list:
    return load_json_data("teacher_schedule.json")


def load_courses() -> dict:
    return load_json_data("courses.json")
