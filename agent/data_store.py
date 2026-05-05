import json
from pathlib import Path

from agent.config import DATA_DIR


class JsonDataProvider:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir

    def load_json_data(self, file_name: str):
        path = self.data_dir / file_name
        return json.loads(path.read_text(encoding="utf-8"))

    def load_teacher_schedule(self) -> list:
        return self.load_json_data("teacher_schedule.json")

    def load_courses(self) -> dict:
        return self.load_json_data("courses.json")

    def get_teacher_schedule(self, teacher_name: str, date: str) -> dict:
        schedules = self.load_teacher_schedule()
        slots = []

        for schedule in schedules:
            if (
                schedule["teacher_name"] == teacher_name
                and schedule["date"] == date
            ):
                slots = schedule["available_slots"]
                break

        return {
            "teacher_name": teacher_name,
            "date": date,
            "available_slots": slots
        }

    def get_course_info(self, course_name: str) -> dict | None:
        courses = self.load_courses()
        return courses.get(course_name)


DATA_PROVIDER = JsonDataProvider()


def get_data_provider() -> JsonDataProvider:
    return DATA_PROVIDER


def set_data_provider(provider: JsonDataProvider) -> None:
    global DATA_PROVIDER
    DATA_PROVIDER = provider


def load_json_data(file_name: str):
    return DATA_PROVIDER.load_json_data(file_name)


def load_teacher_schedule() -> list:
    return DATA_PROVIDER.load_teacher_schedule()


def load_courses() -> dict:
    return DATA_PROVIDER.load_courses()
