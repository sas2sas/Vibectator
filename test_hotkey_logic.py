"""Проверка логики хоткея на заглушке: без модели, микрофона и реальных клавиш."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dictate  # noqa: E402


class Fake:
    def __init__(self, mode="hold"):
        self.state = "idle"
        self.cfg = {"mode": mode}
        self.combo = ["ctrl", "alt", "space"]
        self.log = []

    def start_recording(self):
        self.log.append("start")
        self.state = "recording"

    def stop_recording(self, cancelled=False):
        self.log.append("cancel" if cancelled else "stop")
        self.state = "idle" if cancelled else "working"


def press(app):
    dictate.App._on_press(app)


def release(app):
    """То, что делает опрос при отпускании комбинации в режиме hold."""
    if app.cfg["mode"] == "hold" and app.state == "recording":
        app.stop_recording()


print("== hold (держать) ==")
app = Fake("hold")
press(app)
print("нажали:", app.state, app.log)
release(app)
print("отпустили:", app.state, app.log)

print("\n== toggle ==")
app = Fake("toggle")
press(app)
print("первое нажатие:", app.state, app.log)
press(app)
print("второе нажатие:", app.state, app.log)

print("\n== Esc отменяет ==")
app = Fake("hold")
press(app)
app.stop_recording(cancelled=True)
print("esc:", app.state, app.log)

print("\n== карта клавиш ==")
for name in ("ctrl", "alt", "space", "esc", "a", "enter", "несуществующая"):
    print(f"{name:15} -> {dictate._vk(name)}")
