from pynput import keyboard, mouse
import time
import threading

# ----------------------------
# Keyboard Tracking
# ----------------------------
key_times = []
last_key_time = None
pause_durations = []

def on_key_press(key):
    global last_key_time, key_times, pause_durations
    now = time.time()
    if last_key_time is not None:
        interval = now - last_key_time
        key_times.append(interval)

        # Detect pauses longer than 2 sec
        if interval > 2.0:
            pause_durations.append(interval)

    last_key_time = now

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()

# ----------------------------
# Mouse Tracking
# ----------------------------
mouse_idle_start = time.time()
mouse_idle_time = []
last_mouse_move = time.time()

def on_move(x, y):
    global last_mouse_move, mouse_idle_start
    now = time.time()
    idle_duration = now - last_mouse_move
    if idle_duration > 2.0:  # More than 2 sec idle
        mouse_idle_time.append(idle_duration)
    last_mouse_move = now

def on_click(x, y, button, pressed):
    pass  # We can track clicks if needed

def start_mouse_listener():
    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        listener.join()


def calculate_metrics():
    while True:
        time.sleep(10)  # Every 10 sec, print metrics
        if key_times:
            avg_typing_interval = sum(key_times) / len(key_times)
        else:
            avg_typing_interval = None

        max_pause = max(pause_durations) if pause_durations else 0
        avg_mouse_idle = sum(mouse_idle_time) / len(mouse_idle_time) if mouse_idle_time else 0

        print("\n--- Typing & Mouse Metrics ---")
        print(f"Avg Typing Interval: {avg_typing_interval:.2f} sec" if avg_typing_interval else "No typing data yet")
        print(f"Max Pause Duration: {max_pause:.2f} sec")
        print(f"Avg Mouse Idle: {avg_mouse_idle:.2f} sec")
        print("-------------------------------")


if __name__ == "__main__":
    # Start keyboard + mouse listeners in separate threads
    threading.Thread(target=start_keyboard_listener, daemon=True).start()
    threading.Thread(target=start_mouse_listener, daemon=True).start()

    # Start metrics calculator
    calculate_metrics()
