import importlib.util
import os
import threading

ft_spec = importlib.util.find_spec("flet")
found = ft_spec is not None
ft = None
if not found:
    print("flet  Not Available Installing flet")
    os.system('python -m pip install pip install \'flet[all]\'')
    ft = importlib.import_module('flet')
    spam_spec = importlib.util.find_spec("flet")
    found = spam_spec is not None
    if found:
        print("\nflet Installed and Available Launching GUI")
    else:
        print("\nDone .... Please Rerun Script to continue!")
else:
    print("flet Available Launching GUI")

import flet as ft
import time

def oops(data):
    print("Launching Process ...")
    time.sleep(0.1)

    import OverGround.data_postprocessing as dp
    dp.from_app(data)
    print("Processing ... Done")


def main_app(page: ft.Page):
    # --- 1. Window Configuration ---
    page.title = "Data Processing App"
    page.theme_mode = ft.ThemeMode.LIGHT

    # Window setup
    page.window.width = 600
    page.window.height = 750
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # --- 2. Logic Functions ---
    def get_directory_result(e):
        if e.path:
            dir_input.value = e.path
            dir_input.error_text = None
            dir_input.update()
        else:
            print("Cancelled or empty path.")

    async def handle_get_directory_path(e: ft.Event[ft.Button]):
        dir_input.value = await ft.FilePicker().get_directory_path()

    async def handle_pick_files(e: ft.Event[ft.Button]):
        files = await ft.FilePicker().pick_files(allow_multiple=True)
        csv_input.value = (
            ", ".join(map(lambda f: f.name, files)) if files else "Cancelled!"
        )

    async def submit_data(e: ft.Event[ft.Button]):
        dir_input.error_text = None
        threshold_input.error_text = None

        if not dir_input.value:
            dir_input.error_text = "Please select a directory"
            dir_input.update()
            return

        try:
            threshold_val = float(threshold_input.value)
        except ValueError:
            threshold_input.error_text = "Invalid number"
            threshold_input.update()
            return

        data = {
            "directory": dir_input.value,
            "participant_id": id_input.value,
            "csv_name": csv_input.value,
            "threshold": threshold_val
        }

        result_text.value = f"Success! Processing ID: {data['participant_id']}"
        result_text.color = "green"
        result_text.update()
        my_thread = threading.Thread(target=oops, args=(data,))
        my_thread.start()
        print(f"Data Submitted: {data}")

    # --- 3. UI Setup ---

    # File Picker
    # directory_picker = ft.FilePicker()
    # directory_picker.on_result = get_directory_result

    # Directory Section
    dir_input = ft.TextField(
        label="Directory Path",
        hint_text="Path to folder...",
        icon="folder_open",
        expand=True
    )

    # FIXED: Replaced picky 'IconButton' with a standard 'Button' containing an Icon
    # This is the safest way to make an icon button in v0.80+
    dir_button = ft.Button(
        content="Open",
        on_click=handle_get_directory_path,
        width=100,  # Optional: Make it square-ish
        color="blue"
    )

    dir_row = ft.Row(
        controls=[dir_input, dir_button],
        width=400,
        alignment=ft.MainAxisAlignment.CENTER
    )

    # Participant ID
    id_input = ft.TextField(
        label="Participant ID",
        hint_text="e.g. P_001",
        icon="person",
        width=400
    )

    # CSV Name
    csv_input = ft.TextField(
        label="Info CSV Name",
        value="info.csv",
        icon="table_chart",
        width=300
    )

    csv_button = ft.Button(
        content="Select",
        on_click=handle_pick_files,
        width=100,  # Optional: Make it square-ish
        color="blue"
    )

    csv_row = ft.Row(
        controls=[csv_input, csv_button],
        width=400,
        alignment=ft.MainAxisAlignment.CENTER
    )

    # Threshold Input
    threshold_input = ft.TextField(
        label="Contact Threshold",
        value="20.0",
        suffix=ft.Text("N"),
        keyboard_type=ft.KeyboardType.NUMBER,
        icon="speed",
        width=400
    )

    # FIXED: Submit Button using 'content'
    submit_btn = ft.Button(
        content=ft.Text("Process Data"),
        on_click=submit_data,
        height=50,
        width=200
    )

    result_text = ft.Text(size=16, weight="bold")

    # --- 4. Layout ---
    main_layout = ft.Column(
        controls=[
            ft.Text("Experiment Setup", size=30, weight="bold", color="blue"),

            ft.Container(content=ft.Divider(), width=400),

            dir_row,
            id_input,
            csv_row,
            threshold_input,

            ft.Container(content=ft.Divider(), width=400),

            submit_btn,
            result_text,

        ],
        spacing=20,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    page.add(main_layout)


if __name__ == "__main__":
    ft.run(main_app)
    #ft.app(target=main, view=ft.AppView.WEB_BROWSER)