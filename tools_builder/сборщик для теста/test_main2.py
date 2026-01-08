from pathlib import Path
import pandas as pd
from tkinter import Tk, filedialog, messagebox

from calculate import create_excel


def create_generation_excel(root):
    create_excel(f'{root}/')

    print('Success')


def select_folder_and_combine():
    root = Tk()
    root.withdraw()
    root_folder = Path(filedialog.askdirectory(
        title="Выберите корневую папку"))

    if not root_folder:
        messagebox.showinfo("Информация", "Вы не выбрали папку.")
        return

    create_generation_excel(root_folder)

    messagebox.showwarning("Готово", f"Все готово!")


if __name__ == '__main__':
    select_folder_and_combine()
