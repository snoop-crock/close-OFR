import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def load_file(file_type):
    """Функция для выбора файла через диалоговое окно."""
    file_path = filedialog.askopenfilename(title=f"Выберите {file_type} файл", filetypes=[
                                           ("Excel", "*.xlsx"), ("Текстовый", "*.txt"), ("Все файлы", "*.*")])
    return file_path


def process_files(pre_dataset_path, temp_otbora_path):
    """Функция обработки файлов и добавления столбца Temp в Excel."""
    try:
        # Загрузка Pre_dataset.xlsx
        pre_df = pd.read_excel(pre_dataset_path)
        if 'Gen' not in pre_df.columns:
            raise ValueError("Файл Pre_dataset.xlsx не содержит колонку 'Gen'")

        # Загрузка temp_otbora (табуляция или пробел в качестве разделителя)
        temp_df = pd.read_csv(temp_otbora_path, sep='\s+',
                              header=None, names=["Index", "Value"])
        temp_dict = dict(zip(temp_df["Index"].astype(
            str) + ".Generation", temp_df["Value"]))

        # Добавление нового столбца Temp
        pre_df["Temp"] = pre_df["Gen"].map(temp_dict)

        # Сохранение обновленного файла
        output_path = pre_dataset_path.replace(".xlsx", "_updated.xlsx")
        pre_df.to_excel(output_path, index=False)
        messagebox.showinfo("Готово", f"Файл успешно сохранен: {output_path}")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def main():
    """Главное окно Tkinter."""
    root = tk.Tk()
    root.title("Обновление Pre_dataset.xlsx")
    root.geometry("400x200")

    pre_dataset_path = tk.StringVar()
    temp_otbora_path = tk.StringVar()

    def select_pre_dataset():
        pre_dataset_path.set(load_file("Pre_dataset"))

    def select_temp_otbora():
        temp_otbora_path.set(load_file("temp_otbora"))

    def run_process():
        if not pre_dataset_path.get() or not temp_otbora_path.get():
            messagebox.showwarning("Ошибка", "Выберите оба файла!")
            return
        process_files(pre_dataset_path.get(), temp_otbora_path.get())

    tk.Button(root, text="Выбрать Pre_dataset.xlsx",
              command=select_pre_dataset).pack(pady=5)
    tk.Label(root, textvariable=pre_dataset_path).pack()

    tk.Button(root, text="Выбрать temp_otbora",
              command=select_temp_otbora).pack(pady=5)
    tk.Label(root, textvariable=temp_otbora_path).pack()

    tk.Button(root, text="Обработать файлы", command=run_process).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
