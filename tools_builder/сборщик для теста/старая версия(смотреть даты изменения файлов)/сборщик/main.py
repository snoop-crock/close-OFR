from pathlib import Path
import pandas as pd
from tkinter import Tk, filedialog, messagebox

from calculate import create_excel

def create_generation_excel(root):
    folders = [f.name for f in root.iterdir() if f.is_dir() and not f.name.startswith(".") and not f.name.startswith("__")]

    for folder in folders:
        try:
            create_excel(f'{root}/{folder}/')
        except Exception as e:
            print(e)

    print('Success')

def create_pre_dataset(root):
    all_data = []

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            for file in subfolder.rglob("*.xlsx"):
                print(f"Обрабатываю файл: {file}")
                
                try:
                    df = pd.read_excel(file)
                    all_data.append(df)
                except Exception as e:
                    messagebox.showwarning("Информация", f"Ошибка при обработке {file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        output_file = root / "Pre_dataset.xlsx"
        combined_df.to_excel(output_file, index=False)

        print(f"Объединенные данные сохранены в: {output_file}")
    else:
        messagebox.showwarning("Информация", "Не найдено ни одного файла с расширением .xlsx")

def create_dataset(root):
    input_file = f"{root}/Pre_dataset.xlsx"
    output_file = f"{root}/Dataset.xlsx"
    
    df = pd.read_excel(input_file)
    column_to_remove = ["W_ID", "Gen"]
    for col in column_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Колонки '{col}' удалена.")
        else:
            messagebox.showwarning("Информация", f"Колонка '{col}' не найдена в файле.")

    df.to_excel(output_file, index=False)

    print(f"Обновленный файл сохранен как: {output_file}")

def select_folder_and_combine():
    root = Tk()
    root.withdraw()
    root_folder = Path(filedialog.askdirectory(title="Выберите корневую папку"))
    
    if not root_folder:
        messagebox.showinfo("Информация", "Вы не выбрали папку.")
        return

    create_generation_excel(root_folder)
    create_pre_dataset(root_folder)
    create_dataset(root_folder)

    messagebox.showwarning("Готово", f"Все готово!")

if __name__ == '__main__':
    select_folder_and_combine()

