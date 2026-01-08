import pandas as pd
import os
import re


"""
В некоторых файлах происходит форматирование стобцов и строк для более простого взаимодействия
"""


class Field:
    def __init__(self, dirname: str = ''):
        self.GWC = -1035
        self.dirname = dirname

        self.all_gas_ass_h = self._all_gas_ass_h()
        self.avg_f_perm_zone_well = self._avg_f_perm_zone_well()
        self.avg_perm_zone_well = self._avg_perm_zone_well()
        self.field = self._field()
        self.gwc = self._GWC()
        self.kh_zone_well = self._kh_zone_well()
        try:
            self.ofr = self._ofr()
        except:
            pass
        self.well = self._well()

    def __safe_convert(self, value):
        """
        Функция для безопасного преобразования строк в числа
        """
        try:
            return float(value)
        except ValueError:
            return value

    def _avg_perm_zone_well(self):
        """
        Получаем данные из файла avg_perm_zone_well
        """
        skip_rows = [1] + [x for x in range(2, 1000, 5)]

        df = pd.read_csv(f'{self.dirname}avg_perm_zone_well',
                         sep="\t", header=0, skiprows=skip_rows)
        df.columns = df.columns.str.extract(r"\((.*?)\)", expand=False)
        df.rename(columns={"Voronogo": "Well"}, inplace=True)
        df = df.map(self.__safe_convert)

        reshaped = df["INIT_PERMX"].values.reshape(-1, 4)
        result = pd.DataFrame(
            reshaped, columns=[f"kH_{i}" for i in range(reshaped.shape[1])])
        result["Well"] = result.index.to_series().apply(lambda x: f'W {x+1}')
        result.insert(0, "Well", result.pop("Well"))

        return result

    def _kh_zone_well(self):
        """
        Получаем данные из файла kh_zone_well
        """

        df = pd.read_csv(f'{self.dirname}kh_zone_well',
                         sep="\t", header=0, skiprows=[1])
        df.columns = df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
        df.rename(columns={"Дискретное свойство": "Well"}, inplace=True)
        df = df.map(self.__safe_convert)

        return df

    def _all_gas_ass_h(self):
        """
        Получаем данные из файла all_gas_ass_h
        """

        df = pd.read_csv(f'{self.dirname}all_gas_ass_h',
                         sep="\t", header=0, skiprows=[1])
        df.columns = df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
        df.rename(columns={"Дискретное свойство": "Well"}, inplace=True)
        df = df.map(self.__safe_convert)
        return df

    def _avg_f_perm_zone_well(self):
        """
        Получаем данные из файла avg_f_perm_zone_well
        """

        df = pd.read_csv(f'{self.dirname}avg_f_perm_zone_well',
                         sep="\t", header=0, skiprows=[1])
        df.columns = df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
        df.rename(columns={"Дискретное свойство": "Well"}, inplace=True)
        df = df.map(self.__safe_convert)
        return df

    def _GWC(self):
        """
        Получаем данные из файла GWC
        """

        df = pd.read_csv(f'{self.dirname}GWC', sep=" ",
                         names=['x', 'y', 'z'], header=None)
        df = df.map(self.__safe_convert)
        return df

    def _well(self):
        """
        Получаем данные из файла well
        """
        temp_data = []
        with open(f'{self.dirname}well', 'r', encoding='utf-8') as well_file:
            for line in well_file:
                stripped_line = line.strip()
                if stripped_line:
                    stripped_line = stripped_line.replace(';', '')
                    temp_data.append(stripped_line)

        matches = 'W 0'
        for index, item in enumerate(temp_data[:]):

            if "welltrack" in item:
                matches = re.findall(r"'(.*?)'", item)[0]
            temp_data[index] = "\t".join(
                [matches, item.replace('       ', '\t')])

        for item in temp_data[:]:
            if "welltrack" in item:
                temp_data.remove(item)

        with open(f'{self.dirname}well_temp', 'w', encoding='utf-8') as temp_file:
            for line in temp_data:
                temp_file.write(line + "\n")

        skip_rows = [x for x in range(0, len(temp_data), 3)]
        df = pd.read_csv(f"{self.dirname}well_temp", sep='\t', names=[
                         'Well', 'x', 'y', 'z', 'md'], header=None, skiprows=skip_rows)

        if os.path.exists(f'{self.dirname}well_temp'):
            os.remove(f'{self.dirname}well_temp')

        df = df.map(self.__safe_convert)
        return df

    def _field(self):
        """
        Получаем данные из файла field
        """
        df = pd.read_csv(f'{self.dirname}filed', sep="\t", header=0)
        df.columns = df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
        df.rename(columns={"Проницаемость по X": "INIT_PERMX",
                  "Запасы газа": "gas_volume"}, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        df = df.map(self.__safe_convert)
        return df

    def _ofr(self):
        """
        Получаем данные из файла ofr
        """
        df = pd.read_csv(f'{self.dirname}ofr', sep="\t", header=0)
        VAR_columns = df.loc[:, df.columns.str.contains("VAR_")]
        VAR_columns = VAR_columns.replace(",", ".", regex=True)
        VAR_columns = VAR_columns.map(self.__safe_convert)

        normalized_data = VAR_columns.apply(
            lambda row: row / row.sum(), axis=1)
        return normalized_data
