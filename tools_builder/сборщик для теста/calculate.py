import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import re
import math
import os
from tkinter import messagebox

from loader import Field

# field = Field()


class Calc:
    def __init__(self, well, gwc):
        self.results_contour, self.results_well = self.calculate_min_distances(
            well, gwc)

    def get_kH(self, h_df: pd.DataFrame, k_df: pd.DataFrame) -> pd.DataFrame:
        kH1 = h_df['kH_1'].mul(k_df['h_kh1'])
        kH2 = h_df['kH_2'].mul(k_df['h_kh2'])
        kH3 = h_df['kH_3'].mul(k_df['h_kh3'])
        result = pd.DataFrame({
            'kH1': kH1,
            'kH2': kH2,
            'kH3': kH3,
        })

        return result

    def get_kh_eff(self, h_df: pd.DataFrame, k_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({
            'Well': k_df['Well'],
            'kh_eff': h_df['f_perm'].mul(k_df['eff_kh'])
        })

        return result

    def get_Hwell(self, gah: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({
            'h_all': gah['all_gas_ass_h']
        })

        return result

    def get_NTG(self, k_df: pd.DataFrame, Hwell: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({
            'ntg': k_df['eff_kh'] / Hwell['h_all']
        })

        return result

    def get_H_to_GWC(self, well: pd.DataFrame, GWC: float) -> pd.DataFrame:
        def find_closest(x):
            if x.empty or x.isnull().all():
                return None

            try:
                abs_diff = (x - GWC).abs()
                closest_idx = abs_diff.idxmin()

                if closest_idx in x.index:
                    return x.loc[closest_idx]
                else:
                    return None
            except ValueError:
                return None

        min_well = well.groupby("Well")["z"].apply(
            lambda x: find_closest(x)).reset_index(drop=True)

        result = pd.DataFrame({
            'hgvk': (min_well - GWC).abs()
        })

        return result

    def get_Vdr(self, gas_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({
            'Vdren': gas_df['V_gas']
        })

        return result

    def get_Sdr(self, gas_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame({
            'Sdren': gas_df['S']
        })

        return result

    def calculate_min_distances(self, well: pd.DataFrame, gwc: pd.DataFrame):
        contour_tree = KDTree(gwc[['x', 'y']])
        well["Points"] = well.apply(lambda row: [row["x"], row["y"]], axis=1)
        wells = well.groupby("Well").agg({
            "Points": lambda x: list(x)
        }).reset_index()

        results_contour = []
        results_well = dict()

        for well_id, points in wells[['Well', 'Points']].itertuples(index=False):
            shorter_list = []

            distances_to_contour = [
                contour_tree.query(point)[0] for point in points]
            min_distance_contour = min(distances_to_contour)
            results_contour.append((well_id, min_distance_contour))

            all_other_points = [
                (pt, other_well_id) for other_well_id, other_points in wells[['Well', 'Points']].itertuples(index=False) if other_well_id != well_id for pt in other_points
            ]

            while len(shorter_list) != 3:
                if all_other_points:
                    other_points, other_ids = zip(*all_other_points)
                    other_wells_tree = KDTree(other_points)

                    min_distance_well = float('inf')
                    closest_well_id = None

                    for point in points:
                        distance, index = other_wells_tree.query(point)
                        if distance < min_distance_well:
                            min_distance_well = distance
                            closest_well_id = other_ids[index]

                    for i in range(len(all_other_points) - 1, -1, -1):
                        if closest_well_id in all_other_points[i]:
                            del all_other_points[i]

                    shorter_list.append((min_distance_well, closest_well_id))
                else:
                    shorter_list.append((None, None))

            results_well[well_id] = shorter_list

        def sorter(df):
            def extract_number(s):
                match = re.search(r'(\d+)', s)
                if match:
                    return int(match.group(1))
                return 0
            df['Numeric'] = df['Well'].apply(extract_number)
            df_sorted = df.sort_values(by='Numeric').drop(
                columns='Numeric').reset_index(drop=True)
            return df_sorted

        contour_df = pd.DataFrame(results_contour, columns=["Well", "minLgwc"])
        contour_df_sorted = sorter(contour_df)

        keys = []
        first_values = []
        second_values = []

        for key, value in results_well.items():
            keys.append(key)
            first_values.append([tup[0] for tup in value])
            second_values.append([tup[1] for tup in value])

        well_df = pd.DataFrame({
            'Well': keys,
            'minL1': [v[0] for v in first_values],
            'minL2': [v[1] for v in first_values],
            'minL3': [v[2] for v in first_values],
            'Shrt_well': second_values
        })
        well_df_sorted = sorter(well_df)

        return contour_df_sorted, well_df_sorted

    def get_kh_eff_shrt(self, kh_df: pd.DataFrame) -> pd.DataFrame:
        def get_eff_kh(secondary_wells):
            eff_values = []
            for well in secondary_wells:
                eff_value = kh_df[kh_df['Well'] == well]['kh_eff'].values
                if len(eff_value) > 0:
                    eff_values.append(eff_value[0])
                else:
                    eff_values.append(None)
            return eff_values
        eff_kh_df = pd.DataFrame(self.results_well['Shrt_well'].apply(
            get_eff_kh).tolist(), columns=['kHmin1', 'kHmin2', 'kHmin3'])

        return eff_kh_df

    def get_Scr(self, gah: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
        num_wells = gah.shape[0]
        scr_value = field['S'] / num_wells
        result = pd.DataFrame({'Scr': [scr_value[0]] * num_wells})

        return result

    def get_Lgs(self, well: pd.DataFrame) -> pd.DataFrame:
        vectors_data = []
        grouped = well.groupby('Well')
        for name, group in grouped:
            group = group.sort_index()
            vectors = []
            for i in range(len(group) - 1):
                x1, y1, z1 = group.iloc[i][['x', 'y', 'z']]
                x2, y2, z2 = group.iloc[i + 1][['x', 'y', 'z']]

                dx, dy, dz = x2-x1, y2-y1, z2-z1
                vector = math.sqrt(dx**2 + dy**2 + dz**2)

                vectors_data.append({
                    'Lgs': vector
                })

        vector_df = pd.DataFrame(vectors_data)
        return vector_df['Lgs']

    def get_OFR(self, ofr: pd.DataFrame, basedir: str) -> pd.DataFrame:
        all_values = ofr.values.flatten()
        num_wells = self.num_wells
        num_ofr = all_values.shape[0]
        if num_wells > num_ofr:
            messagebox.showinfo(
                "Информация", f'Не хватает значений в файле {basedir}ofr')
            exit()
        elif num_wells < num_ofr:
            all_values = all_values[:num_wells]

        result = pd.DataFrame({'OFR': all_values})

        return result

    def get_W_ID(self, gah: pd.DataFrame, file_name: str) -> pd.DataFrame:
        self.num_wells = gah.shape[0]
        result = pd.DataFrame({
            'Gen': [file_name] * self.num_wells,
            'W_ID': gah['Well']
        })

        return result


def create_excel(basedir: str = ''):
    try:
        excel_file_name = os.path.basename(os.path.dirname(basedir))

        field = Field(basedir)
        calc = Calc(field.well, field.gwc)

        df_combined = pd.concat([
            calc.get_W_ID(field.all_gas_ass_h, excel_file_name),
            calc.get_kH(field.avg_perm_zone_well, field.kh_zone_well),
            calc.get_kh_eff(field.avg_f_perm_zone_well,
                            field.kh_zone_well)['kh_eff'],
            calc.get_Hwell(field.all_gas_ass_h),
            calc.get_NTG(field.kh_zone_well,
                         calc.get_Hwell(field.all_gas_ass_h)),
            calc.get_Lgs(field.well),
            calc.get_H_to_GWC(field.well, field.GWC),
            calc.results_contour['minLgwc'],
            calc.get_Vdr(field.all_gas_ass_h),
            calc.get_Sdr(field.all_gas_ass_h),
            calc.results_well[['minL1', 'minL2', 'minL3']],
            calc.get_kh_eff_shrt(calc.get_kh_eff(
                field.avg_f_perm_zone_well, field.kh_zone_well)),
            calc.get_Scr(field.all_gas_ass_h, field.field),
            # calc.get_OFR(field.ofr, basedir)
        ],
            axis=1
        )

        df_combined.to_excel(f"{basedir}{excel_file_name}.xlsx", index=False)

    except Exception as e:
        print('Не удалось\n\n', e)
