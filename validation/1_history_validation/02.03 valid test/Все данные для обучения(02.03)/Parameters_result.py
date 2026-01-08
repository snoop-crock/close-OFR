#  Этот файл был сгенерирован tNavigator v22.1-1668-g13327b6d9b7.
#  Copyright (C) RFDynamics 2005-2022.
#  Все права защищены.

# This file is MACHINE GENERATED! Do not edit.

#api_version=v0.0.47

from __main__.tnav.workflow import *
from tnav_debug_utilities import *
from datetime import datetime, timedelta


declare_workflow (workflow_name="Parameters",
      variables=[])


Parameters_variables = {

}

def Parameters (variables = Parameters_variables):
    pass
    check_launch_method ()


    begin_user_imports ()
    end_user_imports ()

    begin_wf_item (index = 1)
    grid_property_create_voronoi_regions_3d (wells=find_object (name="Wells",
          type="gt_wells_entity"),
          grid=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_well_filter=False,
          well_filter=find_object (name="Расстановка скважин",
          type="WellFilter"),
          use_radius=False,
          radius=500)
    end_wf_item (index = 1)


    begin_wf_item (index = 2)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="f_perm",
          type="Grid3dProperty"),
          use_filter=True,
          user_cut_for_filter=find_object (name="SGAS",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="greater",
          value=0.1),
          formula="dynamic_property (\"INIT_PERMX\")*dynamic_property (\"KRG\")",
          variables=variables)
    end_wf_item (index = 2)


    begin_wf_item (index = 3)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="Zone_kH",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="if Z<1030 then 1\nelseif Z<1035 then 2\nelseif Z<1058 then 3\nelse 0\nendif",
          variables=variables)
    end_wf_item (index = 3)


    begin_wf_item (index = 4, name = "h_eff_kh")
    map_2d_create_by_grid_property (grid=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          use_user_cut=True,
          user_cut=find_object (name="SGAS",
          type="gt_tnav_cube_3d_data"),
          comparator=Comparator (rule="greater",
          value=0.1),
          use_user_cut_second=False,
          user_cut_second=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          comparator_second=Comparator (rule="not_equals",
          value=0),
          continuous_properties=True,
          continues_cube_and_map_table=[{"use" : True, "cube" : find_object (name="INIT_NTG",
          type="gt_tnav_cube_3d_data"), "map_2d" : find_object (name="map_eff_kh",
          type="Map2d"), "method" : "net", "smooth" : False, "blocked_wells" : None}],
          discrete_properties=True,
          discrete_cube_and_map_table=[],
          smoothing_radius=10,
          ignore_faults=False,
          set_na_instead_of_zero=False,
          grid_2d_source="custom",
          subdivision=3,
          grid_2d=Grid2D (step_x=100,
          step_y=100,
          area=Rectangle (origin_x=0,
          origin_y=0,
          size_x=1000,
          size_y=1000,
          angle=0)),
          grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True,
          autodetect_box=False,
          min_x=6500,
          min_y=2250,
          length_x=13500,
          length_y=33300,
          margin_x=0,
          margin_y=0,
          consider_blank_nodes=False,
          autodetect_angle=False,
          angle=0,
          autodetect_grid=False,
          grid_adjust_mode="step",
          step_x=100,
          step_y=100,
          counts_x=0,
          counts_y=0,
          ignore_steps=False,
          sample_object=absolute_object_name (name=None,
          typed_name=[typed_object_names (obj_name="main_grid",
          obj_type="Grid3d")])))
    end_wf_item (index = 4)


    begin_wf_item (index = 5, name = "h_kh1")
    map_2d_create_by_grid_property (grid=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          use_user_cut=True,
          user_cut=find_object (name="Zone_kH",
          type="Grid3dProperty"),
          comparator=Comparator (rule="equals",
          value=1),
          use_user_cut_second=False,
          user_cut_second=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          comparator_second=Comparator (rule="not_equals",
          value=0),
          continuous_properties=True,
          continues_cube_and_map_table=[{"use" : True, "cube" : find_object (name="INIT_NTG",
          type="gt_tnav_cube_3d_data"), "map_2d" : find_object (name="map_h_kh_1",
          type="Map2d"), "method" : "net", "smooth" : False, "blocked_wells" : None}],
          discrete_properties=True,
          discrete_cube_and_map_table=[],
          smoothing_radius=10,
          ignore_faults=False,
          set_na_instead_of_zero=False,
          grid_2d_source="custom",
          subdivision=3,
          grid_2d=Grid2D (step_x=100,
          step_y=100,
          area=Rectangle (origin_x=0,
          origin_y=0,
          size_x=1000,
          size_y=1000,
          angle=0)),
          grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True,
          autodetect_box=False,
          min_x=6500,
          min_y=2250,
          length_x=13500,
          length_y=33300,
          margin_x=0,
          margin_y=0,
          consider_blank_nodes=False,
          autodetect_angle=False,
          angle=0,
          autodetect_grid=False,
          grid_adjust_mode="step",
          step_x=100,
          step_y=100,
          counts_x=0,
          counts_y=0,
          ignore_steps=False,
          sample_object=absolute_object_name (name=None,
          typed_name=[typed_object_names (obj_name="main_grid",
          obj_type="Grid3d")])))
    end_wf_item (index = 5)


    begin_wf_item (index = 6, name = "h_kh2")
    map_2d_create_by_grid_property (grid=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          use_user_cut=True,
          user_cut=find_object (name="Zone_kH",
          type="Grid3dProperty"),
          comparator=Comparator (rule="equals",
          value=2),
          use_user_cut_second=False,
          user_cut_second=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          comparator_second=Comparator (rule="not_equals",
          value=0),
          continuous_properties=True,
          continues_cube_and_map_table=[{"use" : True, "cube" : find_object (name="INIT_NTG",
          type="gt_tnav_cube_3d_data"), "map_2d" : find_object (name="map_h_kh_2",
          type="Map2d"), "method" : "net", "smooth" : False, "blocked_wells" : None}],
          discrete_properties=True,
          discrete_cube_and_map_table=[],
          smoothing_radius=10,
          ignore_faults=False,
          set_na_instead_of_zero=False,
          grid_2d_source="custom",
          subdivision=3,
          grid_2d=Grid2D (step_x=100,
          step_y=100,
          area=Rectangle (origin_x=0,
          origin_y=0,
          size_x=1000,
          size_y=1000,
          angle=0)),
          grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True,
          autodetect_box=False,
          min_x=6500,
          min_y=2250,
          length_x=13500,
          length_y=33300,
          margin_x=0,
          margin_y=0,
          consider_blank_nodes=False,
          autodetect_angle=False,
          angle=0,
          autodetect_grid=False,
          grid_adjust_mode="step",
          step_x=100,
          step_y=100,
          counts_x=0,
          counts_y=0,
          ignore_steps=False,
          sample_object=absolute_object_name (name=None,
          typed_name=[typed_object_names (obj_name="main_grid",
          obj_type="Grid3d")])))
    end_wf_item (index = 6)


    begin_wf_item (index = 7, name = "h_kh3")
    map_2d_create_by_grid_property (grid=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          use_user_cut=True,
          user_cut=find_object (name="Zone_kH",
          type="Grid3dProperty"),
          comparator=Comparator (rule="equals",
          value=3),
          use_user_cut_second=False,
          user_cut_second=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          comparator_second=Comparator (rule="not_equals",
          value=0),
          continuous_properties=True,
          continues_cube_and_map_table=[{"use" : True, "cube" : find_object (name="INIT_NTG",
          type="gt_tnav_cube_3d_data"), "map_2d" : find_object (name="map_h_kh_3",
          type="Map2d"), "method" : "net", "smooth" : False, "blocked_wells" : None}],
          discrete_properties=True,
          discrete_cube_and_map_table=[],
          smoothing_radius=10,
          ignore_faults=False,
          set_na_instead_of_zero=False,
          grid_2d_source="custom",
          subdivision=3,
          grid_2d=Grid2D (step_x=100,
          step_y=100,
          area=Rectangle (origin_x=0,
          origin_y=0,
          size_x=1000,
          size_y=1000,
          angle=0)),
          grid_2d_settings=Grid2DSettings (grid_2d_settings_shown=True,
          autodetect_box=False,
          min_x=6500,
          min_y=2250,
          length_x=13500,
          length_y=33300,
          margin_x=0,
          margin_y=0,
          consider_blank_nodes=False,
          autodetect_angle=False,
          angle=0,
          autodetect_grid=False,
          grid_adjust_mode="step",
          step_x=100,
          step_y=100,
          counts_x=0,
          counts_y=0,
          ignore_steps=False,
          sample_object=absolute_object_name (name=None,
          typed_name=[typed_object_names (obj_name="main_grid",
          obj_type="Grid3d")])))
    end_wf_item (index = 7)


    begin_wf_item (index = 8)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="eff_kh",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="map_2d (\"map_eff_kh\")",
          variables=variables)
    end_wf_item (index = 8)


    begin_wf_item (index = 9)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="h_kh1",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="map_2d (\"map_h_kh_1\")",
          variables=variables)
    end_wf_item (index = 9)


    begin_wf_item (index = 10)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="h_kh2",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="map_2d (\"map_h_kh_2\")",
          variables=variables)
    end_wf_item (index = 10)


    begin_wf_item (index = 11)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="h_kh3",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="map_2d (\"map_h_kh_3\")",
          variables=variables)
    end_wf_item (index = 11)


    begin_wf_item (index = 12)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="all_gas_ass_h",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="map_2d (\"Толщины\")",
          variables=variables)
    end_wf_item (index = 12)


    begin_wf_item (index = 13)
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="V_gas",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="dynamic_property (\"PORV\")*dynamic_property (\"SGAS\")*dynamic_property (\"FVFG\")",
          variables=variables)
    end_wf_item (index = 13)


    begin_wf_item (index = 14, name = "S_DX*DY")
    grid_property_calculator (mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          result_grid_property=find_object (name="S",
          type="Grid3dProperty"),
          use_filter=False,
          user_cut_for_filter=find_object (name="DX",
          type="gt_tnav_cube_3d_data"),
          filter_comparator=Comparator (rule="not_equals",
          value=0),
          formula="if k == 40 then dynamic_property (\"DX\")*dynamic_property (\"DY\") \nelse 0 \nendif\n\n",
          variables=variables)
    end_wf_item (index = 14)


    begin_wf_item (index = 15, name = "avg_perm_abs")
    table_create (table=find_object (name="avg_perm_zone_well",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="INIT_PERMX",
          type="gt_tnav_cube_3d_data"), "statistic_type" : "mean", "weights" : None}],
          use_discrete_property_1=True,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=True,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 15)


    begin_wf_item (index = 16, name = "avg_f_perm")
    table_create (table=find_object (name="avg_f_perm_zone_well",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="f_perm",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}],
          use_discrete_property_1=True,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=False,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 16)


    begin_wf_item (index = 17)
    table_create (table=find_object (name="kh_zone_well",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="h_kh1",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="h_kh2",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="h_kh3",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="eff_kh",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}],
          use_discrete_property_1=True,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=False,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 17)


    begin_wf_item (index = 18)
    table_create (table=find_object (name="GIP",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="GIP",
          type="gt_tnav_resource_cube_3d_data"), "statistic_type" : "sum", "weights" : None}],
          use_discrete_property_1=True,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=False,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 18)


    begin_wf_item (index = 19, name = "Field")
    table_create (table=find_object (name="filed",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="GIP",
          type="gt_tnav_resource_cube_3d_data"), "statistic_type" : "sum", "weights" : None}, {"property" : find_object (name="f_perm",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="eff_kh",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="INIT_PERMX",
          type="gt_tnav_cube_3d_data"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="S",
          type="Grid3dProperty"), "statistic_type" : "sum", "weights" : None}],
          use_discrete_property_1=False,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=False,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 19)


    begin_wf_item (index = 20)
    table_create (table=find_object (name="all_gas_ass_h",
          type="Table"),
          use_append_table=False,
          append_table="rows",
          mesh=find_object (name="DynamicModel (Dynamic Model)",
          type="gt_tnav_grid_3d_data"),
          settings_table=[{"property" : find_object (name="all_gas_ass_h",
          type="Grid3dProperty"), "statistic_type" : "mean", "weights" : None}, {"property" : find_object (name="V_gas",
          type="Grid3dProperty"), "statistic_type" : "sum", "weights" : None}, {"property" : find_object (name="S",
          type="Grid3dProperty"), "statistic_type" : "sum", "weights" : None}],
          use_discrete_property_1=True,
          discrete_property_1=find_object (name="Voronogo",
          type="Grid3dProperty"),
          use_discrete_property_2=False,
          discrete_property_2=find_object (name="Zone_kH",
          type="Grid3dProperty"))
    end_wf_item (index = 20)


    begin_wf_item (index = 21, name = "Общая толщина")
    table_export (file_name="../Данные для обучения/1.Generation/all_gas_ass_h",
          table=find_object (name="all_gas_ass_h",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 21)


    begin_wf_item (index = 22)
    table_export (file_name="../Данные для обучения/1.Generation/avg_perm_zone_well",
          table=find_object (name="avg_perm_zone_well",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 22)


    begin_wf_item (index = 23)
    table_export (file_name="../Данные для обучения/1.Generation/avg_f_perm_zone_well",
          table=find_object (name="avg_f_perm_zone_well",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 23)


    begin_wf_item (index = 24)
    table_export (file_name="../Данные для обучения/1.Generation/kh_zone_well",
          table=find_object (name="kh_zone_well",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 24)


    begin_wf_item (index = 25)
    polygon_export_txt_format (dir_name="../Данные для обучения/1.Generation/",
          polygon=find_object (name="GWC",
          type="Curve3d"),
          use_length_unit=False,
          length_unit="metres")
    end_wf_item (index = 25)


    begin_wf_item (index = 26)
    wells_export_welltrack_format (file_name="../Данные для обучения/1.Generation/well",
          use_well_filter=False,
          well_filter=find_object (name="Расстановка скважин",
          type="WellFilter"),
          wells=find_object (name="Wells",
          type="gt_wells_entity"),
          trajectories=find_object (name="Trajectories",
          type="Trajectories"),
          invert_z=True,
          xy_units_system="si",
          z_units_system="si",
          use_xy_units=False,
          xy_units="metres",
          use_z_units=False,
          z_units="metres",
          fraction_digits=1)
    end_wf_item (index = 26)


    begin_wf_item (index = 27)
    table_export (file_name="../Данные для обучения/1.Generation/filed",
          table=find_object (name="filed",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 27)


    begin_wf_item (index = 28)
    table_export (file_name="../Данные для обучения/1.Generation/GIP",
          table=find_object (name="GIP",
          type="Table"),
          delimiter="tab",
          use_delimiter_str=False,
          delimiter_str=",")
    end_wf_item (index = 28)


