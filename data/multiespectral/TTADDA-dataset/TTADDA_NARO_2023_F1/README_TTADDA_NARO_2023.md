#### This readme belongs to the TTADDA_NARO_2023: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset.

# Title: TTADDA_NARO_2023: A subset of the multi-season RGB and multispectral TTADDA-UAV potato dataset.

- Related publication: TTADDA-UAV: A Multi-Season RGB and Multispectral UAV Dataset of Potato Fields Collected in Japan and the Netherlands
- DOI: https://doi.org/10.1016/j.dib.2025.112004 
- Authors: Bart. M. van Marrewijk, Stephen Njehia Njane, Shogo Tsuda, Marcel van Culemborg, Gerrit Polder, Kenji Katayama, Tim van Daalen, Rick van de Zedde

_Corresponding author: Bart M. van Marrewijk & Stephen Njehia Njane
Contact Information: bart.vanmarrewijk@wur.nl;njane.stephen.njehia344@naro.go.jp_

# ===== Background information =====

The TTADDA_NARO_2023 dataset includes drone imagery, manual measurements, and on-site weather data. Weekly RGB and multispectral images were processed with Agisoft Metashape to create a DEM, RGB and multispectral orthomosaics. Manual measurements covered yield and ground coverage. All data was organised and summarized using the MIAPPE format to align with the FAIR data principles (see related MIAPPE file). 
MIAPPE_Minimal_Spreadsheet_Template_TTADDAv4.xlsx. Examples, how to uses this excel sheet can be found on our git: https://github.com/NPEC-NL/MIAPPE_TTADDA_dataset

# ===== FOLDER structure ====
The data structure is schematically visualized below. It contains a metadafile and it has a folder for every study (TADDA_NARO_2022, TADDA_NARO_2023, ... etc). Every study has a field number (F1) with drone_data, measurements and plot_data. 

```bash
MIAPPE_Minimal_Spreadsheet_Template_TTADDAv4.xlsx
TTADDA_DRONE_01/
├── TTADDA_NARO_2023/            #studyId
├── TTADDA_NARO_2023_F1/ # field number, currently only one field
│   │   ├── drone_data/                   # RGB, DEM and multispectral orthomosaics
│   │   │   ├── 2023-05-18/                  # folder
│   │   │   │  ├── 20230518_rgb.tif      # orthomosaic RGB
│   │   │   │  ├── 20230518_dsm.tif    # elevation map
│   │   │   │  ├── 20230518_blue.tif    # blue channel multispectral data
│   │   │   │  ├── 20230518_green.tif  # green channel multispectral data
│   │   │   │  ├── 20230518_red.tif      # red channel multispectral data
│   │   │   │  ├── 20230518_red edge.tif     # red edge channel multispectral data
│   │   │   │  ├── 20230518_nir.tif     # near infrared channel multispectral data
│   │   │   ├── …/                   
│   │   │   ├── 2023-09-28/                   # folder
│   │   ├── measurements/                          # all ground truth data  / measurements
│   │   │  ├── NARO_field_2023_GT_additional.csv
│   │   │  ├── NARO_field_2023_GT_growthphases.csv
│   │   │  ├── NARO_field_2023_GT_yield.csv
│   │   │  ├── NARO_field_2023_GT_weather.csv
│   │   │  ├── NARO_field_2023_GT_weather_online.csv
│   │   ├── metadata/                          # contains shape files for coordinates of plots (obsUnitId)
│   │   │  ├── plot_shapefile.shp        # coordinates
```
# ==== file format ====
- RGB orthomosaics: .tif
- multispectral orthomosaics: .tif
- measurements (weather, yield, growthphases): .csv
- coordinates of every plot: .shp (shapefile)

# ==== licence ====
Dataset is licensed under CC BY-NC-SA 4.0

