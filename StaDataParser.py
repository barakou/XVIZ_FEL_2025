import json
import pandas as pd
import os


# Funkce pro parsování jednotlivého JSON souboru
def parse_json(file_path, station_name):
    with open(file_path, 'r') as f:
        data = json.load(f)

    parsed_data = []
    for record in data:
        # Iterujeme přes jednotlivé měření
        if 'measurements' in record:
            for measure in record['measurements']:
                if isinstance(measure['dataValue'], dict):
                    # Pokud je hodnota složená (např. dictionary), rozbijeme ji
                    for key, value in measure['dataValue'].items():
                        parsed_data.append({
                            "Station": station_name,
                            "MeasureNumber": int(record.get("measureNumber", None)),
                            "DataName": f"{measure['dataName']}_{key}",
                            "Value": value,
                            "Unit": measure.get("dataUnit", None)
                        })
                else:
                    parsed_data.append({
                        "Station": station_name,
                        "MeasureNumber": int(record.get("measureNumber", None)),
                        "DataName": measure['dataName'],
                        "Value": measure['dataValue'],
                        "Unit": measure.get("dataUnit", None)
                    })
        else:
            # Zpracování alternativní struktury
            for key, value in record.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        parsed_data.append({
                            "Station": station_name,
                            "DataName": f"{key}_{sub_key}",
                            "Value": sub_value
                        })
                else:
                    parsed_data.append({
                        "Station": station_name,
                        "DataName": key,
                        "Value": value
                    })

    return pd.DataFrame(parsed_data)


# Funkce pro zpracování všech JSON souborů ve složce
def process_all_files(file_paths):
    combined_data = pd.DataFrame()

    for file_path in file_paths:
        station_name = os.path.basename(file_path).split('_')[2]  # Např. 2B, 2C, 3A
        parsed_data = parse_json(file_path, station_name)
        combined_data = pd.concat([combined_data, parsed_data], ignore_index=True)

    return combined_data

base_dir = os.path.dirname(__file__)  # Cesta ke složce, kde je aktuální skript

# Cesta ke složce server_data
data_dir = os.path.join(base_dir, 'server_data')

# Seznam souborů, které chceme zpracovat
file_paths = [
    data_dir + '/9_2_1AMereni_20250117142535.json',
    data_dir + '/13_2_2BPohybocit_20250121083221.json',
    data_dir + '/3_1_2CPohybocit_20250120152443.json',
    data_dir + '/7_2_3AStereognozie_20250114140457.json',
    data_dir + '/6_2_3BStereognozie_20250114135930.json',
    data_dir + '/5_2_3CStereognozie_20250114135749.json',
    data_dir + '/4_2_4APropriocepce_20250114135540.json',
    data_dir + '/15_1_5AOrientace_20250120152111.json',
    data_dir + '/23_3_7ASila_20250121093203.json'
]

# Zpracování všech souborů
combined_df = process_all_files(file_paths)

# Uložení výsledků do CSV
combined_df.to_csv('combined_measurements.csv', index=False)
print("Data byla úspěšně zpracována a uložena do souboru 'combined_measurements.csv'.")
