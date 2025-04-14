import pandas as pd
import numpy as np

# Nastavení základních parametrů
np.random.seed(42)  # Pro reprodukovatelnost


# Počet vzorků a pokusů na pacienta
num_samples = 1300
attempts = 3

# Definice kategorií sportů

# Rychlostně-silové sporty
rychlostne_silove_sporty = [
    "ATLETIKA-SPRINTY", "DRÁHOVÁ CYKLISTIKA", "PLAVÁNÍ", "RYCHLOBRUSLENÍ",
    "IN-LINE BRUSLENÍ", "BOBY", "VZPÍRÁNÍ", "SILOVÝ TROJBOJ", "ATLETIKA-SKOKY",
    "ATLETIKA-VRHY, HODY", "ALPSKÉ LYŽOVÁNÍ", "SKOKY NA LYŽÍCH", "SNOWBOARDING"
]

# Vytrvalostní sporty
vytrvalostni_sporty = [
    "ATLETIKA-STŘEDNÍ TRATĚ", "DRÁHOVÁ CYKLISTIKA", "PLAVÁNÍ 200-400", "RYCHLOBRUSLENÍ",
    "IN-LINE BRUSLENÍ", "RYCHLOSTNÍ KANOISTIKA", "KANOISTIKA-DIVOKÁ VODA", "VESLOVÁNÍ",
    "ATLETIKA-BĚHY", "ATELTIKA-SPORTOVNÍ CHŮZE", "ORIENTAČNÍ BĚH", "SILNIČNÍ CYKLISTIKA",
    "MTB CYKLISTIKA", "PLAVÁNÍ 800+", "DÁLKOVÉ PLAVÁNÍ", "RYCHLOBRUSLENÍ 3-10km",
    "IN-LINE BRUSLENÍ 5+", "BĚŽECKÉ LYŽOVÁNÍ", "BIATLON"
]

# Sportovní hry
sportovni_hry = [
    "FOTBAL", "SÁLOVÁ KOPANÁ", "NOHEJBAL", "FLORBAL", "BASKETBAL", "VOLEJBAL",
    "HAZENÁ", "LEDNÍ HOKEJ", "POZEMNÍ HOKEJ", "RUGBY", "AMERICKÝ FOTBAL", "BASEBALL",
    "SOFTBALL", "KOLOVÁ", "VODNÍ PÓLO", "TENIS", "STOLNÍ TENIS", "SQUASH", "BADMINTON"
]

# Funkce pro přiřazení numerické kategorie sportu
def classify_sport(sport_name):
    if sport_name in rychlostne_silove_sporty:
        return 0  # Rychlostně-silové sporty
    elif sport_name in vytrvalostni_sporty:
        return 1  # Vytrvalostní sporty
    elif sport_name in sportovni_hry:
        return 2  # Sportovní hry
    else:
        return -1  # Neznámá kategorie

# Definice skupinových charakteristik pro věk, výšku, chodidla
group_specs = {
    "Child": {"height_mean": 130, "height_std": 10, "foot_mean": 18, "foot_std": 1.5},
    "Male": {"height_mean": 175, "height_std": 8, "foot_mean": 27, "foot_std": 1.5},
    "Female": {"height_mean": 165, "height_std": 7, "foot_mean": 25, "foot_std": 1.2},
}


# Definice odchylek pro x-ovou a y-ovou osu na základě fyzického zatížení a pravidelnosti aktivity
def get_activity_variability(physical_load, activity_frequency):
    if physical_load == 1:  # Závodní
        if activity_frequency == 1:  # Pravidelné
            std_x = 0.3
            std_y_range = (0, 0.6)
        else:  # Nepravidelné
            std_x = 0.5
            std_y_range = (0, 1.0)
    else:  # Rekreační
        if activity_frequency == 1:  # Pravidelné
            std_x = 0.5
            std_y_range = (0, 1.0)
        else:  # Nepravidelné
            std_x = 1.0
            std_y_range = (0, 2.0)

    return std_x, std_y_range


# Generování údajů o pacientech a výsledků chůze
def generate_data(attempts=3):
    # Inicializace datových polí
    patient_data = {
        "Patient_ID": [],
        "Age": [],
        "Gender": [],
        "Age_Group":[],
        "Height": [],
        "Foot_Size": [],
        "Hips_Length": [],
        "Sport": [],
        "Sport_Category": [],
        "Physical_Load": [],  # Fyzické zatížení: 0 = Rekreační, 1 = Závodní
        "Activity_Frequency": [],  # Pravidelnost: 0 = Nepravidelné, 1 = Pravidelné
        "1A_Height": [],
        "1A_Foot_Size": [],
        "1A_Hips_Length": [],
        "7A_Continuity_Mean_Torque": [],
        "7A_Continuity_Std_Torque": [],
        "7A_Max_Reaction_Torque": [],
        "7A_Max_Reaction_Torque_Time": [],
        "7A_Movement_Smoothness": [],
        "7A_Continuity_Time": [],
        "7A_Reaction_Time": []
    }

    # Stanoviště 2B: Pohybocit
    for attempt in range(1, 4):  # 3 pokusy
        patient_data[f"2B_Distance_Deviation1_{attempt}"] = []
        patient_data[f"2B_Distance_Deviation2_{attempt}"] = []

    # Stanoviště 2C: Pohybocit
    for attempt in range(1, 7):  # 6 pokusů
        patient_data[f"2C_Test_Set_{attempt}"] = []
        patient_data[f"2C_Answer_{attempt}"] = []
        patient_data[f"2C_Answer_Time_{attempt}"] = []

    # Stanoviště 3A: Stereognozie šíře
    for attempt in range(1, 4):  # 3 pokusy
        patient_data[f"3A_Referential_Width_{attempt}"] = []
        patient_data[f"3A_Estimated_Width_{attempt}"] = []
        patient_data[f"3A_Lower_X_Deviation_{attempt}"] = []
        patient_data[f"3A_Upper_X_Deviation_{attempt}"] = []

    # Počet pokusů na pacienta pro 3B
    num_trials_3B = 5  # Počet porovnání objektů

    # Stanoviště 3B: Stereognozie váha
    for trial in range(1, num_trials_3B + 1):
        patient_data[f"3B_First_Object_{trial}"] = []
        patient_data[f"3B_First_Object_Weight_{trial}"] = []
        patient_data[f"3B_Second_Object_{trial}"] = []
        patient_data[f"3B_Second_Object_Weight_{trial}"] = []
        patient_data[f"3B_Choice_Done_{trial}"] = []

    # Stanoviště 3C: Stereognozie tvar
    for trial in range(1,7):
        patient_data[f"3C_Gesture_Displayed_{trial}"] = []
        patient_data[f"3C_Gesture_Selected_{trial}"] = []
        patient_data[f"3C_Selected_Position_{trial}"] = []

    # Stanoviště 4A: Propriocepce
    for trial in range(1,7):
        patient_data[f"4A_Gesture_Displayed_{trial}"] = []
        patient_data[f"4A_Gesture_Selected_{trial}"] = []

    # Stanoviště 5A:Orientace
    for trial in range (1,4):
        patient_data[f"5A_Target_X_{trial}"] = []
        patient_data[f"5A_Target_Y_{trial}"] = []
        patient_data[f"5A_End_X_{trial}"] = []
        patient_data[f"5A_End_Y_{trial}"] = []


    for i in range(num_samples):
        patient_id = f"P{i + 1}"
        age = np.random.randint(6, 18)
        gender = np.random.choice(['Male', 'Female'])

        # Určení skupiny podle věku a pohlaví
        if age < 12:
            group = "Child"
        else:
            group = "Female" if gender == 'Female' else "Male"

        # Získání parametrů pro danou skupinu
        patient_height = np.random.normal(group_specs[group]["height_mean"], group_specs[group]["height_std"])
        patient_foot_size = np.random.normal(group_specs[group]["foot_mean"], group_specs[group]["foot_std"])
        hips_length = patient_height * np.random.uniform(0.45, 0.55)

        # Generování náhodného sportu
        sport = np.random.choice(rychlostne_silove_sporty + vytrvalostni_sporty + sportovni_hry)

        # Přiřazení kategorie sportu
        sport_category = classify_sport(sport)

        # Generování fyzického zatížení (0 = Rekreační, 1 = Závodní)
        physical_load = np.random.choice([0, 1])

        # Pokud je závodník, aktivita je vždy pravidelná (0 = Nepravidelné, 1 = Pravidelné)
        activity_frequency = 1 if physical_load == 1 else np.random.choice([0, 1])

        # Stanoviště 1A: Měření
        measured_height = np.random.normal(patient_height, 0.02)  # Přesnost ±2 cm
        measured_foot_size = np.random.normal(patient_foot_size, 0.02)  # Přesnost ±2 cm
        measured_hips_length = np.random.normal(hips_length, 0.03)  # Přesnost ±3 cm

        # Stanoviště 2B: Pohybocit
        for attempt in range(1, 4):  # 3 pokusy
            deviation1 = np.random.normal(0.1, 0.2) if sport_category == 0 else np.random.uniform(0.0, 0.2) # Přesnost při prvním pokusu ±0.2 cm
            deviation2 = np.random.normal(0.3, 0.5) if sport_category == 0 else np.random.uniform(0.3, 10)  # Přesnost při pokusu naslepo 0.3±10 cm
            patient_data[f"2B_Distance_Deviation1_{attempt}"].append(abs(deviation1))
            patient_data[f"2B_Distance_Deviation2_{attempt}"].append(abs(deviation2))

        # Stanoviště 2C: Pohybocit
        for attempt in range(1, 7):  # 6 pokusů
            test_set = np.random.choice(["Up", "Down"])
            is_professional = physical_load == 1
            prob_correct = 0.8 if is_professional else 0.6
            if gender == 'Female':  # Ženy mají vyšší přesnost
                prob_correct += 0.1

            # Generování času reakce
            base_time = 10  # Průměrná doba rozpoznání
            if test_set == "Down":
                base_time += 5  # Pokud byl pohyb dolů, trvá déle
            if gender == "Female":
                base_time *= 0.9  # Ženy rozpoznávají rychleji

            # Přidáme variabilitu k času
            time = np.random.normal(base_time, 3) # Malá variabilita kolem průměru

            answer = test_set if np.random.rand() < prob_correct else ("Down" if test_set == "Up" else "Up")
            patient_data[f"2C_Test_Set_{attempt}"].append(test_set)
            patient_data[f"2C_Answer_{attempt}"].append(answer)
            patient_data[f"2C_Answer_Time_{attempt}"].append(time)

        # Stanoviště 3A: Stereognozie šíře
        for attempt in range(1, 4):
            # Skutečná referenční šířka
            reference_width = 15.0
            base_error = np.random.normal(1.5, 0.5)  # Základní odchylka ±1.5 cm
            if gender == "Male":
                base_error *= 0.9  # Muži jsou přesnější
            if is_professional:
                base_error *= 0.8  # Sportovci ve sportovních hrách jsou ještě lepší

            base_error = max(base_error, 0.1)
            estimated_width = reference_width + np.random.normal(0, base_error)

            # Odchylky prstů (nižší a vyšší hodnota)
            lower_x_deviation = np.random.normal(0.5, 0.2) if not is_professional else np.random.normal(0.3, 0.15)
            upper_x_deviation = np.random.normal(0.5, 0.2) if not is_professional else np.random.normal(0.3, 0.15)

            patient_data[f"3A_Referential_Width_{attempt}"].append(reference_width)
            patient_data[f"3A_Estimated_Width_{attempt}"].append(estimated_width)
            patient_data[f"3A_Lower_X_Deviation_{attempt}"].append(lower_x_deviation)
            patient_data[f"3A_Upper_X_Deviation_{attempt}"].append(upper_x_deviation)

        # Možné objekty a jejich hmotnosti (simulace)
        object_weights = {"A": 15, "B": 20, "C": 32, "D": 45, "E": 50}

        # Stanoviště 3B: Stereognozie váha
        for trial in range(1, num_trials_3B + 1):
            first_object, second_object = np.random.choice(list(object_weights.keys()), 2, replace=False)
            first_weight = object_weights[first_object]
            second_weight = object_weights[second_object]

            # Pravděpodobnost správné odpovědi
            prob_correct = 0.9 if is_professional else 0.7
            correct_choice = first_object if first_weight > second_weight else second_object
            choice_done = correct_choice if np.random.rand() < prob_correct else (
                first_object if correct_choice == second_object else second_object)

            patient_data[f"3B_First_Object_{trial}"].append(first_object)
            patient_data[f"3B_First_Object_Weight_{trial}"].append(first_weight)
            patient_data[f"3B_Second_Object_{trial}"].append(second_object)
            patient_data[f"3B_Second_Object_Weight_{trial}"].append(second_weight)
            patient_data[f"3B_Choice_Done_{trial}"].append(choice_done)

        # Stanoviště 3C: Stereognozie váha
            # Možné gesta rukou v boxíku
        gestures = ["2", "6", "7b", "7c", "11"]
            # Pravděpodobnost správné volby
        if age < 12:
            prob_correct = 0.6  # Děti mají nižší úspěšnost
        elif physical_load == 1 and sport_category in [1, 2]:
            prob_correct = 0.85  # Sportovci mimo atletiku jsou lepší
        else:
            prob_correct = 0.75  # Standardní úspěšnost

        for trial in range(1,7):
            gesture_displayed = np.random.choice(gestures)
            correct_choice = gesture_displayed
            gesture_selected = correct_choice if np.random.rand() < prob_correct else np.random.choice(gestures)

            selected_position = np.random.randint(1, 7)  # Nahodilá pozice ruky v boxu (1-6)

            patient_data[f"3C_Gesture_Displayed_{trial}"].append(gesture_displayed)
            patient_data[f"3C_Gesture_Selected_{trial}"].append(gesture_selected)
            patient_data[f"3C_Selected_Position_{trial}"].append(selected_position)

        # Stanoviště 4A: Propriocepce
        gestures_4A = ["1", "2", "3a", "3b", "3c", "4", "5", "6", "7a", "7b", "10", "11"]

        for trial in range(1, 7):
            gesture_displayed = np.random.choice(gestures_4A)
            correct_choice = gesture_displayed
            gesture_selected = correct_choice if np.random.rand() < prob_correct else np.random.choice(gestures_4A)

            patient_data[f"4A_Gesture_Displayed_{trial}"].append(gesture_displayed)
            patient_data[f"4A_Gesture_Selected_{trial}"].append(gesture_selected)

        # Stanoviště 5A: Orientace
        for attempt in range(1, 4):

            target_x = 3.4065
            target_y = 1.164857

            # Odchylka na základě sportovní aktivity
            std_x, y_range = get_activity_variability(physical_load, activity_frequency)

            # Náhodně generované výsledné hodnoty na základě přesnosti
            end_x = np.random.normal(target_x, std_x)
            end_y = np.random.uniform(target_y - 0.2, target_y + 0.2)  # ±20 cm

            patient_data[f"5A_Target_X_{attempt}"].append(target_x)
            patient_data[f"5A_Target_Y_{attempt}"].append(target_y)
            patient_data[f"5A_End_X_{attempt}"].append(end_x)
            patient_data[f"5A_End_Y_{attempt}"].append(end_y)

        # Stanoviště 7A: Reakce na sílu

        gender_factor = 1.0 if gender == "Male" else 1.2
        age_factor = 1.5 if group == 'child' else 1.0
        sport_factor = 0.8 if is_professional else (
            0.9 if activity_frequency == 1 else 1.1)
        variability_factor = gender_factor * age_factor * sport_factor

        continuity_time = np.random.uniform(5, 10)
        continuity_torque = np.random.normal(loc=2.0, scale=0.3 * variability_factor,
                                             size=int(continuity_time * 10))
        continuity_mean_torque = np.mean(continuity_torque)
        continuity_std_torque = np.std(continuity_torque)
        movement_smoothness = 1 / (continuity_std_torque + 1e-6)

        reaction_time = np.random.uniform(5, 20)
        reaction_torque = np.random.normal(loc=5.0, scale=1.5 * variability_factor, size=int(reaction_time * 10))
        max_reaction_torque = np.max(reaction_torque)
        max_reaction_torque_time = np.argmax(reaction_torque) / 10.0

        # Ukládání dat pro každého pacienta
        patient_data["Patient_ID"].append(patient_id)
        patient_data["Age"].append(age)
        patient_data["Gender"].append(gender)
        patient_data["Height"].append(patient_height)
        patient_data["Foot_Size"].append(patient_foot_size)
        patient_data["Hips_Length"].append([hips_length])
        patient_data["Sport"].append(sport)
        patient_data["Sport_Category"].append(sport_category)
        patient_data["Physical_Load"].append(physical_load)
        patient_data["Activity_Frequency"].append(activity_frequency)
        patient_data["Age_Group"].append(group)
        patient_data["1A_Height"].append(measured_height)
        patient_data["1A_Foot_Size"].append(measured_foot_size)
        patient_data["1A_Hips_Length"].append(measured_hips_length)
        patient_data["7A_Continuity_Mean_Torque"].append(continuity_mean_torque)
        patient_data["7A_Continuity_Std_Torque"].append(continuity_std_torque)
        patient_data["7A_Max_Reaction_Torque"].append(max_reaction_torque)
        patient_data["7A_Max_Reaction_Torque_Time"].append(max_reaction_torque_time)
        patient_data["7A_Movement_Smoothness"].append(movement_smoothness)
        patient_data["7A_Continuity_Time"].append(continuity_time)
        patient_data["7A_Reaction_Time"].append(reaction_time)

    # Převedení na DataFrame
    df = pd.DataFrame(patient_data)

    # Výpočet přesnosti
#    df['Accuracy'] = np.sqrt((df['End_X'] - 4.1) ** 2 + (df['End_Y'] - 0.8) ** 2)

    return df


# Generování dat a výpis
pd.set_option('display.max_columns', None)
df_walk = generate_data()
print(df_walk.head())
pd.reset_option('display.max_columns')

df_walk.to_csv('patient_data.csv', index=False)

def correlation_analysis(df):
    # Výběr pouze numerických dat pro výpočet korelační matice
    numeric_data = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)
    return correlation_matrix

# Pro spuštění analýzy korelací
correlation_matrix = correlation_analysis(df_walk)
