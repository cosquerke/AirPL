import json
import requests
import pandas as pd
import logging
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class LoadData:
    def __init__(self, cache_communes_path='communes_data.pkl', cache_population_path='population_data.pkl', cache_entreprise_path='entreprise_data.pkl', cache_pm10_path='pm10_data.pkl', cache_no2_path="no2_data.pkl", cache_brut_path="brut_data.pkl"):
        self.communes_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/234400034_communes-des-pays-de-la-loire/records"
        self.population_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/12002701600563_population_pays_de_la_loire_2019_communes_epci/records"
        self.entreprises_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/120027016_base-sirene-v3-ss/records"
        self.pollution_url = "https://data.airpl.org/api/v1/mesure/horaire/"

        self.communes_params = {
            'select': 'nom_comm,insee_comm,geo_shape,geo_point_2d',
            'limit': 100,
            'offset': 0
        }
        self.population_params = {
            'select': 'code_commune,population_municipale',
            'limit': 100,
            'offset': 0
        }

        self.entreprises_params = {
            'select': 'denominationunitelegale,geolocetablissement,codecommuneetablissement,sectionetablissement,soussectionunitelegale',
            'where': '(etatadministratifetablissement != "Fermé")',
            'limit': 100,
            'offset': 0
        }

        self.pm10_params = {
            'code_configuration_de_mesure__code_point_de_prelevement__code_polluant': '24',
            'code_configuration_de_mesure__code_point_de_prelevement__code_station__code_commune__code_departement__in': '44,49,53,72,85',
            'date_heure_tu__range': '2022-5-25,2024-5-24',
            'export': 'json',
            'limit': 1000,
            'offset': 0
        }

        self.no2_params = {
            'code_configuration_de_mesure__code_point_de_prelevement__code_polluant': '03',
            'code_configuration_de_mesure__code_point_de_prelevement__code_station__code_commune__code_departement__in': '44,49,53,72,85',
            'date_heure_tu__range': '2022-5-25,2024-5-24',
            'export': 'json',
            'limit': 1000,
            'offset': 0
        }

        self.cache_communes_path = cache_communes_path
        self.cache_population_path = cache_population_path
        self.cache_entreprise_path = cache_entreprise_path
        self.cache_pm10_path = cache_pm10_path
        self.cache_no2_path = cache_no2_path
        self.cache_brut_path = cache_brut_path

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, url, params):
        all_records = []
        while True:
            self.logger.info(f"Fetching data from {url} with offset {params['offset']}")
            response = requests.get(url, params=params)
            
            try:
                data = response.json()
                print(len(data.get('results')))
            except ValueError as e:
                self.logger.error(f"Error parsing JSON response: {e}")
                break  # Exit the loop if JSON parsing fails
            
            records = data.get('results', [])
            if not records:
                self.logger.info("No more records found, stopping fetch.")
                break
            
            all_records.extend(records)
            params['offset'] += params['limit']
        
        self.logger.info(f"Fetched {len(all_records)} records from {url}")
        return all_records


    def get_communes_data(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_communes_path):
            self.logger.info(f"Loading cached communes data from {self.cache_communes_path}.")
            communes_data = joblib.load(self.cache_communes_path)
        else:
            self.logger.info("Fetching communes data.")
            communes_data = self.fetch_data(self.communes_url, self.communes_params)
            joblib.dump(communes_data, self.cache_communes_path)
            self.logger.info(f"Communes data cached to {self.cache_communes_path}.")
        return communes_data

    def get_population_data(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_population_path):
            self.logger.info(f"Loading cached population data from {self.cache_population_path}.")
            population_data = joblib.load(self.cache_population_path)
        else:
            self.logger.info("Fetching population data.")
            population_data = self.fetch_data(self.population_url, self.population_params)
            joblib.dump(population_data, self.cache_population_path)
            self.logger.info(f"Population data cached to {self.cache_population_path}.")
        return population_data
    
    def get_entreprise_data(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_entreprise_path):
            self.logger.info(f"Loading cached entreprise data from {self.cache_entreprise_path}.")
            entreprise_data = joblib.load(self.cache_entreprise_path)
        else:
            self.logger.info("Fetching entreprise data.")
            entreprise_data = self.fetch_data(self.entreprises_url, self.entreprises_params)
            joblib.dump(entreprise_data, self.cache_entreprise_path)
            self.logger.info(f"Entreprise data cached to {self.cache_entreprise_path}.")
        return entreprise_data

    def get_pm10_data(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_pm10_path):
            self.logger.info(f"Loading cached pm10 data from {self.cache_pm10_path}.")
            pm10_data = joblib.load(self.cache_pm10_path)
        else:
            self.logger.info("Fetching pm10 data.")
            pm10_data = self.fetch_data(self.pollution_url, self.pm10_params)
            joblib.dump(pm10_data, self.cache_pm10_path)
            self.logger.info(f"pm10 data cached to {self.cache_pm10_path}.")
        return pm10_data
    
    def get_no2_data(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_no2_path):
            self.logger.info(f"Loading cached NO2 data from {self.cache_no2_path}.")
            no2_data = joblib.load(self.cache_no2_path)
        else:
            self.logger.info("Fetching NO2 data.")
            no2_data = self.fetch_data(self.pollution_url, self.no2_params)
            joblib.dump(no2_data, self.cache_no2_path)
            self.logger.info(f"NO2 data cached to {self.cache_no2_path}.")
        return no2_data

    def combine_data(self, communes_data, population_data, entreprise_data, pm10_data, no2_data):
        self.logger.info("Combining data.")
        population_dict = {item['code_commune']: item['population_municipale'] for item in population_data}
        
        for commune in communes_data:
            insee_comm = commune['insee_comm']
            commune['population_municipale'] = population_dict.get(insee_comm, None)
            entreprises = []
            pollutions = {
                "no2": [],
                "pm10": []
            }

            for entreprise in entreprise_data:
                if str(entreprise.get('codecommuneetablissement')) == insee_comm:
                    entreprises.append(entreprise)
            commune['entreprises'] = entreprises

            for record in pm10_data:
                if (str(record.get("code_commune")) == insee_comm) and (str(record.get("validite")) == "True"):
                    pm10_valide = {record.get("date_heure_local"), record.get("valeur")}
                    pollutions['pm10'].append(pm10_valide)
            
            for record in no2_data:
                if (str(record.get("code_commune")) == insee_comm) and (str(record.get("validite")) == "True"):
                    no2_valide = {record.get("date_heure_local"), record.get("valeur")}
                    pollutions['no2'].append(no2_valide)

            commune["pollutions"] = pollutions
            
        self.logger.info("Data combined successfully.")
        return communes_data

    def get_combined_dataframe(self, use_cache=True):
        self.logger.info("Generating combined dataframe.")
        communes_data = self.get_communes_data(use_cache)
        population_data = self.get_population_data(use_cache)
        entreprise_data = self.get_entreprise_data(use_cache)
        pm10_data = self.get_pm10_data(use_cache)
        no2_data = self.get_no2_data(use_cache)
        if os.path.exists(self.cache_brut_path):
            self.logger.info("DataFrame combined load from cache.")
            combined_data = joblib.load(self.cache_brut_path)
        
        else:
            self.logger.info("DataFrame combined")
            combined_data = self.combine_data(communes_data, population_data, entreprise_data, pm10_data, no2_data)
            joblib.dump(combined_data, self.cache_brut_path)

        df = pd.DataFrame(combined_data)
        self.logger.info("Combined dataframe created.")
        return df

    def save_dataframe_to_json(self, df, file_path):
        self.logger.info(f"Saving dataframe to JSON file at {file_path}.")
        df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        self.logger.info(f"Dataframe successfully saved to {file_path}.")

class CleanData():
    def __init__(self, dataframe):
        self.brut_df = dataframe.copy()
        self.cache_file = "clean_df_data.pkl"

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # NOTE: Penser à conserver les communes frontalière en cas d'alerte ?
    def deleteEmptyCityPollution(self):
        self.brut_df = self.brut_df[self.brut_df["pollutions"].apply(lambda x: len(x.get("no2", [])) > 0 or len(x.get("pm10", [])) > 0)]

    def getCleanDataframe(self):
        self.deleteEmptyCityPollution()
        return self.brut_df.copy()
    
    def get_cached_clean_df(self):
        if os.path.exists(self.cache_file):
            # Charge le DataFrame depuis le cache
            self.logger.info("DataFrame load from cache.")
            clean_df = joblib.load(self.cache_file)
            
        else:
            # Nettoie les données et sauvegarde dans le cache
            self.logger.info("DataFrame cleaned et saved into cache.")
            clean_df = self.getCleanDataframe()
            joblib.dump(clean_df, self.cache_file)
            
        return clean_df

class ProcessData():
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.json_seuil_path = 'data/seuils.json'

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def getPollutionAverageByCityandSTrimester(self, pollution):
        records = []
        for index, row in self.dataframe.iterrows():
            nom_comm = row['nom_comm']
            self.logger.info("getPollutionByCityandSemester() -> "+nom_comm)
            for value, date in row['pollutions'][pollution]:
                date = pd.to_datetime(date)
                trimester = (date.month - 1) // 3 + 1  # 1 pour Q1, 2 pour Q2, 3 pour Q3, 4 pour Q4
                year = date.year
                if year!=1970 : 
                    records.append([nom_comm, value, year, trimester])

        pollution_df = pd.DataFrame(records, columns=['nom_comm', 'value', 'year', 'trimester'])

        # Convertir les valeurs en type numérique
        pollution_df['value'] = pd.to_numeric(pollution_df['value'], errors='coerce')

        # Étape 2 : Calculer la moyenne trimestrielle par ville
        average_pollution = pollution_df.groupby(['nom_comm', 'year', 'trimester']).agg({'value': 'mean'}).reset_index()

        # Étape 3 : Créer des graphiques en barres
        cities = average_pollution['nom_comm'].unique()
        for city in cities:
            city_data = average_pollution[average_pollution['nom_comm'] == city]
            plt.figure(figsize=(10, 6))
            plt.bar(city_data['year'].astype(str) + '-T' + city_data['trimester'].astype(str),
                    city_data['value'], color='blue')
            plt.title(f'Moyenne trimestrielle de {pollution.upper()} pour {city}')
            plt.xlabel('Trimestre')
            plt.ylabel(f'Moyenne {pollution.upper()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def getHightPollutionLevelByPopulation(self,pollution):

        with open(self.json_seuil_path) as f:
            seuils = json.load(f)

        # Extraire le seuil d'information pour la pollution
        seuil_qualite_pollution = seuils[pollution]['objectif de qualite']
        
        # Étape 1 : Convertir les données de la pollution en DataFrame
        records = []
        for index, row in self.dataframe.iterrows():
            nom_comm = row['nom_comm']
            population = row['population_municipale']
            self.logger.info(f'getHight{pollution}LevelByPopulation() -> '+nom_comm)
            for value, date in row['pollutions'][pollution]:
                date = pd.to_datetime(date)
                trimester = (date.month - 1) // 3 + 1  # 1 pour Q1, 2 pour Q2, 3 pour Q3, 4 pour Q4
                year = date.year
                if year!=1970 : 
                    records.append([nom_comm, value, year, trimester, population])
               

        pollution_df = pd.DataFrame(records, columns=['nom_comm', 'pollution_value', 'year', 'trimester', 'population'])

        # Convertir les valeurs de pollution en type numérique
        pollution_df['pollution_value'] = pd.to_numeric(pollution_df['pollution_value'], errors='coerce')
        pollution_df['population'] = pd.to_numeric(pollution_df['population'], errors='coerce')
        
        # Filtrer les enregistrements où la pollution dépasse le seuil d'information
        impacted_df = pollution_df[pollution_df['pollution_value'] > seuil_qualite_pollution]
        
        # Agréger par année et trimestre pour trouver le nombre de personnes impactées
        impacted_df = impacted_df.groupby(['year', 'trimester']).agg({'population': 'sum'}).reset_index()
        
        # Étape 2 : Créer des graphiques en barres
        plt.figure(figsize=(10, 6))
        plt.bar(impacted_df['year'].astype(str) + '-T' + impacted_df['trimester'].astype(str),
                impacted_df['population'], color='red')
        plt.title(f'Nombre de personnes impactées par {pollution.upper()} au-delà du seuil de qualité')
        plt.xlabel('Trimestre')
        plt.ylabel('Population impactée')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def getTop5PollutionHoursByCityAndTrimester(self, pollution):
        records = []
        for index, row in self.dataframe.iterrows():
            nom_comm = row['nom_comm']
            self.logger.info("getTop5PollutionHoursByCityAndTrimester() -> " + nom_comm)
            for value, date in row['pollutions'][pollution]:
                date = pd.to_datetime(date)
                hour = date.hour
                trimester = (date.month - 1) // 3 + 1  # 1 pour Q1, 2 pour Q2, 3 pour Q3, 4 pour Q4
                year = date.year
                if year!=1970 : 
                    records.append([nom_comm, value, year, trimester, hour])
        
        pollution_df = pd.DataFrame(records, columns=['nom_comm', 'value', 'year', 'trimester', 'hour'])
        
        # Convertir les valeurs en type numérique
        pollution_df['value'] = pd.to_numeric(pollution_df['value'], errors='coerce')
        
        # Étape 1 : Calculer la moyenne horaire par ville et par trimestre
        hourly_pollution = pollution_df.groupby(['nom_comm', 'year', 'trimester', 'hour']).agg({'value': 'mean'}).reset_index()
        
        # Étape 2 : Pour chaque ville et chaque trimestre, trouver les 3 heures avec les valeurs moyennes les plus élevées
        top_hours = hourly_pollution.groupby(['nom_comm', 'year', 'trimester']).apply(lambda x: x.nlargest(5, 'value')).reset_index(drop=True)
        
        # Étape 3 : Génération des graphiques pour chaque ville et chaque trimestre
        cities = top_hours['nom_comm'].unique()
        for city in cities:
            city_data = top_hours[top_hours['nom_comm'] == city]
            years = city_data['year'].unique()
            for year in years:
                for trimester in range(1, 5):
                    trimester_data = city_data[(city_data['year'] == year) & (city_data['trimester'] == trimester)]
                    if not trimester_data.empty:
                        plt.figure(figsize=(10, 6))
                        plt.bar(trimester_data['hour'].astype(str), trimester_data['value'], color='blue')
                        plt.title(f'Top 5 Heures de Pic de Pollution pour {city} - {year} T{trimester}')
                        plt.xlabel('Heure')
                        plt.ylabel(f'Valeur Moyenne de {pollution.upper()}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()

    def getEnterpriseSectionsByCity(self):
        records = []
        
        for index, row in self.dataframe.iterrows():
            nom_comm = row['nom_comm']
            self.logger.info("plotEnterpriseSectionsByCity() -> " + nom_comm)
            entreprises = row['entreprises']
            
            for entreprise in entreprises:
                section = entreprise['sectionetablissement'].split(';')[0].strip()
                records.append([nom_comm, section])
        
        # Créer un DataFrame avec les sections d'établissement par ville
        sections_df = pd.DataFrame(records, columns=['nom_comm', 'sectionetablissement'])
        
        # Compter le nombre d'entreprises par section pour chaque ville
        section_counts = sections_df.groupby(['nom_comm', 'sectionetablissement']).size().reset_index(name='count')
        
        # Générer des graphiques pour chaque ville
        cities = section_counts['nom_comm'].unique()
        for city in cities:
            city_data = section_counts[section_counts['nom_comm'] == city]
            
            plt.figure(figsize=(12, 8))
            plt.bar(city_data['sectionetablissement'], city_data['count'], color='blue')
            plt.title(f'Nombre d\'Entreprises par Section à {city}')
            plt.xlabel('Section d\'Établissement')
            plt.ylabel('Nombre d\'Entreprises')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def getPollutionEvolutionByCity(self, pollution):
        records = []
        for index, row in self.dataframe.iterrows():
            nom_comm = row['nom_comm']
            self.logger.info("getPollutionEvolutionByCity() -> " + nom_comm)
            for value, date in row['pollutions'][pollution]:
                date = pd.to_datetime(date).tz_localize(None)  # Convertir en tz-naive
                if date.year!=1970 : 
                    records.append([nom_comm, value, date])
        
        pollution_df = pd.DataFrame(records, columns=['nom_comm', 'value', 'date'])
        
        # Convertir les valeurs en type numérique
        pollution_df['value'] = pd.to_numeric(pollution_df['value'], errors='coerce')
        
        # Étape 1 : Calculer les moyennes hebdomadaires de la pollution par ville
        pollution_df['date'] = pd.to_datetime(pollution_df['date'])
        pollution_df['week'] = pollution_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
        
        weekly_pollution = pollution_df.groupby(['nom_comm', 'week']).agg({'value': 'mean'}).reset_index()
        
        # Étape 2 : Génération des graphiques pour chaque ville
        cities = weekly_pollution['nom_comm'].unique()
        for city in cities:
            city_data = weekly_pollution[weekly_pollution['nom_comm'] == city]
            
            plt.figure(figsize=(12, 8))
            plt.plot(city_data['week'], city_data['value'], marker='o', linestyle='-', color='blue')
            plt.title(f'Évolution de la Pollution pour {city}')
            plt.xlabel('Date')
            plt.ylabel(f'Valeur Moyenne Hebdomadaire de {pollution.upper()}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Utilisation de la classe
if __name__ == "__main__":
    loader = LoadData()
    combined_df = loader.get_combined_dataframe()

    cleaner = CleanData(combined_df)
    clean_df = cleaner.get_cached_clean_df()

   # clean_df = joblib.load("clean_df_data.pkl")

    exposer = ProcessData(clean_df)
    #exposer.getPollutionAverageByCityandSTrimester("no2")
    #exposer.getPollutionAverageByCityandSTrimester("pm10")
    exposer.getHightPollutionLevelByPopulation("no2")
    #exposer.getHightPollutionLevelByPopulation("pm10")
    #exposer.getTop5PollutionHoursByCityAndTrimester("no2")
    #exposer.getTop5PollutionHoursByCityAndTrimester("pm10")
    #exposer.getEnterpriseSectionsByCity()
    #exposer.getPollutionEvolutionByCity("no2")
    #exposer.getPollutionEvolutionByCity("pm10")