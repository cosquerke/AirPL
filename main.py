import requests
import pandas as pd
import logging
import joblib
import os

class LoadData:
    def __init__(self, cache_communes_path='communes_data.pkl', cache_population_path='population_data.pkl', cache_entreprise_path='entreprise_data.pkl', cache_pm10_path='pm10_data.pkl', cache_no2_path="no2_data.pkl"):
        self.communes_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/234400034_communes-des-pays-de-la-loire/records"
        self.population_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/12002701600563_population_pays_de_la_loire_2019_communes_epci/records"
        self.entreprises_url = "https://data.paysdelaloire.fr/api/explore/v2.1/catalog/datasets/120027016_base-sirene-v3-ss/records"
        self.pollution_url = "https://data.airpl.org/api/v1/mesure/journaliere/"

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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, url, params):
        all_records = []
        while True:
            self.logger.info(f"Fetching data from {url} with offset {params['offset']}")
            response = requests.get(url, params=params)
            data = response.json()
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
        count = 0
        
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
                    count = count + 1
            
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
        combined_data = self.combine_data(communes_data, population_data, entreprise_data, pm10_data, no2_data)
        df = pd.DataFrame(combined_data)
        self.logger.info("Combined dataframe created.")
        return df

    def save_dataframe_to_json(self, df, file_path):
        self.logger.info(f"Saving dataframe to JSON file at {file_path}.")
        df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        self.logger.info(f"Dataframe successfully saved to {file_path}.")

# Utilisation de la classe
if __name__ == "__main__":
    loader = LoadData()
    combined_df = loader.get_combined_dataframe()

    # Affichage du DataFrame combiné
    #print(combined_df)

    # Sauvegarde du DataFrame en JSON
    loader.save_dataframe_to_json(combined_df, 'data/combined_data.json')