import csv
import json
import os

# Input and output file paths
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def convert_csv_to_json_specify_encoder(csv_file_path, json_file_path):
    """
    Convert a CSV file to a JSON file.
    
    :param csv_file_path: Path to the input CSV file.
    :param json_file_path: Path to the output JSON file.
    """
    # Ensure the input CSV file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_file_path}")

    # Read the CSV file and convert it to a list of dictionaries
    with open(csv_file_path, mode='r', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
        
    # Write the list of dictionaries to a JSON file
    with open(json_file_path, mode='w', encoding='ISO-8859-1') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
        
    print(f"Converted {csv_file_path} to {json_file_path}")
    
def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to a JSON file.
    
    :param csv_file_path: Path to the input CSV file.
    :param json_file_path: Path to the output JSON file.
    """
    # Ensure the input CSV file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_file_path}")
    
    with open(csv_file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

    print(f"Converted {csv_file_path} to {json_file_path}")

if __name__ == "__main__":
    csv_file_path = f'{root_path}/data/twitter/gender_dataset.csv'
    json_file_path = f'{root_path}/data/twitter/gender_dataset.json'
    # Convert CSV to JSON
    convert_csv_to_json(csv_file_path, json_file_path)