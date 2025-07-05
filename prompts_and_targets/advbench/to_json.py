import csv
import json
import argparse

def csv_to_json(input_file, output_file):
    data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                prompt = row[0].strip()
                data.append({
                    "instruction": prompt,
                    "category": None
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert CSV prompts to JSON format.")
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output JSON file')
    args = parser.parse_args()

    csv_to_json(args.input_file, args.output_file)