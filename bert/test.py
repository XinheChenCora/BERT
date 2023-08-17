import pandas as pd

def convert_to_utf8(input_file, output_file):
    # Read the CSV file using pandas with 'latin-1' encoding
    df = pd.read_csv(input_file, encoding='latin-1')

    # Save the DataFrame to a new CSV file in UTF-8 encoding
    df.to_csv(output_file, index=False, encoding='utf-8')

# Example usage
input_file = 'ppcsv.csv'  # Replace 'input.csv' with your input file name
output_file = 'output_utf8.csv'  # Replace 'output_utf8.csv' with your desired output file name

convert_to_utf8(input_file, output_file)
