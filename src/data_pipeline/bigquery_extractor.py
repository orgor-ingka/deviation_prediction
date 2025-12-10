from google.cloud import bigquery
import pandas as pd
import os

def extract_from_bigquery(query: str):
    """
    Extracts data from BigQuery using the provided SQL query.
    Assumes the BigQuery client is authenticated (e.g., via GOOGLE_APPLICATION_CREDENTIALS).
    Saves the extracted data to 'raw/raw_data.csv' within the /mnt/data/ directory.

    Args:
        query (str): The SQL query to execute in BigQuery.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a row of the query result.
    """
    try:
        client = bigquery.Client()
        query_job = client.query(query)
        results = query_job.result()  # Waits for job to complete and retrieves results

        data = []
        for row in results:
            data.append(dict(row))
        
        if data:
            df = pd.DataFrame(data)
            output_dir = '../../data/raw/' # Corrected path
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'raw_data.csv')
            df.to_csv(output_path, index=False)
            print(f"Successfully extracted {len(data)} rows from BigQuery and saved to {output_path}.")
        else:
            print("No data extracted from BigQuery.")

        return data
    except Exception as e:
        print(f"Error extracting data from BigQuery: {e}")
        return []