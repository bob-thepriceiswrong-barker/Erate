import requests
import json

# Test the Form 470 API
url = "https://opendata.usac.org/resource/jp7a-89nd.json"
params = {
    "$where": "billed_entity_state='TX' AND (funding_year='2024' OR funding_year='2025')",
    "$limit": 100
}

headers = {
    "X-App-Token": "MFa7tcJ2Pybss1tK93iriT9qW"
}

try:
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        total = len(data)

        with_c2 = sum(1 for r in data if r.get('category_two_description'))
        with_c1 = sum(1 for r in data if r.get('category_one_description'))
        with_both = sum(1 for r in data if r.get('category_two_description') and r.get('category_one_description'))

        print(f"\nTotal TX records (2024-2025): {total}")
        print(f"With Category 2: {with_c2} ({with_c2/total*100:.1f}%)")
        print(f"With Category 1: {with_c1} ({with_c1/total*100:.1f}%)")
        print(f"With BOTH: {with_both} ({with_both/total*100:.1f}%)")
        print(f"With NEITHER: {total - with_c2 - with_c1 + with_both}")

        # Show a sample record
        if data:
            print("\n=== Sample Record ===")
            sample = data[0]
            print(f"Application: {sample.get('application_number')}")
            print(f"Entity: {sample.get('billed_entity_name')}")
            print(f"City: {sample.get('billed_entity_city')}, {sample.get('billed_entity_state')}")
            print(f"Year: {sample.get('funding_year')}")
            print(f"Has C1: {'Yes' if sample.get('category_one_description') else 'No'}")
            print(f"Has C2: {'Yes' if sample.get('category_two_description') else 'No'}")
    else:
        print(f"Error: {resp.text}")

except Exception as e:
    print(f"Exception: {e}")
