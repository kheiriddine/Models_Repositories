import pandas as pd
import random
import faker
import os

# Initialize Faker instance
fake = faker.Faker()

# Categories for the pharmaceutical domain
categories = [
    "Consultation", "Software", "IT", "Blood Products", 
    "Medical Devices", "Pharmaceutical Supplies", 
    "Research & Development", "Clinical Trials", "Laboratory Equipment", 
    "Regulatory & Compliance", "Packaging Materials", "Healthcare Services"
]
# Generate random mock data
def generate_mock_data(num_records):
    data = []
    for _ in range(num_records):
        supplier = fake.company()
        category = random.choice(categories)
        amount_spent = round(random.uniform(1000, 100000), 2)
        quantity_purchased = random.randint(1, 100)
        country = fake.country()
        price_rate = round(amount_spent / quantity_purchased, 2)
        # Generate item description based on category
        if category == "Consultation":
            item_desc = f"Professional consultation services in healthcare sector for {fake.bs()}."
        elif category == "Software":
            item_desc = f"Software solutions for data management in pharmaceutical companies."
        elif category == "IT":
            item_desc = f"IT infrastructure services, including servers and networking solutions."
        elif category == "Blood Products":
            item_desc = f"Blood plasma, red blood cells, and other blood products for hospitals."
        elif category == "Medical Devices":
            item_desc = f"Medical equipment like diagnostic tools and surgical instruments."
        elif category == "Pharmaceutical Supplies":
            item_desc = f"Various pharmaceutical raw materials and chemical supplies for production."
        elif category == "Research & Development":
            item_desc = f"R&D materials for new drug formulation and clinical trials."
        elif category == "Clinical Trials":
            item_desc = f"Services and supplies for conducting clinical trials and patient studies."
        elif category == "Laboratory Equipment":
            item_desc = f"Laboratory instruments like microscopes, pipettes, and centrifuges."
        elif category == "Regulatory & Compliance":
            item_desc = f"Services ensuring compliance with health regulations and safety standards."
        elif category == "Packaging Materials":
            item_desc = f"Packaging materials including vials, bottles, and blister packs."
        elif category == "Healthcare Services":
            item_desc = f"Healthcare service contracts, such as medical staffing and patient management."
        # Supplier description can be random or tailored
        supplier_desc = f"{supplier} provides high-quality services in the {category.lower()} domain."
        # Add data to list
        data.append({
            "ID": fake.uuid4(),
            "Date": fake.date_this_decade(),
            "Supplier": supplier,
            "Supplier Description": supplier_desc,
            "Category": category,
            "Item Description": item_desc,
            "Amount Spent": amount_spent,
            "Quantity Purchased": quantity_purchased,
            "Country": country,
            "Price Rate": price_rate
        })

    return pd.DataFrame(data)
# Generate mock data (e.g., 100 records)
num_records = 100
df = generate_mock_data(num_records)
# Save to CSV file * faut changer le nom de fichier svp
df.to_csv(f'{os.getcwd()}/mock_files/mock_data_pharma1.csv', index=False,sep='|')

