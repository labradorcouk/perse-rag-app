# config/table_schemas.py

# --- Table schemas for column types ---
TABLE_SCHEMAS = {
    "EPC Non-Domestic Scotland": {
        "numeric": [
            "CURRENT_ENERGY_PERFORMANCE_RATING",
            "PRIMARY_ENERGY_VALUE",
            "BUILDING_EMISSIONS",
            "FLOOR_AREA"
        ],
        "categorical": [
            "CURRENT_ENERGY_PERFORMANCE_BAND",
            "PROPERTY_TYPE"
        ],
        "datetime": [
            "LODGEMENT_DATE"
        ]
    },
    # Add more tables here as needed
} 