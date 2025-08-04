# config/queries.py

# --- Example Queries (customize as needed) ---
QUERIES = {
    "EPC Non-Domestic Scotland": '''
        query($first: Int!) {
            epcNonDomesticScotlands(first: $first) {
                items {
                    CURRENT_ENERGY_PERFORMANCE_BAND
                    CURRENT_ENERGY_PERFORMANCE_RATING
                    LODGEMENT_DATE
                    PRIMARY_ENERGY_VALUE
                    BUILDING_EMISSIONS
                    FLOOR_AREA
                    PROPERTY_TYPE
                }
            }
        }
    ''',
    # Add more queries as needed
} 