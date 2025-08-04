edition = "edition_18"

input_file = "edition_18_0_new_format.gpkg"
cnx = sqlite3.connect(input_file)
table_list = pd.read_sql_query("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name", cnx)

start_time = time.time()

# Iterate over each table in the database
for table_name in table_list['name']:
    sql_query = f"SELECT * FROM {table_name}"
    print(table_name)
    
    chunk_size = 1000000
    chunks = pd.read_sql_query(sql_query, cnx, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunks):
        if not chunk.empty:
            from shapely.ops import transform
            from shapely.wkb import loads
            from shapely.geometry import Polygon, MultiPolygon
            from pyproj import Transformer
            import geojson

            spark_df = spark.createDataFrame(chunk)
            
            # Apply column casting only for edition_17_0_new_format table
            if table_name == f"{edition}_0_new_format":

                # Create UDF for GeoJSON conversion
                convert_bson_to_geojson = F.udf(bson_to_geojson, StringType())

                # Add new GeoJSON column (do this BEFORE converting 'geom' to WKT)
                spark_df = spark_df.withColumn("geojson", convert_bson_to_geojson(spark_df["geom"]))

                print("Converted bson to geojson")

                convert_bson_to_wkt = F.udf(bson_to_wkt, StringType())
                spark_df = spark_df.withColumn("geom", convert_bson_to_wkt(spark_df["geom"]))

                print("Converted bson to wkt")

                spark_df = spark_df.withColumn("bedroom_count", col("bedroom_count").cast(DoubleType()))
                spark_df = spark_df.withColumn("reception_room_count", col("reception_room_count").cast(DoubleType()))
                spark_df = spark_df.withColumn("habitable_rooms", col("habitable_rooms").cast(DoubleType()))
                spark_df = spark_df.withColumn("open_fireplaces", col("open_fireplaces").cast(DoubleType()))
                spark_df = spark_df.withColumn("premise_type_confidence", col("premise_type_confidence").cast(DoubleType()))
                spark_df = spark_df.withColumn("uprn", col("uprn").cast(LongType()))

                print("completed column transformations")
            
            # Write to Delta format
            spark_df.write.mode("append").format("delta").option("mergeSchema", "true").option("delimiter", ",").option("multiline", "true").save(f"{tablePath}/{schema_name}/{table_name}")
            display(i)

# End timer
end_time = time.time()

execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")

abc_link_file_df = spark.sql(f"SELECT * FROM LH_external_datasets.verisk.ukbuildings_{edition}_abc_link_file")
new_format_df = spark.sql(f"SELECT * FROM LH_external_datasets.verisk.{edition}_0_new_format")

v_joined = abc_link_file_df.alias("A") \
    .join(
        new_format_df.alias("B"),
        col("A.upn") == col("B.verisk_premise_id"),
        "inner"
    ) \
    .select(
        col("A.uprn").alias("uprn_link"),
        col("A.upn").alias("upn_link"),
        col("B.*")
    )
v_joined.write.mode("overwrite").format("delta").option("mergeSchema", "true").option("delimiter", ",").option("multiline", "true").save(f"{tablePath}/{schema_name}/v_{edition}_joined")