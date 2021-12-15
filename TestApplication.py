import datetime

from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

token = "Eo8wFg4vDe4nsVMppA-gtsvuMjxCJiYIVjZgb-Ui3UTQ_LWPFNb5a-36zcnfckrBrIVVzl82xzbgOlvfWr_JqQ=="
org = "Maze"
bucket = "AnalyticalData"

client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

# Sample data point
for i in range(4):
    pp = i
    p = Point("AnalyticalSolution").tag("DataDescription", "Testing").field("Particle_X", pp).time(datetime.datetime.utcnow().isoformat())

    # Write to DB
    with client.write_api(write_options=WriteOptions(batch_size=10000,
                                                    flush_interval=10_000,
                                                    jitter_interval=2_000,
                                                    retry_interval=5_000,
                                                    max_retries=5,
                                                    max_retry_delay=30_000,
                                                    exponential_base=2)) as _write_client:
                                                    _write_client.write(bucket, org, p)


# Query from DB
query_api = client.query_api()
query = 'from(bucket: "AnalyticalData") |> range(start: -12h)'
tables = client.query_api().query(query, org=org)
for table in tables:
    for record in table.records:
        print("queried")
