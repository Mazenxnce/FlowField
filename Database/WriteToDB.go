package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"time"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/schollz/progressbar/v3"
)

// Function takes in one row of data (in []string format)
func WriteToDB(DataPoint []string) {
	
	// Create InfluxDB Client
	client := influxdb2.NewClient("http://localhost:8086", "nyfamru803FI19kQi4RCSA3PqAqx3FJ2HOFHEqLpYWZWfYDCj7ZSFc23WQBN_9YhpictlIVAVVfhM2yKi24_Xg==")

	writeAPI := client.WriteAPI("Maze", "AnalyticalDataCSV")

	// Create Point using fluent style
	p := influxdb2.NewPointWithMeasurement("Analytical Solution").
		AddTag("Data Description", "Testing Data (Go)").
		AddField("Time of Measurement", DataPoint[0]).
		AddField("Particle X", DataPoint[2]).
		AddField("Particle Y", DataPoint[3]).
		AddField("Flowrate 1", DataPoint[4]).
		AddField("Flowrate 2", DataPoint[5]).
		AddField("Flowrate 3", DataPoint[6]).
		AddField("Flowrate 4", DataPoint[7]).
		AddField("Flowrate 5", DataPoint[8]).
		AddField("Flowrate 6", DataPoint[9]).
		AddField("Velocity X", DataPoint[10]).
		AddField("Velocity Y", DataPoint[11]).
		SetTime(time.Now())

	// write point asynchronously
	writeAPI.WritePoint(p)

	// Flush writes
	writeAPI.Flush()

	// always close client at the end
	defer client.Close()
}

// get non-blocking write client

func main() {
	// STEP 1: Open the CSV File
	csvfile, err := os.Open("/Users/zain/FlowField/HeleShawAnalyticalSolution.csv")
	if err != nil {
		log.Fatal(err)
	}

	r := csv.NewReader(csvfile)
	bar := progressbar.Default(-1)

	for {
		record, err := r.Read()

		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		WriteToDB(record)

		bar.Add(1)
		time.Sleep(40 * time.Millisecond)
	}
}
