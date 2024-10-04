#!/bin/bash

echo "Starting Docker services..."
docker-compose up -d

if [ ! -d ".dvc" ]; then
  echo "Initializing DVC..."
  dvc init
fi

if [ ! -f "data/raw/Occupancy_Estimation.csv.dvc" ]; then
  echo "Adding raw data to DVC..."
  dvc add data/raw/Occupancy_Estimation.csv
fi

echo "Committing changes to Git..."
git add .
git commit -m "Running full MLOps pipeline"

echo "Running the DVC pipeline..."
dvc repro

if dvc remote list | grep -q 'gdrive_remote'; then
  echo "Pushing DVC artifacts to remote storage..."
  dvc push
fi

echo "Stopping Docker services..."
docker-compose down

echo "Pipeline run complete."
