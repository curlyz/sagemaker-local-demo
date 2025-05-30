#!/bin/bash

# Example test script for FastAPI fraud detection endpoint

set -e

API_URL="http://127.0.0.1:8000/predict"

curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": 0.1,
    "V2": -1.2,
    "V3": 0.5,
    "V4": 1.3,
    "V5": -0.7,
    "V6": 0.0,
    "V7": 0.9,
    "V8": -0.2,
    "V9": 0.8,
    "V10": -1.1,
    "V11": 0.4,
    "V12": 0.3,
    "V13": -0.6,
    "V14": 0.2,
    "V15": 1.0,
    "V16": 0.1,
    "V17": -0.5,
    "V18": 0.6,
    "V19": 0.7,
    "V20": -0.8,
    "V21": 0.2,
    "V22": 0.0,
    "V23": 1.1,
    "V24": -0.1,
    "V25": 0.5,
    "V26": -0.9,
    "V27": 0.3,
    "V28": 0.4
  }'
