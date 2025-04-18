#!/bin/bash

# Shell script to delete Kubernetes pods matching specific patterns

# Define the patterns
PATTERNS="run-evaluation-|run-build-database-|run-pipeline-"

# Get the list of pod names that match the patterns and delete them
kubectl delete pods $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep -E "$PATTERNS")

# Optional: Add a message to confirm the operation is complete
echo "Pods matching the patterns have been deleted."
