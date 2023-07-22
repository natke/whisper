#!/bin/bash

# Set account and subscription
az account set --subscription <subscription ID>
az configure --defaults workspace=<Azure Machine Learning workspace name> group=<resource group>

export ENDPOINT_NAME="whisper-onnx"

# Register a model
az ml model create --name whisper-large --path models/whisper-large-e2e-int8.onnx 

# List models
az ml model list 

# Create the endpoint
az ml online-endpoint create --name $ENDPOINT_NAME --file endpoint.yml

# Show the status of the endpoint
az ml online-endpoint show -n $ENDPOINT_NAME (--local, if you are runninf locally)

# Delete the endpoint. Need to remove the traffic first
az ml online-endpoint delete -n $ENDPOINT_NAME

# Update the traffic to the endpoint
az ml online-endpoint update -n $ENDPOINT_NAME --traffic "green=100"

# Create a deployment. Replace "green" with name of deployment. Replace "deploy-green.yml" with the name of the deployment file
az ml online-deployment create --name green --endpoint $ENDPOINT_NAME -f deploy-green.yml

# Get the logs for the deployment
az ml online-deployment get-logs --name green --endpoint $ENDPOINT_NAME

# Delete a deployment
az ml online-deployment delete --name green --endpoint $ENDPOINT_NAME

# Update the deployment (e.g. if you change the scoring file)
az ml online-deployment update --name green --endpoint $ENDPOINT_NAME -f deploy-green.yml 


