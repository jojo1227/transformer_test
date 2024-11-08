#!/bin/bash

# Variablen
DOWNLOAD_DIR=~/Documents/Uni/cadets_transformer
ZIP_FILE=$DOWNLOAD_DIR/dataset.zip
EXTRACT_DIR=$DOWNLOAD_DIR/extracted_data

# URL für den Download
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/sovitrath/imdb-movie-review-classification-full-and-mini"

# Erstelle das Extraktionsverzeichnis, falls es nicht existiert
mkdir -p $EXTRACT_DIR

# Downloade die Datei
echo "Downloading dataset..."
curl -L -o $ZIP_FILE $KAGGLE_URL

# Überprüfe, ob der Download erfolgreich war
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
    
    # Entpacke die ZIP-Datei
    echo "Extracting files..."
    unzip -o $ZIP_FILE -d $EXTRACT_DIR
    
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully."
        
        # Optional: Lösche die ZIP-Datei nach dem Entpacken
        # rm $ZIP_FILE
        # echo "ZIP file removed."
    else
        echo "Error: Failed to extract the ZIP file."
    fi
else
    echo "Error: Download failed."
fi

echo "Script finished."