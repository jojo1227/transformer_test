#!/bin/bash

# Variablen
SSH_USER="8zm-078-u1"
SSH_HOST="ssh.inform.hs-hannover.de"
DB_HOST="trustdatastore.inform.hs-hannover.de"
LOCAL_PORT=5432
REMOTE_PORT=5432

# Funktion zur Herstellung der SSH-Verbindung
connect_ssh() {
    echo "Verbinde mit SSH..."
    ssh -L $LOCAL_PORT:$DB_HOST:$REMOTE_PORT $SSH_USER@$SSH_HOST
}

# Hauptprogramm
echo "Starte SSH-Verbindung mit Port-Forwarding..."
connect_ssh

# Nach der SSH-Verbindung
echo "SSH-Verbindung beendet."
echo "Sie können nun eine Verbindung zur Datenbank herstellen mit:"
echo "Server: localhost"
echo "Port: $LOCAL_PORT"
echo "Datenbank: cadets-e3"