#!/bin/bash

# ============================================================================
# SCRIPT AVVIO SISTEMA CAPSULE DINAMICHE (FRANKA + VISIONE + PINOCCHIO/CASADI)
# ============================================================================

# --- CONFIGURAZIONE ---
ROBOT_IP="172.16.0.2"
FRANKA_DIR="/home/lab/donaldo_ws/libfranka/build" # Modifica se necessario
PYTHON_SCRIPT="skeleton_yolo_and_transmission.py"
BUILD_DIR="build"
CPP_EXECUTABLE="controllo_pick_and_place_capsule_dinamiche_parallelo"

# Colori
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- FUNZIONE CLEANUP ---
cleanup() {
    echo ""
    echo -e "${YELLOW}--- Chiusura sistema ---${NC}"
    
    if [ -n "$PID_PYTHON" ]; then
        echo -e "${YELLOW}Terminazione Python (PID $PID_PYTHON)...${NC}"
        kill $PID_PYTHON 2>/dev/null
        wait $PID_PYTHON 2>/dev/null
    fi
    
    echo -e "${GREEN}Sistema arrestato.${NC}"
    exit
}

trap cleanup SIGINT SIGTERM

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}  SISTEMA CAPSULE DINAMICHE (PINOCCHIO + CASADI)${NC}"
echo -e "${BLUE}=====================================================${NC}\n"

# --- 1. COMPILAZIONE C++ ---
echo -e "${GREEN}[1/3] Compilazione C++...${NC}"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake .. -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=$FRANKA_DIR > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}ERRORE: CMake fallito. Riprovo con output esplicito:${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=$FRANKA_DIR
    exit 1
fi

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}ERRORE: Compilazione fallita.${NC}"
    exit 1
fi

cd ..
echo -e "${GREEN}✓ Compilazione completata${NC}\n"

# --- 2. AVVIO VISIONE ---
echo -e "${GREEN}[2/3] Avvio modulo Visione (YOLO + RealSense)...${NC}"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}ERRORE: $PYTHON_SCRIPT non trovato!${NC}"
    exit 1
fi

python3 $PYTHON_SCRIPT &
PID_PYTHON=$!

sleep 1
if ! ps -p $PID_PYTHON > /dev/null; then
    echo -e "${RED}ERRORE: Script Python non avviato correttamente.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Visione avviata (PID: $PID_PYTHON)${NC}"

echo -e "${YELLOW}Inizializzazione sistema visione (8s)...${NC}"
for i in {8..1}; do
    echo -ne "\r  $i secondi rimanenti... "
    sleep 1
done
echo -e "\r${GREEN}✓ Sistema pronto                      ${NC}\n"

# --- 3. AVVIO CONTROLLO ---
echo -e "${GREEN}[3/3] Avvio Controllo Robot...${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Robot IP: $ROBOT_IP${NC}"
echo -e "${BLUE}Ottimizzazione: Dinamica Completa (Pinocchio)${NC}"
echo -e "${BLUE}========================================${NC}\n"

if [ ! -f "$BUILD_DIR/$CPP_EXECUTABLE" ]; then
    echo -e "${RED}ERRORE: Eseguibile $CPP_EXECUTABLE non trovato!${NC}"
    cleanup
fi

./$BUILD_DIR/$CPP_EXECUTABLE $ROBOT_IP
CPP_EXIT_CODE=$?

# --- 4. REPORT FINALE ---
echo ""
if [ $CPP_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Task completato con successo.${NC}"
else
    echo -e "${RED}✗ Errore imprevisto (codice: $CPP_EXIT_CODE)${NC}"
fi

cleanup