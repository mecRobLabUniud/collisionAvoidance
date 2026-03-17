
### Descrizione del Progetto
Il progetto "implementazione_dynamic_safety_zones" implementa un sistema di controllo cartesiano per un robot Franka Panda, integrando zone di sicurezza dinamiche basate su capsule geometriche. Queste capsule rappresentano regioni di sicurezza, di dimensione variabile, attorno al robot e agli oggetti umani rilevati, permettendo un'interazione sicura durante operazioni di pick and place.

Il sistema combina:
- **Controllo cartesiano**: Il robot segue traiettorie pianificate in posizione.
- **Tracking scheletrico**: Utilizzo di YOLOv8 per il rilevamento delle pose umane in tempo reale.
- **Comunicazione in tempo reale**: Tramite ZeroMQ per lo scambio di dati tra componenti Python e C++.
- **Capsule dinamiche**: Zone di sicurezza che si adattano dinamicamente alla posizione del robot e degli umani.
- **Architettura Multithread**: Separazione del ciclo di controllo real-time (1 kHz) dal solver di ottimizzazione (CasADi) tramite thread paralleli per garantire il rispetto delle scadenze temporali hard real-time.

Il controllo opera a 1 kHz per garantire risposte in tempo reale.

### File Presenti (Descrizione Dettagliata)

- **CMakeLists.txt**: File di configurazione per la compilazione del progetto utilizzando CMake. Definisce le dipendenze, i target di build e le librerie necessarie (inclusa libfranka). Configura il linking con Eigen, Pinocchio, CasADi e altre librerie per l'ottimizzazione e la cinematica.

- **controllo_pick_and_place_capsule_dinamiche_parallelo.cpp**: File principale contenente l'implementazione del controllo cartesiano con capsule dinamiche per operazioni di pick and place, utilizzando un'architettura parallela. Include:
  - Caricamento del modello URDF del robot Panda usando Pinocchio.
  - **Thread Ottimizzatore**: Esegue il solver CasADi in un thread separato (~50Hz) per calcolare il tempo di stop ottimale senza bloccare il controllo.
  - **Thread Controllo Real-Time**: Loop a 1 kHz che gestisce la comunicazione con il robot, legge i dati condivisi dall'ottimizzatore in modo thread-safe (mutex), e calcola le capsule dinamiche.
  - Definizione di waypoints per traiettorie pick and place.
  - Gestione stati: RUNNING (esecuzione traiettoria), STOPPING (frenata sicura interpolata), PAUSED (robot fermo), RECOVERING (ripresa traiettoria).
  - Calcolo delle capsule geometriche del robot basate sui frame dei giunti.
  - Ricezione dati skeleton umani via ZMQ e calcolo distanze capsule-robot vs capsule-umane.
  - Controllo di sicurezza: se distanza < raggio dinamico, attiva STOPPING utilizzando i parametri calcolati asincronamente dal thread ottimizzatore.

- **examples_common.cpp** e **examples_common.h**: Codice comune e header utilizzati negli esempi, contenenti funzioni di utilità per l'interfaccia con il robot Franka (setDefaultBehavior, calcolo traiettorie polinomiali, etc.).

- **header_capsuleDinamiche.cpp** e **header_capsuleDinamiche.h**: Implementazione e dichiarazione delle classi per gestire le capsule dinamiche. Include:
  - Strutture dati per capsule geometriche (CapsuleGeo) e parametri di sicurezza (DynamicSafetyParams).
  - Funzione optimize_stop_time_casadi_hybrid: usa CasADi e Pinocchio per ottimizzare il tempo di stop del robot, considerando limiti di velocità/accelerazione/jerk/coppia, costruendo traiettorie polinomiali di quinto grado per fermate sicure.
  - Calcolo raggi dinamici delle capsule basati su velocità robot e umana, tempo di reazione.
  - Funzioni per calcolo distanze tra segmenti (capsule).

- **Istruzioni_compilazione.txt**: Documento con istruzioni dettagliate per la compilazione e l'esecuzione del progetto (vedi sezione Istruzioni di Esecuzione).

- **marker_pos.txt**: File contenente la posizione del marker (ArUco) utilizzato per il tracking o la calibrazione della camera.

- **rotation_matrix.txt**: Matrice di rotazione utilizzata per trasformazioni di coordinate, per allineare i sistemi di riferimento tra robot e visione mediante una trasformazione da camera a base robot.

- **run_system_dynamic.sh**: Script shell per l'esecuzione automatica del sistema completo. Gestisce:
  - Compilazione automatica del codice C++.
  - Avvio del processo Python per skeleton tracking.
  - Avvio del controllo C++ con gestione processi (cleanup su SIGINT/SIGTERM).
  - Configurazione IP robot e percorsi.

- **skeleton_yolo_and_transmission.py**: Script Python che gestisce il rilevamento e trasmissione dei dati dello skeleton. Include:
  - Uso di YOLOv8 (modello yolov8x-pose.pt) per rilevamento pose umane da camera RealSense.
  - Filtro One Euro per smoothing dei keypoints 3D, riducendo jitter mantenendo bassa latenza.
  - Conversione keypoints in capsule geometriche (segmenti tra giunti).
  - Trasmissione dati via ZeroMQ (socket IPC) al processo C++.
  - Gestione occlusioni: mantiene ultimi valori validi per 0.5s, poi NaN.
  
- **skeleton_zmq.h**: Header C++ per l'integrazione di ZeroMQ nel codice di controllo. Definisce strutture per buffer capsule skeleton e classe SkeletonZmqSubscriber per ricezione thread-safe dei dati da Python.

- **yolov8x-pose.pt**: Modello pre-addestrato di YOLOv8 per il rilevamento delle pose (skeleton tracking). Rileva 17 keypoints 3D per persona, usato per costruire capsule umane dinamiche.


### Istruzioni di Esecuzione
Il sistema richiede un robot Franka Panda connesso, una camera RealSense per visione, e librerie installate (libfranka, Pinocchio, CasADi, pyrealsense2, ultralytics, zmq).

#### Esecuzione Automatica (Raccomandata)
1. Modifica `run_system_dynamic.sh` per impostare `ROBOT_IP` (default 172.16.0.2) e `FRANKA_DIR` (percorso build libfranka).
2. Rendi eseguibile lo script: `chmod +x run_system_dynamic.sh`
3. Esegui: `./run_system_dynamic.sh`
   - Compila automaticamente il C++.
   - Avvia skeleton tracking Python in background.
   - Avvia controllo C++.
   - Premi Ctrl+C per arrestare tutto.

# Configurazione del file bash
1. **Rendi eseguibile lo script bash**
    - chmod +x run_system_dynamic.sh
2. **Compila il progetto in c++, se non è già stato fatto**
    - mkdir -p build && cd build
    - cmake .. -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=/home/lab/donaldo_ws/home/lab/donaldo_ws/libfranka/build
    - make -j4
    - cd ..
3. **Lancia tutto**
    - ./run_system_dynamic.sh

#### Esecuzione Manuale
1. **Compilazione C++**:
   ```
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=/path/to/libfranka/build
   make
   cp ../*.csv .  # se presenti file CSV
   ```

2. **Avvio Skeleton Tracking** (in terminale separato):
   ```
   python3 skeleton_yolo_and_transmission.py
   ```
   - Assicurati camera RealSense connessa e pyrealsense2 installato.

3. **Avvio Controllo Robot**:
   ```
   cd build
   ./controllo_pick_and_place_capsule_dinamiche_parallelo <ROBOT_IP>
   ```
   - Esempio: `./controllo_pick_and_place_capsule_dinamiche_parallelo 172.16.0.2`

#### Note
- Il controllo opera a 1 kHz; assicurati bassa latenza di rete.
- Log salvati in `log_dynamic_success.csv` o `log_error_dynamic.csv`.
- Per calibrazione, usa `marker_pos.txt` e `rotation_matrix.txt` per allineare sistemi di coordinate.
- Se skeleton non valido, capsule umane ignorate (raggio = 0).



### Setup CasADi e Pinocchio

## Installare CasADi a livello di sistema
# Installare i prerequisiti necessari
sudo apt update
sudo apt install -y build-essential cmake git pkg-config coinor-libipopt-dev gfortran

# Clonare il repository ufficiale
git clone https://github.com/casadi/casadi.git
cd casadi

# Configurazione con CMake
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..

# Compilazione e installazione
make -j$(nproc)
sudo make install
sudo ldconfig *per aggiornare la cache*

## Installare Pinocchio a livello di sistema
# Installare le dipendenze di Pinocchio
sudo apt update
sudo apt install -y cmake git pkg-config
sudo apt install -y libeigen3-dev libboost-all-dev liburdfdom-dev

# Clonare repository ufficiale
cd ~
git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio

# Configurazione con CMake (con supporto Casadi)
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_PYTHON_INTERFACE=OFF \
      -DBUILD_WITH_URDF_SUPPORT=ON \
      -DBUILD_WITH_CASADI_SUPPORT=ON \
      ..

# Compilazione ed installazione
make -j$(nproc)
sudo make install
sudo ldconfig
