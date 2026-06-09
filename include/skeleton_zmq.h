#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <mutex>

#include <Eigen/Dense>
#include <zmq.hpp>

#include "header_capsuleDinamiche.h"

// Questo header gestisce l'interfaccia tra lo script python (lento e asincrono) e il controllo in cpp (veloce e deterministico essendo real time).

// Definizione del protocollo binario (deve corrispondere a struct.pack in Python).
// #pragma pack(push, 1) assicura che non ci sia padding (spazi vuoti) tra i campi,
// garantendo la compatibilità binaria diretta con i dati inviati da Python.
#pragma pack(push, 1)
struct SkeletonCapsuleMsgHeader {
  char magic[4];     // "SKEL" - Identificatore del protocollo per validazione
  uint16_t version;  // 1 - Versione del protocollo
  uint16_t n_caps;   // <= 32 - Numero di capsule contenute nel messaggio
  uint64_t t_ns;     // sender monotonic (solo debug) - Timestamp invio da Python
};

struct SkeletonCapsuleRecord {
  float x1, y1, z1; // Punto iniziale segmento (metri)
  float x2, y2, z2; // Punto finale segmento (metri)
  float radius;     // Raggio ostacolo (metri)
  float conf;       // Confidenza rilevamento (0.0 - 1.0)
};
#pragma pack(pop)
// #pragma pack(pop) dice al compilatore C++ di non aggiungere spazi tra i dati. Senza questo comando ci sarebbero byte vuoti tra le variabili (comportamento di default) cosa però che sarebbe incompatibile con come impacchetta i dati python.

constexpr int MAX_SKELETON_CAPS = 32;

// Buffer interno per memorizzare i dati decodificati e pronti all'uso.
struct SkeletonCapsuleBuffer {
  uint64_t rx_time_ns = 0;   // Timestamp ricezione (steady_clock locale al subscriber)
  uint64_t sender_time_ns = 0; // Timestamp invio (da Python)
  uint16_t n_caps = 0;
  std::array<CapsuleGeo, MAX_SKELETON_CAPS> caps; // Array di capsule geometriche pronte per i calcoli
  // array invece che vector in modo da avere allocazione statica in memoria durante il ciclo di controllo (operazioni di new/malloc sono vietati nei sistemi realtime critici)
};

// Classe Subscriber ZMQ per ricevere dati scheletro in background.
// Utilizza un thread separato per la ricezione e un meccanismo di double-buffering
// per permettere al thread di controllo real-time di leggere sempre l'ultimo dato
// disponibile senza blocchi (lock-free reading).
class SkeletonZmqSubscriber {
public:
  // Costruttore
  explicit SkeletonZmqSubscriber(const std::string& endpoint)
      : ctx_(1), sub_(ctx_, ZMQ_SUB), endpoint_(endpoint) {}

  // Distruttore
  ~SkeletonZmqSubscriber() { stop(); }

  // Avvia il thread di ricezione
  void start() {
    if (running_.exchange(true)) return;
    worker_ = std::thread([this]() { loop(); });
  }

  // Ferma il thread di ricezione
  void stop() {
    if (!running_.exchange(false)) return;
    if (worker_.joinable()) worker_.join();
  }

  // Legge l'ultimo buffer valido in modo thread-safe e non bloccante.
  // Viene chiamata dal loop di controllo real-time (1kHz).
  void readLatest(SkeletonCapsuleBuffer& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out = buffer_;
  }

private:
  // Loop principale del thread worker
  void loop() {
    // Collegamento all'indirizzo ICP definito da python
    sub_.connect(endpoint_);
    sub_.set(zmq::sockopt::subscribe, ""); // Sottoscrizione a tutti i messaggi

    const auto t0 = std::chrono::steady_clock::now();

    while (running_.load(std::memory_order_relaxed)) {
      zmq::message_t msg;
      // Ricezione bloccante (ma siamo in un thread separato, quindi non blocca il robot)
      // Il thread di scrittura da python (non della lettura del robot) si ferma qui, finchè non arriva un pacchetto da python. 
      if (!sub_.recv(msg, zmq::recv_flags::none)) continue;
      
      // Validazione dimensione minima header
      if (msg.size() < sizeof(SkeletonCapsuleMsgHeader)) continue;

      SkeletonCapsuleMsgHeader hdr{};
      std::memcpy(&hdr, msg.data(), sizeof(hdr));
      
      // Validazione Magic e Versione 
      // Controlla che i primi 4 byte siano "SKEL". Se riceve spazzatura o dati da un altro programma per errore, li scarta immediatamente.
      if (std::memcmp(hdr.magic, "SKEL", 4) != 0) continue;
      if (hdr.version != 1) continue;
      if (hdr.n_caps > MAX_SKELETON_CAPS) continue;

      // Validazione dimensione payload
      const size_t need = sizeof(SkeletonCapsuleMsgHeader) + hdr.n_caps * sizeof(SkeletonCapsuleRecord);
      if (msg.size() < need) continue;

      const auto* recs = reinterpret_cast<const SkeletonCapsuleRecord*>(
          static_cast<const uint8_t*>(msg.data()) + sizeof(SkeletonCapsuleMsgHeader));

      const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();

      // Sezione critica: aggiornamento protetto del buffer
      {
          std::lock_guard<std::mutex> lock(mutex_);
          buffer_.rx_time_ns = now_ns;
          buffer_.sender_time_ns = hdr.t_ns;
          buffer_.n_caps = hdr.n_caps;

          for (int i = 0; i < hdr.n_caps; ++i) {
            buffer_.caps[i].p_start = Eigen::Vector3d(recs[i].x1, recs[i].y1, recs[i].z1);
            buffer_.caps[i].p_end   = Eigen::Vector3d(recs[i].x2, recs[i].y2, recs[i].z2);
            buffer_.caps[i].radius  = (double)recs[i].radius;
          }
      }
    }
  }

  // Concetto di double buffering 
  /*
  Il problema principale è: il thread ZMQ sta scrivendo i dati mentre il robot li vuole leggere. Se accedessero alla stessa variabile contemporaneamente, avresti dati corrotti (es. metà scheletro nuovo e metà vecchio).
  */
  mutable std::mutex mutex_;
  SkeletonCapsuleBuffer buffer_;
  /*
  Il thread ZMQ scrive sempre nell'altro buffer (quello libero). Quando ZMQ ha finito di scrivere, cambia active_idx_ atomicamente.
  */

  std::atomic<bool> running_{false};
  std::thread worker_;

  zmq::context_t ctx_;
  zmq::socket_t sub_;
  std::string endpoint_;
};
