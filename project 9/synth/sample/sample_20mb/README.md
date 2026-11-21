# sample_20mb

Small example: MAD noise (no helicopters, no speech) + synthetic drones and explosions, MAD-only backgrounds.

Created 2026-11-21 17:15:09 in 3.5 s. Audio: FLAC, 16 kHz mono, 5 s.
Sources: 60 MAD, 70 synthetic, 0 challenge train.

| split | total | drone_buzz | vehicle_mechanical | explosion | ambient_outdoor |
|---|---|---|---|---|---|
| train | 130 | 50 | 24 | 44 | 12 |
| valid | 8 | 2 | 4 | 1 | 1 |

Labels: dataset/<split>/labels.jsonl (challenge schema), labels.csv, metadata.jsonl.
Full generator settings: manifest.json -> config.
