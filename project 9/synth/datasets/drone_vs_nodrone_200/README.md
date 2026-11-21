# drone_vs_nodrone_200

Binary drone / no-drone baseline: 100 synthetic drones (physics model) over MAD noise, 100 MAD noise clips. Helicopters and speech excluded.

Created 2026-11-21 16:20:24 in 4.9 s. Audio: FLAC, 16 kHz mono, 5 s.
Sources: 100 MAD, 100 synthetic, 0 challenge train.

| split | total | drone_buzz | vehicle_mechanical | explosion | ambient_outdoor |
|---|---|---|---|---|---|
| train | 200 | 100 | 40 | 40 | 20 |
| valid | 8 | 2 | 4 | 1 | 1 |

Labels: dataset/<split>/labels.jsonl (challenge schema), labels.csv, metadata.jsonl.
Full generator settings: manifest.json -> config.
