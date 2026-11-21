import acoular as ac  # import before numpy (Numba/OpenBLAS warning)
import numpy as np, soundfile as sf

FS = 22050
DUR = 10.0
t = np.arange(int(FS * DUR)) / FS

# 1) Synthesize a quadcopter: 4 rotors, 2 blades, slightly different RPMs + jitter
rng = np.random.default_rng(0)
sig = np.zeros_like(t)
for rpm in (5200, 5260, 5310, 5150):
    rpm_t = rpm + 60 * np.cumsum(rng.normal(0, 1, t.size)) / np.sqrt(FS)  # random walk
    phase = 2 * np.pi * np.cumsum(2 * rpm_t / 60) / FS                    # BPF phase
    for h in range(1, 9):
        sig += (1 / h) * np.sin(h * phase)
sig += 0.3 * rng.normal(size=t.size)          # broadband hiss
sig /= np.abs(sig).max()
sf.write("dry.wav", (0.9*sig).astype(np.float32), FS, subtype="PCM_16")

# 2) Mic array: 4 mics, 0.5 m square, on the ground
mics = ac.MicGeom(pos_total=np.array([[-.25, .25, -.25, .25],
                                      [-.25, -.25, .25, .25],
                                      [0, 0, 0, 0]]))

# 3) Straight flyby at 30 m altitude, 15 m/s
traj = ac.Trajectory(points={0.0: (-75.0, 20.0, 30.0), DUR: (75.0, 20.0, 30.0)})

src_sig = ac.GenericSignalGenerator(source=ac.TimeSamples(data=sig[:, None], sample_freq=FS))
moving = ac.MovingPointSource(signal=src_sig, mics=mics, trajectory=traj, conv_amp=True)

# 4) Pull blocks and write a 4-channel WAV (Doppler + 1/r included)
out = np.concatenate(list(moving.result(4096)))
out /= np.abs(out).max()
sf.write("flyby.wav", out.astype(np.float32), FS)
print(out.shape)   # (220500, 4)
