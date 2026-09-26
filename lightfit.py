import numpy as np, torch, glob, math, sys
from PIL import Image

SP = sys.argv[1]
files = sorted(glob.glob('P:/test/Gold-texture/image_stack/*.jpg'))
ids = [f[-8:-4] for f in files]
S = np.load(SP + '/aligned1500.npy')[:, 14:-14, 14:-14]
N, H, W, _ = S.shape
gold = np.load(SP + '/gold_mask.npy')

clip = S.max(-1) >= 250
c = S.astype(np.float32) / 255
lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
Y = lin[..., 0] * 0.0722 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.2126
del c, lin
Ls = np.sort(S.mean(-1), axis=0)[int(0.15 * (N - 1))]
good = (~gold) & (Ls > 60)

B = 20
h, w = 1800 // B, 1460 // B
def block(a):
    return a[..., :h * B, :w * B].reshape(*a.shape[:-2], h, B, w, B).mean((-3, -1))
valid = good[None] & ~clip
wsum = block(valid.astype(np.float32))
Yb = block(Y * valid) / np.maximum(wsum, 1e-6)
wt = (wsum > 0.6).astype(np.float32)
print('usable blocks per frame:', wt.sum((1, 2)).astype(int).tolist())

dev = torch.device('cpu')
logL = torch.tensor(np.log(np.maximum(Yb, 1e-4)), dtype=torch.float64)
Wt = torch.tensor(wt, dtype=torch.float64)
W0 = Wt.clone()
ys = (torch.arange(h, dtype=torch.float64) + 0.5) * B
xs = (torch.arange(w, dtype=torch.float64) + 0.5) * B
GY, GX = torch.meshgrid(ys, xs, indexing='ij')
cx, cy = w * B / 2, h * B / 2
SCALE = 1000.0

loga = (logL * Wt).sum(0) / Wt.sum(0).clamp(min=1)
rat = logL - loga
az0 = []
for i in range(N):
    m = Wt[i] > 0
    A = torch.stack([torch.ones(int(m.sum()), dtype=torch.float64), (GX[m] - cx) / SCALE, (GY[m] - cy) / SCALE], 1)
    sol = torch.linalg.lstsq(A, rat[i][m][:, None]).solution[:, 0]
    az0.append(math.atan2(float(sol[2]), float(sol[1])))

inits = [(r, z) for r in (0.4, 1.2, 3.0) for z in (0.5, 1.2, 3.0)]
K = len(inits)
ux = torch.tensor([[math.cos(az0[i]) * r for (r, z) in inits] for i in range(N)], dtype=torch.float64)
uy = torch.tensor([[math.sin(az0[i]) * r for (r, z) in inits] for i in range(N)], dtype=torch.float64)
lz = torch.tensor([[math.log(z) for (r, z) in inits] for i in range(N)], dtype=torch.float64)

def logshade(px, py, plz):
    X = GX[None, None]; Yg = GY[None, None]
    lx = cx + SCALE * px[..., None, None]
    ly = cy + SCALE * py[..., None, None]
    zz = SCALE * torch.exp(plz)[..., None, None]
    return torch.log(zz) - 1.5 * torch.log((X - lx) ** 2 + (Yg - ly) ** 2 + zz ** 2)

def frame_loss(px, py, plz, loga, keep_parts=False):
    ls = logshade(px, py, plz)
    tgt = (logL - loga)[:, None]
    Wk = Wt[:, None]
    logs = ((tgt - ls) * Wk).sum((-1, -2)) / Wk.sum((-1, -2))
    r = tgt - ls - logs[..., None, None]
    hub = torch.where(r.abs() < 0.08, 0.5 * r ** 2, 0.08 * (r.abs() - 0.04))
    L = (hub * Wk).sum((-1, -2)) / Wk.sum((-1, -2))
    return (L, logs, r) if keep_parts else L

def optimise(px, py, plz, loga, steps, fix_z=False):
    px = px.clone().requires_grad_(True); py = py.clone().requires_grad_(True)
    plz = plz.clone().requires_grad_(not fix_z)
    params = [px, py] + ([] if fix_z else [plz])
    opt = torch.optim.Adam(params, lr=0.02)
    for s in range(steps):
        opt.zero_grad()
        L = frame_loss(px, py, plz, loga)
        L.sum().backward()
        opt.step()
        with torch.no_grad():
            plz.clamp_(math.log(0.05), math.log(20))
    return px.detach(), py.detach(), plz.detach()

for outer in range(6):
    ux, uy, lz = optimise(ux, uy, lz, loga, 250)
    with torch.no_grad():
        L, logs, _ = frame_loss(ux, uy, lz, loga, keep_parts=True)
        best = L.argmin(1)
        bi = torch.arange(N)
        ls = logshade(ux[bi, best][:, None], uy[bi, best][:, None], lz[bi, best][:, None])[:, 0]
        ls = ls + logs[bi, best][:, None, None]
        loga = ((logL - ls) * Wt).sum(0) / Wt.sum(0).clamp(min=1)
        loga = loga - (loga * Wt.sum(0)).sum() / Wt.sum()
        r = logL - loga - ls
        med = (r.abs() * W0).sum() / W0.sum()
        Wt = W0 * (r.abs() < 4 * med + 0.02).double()
    print('round', outer, 'mean loss %.5f' % float(L[bi, best].mean()), 'init choice', best.tolist())

px, py, plz = ux[bi, best], uy[bi, best], lz[bi, best]
L0, logs, r = frame_loss(px[:, None], py[:, None], plz[:, None], loga, keep_parts=True)
tgt = logL - loga
sst = (((tgt - (tgt * Wt).sum((1, 2), keepdim=True) / Wt.sum((1, 2), keepdim=True)) ** 2) * Wt).sum((1, 2))
sse = ((r[:, 0] ** 2) * Wt).sum((1, 2))
r2 = 1 - sse / sst

sens = {}
for f in (0.6, 1.6):
    qx, qy, _ = optimise(px[:, None], py[:, None], (plz + math.log(f))[:, None], loga, 250, fix_z=True)
    sens[f] = frame_loss(qx, qy, (plz + math.log(f))[:, None], loga)[:, 0] / L0[:, 0]

rows = []
print('\nframe   light x   light y   height   (units: painting width)  azimuth  elevation  R2    loss x(h*0.6) x(h*1.6)')
lp = ['%d' % N]
for i in range(N):
    X = float(px[i]) * SCALE / (w * B)
    Yu = -float(py[i]) * SCALE / (w * B)
    Z = math.exp(float(plz[i])) * SCALE / (w * B)
    az = math.degrees(math.atan2(Yu, X))
    el = math.degrees(math.atan2(Z, math.hypot(X, Yu)))
    n = math.sqrt(X * X + Yu * Yu + Z * Z)
    lp.append('%s %.6f %.6f %.6f' % (files[i].split('/')[-1].split('\\')[-1], X / n, Yu / n, Z / n))
    print('%s  %+7.2f  %+7.2f  %6.2f                           %6.0f   %6.0f   %.2f   %5.2f    %5.2f' % (
        ids[i], X, Yu, Z, az, el, float(r2[i]), float(sens[0.6][i]), float(sens[1.6][i])))
open(SP + '/lights_est.lp', 'w').write('\n'.join(lp) + '\n')

with torch.no_grad():
    obs = (logL - loga).numpy()
    mod = (logL - loga - r[:, 0]).numpy()
tiles = []
for i in range(N):
    lo, hi = np.percentile(obs[i][wt[i] > 0], [2, 98])
    o = np.clip((obs[i] - lo) / (hi - lo), 0, 1) * wt[i]
    m = np.clip((mod[i] - lo) / (hi - lo), 0, 1)
    pair = np.hstack([o, np.ones((h, 2)), m])
    tiles.append(np.pad(pair, 3, constant_values=0.3))
tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[j:j + 6]) for j in range(0, 18, 6)])
Image.fromarray((grid * 255).astype(np.uint8)).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(SP + '/lightfit_check.png')
print('saved')
