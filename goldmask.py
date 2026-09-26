import cv2, numpy as np, json, sys, time

D = 'P:/test/Gold-texture'
PREVIEW = sys.argv[1] if len(sys.argv) > 1 else D
t0 = time.time()
def log(*a):
    print('[%5.1fs]' % (time.time() - t0), *a, flush=True)

names = [l.split()[0] for l in open(D + '/rti_input/lights.lp').read().split('\n')[1:] if l.strip()]
info = json.load(open(D + '/rti_rbf/info.json'))
FW, FH = info['width'], info['height']
W, H = FW // 2, FH // 2
S8 = np.stack([cv2.resize(cv2.imread(D + '/rti_input/' + n), (W, H), interpolation=cv2.INTER_AREA) for n in names])
N = len(names)
clip = S8.max(-1) >= 250
S = S8.astype(np.float32)
B, G, R = S[..., 0], S[..., 1], S[..., 2]
L = S.mean(-1)
log('loaded %d frames %dx%d, clipped %.2f%%' % (N, W, H, clip.mean() * 100))

gw, gh = 74, 90
small = np.stack([cv2.resize(l, (gw, gh), interpolation=cv2.INTER_AREA) for l in L])
alb = np.percentile(small, 30, axis=0) + 1
yy, xx = np.mgrid[0:gh, 0:gw]
x = (xx - (gw - 1) / 2) / (gw / 2); y = (yy - (gh - 1) / 2) / (gh / 2)
A = np.stack([np.ones(x.size), x.ravel(), y.ravel(), (x * x).ravel(), (y * y).ravel(), (x * y).ravel()], 1)
Xg, Yg = np.meshgrid(((np.arange(W) + 0.5) / W * gw - gw / 2) / (gw / 2),
                     ((np.arange(H) + 0.5) / H * gh - gh / 2) / (gh / 2))
Xg = Xg.astype(np.float32); Yg = Yg.astype(np.float32)
F = np.empty((N, H, W), np.float32)
for i in range(N):
    c = np.linalg.lstsq(A, (small[i] / alb).ravel(), rcond=None)[0].astype(np.float32)
    F[i] = c[0] + c[1] * Xg + c[2] * Yg + c[3] * Xg * Xg + c[4] * Yg * Yg + c[5] * Xg * Yg
Ln = L / np.maximum(F, 0.2)
valid = (~clip) & (L > 15)
nv = valid.sum(0)

srt = np.sort(np.where(valid, Ln, np.inf), axis=0)
def vpct(q):
    k = np.clip(np.floor(q * (nv - 1)).astype(np.int64), 0, N - 1)
    return np.take_along_axis(srt, k[None], 0)[0]
p15, p75, p90 = vpct(0.15), vpct(0.75), vpct(0.90)
gain = (p90 - p15) / (p15 + 10)

vf = valid.astype(np.float32)
cnt = np.maximum(nv, 1).astype(np.float32)
chrom = (R - B) / (R + B + 1)
Lmu = (Ln * vf).sum(0) / cnt
Cmu = (chrom * vf).sum(0) / cnt
cov = ((Ln - Lmu) * (chrom - Cmu) * vf).sum(0) / cnt
var = (((Ln - Lmu) ** 2) * vf).sum(0) / cnt
slope = cov / (var + 1) * (p15 + 10)
hi = vf * (Ln >= p75)
warm_hi = (chrom * hi).sum(0) / np.maximum(hi.sum(0), 1)
log('stats done')

def n(a, lo, hi_):
    return np.clip((a - lo) / (hi_ - lo), 0, 1)
score = n(gain, np.percentile(gain, 60), np.percentile(gain, 95)) \
      * n(slope, np.percentile(slope, 35), np.percentile(slope, 80)) \
      * n(warm_hi, 0.08, 0.25)
score[nv < 8] = 0
score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 2)
top = np.percentile(score, 99.5)
soft = np.clip((score / top - 0.10) / (0.45 - 0.10), 0, 1)
soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
soft = cv2.GaussianBlur(soft, (0, 0), 1.2)
log('coverage: soft>0.5 %.1f%%, soft>0.1 %.1f%%' % ((soft > 0.5).mean() * 100, (soft > 0.1).mean() * 100))

cv2.imwrite(D + '/web/gold_mask.png', np.clip(soft * 255, 0, 255).astype(np.uint8))

sel = soft > 0.6
best = np.argmax(np.where(valid, Ln, -np.inf), axis=0)
bright = np.take_along_axis(S, best[None, ..., None].repeat(3, -1), 0)[0]
bgr = bright[sel].mean(0) / 255
lin = np.where(bgr <= 0.04045, bgr / 12.92, ((bgr + 0.055) / 1.055) ** 2.4)
rgb = (lin[::-1] / lin.max()).tolist()
json.dump({'color_linear_rgb': [round(v, 4) for v in rgb], 'mask_size': [W, H]}, open(D + '/web/gold.json', 'w'))
log('gold color (linear rgb, normalised):', np.round(rgb, 3))

med = np.median(S, 0)
ov = med * (1 - 0.6 * soft[..., None]) + np.array([0, 215, 255], np.float32) * 0.6 * soft[..., None]
cv2.imwrite(PREVIEW + '/goldmask_overlay.jpg', np.clip(np.hstack([med, ov]), 0, 255).astype(np.uint8)[::3, ::3], [cv2.IMWRITE_JPEG_QUALITY, 88])
mx = S.max(0)
rows = []
for (x0, y0) in [(500, 470), (940, 290), (110, 860)]:
    w, h = 380, 300
    c = lambda a: a[y0:y0 + h, x0:x0 + w]
    rows.append(np.hstack([c(mx), c(ov)]))
cv2.imwrite(PREVIEW + '/goldmask_crops.jpg', np.clip(np.vstack(rows), 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
log('done')
