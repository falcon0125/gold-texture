import cv2, numpy as np, os, shutil, time

D = 'P:/test/Gold-texture'
SRC = D + '/rti_input'
OUT = D + '/rti_input_flat'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a):
    print('[%5.1fs]' % (time.time() - t0), *a, flush=True)

def s2l(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def l2s(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)

names = [l.split()[0] for l in open(SRC + '/lights.lp').read().split('\n')[1:] if l.strip()]
N = len(names)
gw, gh = 74, 90

def small_lum(img8):
    q = cv2.resize(img8, (img8.shape[1] // 4, img8.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    lin = s2l(q)
    lum = lin[..., 2] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 0] * 0.0722
    return cv2.resize(lum, (gw, gh), interpolation=cv2.INTER_AREA)

yy, xx = np.mgrid[0:gh, 0:gw]
x = (xx - (gw - 1) / 2) / (gw / 2); y = (yy - (gh - 1) / 2) / (gh / 2)
A = np.stack([np.ones(x.size), x.ravel(), y.ravel(), (x * x).ravel(), (y * y).ravel(), (x * y).ravel()], 1)

def fit(ratio):
    w = np.ones(ratio.size)
    for _ in range(3):
        c = np.linalg.lstsq(A * w[:, None], ratio.ravel() * w, rcond=None)[0]
        r = ratio.ravel() - A @ c
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = (np.abs(r) < 2.5 * s).astype(np.float64)
    return c

def strength(c):
    f = (A @ c).reshape(gh, gw)
    return float((f.max() - f.min()) / f.mean())

imgs = [cv2.imread(SRC + '/' + n) for n in names]
H, W = imgs[0].shape[:2]
smalls = np.stack([small_lum(im) for im in imgs])
alb = np.percentile(smalls, 30, axis=0) + 1e-4
log('loaded %d frames %dx%d' % (N, W, H))

Xf, Yf = np.meshgrid(((np.arange(W) + 0.5) / W * gw - gw / 2) / (gw / 2),
                     ((np.arange(H) + 0.5) / H * gh - gh / 2) / (gh / 2))
Xf = Xf.astype(np.float32); Yf = Yf.astype(np.float32)
after = []
for i, (n, im) in enumerate(zip(names, imgs)):
    c = fit(smalls[i] / alb).astype(np.float32)
    F = c[0] + c[1] * Xf + c[2] * Yf + c[3] * Xf * Xf + c[4] * Yf * Yf + c[5] * Xf * Yf
    gain = np.clip(c[0] / np.maximum(F, 1e-3), 0.25, 4.0)
    lin = s2l(im.astype(np.float32) / 255) * gain[..., None]
    clipped = float((lin.max(-1) > 1).mean() * 100)
    out = np.clip(l2s(np.clip(lin, 0, 1)) * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(OUT + '/' + n, out, [cv2.IMWRITE_JPEG_QUALITY, 98])
    after.append(small_lum(out))
    log('%s falloff strength %.3f  gain range %.2f-%.2f  newly clipped %.2f%%' % (n[-8:-4], strength(c), gain.min(), gain.max(), clipped))

after = np.stack(after)
alb2 = np.percentile(after, 30, axis=0) + 1e-4
for n, s in zip(names, after):
    log('%s residual strength after flattening %.3f' % (n[-8:-4], strength(fit(s / alb2))))
shutil.copy(SRC + '/lights.lp', OUT + '/lights.lp')
log('done')
