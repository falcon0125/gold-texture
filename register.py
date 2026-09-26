import cv2, numpy as np, glob, json, os, sys, time

SRC = 'P:/test/Gold-texture/image_stack'
OUT = 'P:/test/Gold-texture/registered'
REF = sys.argv[1] if len(sys.argv) > 1 else '7366'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time() - t0), *a, flush=True)

files = sorted(glob.glob(SRC + '/*.jpg'))
ids = [os.path.basename(f)[-8:-4] for f in files]
full = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
H, W = full[0].shape[:2]

def feat(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (np.percentile(m, 99.5) + 1e-6)

half = [cv2.resize(im, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for im in full]
r = ids.index(REF)
tmpl = feat(half[r])
S = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], np.float64)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)

Hs = {}
for i, k in enumerate(ids):
    if i == r:
        Hs[k] = np.eye(3)
        continue
    M = np.eye(3, dtype=np.float32)
    cc, M = cv2.findTransformECC(tmpl, feat(half[i]), M, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    Hs[k] = S @ M.astype(np.float64) @ np.linalg.inv(S)
    log(k, 'cc %.4f' % cc, 'shift %+.1f %+.1f px' % (Hs[k][0, 2], Hs[k][1, 2]))

corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], np.float64).T
margin = 0
for k, Hm in Hs.items():
    p = Hm @ corners
    p = p[:2] / p[2]
    margin = max(margin, float(np.abs(p - corners[:2]).max()))
m = int(np.ceil(margin)) + 2
log('crop margin %d px' % m)

reg = {}
for i, k in enumerate(ids):
    al = cv2.warpPerspective(full[i], Hs[k], (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    al = al[m:H - m, m:W - m]
    reg[k] = al
    cv2.imwrite(os.path.join(OUT, os.path.basename(files[i])[:-4] + '.tif'), al)
log('wrote %d TIFFs %dx%d to %s' % (len(ids), W - 2 * m, H - 2 * m, OUT))

ref_f = feat(reg[REF])
hh, ww = ref_f.shape
win = cv2.createHanningWindow((512, 512), cv2.CV_32F)
report = {}
for k in ids:
    if k == REF:
        continue
    f = feat(reg[k])
    res = []
    for gy in range(3):
        for gx in range(3):
            y0 = int((hh - 512) * gy / 2); x0 = int((ww - 512) * gx / 2)
            (dx, dy), resp = cv2.phaseCorrelate(ref_f[y0:y0 + 512, x0:x0 + 512], f[y0:y0 + 512, x0:x0 + 512], win)
            res.append(float(np.hypot(dx, dy)))
    report[k] = res
    log(k, 'residual px (3x3 grid) max %.2f median %.2f' % (max(res), float(np.median(res))))

json.dump({'reference': REF, 'crop_margin': m, 'size': [W - 2 * m, H - 2 * m],
           'homographies_full_res': {k: Hs[k].tolist() for k in ids},
           'residual_px_grid3x3': report}, open(os.path.join(OUT, 'registration.json'), 'w'), indent=1)
log('done')
