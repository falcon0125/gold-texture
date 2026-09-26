import cv2, json, os, shutil, sys

D = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else 'rti_rbf_flat')
divisor = int(sys.argv[2]) if len(sys.argv) > 2 else 2
quality = int(sys.argv[3]) if len(sys.argv) > 3 else 92
dst = src + {2: '_half', 4: '_quarter'}.get(divisor, '_div%d' % divisor)
os.makedirs(dst, exist_ok=True)
info = json.load(open(os.path.join(src, 'info.json')))
W, H = info['width'] // divisor, info['height'] // divisor
for i in range((info['nplanes'] + 2) // 3):
    name = 'plane_%d.%s' % (i, info['format'])
    im = cv2.imread(os.path.join(src, name))
    cv2.imwrite(os.path.join(dst, name), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(name, '%.1f MB' % (os.path.getsize(os.path.join(dst, name)) / 1e6))
info['width'], info['height'] = W, H
json.dump(info, open(os.path.join(dst, 'info.json'), 'w'))
if os.path.exists(os.path.join(src, 'materials.png')):
    shutil.copy(os.path.join(src, 'materials.png'), dst)
print('wrote', dst, '%dx%d' % (W, H))
