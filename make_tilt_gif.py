import argparse, math, os, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageChops

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
ROOT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description='Render the tilt page swinging into a looping GIF or MP4.')
ap.add_argument('--url', default='http://127.0.0.1:8765/web/tilt.html')
ap.add_argument('--frames', type=int, default=None, help='frames per swing (default: 36 for gif, fps*period for mp4)')
ap.add_argument('--fps', type=float, default=30, help='mp4 frame rate')
ap.add_argument('--amp', type=float, default=25, help='swing amplitude in degrees')
ap.add_argument('--axis', choices=['tx', 'ty'], default='tx')
ap.add_argument('--period', type=float, default=4.0, help='seconds per full swing')
ap.add_argument('--cycles', type=int, default=1, help='number of swings in an mp4')
ap.add_argument('--width', type=int, default=None, help='output width (default: 600 gif, 1080 mp4)')
ap.add_argument('--window', default=None, help='browser window WxH (default 900x1000 gif, 1080x1200 mp4)')
ap.add_argument('--crf', type=int, default=18, help='mp4 quality, lower is better')
ap.add_argument('--extra', default='gold=1', help='extra query parameters, e.g. "gold=1&z=2.5"')
ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'tilt_swing.gif'))
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()

video = args.out.lower().endswith(('.mp4', '.webm'))
frames_n = args.frames or (round(args.fps * args.period) if video else 36)
width = args.width or (1080 if video else 600)
win = args.window or ('1080,1200' if video else '900,1000')
win = win.replace('x', ',')

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
tmp = tempfile.mkdtemp(prefix='tiltframes_')
t0 = time.time()

def render(k):
    angle = args.amp * math.sin(2 * math.pi * k / frames_n)
    path = os.path.join(tmp, 'f%04d.png' % k)
    prof = os.path.join(tmp, 'profile%d' % (k % args.workers))
    url = '%s?embed=1&%s=%.3f&%s' % (args.url, args.axis, angle, args.extra)
    for _ in range(3):
        subprocess.run([EDGE, '--headless=new', '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
                        '--window-size=' + win, '--virtual-time-budget=30000', '--hide-scrollbars',
                        '--user-data-dir=' + prof, '--screenshot=' + path, url],
                       capture_output=True, timeout=180)
        if os.path.exists(path):
            return path
    raise RuntimeError('frame %d failed' % k)

buckets = [list(range(frames_n))[w::args.workers] for w in range(args.workers)]
with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(lambda ks: [render(k) for k in ks], buckets))
print('rendered %d frames in %.0fs' % (frames_n, time.time() - t0), flush=True)

frames = [Image.open(os.path.join(tmp, 'f%04d.png' % k)).convert('RGB') for k in range(frames_n)]
bg = Image.new('RGB', frames[0].size, (18, 18, 18))
box = None
for f in frames:
    b = ImageChops.difference(f, bg).convert('L').point(lambda v: 255 if v > 12 else 0).getbbox()
    if b:
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
pad = 12
box = (max(0, box[0] - pad), max(0, box[1] - pad), min(frames[0].width, box[2] + pad), min(frames[0].height, box[3] + pad))
height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
if video:
    width -= width % 2
    height -= height % 2
out = [f.crop(box).resize((width, height), Image.LANCZOS) for f in frames]

if not video:
    pal = [im.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for im in out]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=round(args.period * 1000 / frames_n),
                loop=0, optimize=True, disposal=1)
else:
    seq = os.path.join(tmp, 'seq')
    os.makedirs(seq)
    n = 0
    for _ in range(args.cycles):
        for im in out:
            im.save(os.path.join(seq, 'c%05d.png' % n))
            n += 1
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fps = frames_n / args.period
    if args.out.lower().endswith('.webm'):
        codec = ['-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', str(args.crf + 14), '-row-mt', '1']
    else:
        codec = ['-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', '%.4f' % fps,
                    '-i', os.path.join(seq, 'c%05d.png')] + codec + [args.out], check=True)
print('saved %s  %dx%d  %d frames  %.1f MB' % (args.out, width, height, frames_n * (args.cycles if video else 1),
                                                os.path.getsize(args.out) / 1e6))
shutil.rmtree(tmp, ignore_errors=True)
