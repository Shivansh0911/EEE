"""
Small toolkit for the demo videos: narration (Windows SAPI voice), title cards,
terminal replays of real recorded runs, image holds, trajectory animations,
narrated browser sessions (Playwright), and ffmpeg assembly.

Every terminal scene replays a recording made by demo/capture.py; every browser
scene drives the real web/index.html. Nothing shown is typed in by hand.
"""

import json
import math
import os
import subprocess
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1280, 720, 24
CAP_H = 104                                   # caption band height
VOICE = "Microsoft Zira Desktop"

FONT_DIR = "C:/Windows/Fonts"
F = lambda name, size: ImageFont.truetype(os.path.join(FONT_DIR, name), size)
MONO, MONO_B = "consola.ttf", "consolab.ttf"
SANS, SANS_B = "segoeui.ttf", "segoeuib.ttf"

BG = (18, 20, 26)
INK = (232, 236, 243)
INK2 = (167, 176, 192)
ACC = (90, 162, 255)
WARN = (233, 179, 74)
GOOD = (78, 201, 160)
BAD = (239, 112, 112)


# --- narration -----------------------------------------------------------------

def tts(text, path, rate=0):
    """Speak `text` to a WAV with the Windows voice; return its duration in s."""
    if not os.path.exists(path):
        esc = text.replace("'", "''")
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SelectVoice('{VOICE}'); $s.Rate={rate}; "
              f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{esc}'); $s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return a, sr


def write_wav(path, a, sr):
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(a, -32767, 32767).astype(np.int16).tobytes())


def mix(placements, total_s, path):
    """Place (wav_path, offset_s) on a silent track of total_s seconds."""
    sr = 22050
    track = np.zeros(int(total_s * sr) + sr, dtype=np.float32)
    for p, off in placements:
        a, r = read_wav(p)
        assert r == sr, r
        i = int(off * sr)
        track[i:i + len(a)] += a[: max(0, len(track) - i)]
    write_wav(path, track[: int(total_s * sr)], sr)


# --- drawing helpers -------------------------------------------------------------

def wrap(text, font, width, draw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def caption(img, text):
    """The narration as a caption band at the bottom of a frame."""
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - CAP_H, W, H], fill=(10, 11, 15))
    d.line([0, H - CAP_H, W, H - CAP_H], fill=(44, 49, 61), width=1)
    for size, lh, most in ((21, 28, 3), (18, 23, 4), (16, 20, 5)):   # shrink to fit
        f = F(SANS, size)
        lines = wrap(text, f, W - 80, d)
        if len(lines) <= most:
            break
    y = H - CAP_H + (CAP_H - lh * len(lines)) // 2
    for ln in lines:
        d.text((40, y), ln, font=f, fill=INK)
        y += lh
    return img


def card(title, subtitle, text, kicker=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8, H - CAP_H], fill=ACC)
    y = 170
    if kicker:
        d.text((90, y), kicker.upper(), font=F(MONO, 18), fill=ACC)
        y += 44
    for ln in wrap(title, F(SANS_B, 46), W - 180, d):
        d.text((90, y), ln, font=F(SANS_B, 46), fill=INK)
        y += 60
    y += 14
    for ln in wrap(subtitle, F(SANS, 24), W - 180, d):
        d.text((90, y), ln, font=F(SANS, 24), fill=INK2)
        y += 34
    return caption(img, text)


# --- ffmpeg ----------------------------------------------------------------------

def encode_frames(frames, wav, dur, out):
    """frames: iterator of PIL images at FPS; audio padded/cut to dur."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav, "-af", "apad",
           "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-ar", "44100", "-ac", "1",
           "-b:a", "128k", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for im in frames:
        p.stdin.write(im.convert("RGB").tobytes())
        n += 1
        if n >= int(dur * FPS) + 1:
            break
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, out


def still(img, wav, dur, out):
    encode_frames((img for _ in range(int(dur * FPS) + 2)), wav, dur, out)


def concat(parts, out):
    lst = out + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", out], check=True)
    os.remove(lst)


# --- scenes ----------------------------------------------------------------------

class Scenes:
    """Collects scenes for one video and renders them into one MP4."""

    def __init__(self, workdir, name):
        self.dir = os.path.join(workdir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.name, self.parts, self.n = name, [], 0

    def _wav(self, text):
        self.n += 1
        p = os.path.join(self.dir, f"n{self.n:02d}.wav")
        return p, tts(text, p)

    def _out(self):
        return os.path.join(self.dir, f"s{self.n:02d}.mp4")

    def card(self, title, subtitle, text, kicker=""):
        wav, d = self._wav(text)
        out = self._out()
        still(card(title, subtitle, text, kicker), wav, d + 0.8, out)
        self.parts.append(out)

    def image(self, path, text, title=""):
        wav, d = self._wav(text)
        img = Image.new("RGB", (W, H), BG)
        top = 50 if title else 12
        if title:
            ImageDraw.Draw(img).text((40, 12), title, font=F(SANS_B, 24), fill=INK)
        src = Image.open(path).convert("RGB")
        box = (W - 40, H - CAP_H - top - 10)
        src.thumbnail(box)
        img.paste(src, ((W - src.width) // 2, top + (box[1] - src.height) // 2))
        out = self._out()
        still(caption(img, text), wav, d + 0.8, out)
        self.parts.append(out)

    def terminal(self, record, text, title="", show_from=None, max_speed=None):
        """Replay a recorded run: the command is typed, then output lines appear
        at their real times, sped up to fit the narration (labelled)."""
        with open(record, encoding="utf-8") as f:
            rec = json.load(f)
        wav, d = self._wav(text)
        lines = rec["lines"]
        # show the scratch clone's absolute path as "." (it is a clone of the project)
        roots = {rec["cwd"], rec["cwd"].replace("/", "\\"), rec["cwd"].replace("\\", "/")}
        def tidy(s):
            for r in roots:
                s = s.replace(r + "\\", ".\\").replace(r + "/", "./").replace(r, ".")
            return s
        lines = [[t, tidy(l)] for t, l in lines]
        if show_from:
            lines = [l for l in lines if l[1].strip() and not any(s in l[1] for s in show_from)]
        # Shorten long silent waits (a solver computing) to MAX_GAP so the screen
        # is not empty for most of the narration; the badge says so.
        MAX_GAP, prev, tc, shortened = 2.0, 0.0, 0.0, False
        for l in lines:
            gap = l[0] - prev
            prev = l[0]
            if gap > MAX_GAP:
                gap, shortened = MAX_GAP, True
            tc += gap
            l[0] = tc
        t_end = max([l[0] for l in lines] + [0.5])
        type_s, hold = 1.6, 2.0
        speed = max(1.0, t_end / max(1.0, d - type_s - 1.0))
        if max_speed:
            speed = min(speed, max_speed)
        dur = max(d + 0.8, type_s + t_end / speed + hold)
        fm, fb = F(MONO, 15), F(MONO_B, 15)
        lh, x0, y0 = 19, 24, 52
        rows = (H - CAP_H - y0 - 10) // lh
        cmd = rec["command"].replace("python -m ", "python -m ")
        badge = f"real run, {rec['seconds']:.0f} s"
        if shortened:
            badge += ", waiting time shortened"
        if speed > 1.5:
            badge += f", replayed {speed:.0f}x faster"
        base = Image.new("RGB", (W, H), (12, 13, 17))
        bd = ImageDraw.Draw(base)
        bd.rectangle([0, 0, W, 36], fill=(30, 33, 42))
        for i, c in enumerate([(239, 112, 112), (233, 179, 74), (78, 201, 160)]):
            bd.ellipse([16 + i * 22, 12, 28 + i * 22, 24], fill=c)
        bd.text((100, 8), title or "terminal", font=F(SANS, 16), fill=INK2)
        tw = bd.textlength(badge, font=F(MONO, 14))
        bd.rounded_rectangle([W - tw - 40, 7, W - 14, 29], 6, fill=(30, 58, 95))
        bd.text((W - tw - 27, 10), badge, font=F(MONO, 14), fill=(168, 201, 245))
        caption(base, text)

        def frames():
            nf = int(dur * FPS) + 2
            for k in range(nf):
                t = k / FPS
                img = base.copy()
                dr = ImageDraw.Draw(img)
                typed = cmd[: int(len(cmd) * min(1.0, t / type_s))]
                buf = [("$ " + typed, "cmd")]
                if t >= type_s:
                    rt = (t - type_s) * speed
                    buf += [(l[1], "out") for l in lines if l[0] <= rt]
                vis = buf[-rows:]
                y = y0
                for s, kind in vis:
                    s = s.replace("\t", "    ")[:132]
                    col = GOOD if kind == "cmd" else INK
                    if kind == "out":
                        u = s.upper()
                        if "PASS" in u or " passed" in s:
                            col = GOOD
                        elif "FAIL" in u or "ERROR" in u or "REFUSED" in u:
                            col = BAD
                        elif s.strip().startswith(("PIERCE:", "design:", "deck written")):
                            col = (168, 201, 245)
                    dr.text((x0, y), s, font=fb if kind == "cmd" else fm, fill=col)
                    y += lh
                if (k // 12) % 2 == 0:
                    dr.rectangle([x0, y + 2, x0 + 9, y + 16], fill=INK2)
                yield img

        out = self._out()
        encode_frames(frames(), wav, dur, out)
        self.parts.append(out)

    def frames(self, gen_factory, text, min_dur=0):
        """Custom animation: gen_factory(dur) yields PIL frames for dur seconds."""
        wav, d = self._wav(text)
        dur = max(d + 0.8, min_dur)
        out = self._out()
        encode_frames((caption(im, text) for im in gen_factory(dur)), wav, dur, out)
        self.parts.append(out)

    def browser(self, url, steps, setup=None):
        """
        Narrated browser session. steps: list of (narration, action) where action
        is a callable(page) run at the start of that narration. The step lasts as
        long as its narration (or the action, if longer). Recorded with Playwright
        in Edge; the narration is placed at each step's real start time.
        """
        from playwright.sync_api import sync_playwright
        wavs = [self._wav(t) for t, _ in steps]
        vid_dir = os.path.join(self.dir, f"web{self.n:02d}")
        os.makedirs(vid_dir, exist_ok=True)
        offsets = []
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                record_video_dir=vid_dir, record_video_size={"width": W, "height": H})
            page = ctx.new_page()
            t0 = time.time()
            page.goto(url)
            page.wait_for_timeout(600)
            page.add_style_tag(content=CAPTION_CSS)
            if setup:
                setup(page)
            for (text, action), (wavp, d) in zip(steps, wavs):
                start = time.time() - t0
                offsets.append((wavp, start))
                page.evaluate(SET_CAPTION, text)
                ta = time.time()
                if action:
                    action(page)
                left = d + 0.7 - (time.time() - ta)
                if left > 0:
                    page.wait_for_timeout(int(left * 1000))
            total = time.time() - t0
            webm = page.video.path()
            ctx.close()
            b.close()
        audio = os.path.join(self.dir, f"web{self.n:02d}.wav")
        mix(offsets, total, audio)
        out = self._out()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", webm, "-i", audio,
                        "-vf", f"scale={W}:{H},fps={FPS}", "-af", "apad", "-t", f"{total:.3f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-ar", "44100", "-ac", "1", "-b:a", "128k", out], check=True)
        self.parts.append(out)

    def render(self, out):
        concat(self.parts, out)
        return out


CAPTION_CSS = """
#demo-cap { position: fixed; left: 0; right: 0; bottom: 0; z-index: 99999;
  background: rgba(10,11,15,0.94); color: #e8ecf3; border-top: 1px solid #2c313d;
  font: 21px/28px 'Segoe UI', sans-serif; padding: 14px 40px; min-height: 76px; }
.demo-hl { outline: 3px solid #e9b34a !important; outline-offset: 3px; border-radius: 6px; }
"""
SET_CAPTION = """(t) => { let c = document.getElementById('demo-cap');
  if (!c) { c = document.createElement('div'); c.id = 'demo-cap'; document.body.appendChild(c); }
  c.textContent = t; }"""


def highlight(page, sel):
    page.evaluate("""(s) => { document.querySelectorAll('.demo-hl').forEach(e => e.classList.remove('demo-hl'));
      const e = document.querySelector(s); if (e) { e.classList.add('demo-hl');
      e.scrollIntoView({behavior: 'smooth', block: 'center'}); } }""", sel)


def clear_highlight(page):
    page.evaluate("() => document.querySelectorAll('.demo-hl').forEach(e => e.classList.remove('demo-hl'))")


def slow_fill(page, sel, value, delay=90):
    page.fill(sel, "")
    page.type(sel, str(value), delay=delay)
