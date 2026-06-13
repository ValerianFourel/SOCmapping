"""Build the LUCAS windowed training set for -large-sentinel2 (resolution-aware).
Per fine band-year: find+download its GeoTIFF from whichever account's Drive has
it (Drive API on EE creds), extract an 11x11 window around each LUCAS point,
save [N,11,11] float32, delete the GeoTIFF. Resumable (skips done). ~4 GB out."""
import json, os, time, urllib.request, numpy as np, rasterio, pandas as pd, ee
from rasterio.windows import Window
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

CFG = "/home/valerian/.config/earthengine"
DATA = os.environ['SOC_DATA_DIR']
OUT = os.path.join(DATA, 'windows', 'lucas'); os.makedirs(OUT, exist_ok=True)
TIFD = os.path.join(DATA, '_tif'); os.makedirs(TIFD, exist_ok=True)
W = 11
ACCTS = ['creds_main.json','creds_enc1.json','creds_e2.json','creds_e3.json','creds_e4.json']

# fine band manifest: (band, kind, drive-name-template, years-or-None)
SRC = ['SRC_Blue','SRC_Green','SRC_Red','SRC_NIR','SRC_SWIR1','SRC_SWIR2',
       'SRC_RCC','SRC_BCC','SRC_NBR2','SRC_BSI','SRC_ExposureCount']
TERRAIN = ['Elevation','Slope','Aspect','TWI','TPI_90','TPI_300','TPI_1000','TRI','Roughness']
JOBS = []
for b in SRC:
    for y in range(2002, 2024):
        JOBS.append((f"{b}_{y}", f"landsat_{b}_{y}.tif"))
for b in TERRAIN:
    JOBS.append((f"{b}_static", f"topo_{b}_static.tif"))

_cred_cache = {}
def cred(f):
    c = Credentials(None, refresh_token=json.load(open(f"{CFG}/{f}"))['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token', client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET, scopes=ee.oauth.SCOPES)
    c.refresh(Request()); return c

def find_download(name, dst):
    for f in ACCTS:
        c = cred(f); H = {"Authorization": f"Bearer {c.token}"}
        u = f"https://www.googleapis.com/drive/v3/files?q=name='{name}'+and+trashed=false&fields=files(id)&pageSize=1"
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60))
        except Exception:
            continue
        if j.get('files'):
            fid = j['files'][0]['id']
            with urllib.request.urlopen(urllib.request.Request(
                    f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media", headers=H), timeout=1200) as r, open(dst,'wb') as o:
                o.write(r.read())
            return f
    return None

df = pd.read_excel(os.path.join(DATA, 'LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx'))
lats = df['GPS_LAT'].to_numpy(float); lons = df['GPS_LONG'].to_numpy(float)
ok = np.isfinite(lats) & np.isfinite(lons)
N = len(df)
np.save(os.path.join(OUT, '_points.npy'),
        np.column_stack([lats, lons]).astype(np.float64))   # [N,2] lat,lon for alignment
print(f"LUCAS N={N} ({ok.sum()} finite)  jobs={len(JOBS)}", flush=True)

def extract(tif):
    out = np.full((N, W, W), np.nan, dtype=np.float32)
    with rasterio.open(tif) as ds:
        b = ds.bounds
        for i in np.where(ok)[0]:
            lo, la = lons[i], lats[i]
            if not (b.left <= lo <= b.right and b.bottom <= la <= b.top): continue
            r, c = ds.index(lo, la)
            try:
                win = ds.read(1, window=Window(int(c)-W//2, int(r)-W//2, W, W), boundless=True, fill_value=np.nan)
            except Exception:
                continue
            if win.shape == (W, W): out[i] = win
    return out

done = miss = 0
t0 = time.time()
for k, (key, name) in enumerate(JOBS):
    outp = os.path.join(OUT, f"{key}.npy")
    if os.path.exists(outp):
        done += 1; continue
    dst = os.path.join(TIFD, name)
    src = find_download(name, dst) if not os.path.exists(dst) else 'local'
    if not src:
        print(f"  [{k+1}/{len(JOBS)}] MISS {name} (not in any Drive)", flush=True); miss += 1; continue
    arr = extract(dst)
    np.save(outp, arr)
    cov = 100*np.isfinite(arr).any(axis=(1,2)).mean()
    os.remove(dst)
    done += 1
    el = time.time()-t0
    print(f"  [{k+1}/{len(JOBS)}] {key:24s} {arr.nbytes/1e6:.0f}MB cov={cov:.0f}% | done={done} miss={miss} | {el/60:.0f}min", flush=True)
print(f"DONE windows: {done}/{len(JOBS)}  missing={miss}  out={OUT}", flush=True)
