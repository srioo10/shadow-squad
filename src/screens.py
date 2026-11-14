"""Start the Streamlit app and save cropped screenshots of its panels for each case. Usage: python src/screens.py <cases.json>"""
import json, subprocess, sys, time
from urllib.parse import quote
from playwright.sync_api import sync_playwright
cases = json.load(open(sys.argv[1]))
srv = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app/shadow_squad.py', '--server.headless', 'true', '--server.port', '8765', '--browser.gatherUsageStats', 'false'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def shot(pg, url, sel, path, wait):
    pg.goto('http://localhost:8765/' + url); pg.wait_for_selector(wait, timeout=120000); time.sleep(2.5); pg.locator(sel).first.screenshot(path=path); print(path)
try:
    time.sleep(8)
    with sync_playwright() as p:
        br = p.chromium.launch(); pg = br.new_page(viewport={'width': 1180, 'height': 2400}, device_scale_factor=2)
        for c in cases:
            base = f"?season={c['season']}&club={quote(c['club'])}&starter={c['starter']}"; t = f"outputs/screens/{c['tag']}"
            shot(pg, base + '&view=Squad', '[data-testid="stMainBlockContainer"]', t + '_squad.png', '.st-key-pitch svg')
            shot(pg, base + '&view=Shadow%20list', '.st-key-cards', t + '_cards.png', '.st-key-cards .card')
            if c['kind'] != 'miss':
                shot(pg, base + '&view=Compare' + (f"&cand={c['cand']}" if c['cand'] else ''), '[data-testid="stMainBlockContainer"]', t + '_compare.png', '.st-key-radar img')
            if c['kind'] == 'live': shot(pg, base + '&view=Budget%20rebuild', '[data-testid="stMainBlockContainer"]', t + '_rebuild.png', '[data-testid="stDataFrame"]')
        br.close()
finally: srv.terminate()
