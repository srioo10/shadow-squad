"""Start the Streamlit app, open each case-study URL in a headless browser and save a screenshot. Usage: python src/screens.py <cases.json>"""
import json, subprocess, sys, time
from playwright.sync_api import sync_playwright
cases = json.load(open(sys.argv[1]))
srv = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app/shadow_squad.py', '--server.headless', 'true', '--server.port', '8765', '--browser.gatherUsageStats', 'false'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(8)
    with sync_playwright() as p:
        br = p.chromium.launch(); pg = br.new_page(viewport={'width': 1040, 'height': 1900}, device_scale_factor=2)
        for c in cases:
            pg.goto('http://localhost:8765/' + c['url']); pg.wait_for_selector('text=Why this rank', timeout=90000); pg.wait_for_function("document.querySelectorAll('img').length >= 2", timeout=90000); time.sleep(3)
            # two readable crops instead of the whole page: the ranked list, and the radar + SHAP + value panels
            pg.locator('[data-testid="stDataFrame"]').first.screenshot(path=f"outputs/screens/{c['tag']}_list.png"); print(f"outputs/screens/{c['tag']}_list.png")
            pg.locator('[data-testid="stHorizontalBlock"]').last.screenshot(path=f"outputs/screens/{c['tag']}_panels.png"); print(f"outputs/screens/{c['tag']}_panels.png")
        br.close()
finally: srv.terminate()
