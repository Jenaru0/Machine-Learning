# Roadmap/stats.py
from datetime import datetime

with open("stats_output.txt", "w") as f:
    f.write(f"📊 Reporte generado el {datetime.now()}\n")
    f.write("✔️ Sistema funcionando correctamente.\n")
