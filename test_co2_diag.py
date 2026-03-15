import json
import urllib.request


def main():
    stars = [0, 5, 10, 20, 50, 100]
    for sid in stars:
        try:
            response = urllib.request.urlopen(f"http://127.0.0.1:5000/api/star/{sid}?t=13.8")
            data = json.loads(response.read().decode("utf-8"))
            for pi, planet in enumerate(data.get("planets", [])):
                atmo = planet.get("atmosphere")
                if not atmo or planet["type"] not in ("rocky", "hot_rocky"):
                    continue
                comp = atmo.get("composition", {})
                co2 = comp.get("CO2", {})
                n2 = comp.get("N2", {})
                psf = atmo.get("surface_pressure_atm", 0) or 0
                tsurf = atmo.get("surface_temp_K", 0) or 0
                co2_pct = co2.get("pct", 0)
                n2_pct = n2.get("pct", 0)
                co2_pp = co2.get("partial_P_atm", 0)
                hab = planet.get("is_habitable", False)
                print(
                    "S{:3d} P{} {:14s} P={:8.2f}atm T={:6.0f}K CO2={:6.2f}%({:.4f}atm) N2={:5.1f}% hab={}".format(
                        sid, pi, planet["type"], psf, tsurf, co2_pct, co2_pp, n2_pct, hab
                    )
                )
        except Exception as exc:
            print(f"Star {sid} error: {exc}")


if __name__ == '__main__':
    main()
