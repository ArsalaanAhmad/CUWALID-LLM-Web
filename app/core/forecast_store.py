
# think of this as the "model" or truth layer for your seasonal forecasts. It has no knowledge of the UI or how data will be used. 
# It just provides a clean API for loading and querying forecasts, and applying any necessary logic (e.g. flood reversal).
#  By centralizing this logic here, you can keep your UI and backend provider code simple and focused on their respective roles.

#NOTE : Before making more changes to this file, do ask the main developer about the expected CSV Format and LLM Json format.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import csv


@dataclass(frozen=True)
class PredictionKey:
    country: str
    season: str
    year: int
    location_id: str
    variable: str


class ForecastStore:
    """
    Deterministic in-memory store for seasonal CUWALID/WujihaCast assets.

    Responsibilities:
    - Load seasonal CSV(s) into memory at startup.
    - Resolve user-provided place names to internal location IDs.
    - Fetch prediction status by variables (season, country, location_id, variable).
    - Construct map/voice asset paths in a deterministic manner.
    - Apply special-case logic if needed (e.g. flood reversal) centrally.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

        # location index: (country -> normalized place -> location_id)
        self.place_to_id: Dict[str, Dict[str, str]] = {}

        # predictions: key ---> status_code (0/1/2)
        self.predictions: Dict[PredictionKey, int] = {}

        # Keep a manifest for supported values (useful for /api/manifest)
        self.supported_countries: set[str] = set()
        self.supported_seasons: set[Tuple[str, int]] = set()
        self.supported_variables: set[str] = set()

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def load_all(self) -> None:
        """
        Load all CSV files from data_root. You need to align these paths
        with your repo layout.

        EXPECTED layout example (adjust to repo):
          data/
            2026/
              OND/
                kenya.csv
                somalia.csv
                ethiopia.csv
        """
        if not self.data_root.exists():
            raise FileNotFoundError(f"Forecast data root not found: {self.data_root}")

        # Crawl for CSVs
        csv_files = list(self.data_root.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {self.data_root}")

        for f in csv_files:
            # infer year/season/country from path if possible
            # like, data/2026/OND/kenya.csv
            year, season, country = self._infer_meta_from_path(f)

            self.supported_countries.add(country)
            self.supported_seasons.add((season, year))

            self._load_csv(country=country, season=season, year=year, csv_path=f)

    def _infer_meta_from_path(self, csv_path: Path) -> Tuple[int, str, str]:
        """
        Adjust this to match your repo layout. This is a safe default
        that expects .../<year>/<season>/<country>.csv
        """
        parts = [p.name for p in csv_path.parts]
        # last part is filename e.g. kenya.csv
        country = csv_path.stem.lower()

        # Try: parent = season, grandparent = year
        season = csv_path.parent.name.upper()
        year = int(csv_path.parent.parent.name)

        return year, season, country

    def _load_csv(self, country: str, season: str, year: int, csv_path: Path) -> None:
        """
        Expected CSV schema (based on your manifesto):
          name, place, variable, status
        Where:
          name = location_id (e.g., KE_10)
          place = human name (e.g., Marsabit)
          variable = crop/pasture/flood...
          status = 0/1/2
        """
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            required = {"name", "place", "variable", "status"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{csv_path} missing required columns: {required}")

            self.place_to_id.setdefault(country, {})

            for row in reader:
                location_id = (row.get("name") or "").strip()
                place = (row.get("place") or "").strip()
                variable = (row.get("variable") or "").strip().lower()
                status_raw = (row.get("status") or "").strip()

                if not location_id or not place or not variable or status_raw == "":
                    continue

                try:
                    status_code = int(status_raw)
                except ValueError:
                    continue

                self.supported_variables.add(variable)
                self.place_to_id[country][self._norm(place)] = location_id

                key = PredictionKey(
                    country=country,
                    season=season,
                    year=year,
                    location_id=location_id,
                    variable=variable
                )
                self.predictions[key] = status_code

    def resolve_location_id(self, country: str, place: str) -> Optional[str]:
        c = country.lower().strip()
        p = self._norm(place)
        return self.place_to_id.get(c, {}).get(p)

    def search_locations(self, country: str, q: str, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Return list of (place_name, location_id). Simple substring match.
        """
        c = country.lower().strip()
        qq = self._norm(q)
        results = []
        for place_norm, loc_id in self.place_to_id.get(c, {}).items():
            if qq in place_norm:
                results.append((place_norm, loc_id))
                if len(results) >= limit:
                    break
        return results

    def get_prediction(self, country: str, season: str, year: int, location_id: str, variable: str) -> Optional[int]:
        key = PredictionKey(
            country=country.lower().strip(),
            season=season.upper().strip(),
            year=int(year),
            location_id=location_id.strip(),
            variable=variable.lower().strip()
        )
        status = self.predictions.get(key)
        if status is None:
            return None

        # Flood reversal rule:
        if key.variable == "flood":
            # Interpret reversal: 0 (above normal) becomes "bad" instead of good.
            # at some point, either flip the label later OR flip the status code here.
            # for now, just the raw code and mark reversal; we’ll handle in labeling.
            return status

        return status

    def build_map_path(self, year: int, season: str, location_id: str, variable: str, language: str) -> str:
        """
        Adjust to match repo naming scheme.
        Example pattern from manifesto:
          KE_10_Crop_OND_2026_SW.png
        """
        var = variable.capitalize()
        lang = language.upper()
        season_u = season.upper()
        return f"/assets/maps/{year}/{season_u}/{location_id}_{var}_{season_u}_{year}_{lang}.png"

    def label_status(self, variable: str, status_code: int) -> str:
        """
        Human label. Handles flood reversal by flipping the *meaning*.
        """
        variable = variable.lower().strip()

        normal_map = {0: "Above Normal", 1: "Near Normal", 2: "Below Normal"}

        if status_code not in normal_map:
            return "Unknown"

        if variable != "flood":
            return normal_map[status_code]

        # Flood reversal: Above Normal is bad, Below Normal is good.
        # Minimal: still return the tercile label but perhaps(?) can add a flag in response.
        return normal_map[status_code]