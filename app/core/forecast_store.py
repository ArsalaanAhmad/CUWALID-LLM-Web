# Think of this as the "truth layer" for seasonal forecasts.
# It does not know anything about the UI, prompting, MCP, or how results are rendered.
# It only provides a deterministic API for:
# - loading forecasts from static CSV files
# - resolving place names into internal location IDs
# - retrieving forecast status values
# - constructing asset paths
# - applying domain-specific interpretation logic (e.g. flood reversal)
#
# By centralizing this logic here, the rest of the application can stay simple:
# - LLM / extractor layer = understands user language
# - FastAPI layer = orchestrates requests and responses
# - ForecastStore = source of truth
#
# NOTE:
# Before making major schema/path assumptions permanent, confirm with the main developer:
# - exact CSV format
# - exact asset naming / folder structure
# - exact LLM JSON contract

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
    Deterministic in-memory store for seasonal CUWALID / WujihaCast forecast assets.

    Responsibilities:
    - Load seasonal CSVs into memory on startup
    - Resolve user-facing place names to internal location IDs
    - Retrieve prediction status by:
        (country, season, year, location_id, variable)
    - Construct forecast asset paths deterministically
    - Apply domain-aware interpretation logic, e.g. flood reversal
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

        # country -> normalized_place_name -> location_id
        self.place_to_id: Dict[str, Dict[str, str]] = {}

        # country -> location_id -> original/human-readable place name
        self.id_to_place: Dict[str, Dict[str, str]] = {}

        # prediction lookup table
        # key -> status_code (0 / 1 / 2)
        self.predictions: Dict[PredictionKey, int] = {}

        # manifests / debug info
        self.supported_countries: set[str] = set()
        self.supported_seasons: set[Tuple[str, int]] = set()
        self.supported_variables: set[str] = set()

    @staticmethod
    def _norm(text: str) -> str:
        """
        Normalize place names for matching.
        """
        return " ".join(text.strip().lower().split())

    def load_all(self) -> None:
        """
        Load all CSV files under data_root.

        Expected default layout:
            data/
              2026/
                OND/
                  kenya.csv
                  somalia.csv
                  ethiopia.csv

        This can be adjusted if the real repo layout differs.
        """
        if not self.data_root.exists():
            raise FileNotFoundError(f"Forecast data root not found: {self.data_root}")

        csv_files = list(self.data_root.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {self.data_root}")

        for csv_file in csv_files:
            year, season, country = self._infer_meta_from_path(csv_file)

            self.supported_countries.add(country)
            self.supported_seasons.add((season, year))

            self._load_csv(country=country, season=season, year=year, csv_path=csv_file)

    def _infer_meta_from_path(self, csv_path: Path) -> Tuple[int, str, str]:
        """
        Infer:
            year, season, country

        from a path like:
            data/<year>/<season>/<country>.csv

        Example:
            data/2026/OND/kenya.csv
        """
        country = csv_path.stem.lower()
        season = csv_path.parent.name.upper()

        try:
            year = int(csv_path.parent.parent.name)
        except ValueError as e:
            raise ValueError(
                f"Could not infer year from path '{csv_path}'. "
                f"Expected structure like data/<year>/<season>/<country>.csv"
            ) from e

        return year, season, country

    def _load_csv(self, country: str, season: str, year: int, csv_path: Path) -> None:
        """
        Expected CSV schema (based on current documentation / manifesto):
            name, place, variable, status

        Where:
            name     = internal location ID (e.g. KE_10)
            place    = human-readable place name (e.g. Marsabit)
            variable = forecast variable (e.g. crop, pasture, flood)
            status   = forecast code (0 / 1 / 2)
        """
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)

            required = {"name", "place", "variable", "status"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{csv_path} missing required columns: {required}")

            self.place_to_id.setdefault(country, {})
            self.id_to_place.setdefault(country, {})

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

                normalized_place = self._norm(place)
                self.place_to_id[country][normalized_place] = location_id
                self.id_to_place[country][location_id] = place

                key = PredictionKey(
                    country=country,
                    season=season,
                    year=year,
                    location_id=location_id,
                    variable=variable,
                )
                self.predictions[key] = status_code

    def resolve_location_id(self, country: str, place: str) -> Optional[str]:
        """
        Resolve a user-provided place name to an internal location ID.
        """
        c = country.lower().strip()
        p = self._norm(place)
        return self.place_to_id.get(c, {}).get(p)

    def get_place_name(self, country: str, location_id: str) -> Optional[str]:
        """
        Resolve a location ID back to the original human-readable place name.
        """
        c = country.lower().strip()
        return self.id_to_place.get(c, {}).get(location_id)

    def search_locations(self, country: str, q: str, limit: int = 10) -> List[Tuple[str, str]]:
        """
        Return fuzzy-ish substring matches as:
            [(human_readable_place, location_id), ...]

        This is currently a simple substring match against normalized names.
        """
        c = country.lower().strip()
        qq = self._norm(q)
        results: List[Tuple[str, str]] = []

        for place_norm, loc_id in self.place_to_id.get(c, {}).items():
            if qq in place_norm:
                human_place = self.id_to_place.get(c, {}).get(loc_id, place_norm)
                results.append((human_place, loc_id))
                if len(results) >= limit:
                    break

        return results

    def get_prediction(
        self,
        country: str,
        season: str,
        year: int,
        location_id: str,
        variable: str
    ) -> Optional[int]:
        """
        Return raw forecast status code for the given lookup parameters.

        Output:
            0 / 1 / 2
        or:
            None if no matching forecast exists
        """
        key = PredictionKey(
            country=country.lower().strip(),
            season=season.upper().strip(),
            year=int(year),
            location_id=location_id.strip(),
            variable=variable.lower().strip()
        )

        return self.predictions.get(key)

    def build_map_path(
        self,
        year: int,
        season: str,
        location_id: str,
        variable: str,
        language: str
    ) -> str:
        """
        Construct map asset path.

        Current assumed naming pattern:
            /assets/maps/<year>/<season>/<location_id>_<Variable>_<SEASON>_<year>_<LANG>.png

        Example:
            /assets/maps/2026/OND/KE_10_Crop_OND_2026_SW.png

        IMPORTANT:
        This is based on current documentation assumptions and should be verified
        against the real WujihaCast/CUWALID asset structure.
        """
        var = variable.capitalize()
        lang = language.upper()
        season_u = season.upper()

        return f"/assets/maps/{year}/{season_u}/{location_id}_{var}_{season_u}_{year}_{lang}.png"

    def label_status(self, variable: str, status_code: int) -> str:
        """
        Return the raw tercile/category label associated with the forecast code.

        For most variables:
            0 -> Above Normal
            1 -> Near Normal
            2 -> Below Normal

        NOTE:
        This method intentionally returns the raw category label only.
        If you want user-facing meaning (e.g. flood reversal), use interpret_status().
        """
        normal_map = {
            0: "Above Normal",
            1: "Near Normal",
            2: "Below Normal"
        }
        return normal_map.get(status_code, "Unknown")

    def interpret_status(self, variable: str, status_code: int) -> str:
        """
        Return the user-facing interpretation of a forecast status.

        For most variables:
            0 -> Above Normal
            1 -> Near Normal
            2 -> Below Normal

        For flood:
            0 -> High Flood Hazard
            1 -> Near-Normal Flood Hazard
            2 -> Low Flood Hazard

        This is where flood reversal is handled semantically.
        """
        variable = variable.lower().strip()

        if variable == "flood":
            flood_map = {
                0: "High Flood Hazard",
                1: "Near-Normal Flood Hazard",
                2: "Low Flood Hazard"
            }
            return flood_map.get(status_code, "Unknown")

        normal_map = {
            0: "Above Normal",
            1: "Near Normal",
            2: "Below Normal"
        }
        return normal_map.get(status_code, "Unknown")