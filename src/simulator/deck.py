import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _tokenize_deck(text: str) -> List[str]:
    """
    Разбивает текст ECL/CMG deck на токены.
    Удаляем комментарии (-- ...), запятые, переводим в верхний регистр для ключевых слов.
    """
    cleaned = []
    for raw_line in text.splitlines():
        line = raw_line.split("--", 1)[0].strip()
        if not line:
            continue
        line = line.replace(",", " ")
        cleaned.append(line)
    merged = "\n".join(cleaned)
    # Разделяем по пробелам, оставляя "/" отдельным токеном
    tokens = re.findall(r"[^\s/]+|/", merged)
    return tokens


def _collect_rows(tokens: List[str], start_index: int) -> Tuple[List[List[float]], int]:
    """
    Собирает строки чисел до ближайшего '/'.
    Возвращает список строк (как float) и индекс, на котором остановились.
    """
    rows: List[List[float]] = []
    row: List[float] = []
    i = start_index
    while i < len(tokens):
        tok = tokens[i]
        if tok == "/":
            if row:
                rows.append(row)
                row = []
            else:
                break  # двойной '/' — конец секции
            i += 1
            break
        else:
            try:
                row.append(float(tok))
            except ValueError:
                # Нестандартный токен — заканчиваем секцию
                break
        if (i + 1) < len(tokens) and tokens[i + 1] == "/":
            rows.append(row)
            row = []
            i += 1  # пропускаем '/'
        i += 1
    return rows, i


SUPPORTED_KEYWORDS = {
    "PVTO", "PVCO", "PVCDO", "PVDO",
    "PVTW", "PVTG", "PVDG",
    "DENSITY",
    "SWOF", "SGOF", "SWFN", "SGFN", "SOF3", "SOF2",
    "PLYROCK", "PLYVISC", "PLYADS", "PLYVISCEL", "PLYMULCT", "PLYDENS",
    "SURFROCK", "SURFADS", "SURFMULT", "SURFSTOR",
    "THERM", "THPVT", "TEMPVD", "ROCKTAB",
    "TABDIMS",
}


def parse_eclipse_deck(path: str, keywords: Optional[List[str]] = None) -> Dict[str, List[List[float]]]:
    """
    Простейший парсер deck-файлов (Eclipse/CMG) для ключевых секций PVTO, PVTW, PVDG, SWOF, SGOF.
    Возвращает словарь {keyword: [row1, row2, ...]}.
    """
    deck_path = Path(path)
    if not deck_path.exists():
        raise FileNotFoundError(f"ECL/CMG deck '{path}' не найден")

    tokens = _tokenize_deck(deck_path.read_text(encoding="utf-8"))
    keyword_set = {kw.upper() for kw in (keywords or SUPPORTED_KEYWORDS)}

    i = 0
    result: Dict[str, List[List[float]]] = {kw: [] for kw in keyword_set}

    while i < len(tokens):
        tok = tokens[i].upper()
        if tok in keyword_set:
            rows, new_i = _collect_rows(tokens, i + 1)
            if rows:
                result[tok].extend(rows)
            i = new_i
        else:
            i += 1

    return result


def _sort_and_fill(table: Dict[str, List[float]], fill_defaults: Dict[str, float]) -> None:
    order = sorted(range(len(table["pressure_MPa"])), key=lambda idx: table["pressure_MPa"][idx])
    for key, series in table.items():
        table[key] = [series[idx] for idx in order]
    for key, default in fill_defaults.items():
        values = table.get(key)
        if values is None:
            continue
        last = default
        filled = []
        for v in values:
            if v is None:
                filled.append(last)
            else:
                filled.append(v)
                last = v
        table[key] = filled


def _parse_pvto(rows: List[List[float]]) -> Dict[str, Dict[str, List[float]]]:
    tables: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        rs = float(row[0])
        pressure_mpa = float(row[1]) / 1e3  # предполагаем вход в кПа
        bo = float(row[2])
        mu = float(row[3]) if len(row) >= 4 else None
        rv = float(row[4]) if len(row) >= 5 else None
        entry = tables.setdefault(rs, {
            "pressure_MPa": [],
            "Bo": [],
            "mu_o_cP": [],
            "Rv_m3m3": [],
        })
        entry["pressure_MPa"].append(pressure_mpa)
        entry["Bo"].append(bo)
        entry["mu_o_cP"].append(mu)
        entry["Rv_m3m3"].append(rv if rv is not None else 0.0)

    for rs_value, table in tables.items():
        _sort_and_fill(table, {"mu_o_cP": table["mu_o_cP"][0] if table["mu_o_cP"][0] is not None else 1.0})
    return tables


def _parse_simple_pv_table(rows: List[List[float]], columns: List[str], pressure_index: int = 0, unit_scale: float = 1e-3) -> Dict[str, List[float]]:
    if not rows:
        return {}
    data: Dict[str, List[float]] = {col: [] for col in ["pressure_MPa", *columns]}
    for row in rows:
        if len(row) <= pressure_index:
            continue
        pressure = float(row[pressure_index]) * unit_scale
        data["pressure_MPa"].append(pressure)
        for idx, col in enumerate(columns, start=pressure_index + 1):
            value = float(row[idx]) if idx < len(row) else None
            data[col].append(value)
    _sort_and_fill(data, {col: data[col][0] if data[col][0] is not None else 0.0 for col in columns})
    return data


def _parse_polymer_tables(deck: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    polymer: Dict[str, Dict[str, List[float]]] = {}
    if deck.get("PLYROCK"):
        rows = deck["PLYROCK"]
        conc = [float(r[0]) for r in rows if r]
        mult = [float(r[1]) if len(r) > 1 else 1.0 for r in rows if r]
        polymer["rock"] = {"concentration": conc, "multiplier": mult}
    if deck.get("PLYVISC"):
        rows = deck["PLYVISC"]
        conc = [float(r[0]) for r in rows if r]
        mult = [float(r[1]) if len(r) > 1 else 1.0 for r in rows if r]
        polymer["viscosity"] = {"concentration": conc, "multiplier": mult}
    if deck.get("PLYADS"):
        rows = deck["PLYADS"]
        conc = [float(r[0]) for r in rows if r]
        ads = [float(r[1]) if len(r) > 1 else 0.0 for r in rows if r]
        polymer["adsorption"] = {"concentration": conc, "adsorption": ads}
    return polymer


def _parse_surfactant_tables(deck: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    surf: Dict[str, Dict[str, List[float]]] = {}
    if deck.get("SURFROCK"):
        rows = deck["SURFROCK"]
        conc = [float(r[0]) for r in rows if r]
        mult = [float(r[1]) if len(r) > 1 else 1.0 for r in rows if r]
        surf["rock"] = {"concentration": conc, "multiplier": mult}
    if deck.get("SURFADS"):
        rows = deck["SURFADS"]
        conc = [float(r[0]) for r in rows if r]
        ads = [float(r[1]) if len(r) > 1 else 0.0 for r in rows if r]
        surf["adsorption"] = {"concentration": conc, "adsorption": ads}
    return surf


def _parse_thermal_tables(deck: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
    thermal: Dict[str, List[List[float]]] = {}
    for key in ("THERM", "THPVT", "TEMPVD"):
        if deck.get(key):
            thermal[key.lower()] = deck[key]
    return thermal


def load_pvt_from_deck(path: str) -> Dict[str, List[float]]:
    """
    Преобразует данные из deck-файла в словарь, совместимый с PVTTable.
    Поддерживает минимальный набор: PVTO (Rs, P, Bo, mu_o), PVTW (P, Bw, mu_w), PVDG (P, Bg, mu_g).
    Для PVTO выбираем строки с минимальным значением Rs (обычно 0). Если таких несколько, берём данные по возрастанию давления.
    """
    deck = parse_eclipse_deck(path)

    extended: Dict[str, Dict[str, Any]] = {
        "oil": {},
        "water": {},
        "gas": {},
        "polymer": {},
        "surfactant": {},
        "thermal": {},
        "meta": {},
    }
    data: Dict[str, List[float]] = {}

    # Oil (PVTO, PVDO, PVCO)
    oil_tables = _parse_pvto(deck.get("PVTO") or [])
    if oil_tables:
        rs_values = sorted(oil_tables.keys())
        extended["oil"]["rs_values"] = rs_values
        extended["oil"]["tables"] = oil_tables
        base_rs = rs_values[0]
        base_table = oil_tables[base_rs]
        data["pressure_MPa"] = list(base_table["pressure_MPa"])
        data["Bo"] = list(base_table["Bo"])
        data["mu_o_cP"] = list(base_table["mu_o_cP"])
        data["Rs_m3m3"] = [base_rs] * len(base_table["pressure_MPa"])
        data["Rv_m3m3"] = list(base_table["Rv_m3m3"])
    pvdo = _parse_simple_pv_table(deck.get("PVDO") or [], ["Bo", "mu_o_cP"])
    if pvdo:
        extended["oil"]["undersaturated"] = pvdo
        if "Bo" not in data:
            data["pressure_MPa"] = list(pvdo["pressure_MPa"])
            data["Bo"] = list(pvdo["Bo"])
            data["mu_o_cP"] = list(pvdo["mu_o_cP"])
    pvco = _parse_simple_pv_table(deck.get("PVCO") or deck.get("PVCDO") or [], ["co_1overMPa", "mu_o_cP"])
    if pvco:
        extended["oil"]["compressibility"] = pvco

    # Water
    pvtw = _parse_simple_pv_table(deck.get("PVTW") or [], ["Bw", "mu_w_cP", "cw_1overMPa", "ct_1overMPa"])
    if pvtw:
        extended["water"]["table"] = pvtw
        data.setdefault("pressure_MPa", list(pvtw["pressure_MPa"]))
        data["Bw"] = list(pvtw["Bw"])
        data["mu_w_cP"] = list(pvtw["mu_w_cP"])

    # Gas
    pvdg = _parse_simple_pv_table(deck.get("PVDG") or [], ["Bg", "mu_g_cP"])
    if pvdg:
        extended["gas"]["table"] = pvdg
        data.setdefault("pressure_MPa", list(pvdg["pressure_MPa"]))
        data["Bg"] = list(pvdg["Bg"])
        data["mu_g_cP"] = list(pvdg["mu_g_cP"])
    pvtg = _parse_simple_pv_table(deck.get("PVTG") or [], ["Bg", "mu_g_cP"], pressure_index=1)
    if pvtg:
        temperatures = [float(row[0]) for row in deck.get("PVTG") or [] if row]
        pvtg["temperature_C"] = temperatures[:len(pvtg["pressure_MPa"])]
        extended["gas"]["temperature"] = pvtg

    # Polymer, surfactant, thermal
    polymer_tables = _parse_polymer_tables(deck)
    if polymer_tables:
        extended["polymer"] = polymer_tables
    surf_tables = _parse_surfactant_tables(deck)
    if surf_tables:
        extended["surfactant"] = surf_tables
    thermal_tables = _parse_thermal_tables(deck)
    if thermal_tables:
        extended["thermal"] = thermal_tables

    # Densities metadata
    density_rows = deck.get("DENSITY") or []
    if density_rows:
        row = density_rows[0]
        if row:
            extended["meta"]["density"] = {
                "oil": float(row[0]) if len(row) > 0 else None,
                "water": float(row[1]) if len(row) > 1 else None,
                "gas": float(row[2]) if len(row) > 2 else None,
            }

    if "pressure_MPa" not in data:
        raise ValueError("В deck-файле не найдены секции PVTO/PVTW/PVDG с давлением")

    # Убедимся, что все массивы одной длины
    n = len(data["pressure_MPa"])
    for key, arr in list(data.items()):
        if key == "pressure_MPa":
            continue
        if len(arr) != n:
            raise ValueError(f"Размер колонки {key} ({len(arr)}) не совпадает с длиной давления ({n})")

    # Заполним отсутствующие столбцы нулями/единицами для совместимости
    defaults = {
        "Bo": 1.0,
        "Bw": 1.0,
        "Bg": 1.0,
        "mu_o_cP": 1.0,
        "mu_w_cP": 0.5,
        "mu_g_cP": 0.02,
        "Rs_m3m3": 0.0,
        "Rv_m3m3": 0.0,
    }
    for key, default in defaults.items():
        if key not in data:
            data[key] = [default] * n
    data["extended"] = extended

    return data


def load_relperm_tables_from_deck(path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Загружает таблицы относительных проницаемостей из секций SWOF и SGOF deck-файла.
    Возвращает словарь вида:
      {
         'swof': {'sw': [...], 'krw': [...], 'kro': [...], 'pcow': [...]},
         'sgof': {'sg': [...], 'krg': [...], 'kro': [...], 'pcog': [...]}
      }
    """
    deck = parse_eclipse_deck(path)
    result: Dict[str, Dict[str, List[float]]] = {}

    swof = deck.get("SWOF") or []
    if swof:
        sw = [row[0] for row in swof]
        krw = [row[1] for row in swof]
        kro = [row[2] for row in swof]
        pcow = [row[3] for row in swof] if len(swof[0]) >= 4 else [0.0] * len(swof)
        result["swof"] = {
            "sw": sw,
            "krw": krw,
            "kro": kro,
            "pcow": pcow,
        }

    sgof = deck.get("SGOF") or []
    if sgof:
        sg = [row[0] for row in sgof]
        krg = [row[1] for row in sgof]
        kro = [row[2] for row in sgof]
        pcog = [row[3] for row in sgof] if len(sgof[0]) >= 4 else [0.0] * len(sgof)
        result["sgof"] = {
            "sg": sg,
            "krg": krg,
            "kro": kro,
            "pcog": pcog,
        }

    return result

