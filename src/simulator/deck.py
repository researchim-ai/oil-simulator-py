import re
from pathlib import Path
from typing import Dict, List, Tuple


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


def parse_eclipse_deck(path: str) -> Dict[str, List[List[float]]]:
    """
    Простейший парсер deck-файлов (Eclipse/CMG) для ключевых секций PVTO, PVTW, PVDG, SWOF, SGOF.
    Возвращает словарь {keyword: [row1, row2, ...]}.
    """
    deck_path = Path(path)
    if not deck_path.exists():
        raise FileNotFoundError(f"ECL/CMG deck '{path}' не найден")

    tokens = _tokenize_deck(deck_path.read_text(encoding="utf-8"))
    i = 0
    result: Dict[str, List[List[float]]] = {
        "PVTO": [],
        "PVTW": [],
        "PVDG": [],
        "SWOF": [],
        "SGOF": [],
    }

    keywords = set(result.keys())

    while i < len(tokens):
        tok = tokens[i].upper()
        if tok in keywords:
            rows, new_i = _collect_rows(tokens, i + 1)
            if rows:
                result[tok].extend(rows)
            i = new_i
        else:
            i += 1

    return result


def load_pvt_from_deck(path: str) -> Dict[str, List[float]]:
    """
    Преобразует данные из deck-файла в словарь, совместимый с PVTTable.
    Поддерживает минимальный набор: PVTO (Rs, P, Bo, mu_o), PVTW (P, Bw, mu_w), PVDG (P, Bg, mu_g).
    Для PVTO выбираем строки с минимальным значением Rs (обычно 0). Если таких несколько, берём данные по возрастанию давления.
    """
    deck = parse_eclipse_deck(path)
    data: Dict[str, List[float]] = {}

    # Обработка PVTO
    pvto_rows = deck.get("PVTO") or []
    if pvto_rows:
        pvto_rows_sorted = sorted(pvto_rows, key=lambda r: (r[0], r[1]))  # sort by Rs, then pressure
        # выбираем минимальный Rs
        if pvto_rows_sorted:
            min_rs = pvto_rows_sorted[0][0]
            filtered = [row for row in pvto_rows_sorted if abs(row[0] - min_rs) < 1e-6]
            filtered = sorted(filtered, key=lambda r: r[1])
            pressures = [row[1] / 1e3 for row in filtered]  # deck в psi? предположим kPa? Примем, что вход в кПа -> МПа. Если psi, нужно конвертировать.
            data["pressure_MPa"] = pressures
            data["Bo"] = [row[2] for row in filtered]
            if len(filtered[0]) >= 4:
                data["mu_o_cP"] = [row[3] for row in filtered]
            data["Rs_m3m3"] = [row[0] for row in filtered]
    # PVTW
    pvtw_rows = deck.get("PVTW") or []
    if pvtw_rows:
        pvtw_sorted = sorted(pvtw_rows, key=lambda r: r[0])
        data["pressure_MPa"] = [row[0] / 1e3 for row in pvtw_sorted]
        data["Bw"] = [row[1] for row in pvtw_sorted]
        if len(pvtw_sorted[0]) >= 3:
            data["mu_w_cP"] = [row[2] for row in pvtw_sorted]
    # PVDG
    pvdg_rows = deck.get("PVDG") or []
    if pvdg_rows:
        pvdg_sorted = sorted(pvdg_rows, key=lambda r: r[0])
        data["pressure_MPa"] = [row[0] / 1e3 for row in pvdg_sorted]
        data["Bg"] = [row[1] for row in pvdg_sorted]
        if len(pvdg_sorted[0]) >= 3:
            data["mu_g_cP"] = [row[2] for row in pvdg_sorted]

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

