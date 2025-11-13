import torch
import numpy as np
from pathlib import Path

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - fallback if SciPy missing
    gaussian_filter = None

class Reservoir:
    """
    Класс для представления модели пласта.
    """
    def __init__(self, config, device=None):
        """
        Инициализация модели пласта.

        :param config: Словарь с параметрами пласта.
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Размеры пласта
        self.dimensions = config.get('dimensions', [10, 10, 1])
        self.nx, self.ny, self.nz = self.dimensions
        
        # Конфигурация сетки (поддержка corner-point / LGR через индивидуальные шаги)
        grid_cfg = config.get('grid', {}) or {}
        base_grid_size = config.get('grid_size', [10.0, 10.0, 10.0])
        self.is_uniform_grid = True
        self._init_spacing(grid_cfg, base_grid_size)

        # Пористость (может быть скаляром, путем к файлу или задана стохастически)
        porosity_value = config.get('porosity', 0.2)
        self.porosity = self._create_field(porosity_value, default=0.2, name='porosity')

        # Проницаемость
        perm_value = config.get('permeability', 100.0)  # мД
        md_to_m2 = 9.869233e-16
        perm_si = self._create_field(perm_value, default=100.0, name='permeability') * md_to_m2

        k_v_fraction = config.get('k_vertical_fraction', 0.1)
        self.permeability_x = perm_si.clone()
        self.permeability_y = perm_si.clone()
        self.permeability_z = perm_si * k_v_fraction

        # Стохастические поля (опционально)
        stochastic_cfg = config.get('stochastic') or {}
        if stochastic_cfg:
            self._apply_stochastic_fields(stochastic_cfg, md_to_m2, k_v_fraction)

        # Сжимаемость породы
        self.rock_compressibility = config.get('c_rock', 1e-5) / 1e6  # 1/МПа -> 1/Па

        # Геометрия
        self._build_geometric_properties()

        # Выводим информацию
        print("Создание модели пласта...")
        print(f"  Размеры грида: {self.nx}x{self.ny}x{self.nz} ячеек")
        print(f"  Пористость: mean={float(self.porosity.mean()):.3f}, std={float(self.porosity.std()):.3f}")
        print(f"  Горизонтальная проницаемость (мД): mean={float((self.permeability_x / md_to_m2).mean()):.1f}")
        print(f"  Вертикальная проницаемость/горизонтальная: {k_v_fraction}")
        print(f"  Тензоры размещены на: {self.device}")

    @property
    def permeability_tensors(self):
        """
        Возвращает тензоры проницаемости.
        
        Returns:
            Кортеж из трех тензоров проницаемости (k_x, k_y, k_z) в мД
        """
        return self.permeability_x, self.permeability_y, self.permeability_z
        
    def get_cell_indices(self, i, j, k):
        """
        Возвращает линейный индекс ячейки по трехмерным координатам.
        
        Args:
            i, j, k: Координаты ячейки
            
        Returns:
            Линейный индекс ячейки
        """
        return i + j * self.nx + k * self.nx * self.ny

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _init_spacing(self, grid_cfg, base_spacing):
        dx_default, dy_default, dz_default = base_spacing

        def _axis_spacing(axis_name, n, default_val):
            spacing = None
            if grid_cfg:
                nodes_key = f"{axis_name}_nodes"
                spacing_key = f"d{axis_name}"
                if nodes_key in grid_cfg:
                    nodes = np.asarray(grid_cfg[nodes_key], dtype=np.float64)
                    if nodes.shape[0] != n + 1:
                        raise ValueError(f"{nodes_key} length {nodes.shape[0]} != {n+1}")
                    spacing = np.diff(nodes)
                    self.is_uniform_grid = False
                elif spacing_key in grid_cfg:
                    spacing = np.asarray(grid_cfg[spacing_key], dtype=np.float64)
                    if spacing.shape[0] != n:
                        raise ValueError(f"{spacing_key} length {spacing.shape[0]} != {n}")
                    self.is_uniform_grid = False
            if spacing is None:
                spacing = np.full(n, float(default_val), dtype=np.float64)
            return torch.from_numpy(spacing).to(self.device)

        self.dx_vector = _axis_spacing('x', self.nx, dx_default)
        self.dy_vector = _axis_spacing('y', self.ny, dy_default)
        self.dz_vector = _axis_spacing('z', self.nz, dz_default)

        # Сохраненная средняя величина для обратной совместимости
        self.grid_size = torch.tensor([
            self.dx_vector.mean(),
            self.dy_vector.mean(),
            self.dz_vector.mean()
        ], device=self.device)

    def _create_field(self, value, default, name):
        """
        Создает тензор поля (пористость или проницаемость) из заданного значения / файла.
        """
        shape = self.dimensions
        if isinstance(value, (int, float)):
            return torch.full(shape, float(value), device=self.device, dtype=torch.float64)
        if isinstance(value, str):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"{name}: файл '{value}' не найден")
            if path.suffix.lower() == '.npy':
                arr = np.load(path)
            elif path.suffix.lower() == '.npz':
                with np.load(path) as data:
                    arr = data[data.files[0]]
            else:
                raise ValueError(f"{name}: неподдерживаемый формат '{path.suffix}'")
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name}: ожидаемая форма {shape}, получено {arr.shape}")
            return torch.from_numpy(arr).to(self.device)
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name}: ожидаемая форма {shape}, получено {arr.shape}")
            return torch.from_numpy(arr).to(self.device)
        raise ValueError(f"{name}: неподдерживаемый тип {type(value)}")

    def _apply_stochastic_fields(self, cfg, md_to_m2, k_v_fraction):
        seed = cfg.get('seed', 0)
        if 'porosity' in cfg:
            params = cfg['porosity']
            self.porosity = self._generate_random_field(
                params,
                base_mean=float(self.porosity.mean()),
                base_std=float(self.porosity.std() or 0.01),
                seed=seed,
                clamp=(0.01, 0.9),
            )
        if 'permeability' in cfg:
            params = cfg['permeability']
            perm_field_md = self._generate_random_field(
                params,
                base_mean=float((self.permeability_x / md_to_m2).mean()),
                base_std=float((self.permeability_x / md_to_m2).std() or 1.0),
                seed=seed + 1,
                clamp=(params.get('min_md', 0.01), params.get('max_md', None)),
            )
            perm_si = perm_field_md * md_to_m2
            self.permeability_x = perm_si.clone()
            self.permeability_y = perm_si.clone()
            vfrac = params.get('k_vertical_fraction', k_v_fraction)
            self.permeability_z = perm_si * vfrac

    def _generate_random_field(self, params, base_mean, base_std, seed, clamp=None):
        shape = self.dimensions
        rng = np.random.default_rng(params.get('seed', seed))
        noise = rng.standard_normal(shape)
        corr = params.get('corr_length', 0.0)
        if gaussian_filter is not None and corr and corr > 0:
            noise = gaussian_filter(noise, sigma=corr, mode='reflect')
        elif corr and corr > 0:
            # fallback smoothing by simple averaging
            for _ in range(int(max(1, corr))):
                noise = self._simple_smooth(noise)
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        distribution = params.get('distribution', 'normal').lower()
        if distribution == 'lognormal':
            sigma = params.get('log_std', 0.5)
            mean_log = params.get('log_mean', np.log(max(base_mean, 1e-6)))
            field = np.exp(mean_log + sigma * noise)
        else:
            mean = params.get('mean', base_mean)
            std = params.get('std', base_std)
            field = mean + std * noise

        if clamp:
            lo, hi = clamp
            if lo is not None:
                field = np.maximum(field, lo)
            if hi is not None:
                field = np.minimum(field, hi)

        return torch.from_numpy(field.astype(np.float64)).to(self.device)

    @staticmethod
    def _simple_smooth(arr):
        # простое усреднение соседей (numpy) без SciPy
        padded = np.pad(arr, 1, mode='edge')
        result = np.zeros_like(arr, dtype=np.float64)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    result += padded[dx:dx+arr.shape[0], dy:dy+arr.shape[1], dz:dz+arr.shape[2]]
        return result / 27.0

    def _build_geometric_properties(self):
        self.dx_face = 0.5 * (self.dx_vector[:-1] + self.dx_vector[1:]) if self.nx > 1 else torch.zeros(0, device=self.device)
        self.dy_face = 0.5 * (self.dy_vector[:-1] + self.dy_vector[1:]) if self.ny > 1 else torch.zeros(0, device=self.device)
        self.dz_face = 0.5 * (self.dz_vector[:-1] + self.dz_vector[1:]) if self.nz > 1 else torch.zeros(0, device=self.device)

        dx = self.dx_vector.view(self.nx, 1, 1)
        dy = self.dy_vector.view(1, self.ny, 1)
        dz = self.dz_vector.view(1, 1, self.nz)
        self.cell_volume = (dx * dy * dz).to(self.device)
        self.cell_volume_flat = self.cell_volume.reshape(-1)

        self.porous_volume = self.cell_volume * self.porosity

        # Центры ячеек и размер домена
        self.x_edges = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(self.dx_vector, dim=0)])
        self.y_edges = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(self.dy_vector, dim=0)])
        self.z_edges = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(self.dz_vector, dim=0)])
        self.domain_lengths = torch.tensor([
            float(self.x_edges[-1]),
            float(self.y_edges[-1]),
            float(self.z_edges[-1])
        ], device=self.device)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
