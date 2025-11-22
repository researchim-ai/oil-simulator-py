import torch
import numpy as np

class SparseMatrixBuilder:
    """
    Помощник для создания структуры разреженной матрицы (CSR) для 3D сетки.
    Строит indices и indptr для 7-точечного шаблона (Heptadiagonal).
    
    Также создает маппинги для векторизованного заполнения (assembly):
    - diag_indices: индексы в values[], где лежат диагональные блоки
    - ax_pos_indices: индексы связей i -> i+1 (X+)
    - ax_neg_indices: индексы связей i -> i-1 (X-)
    и так далее для Y и Z.
    """
    def __init__(self, nx, ny, nz, block_size=1, device='cpu'):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n_cells = nx * ny * nz
        self.block_size = block_size
        self.device = device
        
        # Структура CSR
        self.indices = None
        self.indptr = None
        self.n_nonzero = 0
        
        # Маппинги для Assembly (Tensor int64)
        # Размерность маппингов равна числу ячеек (n_cells), кроме граничных - там будет -1
        # То есть diag_indices[idx] дает позицию блока (idx, idx) в массиве values.
        self.diag_map = None
        self.x_pos_map = None # (idx, idx+1)
        self.x_neg_map = None # (idx, idx-1)
        self.y_pos_map = None
        self.y_neg_map = None
        self.z_pos_map = None
        self.z_neg_map = None
        
        self._build_structure()

    def _build_structure(self):
        """
        Строит CSR структуру и маппинги.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        n_cells = self.n_cells
        
        # Списки для CSR
        indptr_list = [0]
        indices_list = []
        
        # Списки для маппингов (будем заполнять по ходу)
        # Храним глобальный индекс в values для каждой связи каждой ячейки
        diag_map = np.full(n_cells, -1, dtype=np.int64)
        x_pos_map = np.full(n_cells, -1, dtype=np.int64)
        x_neg_map = np.full(n_cells, -1, dtype=np.int64)
        y_pos_map = np.full(n_cells, -1, dtype=np.int64)
        y_neg_map = np.full(n_cells, -1, dtype=np.int64)
        z_pos_map = np.full(n_cells, -1, dtype=np.int64)
        z_neg_map = np.full(n_cells, -1, dtype=np.int64)
        
        current_offset = 0
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = k * (nx * ny) + j * nx + i
                    
                    # Собираем соседей и их смещения
                    # Порядок важен для CSR (возрастание индексов столбцов)
                    neighbors = []
                    
                    # Z- (k-1) -> idx - nx*ny
                    if k > 0: neighbors.append((idx - nx*ny, 'zm'))
                    # Y- (j-1) -> idx - nx
                    if j > 0: neighbors.append((idx - nx, 'ym'))
                    # X- (i-1) -> idx - 1
                    if i > 0: neighbors.append((idx - 1, 'xm'))
                    
                    # Self
                    neighbors.append((idx, 'diag'))
                    
                    # X+ (i+1) -> idx + 1
                    if i < nx - 1: neighbors.append((idx + 1, 'xp'))
                    # Y+ (j+1) -> idx + nx
                    if j < ny - 1: neighbors.append((idx + nx, 'yp'))
                    # Z+ (k+1) -> idx + nx*ny
                    if k < nz - 1: neighbors.append((idx + nx*ny, 'zp'))
                    
                    # Сортировка не нужна, так как мы добавляем строго в порядке возрастания индексов
                    # (idx-N < idx-1 < idx < idx+1 < idx+N)
                    # Проверим порядок:
                    # zm (-N^2), ym (-N), xm (-1), diag (0), xp (+1), yp (+N), zp (+N^2)
                    # Все верно, порядок возрастающий.
                    
                    for col_idx, type_str in neighbors:
                        indices_list.append(col_idx)
                        
                        # Записываем маппинг
                        if type_str == 'diag': diag_map[idx] = current_offset
                        elif type_str == 'xm': x_neg_map[idx] = current_offset
                        elif type_str == 'xp': x_pos_map[idx] = current_offset
                        elif type_str == 'ym': y_neg_map[idx] = current_offset
                        elif type_str == 'yp': y_pos_map[idx] = current_offset
                        elif type_str == 'zm': z_neg_map[idx] = current_offset
                        elif type_str == 'zp': z_pos_map[idx] = current_offset
                        
                        current_offset += 1
                    
                    indptr_list.append(current_offset)
        
        # Конвертируем в тензоры
        self.indices = torch.tensor(indices_list, dtype=torch.long, device=self.device)
        self.indptr = torch.tensor(indptr_list, dtype=torch.long, device=self.device)
        self.n_nonzero = len(indices_list)
        
        # Маппинги тоже на GPU
        self.diag_map = torch.tensor(diag_map, dtype=torch.long, device=self.device)
        self.x_pos_map = torch.tensor(x_pos_map, dtype=torch.long, device=self.device)
        self.x_neg_map = torch.tensor(x_neg_map, dtype=torch.long, device=self.device)
        self.y_pos_map = torch.tensor(y_pos_map, dtype=torch.long, device=self.device)
        self.y_neg_map = torch.tensor(y_neg_map, dtype=torch.long, device=self.device)
        self.z_pos_map = torch.tensor(z_pos_map, dtype=torch.long, device=self.device)
        self.z_neg_map = torch.tensor(z_neg_map, dtype=torch.long, device=self.device)

    def get_bsr_tensor(self, block_values):
        """
        Создает BSR тензор PyTorch.
        block_values: Tensor [n_nonzero, B, B]
        """
        return torch.sparse_bsr_tensor(
            self.indptr, 
            self.indices, 
            block_values, 
            size=(self.n_cells * self.block_size, self.n_cells * self.block_size),
            dtype=torch.float64
        )

if __name__ == "__main__":
    # Тест
    nx, ny, nz = 3, 3, 3
    builder = SparseMatrixBuilder(nx, ny, nz, block_size=2)
    print(f"Grid {nx}x{ny}x{nz} ({nx*ny*nz} cells)")
    print(f"NNZ blocks: {builder.n_nonzero}")
    print(f"Diag map sample: {builder.diag_map[:5]}")
