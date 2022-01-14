from PIL import Image
import numpy as np
import glob
import random


class ImageGenerator():

    """
        Classe di base per un generatore di immagini
    """
    def __init__(self, n_blocks: int = 1) -> None:
        """
            Costrtuttore di base di un generatore di immagini.

            Parametri:
            - n_blocks: numero di blocchi logici nel quale la sorgente di immagini è suddivisa.
        """
        self.n_blocks = max(1, n_blocks)

    def next(self, block: int = 0):
        """
            Restituisce la prossima immagine.

            Parametri:
            - block: indice del blocco logico dal quale estrarre l'immagine.
        """
        pass

    def next_batch(self, batch_size: int, block: int = 0) -> list:
        """
            Restituisce il prossimo batch di immagini.

            Parametri:
            - batch_size: dimensione del batch di immagini.
            - block: indice del blocco logico dal quale estrarre il batch.
        """
        batch = [None] * batch_size
        for i in range(batch_size):
            batch[i] = self.next(block)
        return batch

    def get_all(self) -> 'list | None':
        """
            Restituisce tutte le immagini.
        """
        pass

    def get_block(self, block: int = 0) -> 'list | None':
        """
            Restituisce tutte le immagini incluse in un blocco.
        """
        pass

    def reset(self) -> None:
        """
            Resetta il generatore
        """
        pass

class BlockNumberError(Exception):
    # TODO: completare
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FSRandomImageGenerator(ImageGenerator):
    """
        Generatore casuale di immagini da filesystem.
    """
    def __init__(self, data_path: "str | list", image_shape: tuple = (128, 128, 1), rescale: float = 1./255, limit: int = None, shuffle: bool = True, n_blocks: int = 1) -> None:
        super().__init__(n_blocks=n_blocks)
        if isinstance(data_path, str):
            self.data_path = [data_path]
        elif isinstance(data_path, list):
            self.data_path = data_path
        else:
            raise ValueError()
        self.image_shape = image_shape
        self.channels = self.image_shape[2]
        self.rescale = rescale

        self.filenames = []
        for path in self.data_path:
            self.filenames.extend(glob.glob(path, recursive = False))
        if shuffle:
            random.shuffle(self.filenames)

        if limit != None:
            self.filenames = self.filenames[:limit]
        
        n_files = len(self.filenames)
        if n_blocks > n_files:
            raise BlockNumberError()

        self.block_size = n_files // n_blocks
        self.last_block_size = n_files % n_blocks

        self.file_idx = [0] * self.n_blocks

        self.images = [None] * self.n_blocks
        self.all_in_memory = [False] * self.n_blocks

    def read_img(self, filename: str) -> np.ndarray:
        if self.channels == 1:
            im = Image.open(filename).convert("L")
        else:
            im = Image.open(filename)
        im = im.resize((self.image_shape[0], self.image_shape[1]))
        im = np.array(list(im.getdata())) * self.rescale
        im = im.reshape(self.image_shape)
        
        return im

    def load_all(self):
        """
            Carica in memoria tutte le immagini presenti nella cartella specificata
        """
        self.images = [[]] * self.n_blocks
        for block in range(self.n_blocks):
            block_start = block * self.block_size
            block_end = (block + 1) * self.block_size
            for filename in self.filenames[block_start : block_end]:
                if self.images[block] == None:
                    self.images[block] = []
                self.images[block].append(self.read_img(filename))

        self.all_in_memory = [True] * self.n_blocks
        self.file_idx = [self.block_size] * self.n_blocks
        self.file_idx[-1] = self.last_block_size

    def load_block(self, block: int):
        """
            Carica in memoria tutte le immagini appartenenti ad un blocco
        """
        if self.all_in_memory[block]:
            return
        
        start = block * self.block_size + self.file_idx[block]
        end = (block + 1) * self.block_size
        for filename in self.filenames[start : end]:
            if self.images[block] == None:
                self.images[block] = []
            self.images[block].append(self.read_img(filename))

        self.all_in_memory[block] = True
        if self.n_blocks > 1 and block == self.n_blocks - 1:
            self.file_idx[block] = self.last_block_size
        else:
            self.file_idx[block] = self.block_size

    def next(self, block: int = 0):
        if self.all_in_memory[block]:
            return self.images[block][random.randint(0, len(self.images[block]) - 1)]
        
        file_idx = block * self.block_size + self.file_idx[block]
        filename = self.filenames[file_idx]
        im = self.read_img(filename)
        if self.images[block] == None:
            self.images[block] = []
        self.images[block].append(im)

        self.file_idx[block] += 1
        if self.file_idx[block] >= self.block_size or \
           (self.n_blocks > 1 and block == self.n_blocks - 1 and self.file_idx[block] >= self.last_block_size):
            self.all_in_memory[block] = True

        return im

    def get_all(self) -> 'list | None':
        if all(self.all_in_memory):
            return [img for block in self.images for img in block]
        self.load_all()
        return [img for block in self.images for img in block]

    def get_block(self, block: int) -> 'list | None':
        if self.all_in_memory[block]:
            return self.images[block]
        self.load_block(block)
        return self.images[block]

    def reset(self):
        self.file_idx = [0] * self.n_blocks
        self.images = [None] * self.n_blocks
        self.all_in_memory = [False] * self.n_blocks