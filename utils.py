from argparse import ArgumentParser


def get_arguments():
    """
    Обрабатывает аргументы командной строки.
    :return: Словарь аргументов.
    """
    ap = ArgumentParser()
    ap.add_argument("-d", "--device", type=str, default='mps', help="device: 'mps', 'cuda' or 'cpu'")
    ap.add_argument("-c", "--camera", type=str, default=None, help="path to the optional video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="maximum buffer size for trajectory")
    ap.add_argument("-t", "--timeout", type=int, default=30, help="number of frames before switching to full image search")
    ap.add_argument("-o", "--overlay", type=int, default=50, help="size of the ROI overlay")
    return vars(ap.parse_args())
