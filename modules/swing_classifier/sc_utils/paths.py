"""Path Configs"""
from datetime import datetime
from pathlib import Path

class Paths:
    """Paths class which contains logidr, summary_dir, ckpt_dir, metric_dir"""
    
    def __init__(self, logdir, summary_dir, ckpt_dir, metric_dir):
        self.logdir = logdir
        self.summary_dir = summary_dir
        self.ckpt_dir = ckpt_dir
        self.metric_dir = metric_dir
    
    @classmethod
    def make_dirs(cls, logdir: str) -> object:
        """Make log directory
        Parameters
        ----------
        logdir : str
            parent log directory
        Returns
        -------
        Tuple of pathlib.Path
            directory paths
        """
        now = datetime.now().isoformat()

        # log parent directory
        logdir = Path(logdir) / now
        logdir.mkdir(exist_ok=True, parents=True)

        # summary directory
        summary_dir = logdir / 'tensorboard'
        summary_dir.mkdir(exist_ok=True)

        # checkpoint directory
        ckpt_dir = logdir / 'ckpt'
        ckpt_dir.mkdir(exist_ok=True)

        # metric directory
        metric_dir = logdir / 'metric'
        metric_dir.mkdir(exist_ok=True)

        return cls(logdir, summary_dir, ckpt_dir, metric_dir)