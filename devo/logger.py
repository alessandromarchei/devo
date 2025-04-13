import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, name, scheduler, total_steps=0, step=1, tensorboard_update_step=100, args_config=None):
        self.name = name
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.step = step
        self.tensorboard_update_step = tensorboard_update_step
        self.args_config = args_config
        self.running_loss = {}
        self.writer = None
        self._config_logged = False  # Ensures config is logged only once

    def _init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(f"runs/{self.name}")
            self._log_config_once()
            print(f"TensorBoard writer initialized for run: {self.name}")

    def _log_config_once(self):
        if self.writer is not None and self.args_config is not None and not self._config_logged:
            # Manually convert Namespace to string
            config_text = '\n'.join(f"{k}: {v}" for k, v in vars(self.args_config).items())
            config_text = config_text.replace("\n", "  \n")  # Markdown-friendly for TensorBoard
            self.writer.add_text("config", f"```bash\n{config_text}\n```", global_step=0)
            print("Logged config to TensorBoard.")
            self._config_logged = True


    def _print_training_status(self):
        self._init_writer()

        lr = self.scheduler.get_last_lr()[-1]
        metrics_data = [self.running_loss[k] / self.tensorboard_update_step for k in self.running_loss]
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        training_str = f"[{self.total_steps * self.step + 1:6d}, {lr:10.7f}] "

        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / self.tensorboard_update_step
            self.writer.add_scalar(key, val, self.total_steps * self.step)
            self.running_loss[key] = 0.0

        self.writer.add_scalar("lr", lr, self.total_steps * self.step)

    def push(self, metrics):
        for key, value in metrics.items():
            self.running_loss[key] = self.running_loss.get(key, 0.0) + value

        if self.total_steps % self.tensorboard_update_step == self.tensorboard_update_step - 1:
            self._print_training_status()

        self.total_steps += 1

    def write_dict(self, results):
        self._init_writer()

        for key, value in results.items():
            self.writer.add_scalar(key, value, self.total_steps * self.step)

    def write_figures(self, figures):
        self._init_writer()

        for key, figure in figures.items():
            self.writer.add_figure(key, figure, self.total_steps * self.step)

    def close(self):
        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed.")
