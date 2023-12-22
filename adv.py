from typing import Any, Dict, Iterator, List, Optional, cast

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _get_params(optimizer: Optimizer) -> Iterator[Tensor]:
    for param_group in cast(List[Dict[Any, Any]], optimizer.param_groups):
        for param in param_group["params"]:
            if not isinstance(param, Tensor):
                raise TypeError(f"expected Tensor, but got: {type(param)}")
            yield param


def _get_loss(step_output: STEP_OUTPUT) -> Optional[Tensor]:
    if step_output is None:
        return None
    if isinstance(step_output, Tensor):
        return step_output
    return step_output.get("loss")

class FGM(torch.nn.Module):
    def __init__(self, model, emb_name="word_embeddings"):
        super().__init__()
        self.model = model
        self.backup = {}
        self.emb_name = emb_name
    
    def forward(self ,x):
        return self.model(x)

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGMCallBack(Callback):
    _batch: Any
    _batch_idx: int

    def __init__(self, emb_name="word_embeddings") -> None:
        super().__init__()
        self.emb_name = emb_name

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self._batch = batch
        self._batch_idx = batch_idx

    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        # org_weights = self._first_step(optimizer)
        with torch.enable_grad():
            step_output = pl_module.training_step(self._batch, self._batch_idx)
            loss = _get_loss(step_output)
            if loss is not None:
                trainer.accelerator.backward(
                    loss, optimizer=optimizer, optimizer_idx=opt_idx
                )
            pl_module.model.attack()
            adv_loss = _get_loss(pl_module.training_step(self._batch, self._batch_idx))
            trainer.accelerator.backward(
                adv_loss, optimizer=optimizer, optimizer_idx=opt_idx
            )
            pl_module.model.restore()


    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.model = FGM(model=pl_module.model)
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.model = pl_module.model.model



if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning import LightningModule, Trainer



    class RandomDataset(Dataset):
        def __init__(self, size, num_samples):
            self.data = torch.randn(num_samples, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.word_embeddings = torch.nn.Linear(32, 2)
            self.mlp = torch.nn.Linear(2,2)
        
        def forward(self, x):
            return self.mlp(self.word_embeddings(x))

    class BoringModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = Net()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            return self(batch).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.1)


    model = BoringModel()
    # trainer = Trainer(max_epochs=3, callbacks=[FGMCallBack()])
    trainer = Trainer(max_epochs=3, callbacks=[])
    trainer.fit(model, train_dataloaders=DataLoader(RandomDataset(32, 64), batch_size=2))