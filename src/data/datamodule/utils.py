from pytorch_lightning import LightningDataModule

from .msra_chn_ner import MsraCHNNERDataModule

_AVALIABLE_DATAMODULE = {
    "msra_chn": MsraCHNNERDataModule,
}


def get_datamodule(
    datamodule_name: str,
    **datamodule_kwargs
) -> LightningDataModule:
    if (datamodule := _AVALIABLE_DATAMODULE.get(datamodule_name)) is not None:
        return datamodule(**datamodule_kwargs)

    raise ValueError(f"datamodule `{datamodule_name}` is not supported!")