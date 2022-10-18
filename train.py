from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=8,
            root="./experiments",
        )
        trainer = Trainer(
            triples="data/msmarco-passage/triples.train.tiny.yqx.tsv",
            queries="data/msmarco-passage/queries.train.tsv",
            collection="data/msmarco-passage/collection.tsv",
            config=config,
        )
        print("finish initialization, start training")
        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")
