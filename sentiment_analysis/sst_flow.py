from metaflow import conda_base, FlowSpec, IncludeFile, Parameter, step, S3


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.
    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def evaluate(model, val_loader, criterion, device):
    import torch
    from tqdm import tqdm
    from utils import get_accuracy_from_logits, read_yaml

    
    model.eval()
    batch_accuracy_summation, loss, num_batches = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_accuracy_summation += get_accuracy_from_logits(logits, labels)
            loss += criterion(logits.squeeze(-1), labels.float()).item()
            num_batches += 1
    accuracy = batch_accuracy_summation / num_batches
    return accuracy.item(), loss


def train(model, train_loader, optimizer, criterion, device):
    import torch
    from tqdm import tqdm
    
    model.train()
    for input_ids, attention_mask, labels in tqdm(
        iterable=train_loader, desc="Training"
    ):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(input=logits.squeeze(-1), target=labels.float())
        loss.backward()
        optimizer.step()


@conda_base(
    python="3.8.10",
    libraries={"pandas": "1.3.4", "transformers": "4.12.3", "pytorch-gpu": "1.10.1", "pyyaml": "6.0.0"},
)
class SSTFlow(FlowSpec):
    """
    Stanford Sentiment Treebank model training, fine-tuned [BERT, ALBERT, DistilBERT].
    """

    config_file = Parameter(
        "config-file", help="Configuration file for experiment", default="config/bert-sst.yaml"
    )

    train_fname = Parameter(
        "train-path", help="The path to sst train file", default="data/sst/train.tsv"
    )

    val_fname = Parameter(
        "val-path", help="The path to sst val file", default="data/sst/dev.tsv"
    )
    # train_fname = IncludeFile(
    #     "train_fname",
    #     help="The path to sst train file.",
    #     default=script_path("data/sst/train.tsv"),
    # )

    # val_fname = IncludeFile(
    #     "val_fname",
    #     help="The path to sst val file.",
    #     default=script_path("data/sst/dev.tsv"),
    # )

    @step
    def start(self):
        from utils import get_accuracy_from_logits, read_yaml

        self.experiment_config = read_yaml(self.config_file)
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
        from datasets.sst import SSTDataset

        model_name_or_path = self.experiment_config.get("model_name_or_path")
        maxlen_train = int(self.experiment_config.get("maxlen_train"))
        maxlen_val = int(self.experiment_config.get("maxlen_val"))
        batch_size = int(self.experiment_config.get("batch_size"))
        num_threads = int(self.experiment_config.get("num_threads"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

        train_set = SSTDataset(
            file_path=self.train_fname,
            maxlen=maxlen_train,
            tokenizer=self.tokenizer,
        )
        print('loaded train_set')
        val_set = SSTDataset(
            file_path=self.val_fname,
            maxlen=maxlen_val,
            tokenizer=self.tokenizer,
        )
        print('loaded val_set')

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_threads,
        )
        self.val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            num_workers=num_threads,
        )
        self.next(self.train)

    @step
    def train(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm, trange
        
        from modeling import (
            BertForSentimentClassification,
            AlbertForSentimentClassification,
            DistilBertForSentimentClassification,
        )

        model_type = self.experiment_config.get("model_type")
        model_name_or_path = self.experiment_config.get("model_name_or_path")
        lr = float(self.experiment_config.get("lr"))
        num_eps = int(self.experiment_config.get("num_eps"))

        if model_type == "bert":
            self.model = BertForSentimentClassification.from_pretrained(
                model_name_or_path
            )
        elif model_type == "albert":
            self.model = AlbertForSentimentClassification.from_pretrained(
                model_name_or_path
            )
        elif model_type == "distilbert":
            self.model = DistilBertForSentimentClassification.from_pretrained(
                model_name_or_path
            )
        else:
            raise ValueError(f'"{model_type}" transformer model is not supported yet.')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            params=self.model.parameters(), lr=lr
        )

        best_accuracy = 0
        for epoch in trange(num_eps, desc="Epoch"):
            train(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_accuracy, val_loss = evaluate(
                model=self.model, val_loader=self.val_loader, criterion=criterion, device=device
            )
            print(
                f"Epoch {epoch} complete! Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}"
            )

            if val_accuracy > best_accuracy:
                print(
                    f"Best validation accuracy improved from {best_accuracy} to {val_accuracy} ..."
                )
                best_accuracy = val_accuracy
        self.next(self.end)

    @step
    def end(self):
        print("model finished training!!")


if __name__ == "__main__":
    SSTFlow()
