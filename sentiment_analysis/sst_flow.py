from metaflow import conda_base, FlowSpec, IncludeFile, Parameter, step


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
    model.eval()
    batch_accuracy_summation, loss, num_batches = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(
            val_loader, desc="Evaluating"
        ):
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


@conda_base(python='3.8.10', libraries={'pandas': '1.3.4',
                                        'transformers': '4.12.3',
                                        'torch': '1.10.1'})
class SSTFlow(FlowSpec):
    """
    Stanford Sentiment Treebank model training, fine-tuned [BERT, ALBERT, DistilBERT].
    """

    config_file = Parameter('config-file',
                            help='Configuration file for experiment',
                            default='bert.yaml')
    
    train_fname = IncludeFile(
        "fname",
        help="The path to sst train file.",
        default=script_path("data/train.tsv"),
    )

    val_fname = IncludeFile(
        "fname",
        help="The path to sst val file.",
        default=script_path("data/dev.tsv"),
    )

    @step
    def start(self):
        self.experiment_config = read_yaml(self.config_file)
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.experiment_config.get('model_name_or_path'))

        train_set = SSTDataset(
            filename=self.train_fname,
            maxlen=self.experiment_config.get('maxlen_train'),
            tokenizer=self.tokenizer,
        )
        val_set = SSTDataset(
            filename=self.val_fname, maxlen=self.experiment_config.get('maxlen_val'), tokenizer=self.tokenizer
        )

        self.train_loader = DataLoader(
            dataset=train_set, batch_size=self.experiment_config.get('batch_size'), num_workers=self.experiment_config.get('num_threads')
        )
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=self.experiment_config.get('batch_size'), num_workers=self.experiment_config.get('num_threads')
        )
        self.next(self.train)
    
    @step
    def train(self):
        import torch
        from transformers import AutoTokenizer, AutoConfig
        from tqdm import tqdm

        from modeling import (
            BertForSentimentClassification,
            AlbertForSentimentClassification,
            DistilBertForSentimentClassification,
        )

        if self.experiment_config.get('model_type') == "bert":
            self.model = BertForSentimentClassification.from_pretrained(
                self.experiment_config.get('model_name_or_path')
            )
        elif elf.experiment_config.get('model_type') == "albert":
            self.model = AlbertForSentimentClassification.from_pretrained(
                self.experiment_config.get('model_name_or_path')
            )
        elif elf.experiment_config.get('model_type') == "distilbert":
            self.model = DistilBertForSentimentClassification.from_pretrained(
                self.experiment_config.get('model_name_or_path')
            )
        else:
            raise ValueError("This transformer model is not supported yet.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.experiment_config.get('lr'))

        best_accuracy = 0
        for epoch in trange(args.num_eps, desc="Epoch"):
            train(
                model=self.model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device
            )
            val_accuracy, val_loss = evaluate(
                model=model, val_loader=val_loader, criterion=criterion, device=device
            )
            print(
                f"Epoch {epoch} complete! Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}"
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(
                    f"Best validation accuracy improved from {best_accuracy} to {val_accuracy}, saving analyzer..."
                )
                with S3(run=self) as s3:
                    model_url = s3.put('best_val_acc_model', self.model)
                    tokenizer_url = s3.put('best_val_acc_tokenizer', self.tokenizer)
        self.next(self.end)

    @step
    def end(self):
        print("model finished training!!")

if __name__ == "__main__":
    SSTFlow()
