from metaflow import conda_base, FlowSpec, IncludeFile, Parameter, step


@conda_base(python='3.8.10', libraries={'pandas': '1.3.4',
                                        'transformers': '4.12.3'})
class SSTFlow(FlowSpec):
    """
    Stanford Sentiment Treebank model training, fine-tuned [BERT, ALBERT, DistilBERT].
    """

    model_type = Parameter("model-type", default="bert")
    
    train_fname = IncludeFile(
        "fname",
        help="The path to sst train file.",
        default=utils.script_path("data/train.tsv"),
    )

    val_fname = IncludeFile(
        "fname",
        help="The path to sst val file.",
        default=utils.script_path("data/dev.tsv"),
    )

    @step
    def start(self):
        train_set = SSTDataset(
            filename=self.train_fname,
            maxlen=args.maxlen_train,
            tokenizer=analyzer.tokenizer,
        )

        val_set = SSTDataset(
            filename=self.val_fname, maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
        )

        self.train_loader = DataLoader(
            dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads
        )
        self.val_loader = DataLoader(
            dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads
        )
        self.next(self.train)
    
    @step
    def train(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=analyzer.model.parameters(), lr=args.lr)

        best_accuracy = 0
        for epoch in trange(args.num_eps, desc="Epoch"):
            analyzer.train(
                train_loader=train_loader, optimizer=optimizer, criterion=criterion
            )
            val_accuracy, val_loss = analyzer.evaluate(
                val_loader=val_loader, criterion=criterion
            )
            print(
                f"Epoch {epoch} complete! Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}"
            )
            # Save analyzer if validation accuracy imporoved.
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(
                    f"Best validation accuracy improved from {best_accuracy} to {val_accuracy}, saving analyzer..."
                )
                # analyzer.save()

