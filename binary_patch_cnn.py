import time
import models, data, options, helpers
from torch.utils.data import DataLoader


args = options.PatchCNNOptions().parse()
classifier = models.PatchCNN(args)
log_file = open(args.log_file, "w") if args.log_file != "" else None

if args.use_cuda:
    classifier.to_cuda()

if args.mode == "train":
    train_loader = DataLoader(
            data.BinaryPatchDataset(args, "train"),
            shuffle=True,
            batch_size=args.batch_size
        )
    val_loader = DataLoader(
            data.BinaryPatchDataset(args, "val"),
            shuffle=True,
            batch_size=args.batch_size
        )

    epoch_num, epochs_since_anneal, anneal_num, epochs_since_improvement, best_val_accuracy = 0, 0, 0, 0, 0.0
    anneal_lr_accuracies = []

    while True:
        start_time = time.time()
        epoch_num += 1
        epochs_since_anneal += 1
        epochs_since_improvement += 1
        total_val_accuracy = 0.0
        anneal_lr_accuracy_change = 0.0

        #train
        print("\nBeginning Training Epoch %d\n" % epoch_num, file=log_file)

        classifier.train()

        for batch_num, (patches, targets) in enumerate(train_loader):
            classifier.train_on_batch(epoch_num, batch_num, len(train_loader), patches, targets, log_file=log_file)

        print("\nEpoch time: " +  str(int((time.time() - start_time) // 60)) + " minutes", file=log_file)

        #validate
        print("\nBeginning Validation\n", file=log_file)

        classifier.eval()

        for patches, targets in val_loader:
            preds = classifier.classify(patches)
            batch_results = helpers.get_topk_accuracies(preds, targets, top_k=(1,))
            total_val_accuracy += batch_results[0]

        avg_val_accuracy = total_val_accuracy / len(val_loader)

        if epoch_num >= args.anneal_lr_epochs:
            anneal_lr_accuracies.pop(0)

        anneal_lr_accuracies.append(avg_val_accuracy)
        anneal_lr_accuracy_change = anneal_lr_accuracies[-1] - anneal_lr_accuracies[0]

        print("Epoch %d Val Accuracy: %.4f" % (epoch_num, avg_val_accuracy), file=log_file)
        print("%d Epochs Val Accuracy Change: %.4f" % (args.anneal_lr_epochs, anneal_lr_accuracy_change), file=log_file)

        #early stopping
        if epochs_since_improvement >= args.early_stopping_epochs and anneal_num >= args.min_num_anneals \
                and avg_val_accuracy < (best_val_accuracy + args.early_stopping_threshold):
            print("\nValidation accuracy has not improved in last %d epochs, training has ended!" \
                % args.early_stopping_epochs, file=log_file)
            break

        #save weights
        if avg_val_accuracy > best_val_accuracy:
            epochs_since_improvement = 0
            classifier.save_weights()
            best_val_accuracy = avg_val_accuracy
            print("New best accuracy achieved, saving weights!", file=log_file)

        #anneal lr
        if epochs_since_anneal > args.anneal_lr_epochs and anneal_lr_accuracy_change < args.anneal_lr_threshold:
            epochs_since_anneal = 0
            anneal_num += 1
            classifier.anneal_lr(log_file=log_file)
else:
    test_loader = DataLoader(
            data.BinaryPatchDataset(args, "test"),
            shuffle=True,
            batch_size=args.batch_size
        )

    classifier.eval()
    total_top1 = 0.0

    for patches, targets in test_loader:
        preds = classifier.classify(patches)
        batch_results = helpers.get_topk_accuracies(preds, targets, top_k=(1,))
        total_top1 += batch_results[0]

    print("Top 1 Accuracy: %.4f" % (total_top1 / len(test_loader)), file=log_file)

if log_file != None:
    log_file.close()
